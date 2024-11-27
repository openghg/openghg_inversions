from collections import defaultdict
from copy import deepcopy
import inspect
from pathlib import Path
import re
from types import UnionType
import typing
from typing import Any, Sequence, Union

from openghg_inversions.models.components import ModelComponent

try:
    import tomllib as toml
except ModuleNotFoundError:
    import pip._vendor.tomli as toml

import matplotlib.pyplot as plt
import networkx as nx
import pymc as pm


class Node:
    def __init__(self, name):
        self.name = name
        self.type = "default"
        self.children = []
        self.inputs = []

    def __str__(self):
        return f"{self.name}, type={self.type}, children={self.children}, inputs={self.inputs}"

    def __repr__(self):
        return f"Node({self.__str__()})"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return self.name == other.name

    def __lt__(self, other) -> bool:
        return self.name in other.children or self.name in other.inputs


def is_node(conf: Any, name: str | None = None, cache: dict[str, bool] | None = None) -> bool:

    if cache is not None and name is not None and name in cache:
        return cache[name]

    if not isinstance(conf, dict):
        result = False
    elif "type" in conf:
        result = True
    else:
        prefix = "" if name is None else name + "."
        result = any(is_node(v, name=prefix + k, cache=cache) for k, v in conf.items() if isinstance(v, dict))

    if cache is not None and name is not None:
        cache[name] = result

    return result


def get_children(conf: dict, name: str | None = None, cache: dict[str, bool] | None = None) -> list[str]:
    prefix = "" if name is None else name + "."
    result = [
        prefix + k
        for k, v in conf.items()
        if isinstance(v, dict) and is_node(v, name=prefix + k, cache=cache)
    ]

    return result


def get_nodes(conf: dict) -> list[Node]:
    cache = {}

    # depth first search for nodes
    stack = [deepcopy(conf)]
    nodes = []

    while stack:
        top = stack.pop()

        for name, values in top.items():
            if is_node(values, name, cache):
                node = Node(name)
                node.children = get_children(values, name, cache)

                for child in node.children:
                    # append dict in the form {name: value}
                    # take last part of child name, since it includes all ancestors' names
                    stack.append({child: values.pop(child.split(".")[-1])})

                if "inputs" in values:
                    node.inputs = values.pop("inputs")

                if "type" in values:
                    node.type = values.pop("type")

                # store all remaining values in node
                node.__dict__.update(values)

                nodes.append(node)

    return nodes


def prepare_graph(nodes: list[Node]):
    nodes_hash = {node.name: node for node in nodes}
    edges = defaultdict(list)
    for node in nodes:
        for child in node.children:
            edges[nodes_hash[child]].append(node)
        for input_ in node.inputs:
            edges[nodes_hash[input_]].append(node)
    return edges


def make_nx_graph(config: dict) -> nx.DiGraph:
    graph_nodes = get_nodes(config)
    graph = prepare_graph(graph_nodes)
    G = nx.DiGraph()
    G.add_nodes_from(graph_nodes)
    G.add_edges_from((k, x) for k, v in graph.items() for x in v)
    return G


def plot_graph(G, title="", show_types=False, show_inputs=False, edge_labels=None, **kwargs):
    for layer, nodes in enumerate(nx.topological_generations(G)):
    # `multipartite_layout` expects the layer as a node attribute, so add the
    # numeric layer value as a node attribute
        for node in sorted(nodes, key=lambda x: x.name):
            G.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
    pos = dict(sorted(pos.items(), key=lambda x: (x[1][1], x[1][0])))
    layers = {v[1] for v in pos.values()}
    for layer in layers:
        nodes, positions = zip(*((k, v) for k, v in pos.items() if v[1] == layer))
        pos.update(dict(zip(sorted(nodes, key=lambda x: x.name), positions)))
    fig, ax = plt.subplots()
    options = dict(node_shape="s",  node_color="none", bbox=dict(facecolor="white", edgecolor='black', boxstyle='round,pad=0.2'))
    options.update(kwargs)
    if show_types:
        node_labels = {node: f"{node.name}\ntype: {node.type}" for node in G.nodes}
    else:
        node_labels = {node: node.name for node in G.nodes}
    nx.draw_networkx(G, pos=pos, ax=ax, labels=node_labels, **options)  # type: ignore
    if show_inputs:
        edge_labels = edge_labels or {}
        for edge in G.edges:
            if edge[0].name in edge[1].inputs:
                edge_labels[edge] = "input"
    if edge_labels is not None:
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def nodes_to_components(G: nx.DiGraph, topo_sort: bool = True) -> dict[Node, type[ModelComponent]]:
    components = {}
    pat = re.compile(r"\s+")

    if topo_sort:
        nodes = nx.lexicographical_topological_sort(G, key=lambda node: node.name)
    else:
        nodes = G.nodes

    for node in nodes:
        node_type = node.type
        node_type = pat.sub("_", node_type)
        try:
            components[node] = ModelComponent._component_registry[node_type]
        except KeyError:
            raise TypeError(f"No `ModelComponent` registered with name {node_type}.")

    return components


def _valid_arg(type_name, ann) -> bool:
    if typing.get_origin(ann) is UnionType:
        return any((type_name is sub_ann) or (type_name.__name__ == sub_ann.__name__) for sub_ann in typing.get_args(ann))
    return (type_name is ann) or (type_name.__name__ == ann.__name__)


def create_components(G: nx.DiGraph, data_dict: dict, verbose: bool = False) -> dict[Node, ModelComponent]:
    """Create instantiated model components.

    Example args:

    data_dict = {"forward.flux": {"h_matrix": flux_h_matrix},
                 "forward.baseline.bc": {"h_matrix": bc_h_matrix},
                 "likelihood": {"y_obs": y_obs, "error": error, "site_indicator": site_indicator},
                 "forward.baseline.offset": {"site_indicator": site_indicator, "prior_args": {"pdf": "normal", "mu": 0.0, "sigma": 1.0}}}

    """
    component_type_dict = nodes_to_components(G)

    component_dict = {}

    for node, comp_type in component_type_dict.items():
        print("processing component for node", node.name)
        node_type = re.sub(r"\s+", "_", node.type)
        if hasattr(node, f"use_{node_type}") and not getattr(node, f"use_{node_type}"):
            try:
                comp = comp_type()
            except TypeError:
                comp = None
            component_dict[node] = comp
        else:
            kwargs = data_dict.get(node.name, {}).copy()

            children = {k: v for k, v in component_dict.items() if k.name in node.children}

            # try to infer parameter names for children
            # TODO: need a way to specify this explicitly
            annotations = inspect.get_annotations(comp_type.__init__, eval_str=True)
            del annotations["return"]

            parameters = inspect.signature(comp_type.__init__).parameters
            parameter_values = list(parameters.values())

            var_positionals = [param.name for param in parameter_values if param.kind == param.VAR_POSITIONAL]

            remaining_children = {}
            for k, child in children.items():
                for param, ann in annotations.items():
                    if param not in var_positionals and _valid_arg(type(child), ann):
                        kwargs[param] = child
                        break
                else:  # no keyword found
                    remaining_children[k] = child


            # remaining children need to be positional args; check this is valid
            if var_positionals:
                var_pos_type = annotations[var_positionals[0]]
                bad_remaining_children = [k.name for k, child in remaining_children.items() if not isinstance(child, var_pos_type)]

                if bad_remaining_children:
                    raise ValueError(f"The following child nodes must be specified by name: {', '.join(brc.name for brc in bad_remaining_children)}")
            elif remaining_children:
                raise ValueError(f"The following child nodes  must be specified by name: {', '.join(rc.name for rc in remaining_children)}")


            kwargs.update(node.__dict__.get("options", {}))

            # grab any other arguments available
            for param in parameter_values:
                if param.name in kwargs:
                    continue

                try:
                    val = getattr(node, param.name)
                except AttributeError:
                    continue
                else:
                    if param.name == "name":
                        val = val.split(".")[-1]  # if name has dots, we only want the last part
                    kwargs[param.name] = val

            # separate any positional args that must come before variable args
            args = []
            if var_positionals:
                for k, param in parameters.items():
                    if k in var_positionals:
                        break
                    if k in kwargs and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                        args.append(kwargs.pop(k))

            # if node has inputs, get the built components
            if node.inputs:
                kwargs["inputs"] = [v for k, v in component_dict.items() if k.name in node.inputs]

            # try to fix var args order
            if remaining_children:
                full_build_order = nx.topological_sort(G)
                temp = {}
                for n in full_build_order:
                    if n in remaining_children:
                        temp[n] = remaining_children[n]
                remaining_children = temp

            if verbose:
                print_kwargs = {}
                for k, v in kwargs.items():
                    import numpy as np
                    import pandas as pd
                    if isinstance(v, pd.Series | pd.Index | np.ndarray):
                        print_kwargs[k] = type(v)
                    else:
                        print_kwargs[k] = v
                print("args:", args, "var args:", [k.name for k in remaining_children.keys()], "kwargs:", print_kwargs)
                print()

            component_dict[node] = comp_type(*args, *remaining_children.values(), **kwargs)

    return component_dict


def build_model(G: nx.DiGraph, component_dict: dict[Node, ModelComponent], verbose: bool = False) -> pm.Model:
    all_children = set()
    for node in G.nodes:
        all_children = all_children.union(set(node.children))

    build_order = list(nx.topological_sort(G))

    if verbose:
        print("Full build order:")
        for i, k in enumerate(build_order):
            print(f"{i + 1}) {k.name}")
        print()

    to_build = {k: component_dict[k] for k in build_order if k.name not in all_children}

    if verbose:
        print("Build order:")
        for i, k in enumerate(to_build.keys()):
            print(f"{i + 1}) {k.name}")
        print()

    with pm.Model() as model:
        for node, component in to_build.items():
            if verbose:
                print("Building component", node.name)

            inputs = [comp for k, comp in component_dict.items() if k.name in node.inputs]

            if inputs and verbose:
                print("\tcomponent inputs:", [comp.name for comp in inputs])

            component.build(*inputs)

    return model


def model_from_config(config_path: str | Path, data_dict: dict, verbose: bool = False) -> pm.Model:
    config_path = Path(config_path)

    with open(config_path, "rb") as f:
        config = toml.load(f)

    if "model" in config:
        G = make_nx_graph(config["model"])
    else:
        G = make_nx_graph(config)

    comp_dict = create_components(G, data_dict, verbose)

    return build_model(G, comp_dict, verbose)
