from __future__ import annotations

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
    def __init__(self, name: str, type_: str = "default", children: list[str] | None = None, inputs: list[str] | None = None, **kwargs) -> None:
        self.name = name
        self.type = type_
        self.children_names = children or []
        self.input_names = inputs or []

        self.__dict__.update(kwargs)

        try:
            self.component_type = ModelComponent._component_registry[self.type]
        except KeyError:
            raise TypeError(f"No `ModelComponent` registered with name {self.type}.")

        # info on required arguments
        self.annotations = inspect.get_annotations(self.component_type.__init__, eval_str=True)
        if "return" in self.annotations:
            del self.annotations["return"]

        self.signature =  inspect.signature(self.component_type)
        self.parameters = self.signature.parameters

        var_args = [param.name for param in self.parameters.values() if param.kind == param.VAR_POSITIONAL]
        self.var_args = var_args[0] if var_args else None


        # these attributes will be set when the model graph is created.
        self.children = []
        self.inputs = []


    def __str__(self):
        return self.name

    def __repr__(self):
        # repr_str = f"Node('{self.name}', type='{self.type}'"

        # if self.children_names:
        #     repr_str += f", children={self.children_names}"

        # if self.input_names:
        #     repr_str += f", inputs={self.input_names}"

        # return repr_str + ")"

        return f"Node({self.name})"  # full repr is too long for debugging...

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Node):
            return False
        return self.name == other.name

    def __lt__(self, other) -> bool:
        return self.name in other.children or self.name in other.inputs

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, value: str) -> None:
        self._type = re.sub(r"\s+", "_", value)

    @property
    def short_name(self) -> str:
        return self.name.split(".")[-1]

    def check_component_parameters(self, verbose: bool = False, partial: bool = True) -> inspect.BoundArguments:
        args = []
        var_args = []
        kwargs = {}
        if self.var_args:
            past_var_args = False

            for k, param in self.parameters.items():
                if k in self.var_args:
                    past_var_args = True
                elif hasattr(self, k):
                    if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD) and not past_var_args:
                        args.append(getattr(self, k))
                    else:
                        kwargs[k] = getattr(self, k)

            var_args = [child for child in self.children if child not in list(kwargs.values())]

        else:
            for k, param in self.parameters.items():
                if hasattr(self, k):
                    kwargs[k] = getattr(self, k)

        remaining_children = {}
        kwarg_values = list(kwargs.values())
        for child in self.children:
            if child not in kwarg_values and child not in var_args:
                for param, ann in self.annotations.items():
                    if param != self.var_args and _valid_arg(child.component_type, ann):
                        kwargs[param] = child
                        break
                else:  # no keyword found
                    remaining_children[child.name] = child

        if verbose:
            print(self.name, args, var_args, kwargs)
            if remaining_children:
                print("remaining children:", ", ".join(rc for rc in remaining_children))

        if partial:
            return self.signature.bind_partial(*args, *var_args, **kwargs)

        return self.signature.bind(*args, *var_args, **kwargs)


    @property
    def missing_params(self) -> tuple[set[str], set[str]]:
        required = []
        optional = []

        params_we_have = self.check_component_parameters(partial=True).arguments

        for param in self.parameters.values():
            if param.name not in params_we_have:
                if param.kind != param.VAR_POSITIONAL and param.default == param.empty:
                    required.append(param.name)
                else:
                    optional.append(param.name)

        return set(required), set(optional)

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
                node_kwargs = {}
                if "inputs" in values:
                    node_kwargs["inputs"] = values.pop("inputs")

                if "type" in values:
                    node_kwargs["type_"] = values.pop("type")

                node = Node(name, **node_kwargs)
                node.children_names = get_children(values, name, cache)

                for child in node.children_names:
                    # append dict in the form {name: value}
                    # take last part of child name, since it includes all ancestors' names
                    stack.append({child: values.pop(child.split(".")[-1])})

                # store all remaining values in node
                node.__dict__.update(values)

                nodes.append(node)

    return nodes


def make_nx_graph(config: dict) -> nx.DiGraph:
    nodes = get_nodes(config)
    nodes_hash = {node.name: node for node in nodes}

    child_edges = defaultdict(list)
    input_edges = defaultdict(list)

    for node in nodes:
        for child in node.children_names:
            child_edges[nodes_hash[child]].append(node)
        for input_ in node.input_names:
            input_edges[nodes_hash[input_]].append(node)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    G.add_edges_from(((k, x) for k, v in child_edges.items() for x in v), kind="child")
    G.add_edges_from(((k, x) for k, v in input_edges.items() for x in v), kind="input")

    for node in G.nodes:
        for pred in G.predecessors(node):
            kind = G.edges[pred, node]["kind"]

            if kind == "child":
                node.children.append(pred)

            if kind == "input":
                node.inputs.append(pred)

    # put child nodes into build order
    build_order = list(nx.topological_sort(G))

    for node in G.nodes:
        temp = []
        for bnode in build_order:
            if bnode in node.children:
                temp.append(bnode)
        node.children = temp

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
    nodes = nx.lexicographical_topological_sort(G, key=lambda node: node.name)

    component_dict = {}

    for node in nodes:
        comp_type = node.component_type

        print("processing component for node", node.name)

        # try to handle nodes that should be skipped (e.g. if `use_bc = False`)
        if hasattr(node, f"use_{node.type}") and not getattr(node, f"use_{node.type}"):
            try:
                comp = comp_type()
            except TypeError:
                comp = None
            component_dict[node] = comp
        else:
            node_params = node.check_component_parameters(partial=True)
            missing_required, missing_optional = node.missing_params

            # TODO: add "options" from node?
            node_data = data_dict.get(node.name, {}).copy()
            node_data_keys = set(node_data.keys())

            if not missing_required.issubset(node_data_keys):
                still_missing = missing_required - node_data_keys
                raise ValueError(f"Node {node.name} is missing required parameters: {still_missing}.")

            all_missing = missing_required | missing_optional

            for k, v in node_data.items():
                if k in all_missing:
                    node_params.arguments[k] = v
                else:
                    if verbose:
                        print(f"Ignoring value for {k} from data_dict; it was already supplied.")

            # get components from child nodes
            for k, v in node_params.arguments.items():
                if isinstance(v, Node):
                    try:
                        v_comp = component_dict[v]
                    except KeyError:
                        # TODO make a new exception class
                        raise ValueError(f"Child component {v.name} of node {node.name} has not been created yet.")
                    else:
                        node_params.arguments[k] = v_comp
                elif isinstance(v, tuple):
                    temp = []

                    for item in v:
                        if isinstance(item, Node):
                            try:
                                item_comp = component_dict[item]
                            except KeyError:
                                # TODO make a new exception class
                                raise ValueError(f"Child component {item.name} of node {node.name} has not been created yet.")
                            else:
                                temp.append(item_comp)
                        else:
                            temp.append(item)

                    node_params.arguments[k] = tuple(temp)

            # if node has inputs, get the built components
            if node.inputs:
                node_params.arguments["inputs"] = [component_dict[inp] for inp in node.inputs]

            # use short name for component
            node_params.arguments["name"] = node.short_name

            if verbose:
                import numpy as np
                import pandas as pd
                print_kwargs = {}
                for k, v in node_params.arguments.items():
                    if isinstance(v, np.ndarray | pd.Series | pd.Index):
                        print_kwargs[k] = type(v)
                    else:
                        print_kwargs[k] = v
                print(print_kwargs, "\n")

            component_dict[node] = node.component_type(*node_params.args, **node_params.kwargs)

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

    to_build = {k: component_dict[k] for k in build_order if k not in all_children}

    if verbose:
        print("Build order:")
        for i, k in enumerate(to_build.keys()):
            print(f"{i + 1}) {k.name}")
        print()

    with pm.Model() as model:
        for node, component in to_build.items():
            if verbose:
                print("Building component", node.name)

            inputs = [comp for k, comp in component_dict.items() if k in node.inputs]

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
