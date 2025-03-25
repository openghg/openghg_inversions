from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import inspect
from pathlib import Path
import re
from types import UnionType
import typing
from typing import Any, Iterable, Literal, cast
from typing_extensions import Self

from openghg_inversions.models.components import ModelComponent

try:
    import tomllib as toml
except ModuleNotFoundError:
    import pip._vendor.tomli as toml

import matplotlib.pyplot as plt
import networkx as nx
import pymc as pm


def _valid_arg(type_name, ann) -> bool:
    if typing.get_origin(ann) is UnionType:
        return any(
            (type_name is sub_ann) or (type_name.__name__ == sub_ann.__name__)
            for sub_ann in typing.get_args(ann)
        )
    return (type_name is ann) or (type_name.__name__ == ann.__name__)


class ModelBuildError(Exception): ...


class Node:
    def __init__(
        self,
        name: str,
        type_: str = "default",
        children: list[str] | None = None,
        inputs: list[str] | None = None,
        **kwargs,
    ) -> None:
        self.name = name
        self.type = type_
        self.children_names = children or []
        self.input_names = inputs or []

        if f"use_{self.type}" in kwargs:
            self.skip = not kwargs.pop(f"use_{self.type}")
        else:
            self.skip = False

        # args to be used for finding data
        if "data_args" in kwargs:
            self.data_args = kwargs.pop("data_args")
        else:
            self.data_args = {}

        # data used to create component
        self.data = {}

        # store all other args
        self.__dict__.update(kwargs)

        try:
            self.component_type = ModelComponent._component_registry[self.type]
        except KeyError:
            raise TypeError(f"No `ModelComponent` registered with name {self.type}.")

        # info on required arguments
        self.annotations = inspect.get_annotations(self.component_type.__init__, eval_str=True)
        if "return" in self.annotations:
            del self.annotations["return"]

        self.signature = inspect.signature(self.component_type)
        self.parameters = self.signature.parameters

        var_args = [param.name for param in self.parameters.values() if param.kind == param.VAR_POSITIONAL]
        self.var_args = var_args[0] if var_args else None

        # data needed to build component; this should be updated before using the component
        self.data = {}

        # these attributes will be set when the model graph is created.
        self.children = []
        self.inputs = []

        # model component; initially set to None
        self._component: ModelComponent | None = None

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

    def component_args(
        self, verbose: bool = False, partial: bool = True, create_children: bool = False
    ) -> inspect.BoundArguments:
        args = []
        kwargs = {}
        var_args = []

        if self.var_args is not None:
            past_var_args = False

            for k, param in self.parameters.items():
                val = None
                if k == self.var_args:
                    past_var_args = True
                elif hasattr(self, k):
                    val = getattr(self, k)
                elif k in self.data:
                    val = self.data[k]

                if val is not None:
                    if (
                        param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
                        and not past_var_args
                    ):
                        args.append(getattr(self, k))
                    else:
                        kwargs[k] = getattr(self, k)

        else:
            for k, param in self.parameters.items():
                if hasattr(self, k):
                    kwargs[k] = getattr(self, k)
                elif k in self.data:
                    kwargs[k] = self.data[k]

        remaining_children = {}
        for child in self.children:
            if child.skip is True:
                if verbose:
                    print(f"Skipping child {child.name}")
                continue
            for param, ann in self.annotations.items():
                if param != self.var_args and _valid_arg(child.component_type, ann):
                    if create_children:
                        kwargs[param] = child.component(verbose=verbose)
                    else:
                        kwargs[param] = child
                    break
            else:  # no keyword found
                remaining_children[child.name] = child

        if remaining_children:
            if create_children:
                var_args = [child.component(verbose=verbose) for child in remaining_children.values()]
            else:
                var_args = list(remaining_children.values())

        if partial:
            result = self.signature.bind_partial(*args, *var_args, **kwargs)
        else:
            result = self.signature.bind(*args, *var_args, **kwargs)

        return result

    def print_args(self) -> None:
        import numpy as np
        import pandas as pd

        print_kwargs = {}
        for k, v in self.component_args().arguments.items():
            if isinstance(v, np.ndarray | pd.Series | pd.Index | xr.DataArray):
                print_kwargs[k] = type(v)
            else:
                print_kwargs[k] = v
        print(self.name, print_kwargs, "\n")

    @property
    def missing_params(self) -> tuple[set[str], set[str]]:
        required = []
        optional = []

        params_we_have = self.component_args(partial=True).arguments

        for param in self.parameters.values():
            if param.name not in params_we_have:
                if param.kind != param.VAR_POSITIONAL and param.default == param.empty:
                    required.append(param.name)
                else:
                    optional.append(param.name)

        return set(required), set(optional)

    def _create_component(self, verbose: bool = False) -> None:
        try:
            comp_args = self.component_args(partial=False, create_children=True, verbose=verbose)
        except TypeError as e:
            required_missing, _ = self.missing_params
            raise ModelBuildError(
                f"Cannot create component {self.name}; "
                "the following required arguments are missing: "
                f"{', '.join(required_missing)}"
            ) from e

        comp_args.arguments["name"] = self.short_name

        # add inputs
        # TODO: remove inputs from ModelComponent subclass __init__ methods
        if self.inputs:
            comp_args.arguments["inputs"] = [inp.component() for inp in self.inputs if not inp.skip]

        self._component = self.component_type(*comp_args.args, **comp_args.kwargs)

    def component(self, verbose: bool = False) -> ModelComponent:
        if self._component is None:
            if verbose:
                print(f"Creating component {self.name}")
            self._create_component(verbose=verbose)

        return cast(ModelComponent, self._component)

    def instantiate_model(self, verbose: bool = False) -> None:
        if self.skip:
            return None
        if verbose:
            print(f"{repr(self)} instantiating model")
        self.component().instantiate_model()

        with self.component().model:
            for child in self.children:
                child.instantiate_model(verbose=verbose)


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

                node_children_names = get_children(values, name, cache)
                node_kwargs["children"] = node_children_names

                for child in node_children_names:
                    # append dict in the form {name: value}
                    # take last part of child name, since it includes all ancestors' names
                    stack.append({child: values.pop(child.split(".")[-1])})

                # store all remaining values in node
                node_kwargs.update(values)

                node = Node(name, **node_kwargs)

                nodes.append(node)

    return nodes


class ModelGraph:
    def __init__(
        self,
        nodes: Iterable[Node],
        child_edges: Iterable[tuple[Node, Node]],
        input_edges: Iterable[tuple[Node, Node]] | None = None,
    ) -> None:
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(child_edges, kind="child")
        if input_edges:
            self.graph.add_edges_from(input_edges, kind="input")

        self.build_order = [node for node in nx.topological_sort(self.graph) if not node.skip]

        # create .child and .inputs attributes for each Node
        for node in self.graph.nodes:
            for pred in self.graph.predecessors(node):
                kind = self.graph.edges[pred, node]["kind"]

                if kind == "child":
                    node.children.append(pred)

                if kind == "input":
                    node.inputs.append(pred)

        # put child nodes in build order
        # this might not be necessary once we can set the name spaces of the model components ahead of time
        for node in self.nodes:
            temp = []
            for bnode in self.build_order:
                if bnode in node.children:
                    temp.append(bnode)
            node.children = temp

        self._node_hash = {node.name: node for node in self.nodes}

        self._model = None

        # data related attributes
        self.data_args = {}
        self.component_data = {}

    @property
    def model(self) -> pm.Model:
        if self._model is None:
            raise ModelBuildError("Model has not been created yet.")
        return self._model

    @property
    def nodes(self) -> list[Node]:
        """Return copy of nodes in graph."""
        return list(self.graph.nodes)

    def subgraph(self, kind: Literal["child", "input"] = "child") -> nx.DiGraph:
        result = self.graph.edge_subgraph(
            (u, v) for u, v, d in self.graph.edges(data=True) if d["kind"] == kind
        )
        return cast(nx.DiGraph, result)  # this cast shouldn't be necessary...

    def parents(self, node: Node, subgraph_kind: Literal["child", "input"] | None = "child") -> list[Node]:
        if subgraph_kind is not None:
            return list(nx.descendants(self.subgraph(kind=subgraph_kind), node))
        return list(nx.descendants(self.graph, node))

    @classmethod
    def from_config(cls, config: dict) -> Self:
        config = config["model"] if "model" in config else config

        nodes = get_nodes(config)
        nodes_hash = {node.name: node for node in nodes}

        child_edges = defaultdict(list)
        input_edges = defaultdict(list)

        for node in nodes:
            for child in node.children_names:
                child_edges[nodes_hash[child]].append(node)
            for input_ in node.input_names:
                input_edges[nodes_hash[input_]].append(node)

        return cls(
            nodes,
            ((k, x) for k, v in child_edges.items() for x in v),
            ((k, x) for k, v in input_edges.items() for x in v),
        )

    def plot(self, title="", show_types=False, show_inputs=False, edge_labels=None, short_names=False, **kwargs) -> None:
        # TODO: add option to display short names instead of long names
        for layer, nodes in enumerate(nx.topological_generations(self.graph)):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                self.graph.nodes[node]["layer"] = layer

        # Compute the multipartite_layout using the "layer" node attribute
        pos = nx.multipartite_layout(self.graph, subset_key="layer", align="horizontal")
        pos = dict(sorted(pos.items(), key=lambda x: (x[1][1], x[1][0])))
        layers = {v[1] for v in pos.values()}
        for layer in layers:
            nodes, positions = zip(*((k, v) for k, v in pos.items() if v[1] == layer))
            pos.update(dict(zip(sorted(nodes, key=lambda x: x.name), positions)))

        # plot
        fig, ax = plt.subplots()
        options = dict(
            node_shape="s",
            node_color="none",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"),
        )
        options.update(kwargs)

        # make node labels
        if short_names:
            def name_func(name):
                return name.split(".")[-1]
        else:
            def name_func(name):
                return name

        if show_types:
            node_labels = {node: f"{name_func(node.name)}\ntype: {node.type}" for node in self.nodes}
        else:
            node_labels = {node: name_func(node.name) for node in self.nodes}

        # draw
        nx.draw_networkx(self.graph, pos=pos, ax=ax, labels=node_labels, **options)  # type: ignore

        # add edge labels
        if show_inputs:
            edge_labels = edge_labels or {}
            for edge in self.graph.edges:
                if edge[0].name in edge[1].inputs:
                    edge_labels[edge] = "input"

        if edge_labels is not None:
            nx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=edge_labels, ax=ax)

        # draw plot
        ax.set_title(title)
        fig.tight_layout()
        plt.show()

    def add_data(self, data_dict: dict) -> None:
        for node in self.nodes:
            node.data.update(data_dict.get(node.name, {}))

    def create_components(self, data_dict: dict | None = None, verbose: bool = False) -> None:
        """Create ModelComponent for each Node in the ModelGraph

        Nodes are built according to the `build_order` for this graph.
        """
        if data_dict is not None:
            self.add_data(data_dict)

        for node in self.build_order:
            if node.skip:
                continue

            node.component(verbose=verbose)

    def instantiate_model(self, verbose: bool = False) -> None:
        """Instantiate models to create namespaces corresponding to parent-child relationships.

        For instance, if node1 is a child of node2, then we need to instantiate node2, then
        instantiate node1 inside the model context of node2:

        node2.instantiate_model()
        with node2.model:
            node1.instantiate_model()

        and so on...

        The actual recursion is carried out by the `instantiate_model` method of the Nodes.
        """
        if verbose:
            print("Instantiating model for model graph.")
        all_children = set()
        for node in self.nodes:
            all_children = all_children.union(set(node.children))

        self._model = pm.Model()

        with self.model:
            for node in self.nodes:
                if node in all_children:
                    continue
                node.instantiate_model(verbose=verbose)

    def build_model(self, verbose: bool = False) -> None:
        if self._model is None:
            self.instantiate_model(verbose=verbose)

        if verbose:
            print("Building model for model graph.")
        all_children = set()
        for node in self.nodes:
            all_children = all_children.union(set(node.children))

        with self.model:
            for node in self.build_order:
                if node.skip is True:
                    continue

                if verbose:
                    print(f"Building {repr(node)}")

                if node.inputs:
                    node.component().build(*(inp.component() for inp in node.inputs))
                else:
                    node.component().build()
                if verbose:
                    print(node.component().model.basic_RVs)

                # HACK: to prevent the same node from building twice
                node.component()._temp_build = node.component().build
                node.component().build = lambda *x: None

        # restore build methods
        for node in self.nodes:
            if node.skip is True:
                continue

            node.component().build = node.component()._temp_build  # type: ignore
            delattr(node.component(), "_temp_build")

def make_nx_graph(config: dict) -> nx.DiGraph:
    return ModelGraph.from_config(config).graph


def model_from_config(config_path: str | Path, data_dict: dict, verbose: bool = False) -> pm.Model:
    config_path = Path(config_path)

    with open(config_path, "rb") as f:
        config = toml.load(f)

    model_graph = ModelGraph.from_config(config)
    model_graph.create_components(data_dict=data_dict, verbose=verbose)
    model_graph.instantiate_model(verbose=verbose)
    model_graph.build_model(verbose=verbose)

    return model_graph.model
