from collections import ChainMap, defaultdict
from pathlib import Path

import networkx as nx

from openghg_inversions.data_functions import align_and_merge, ComponentData
from openghg_inversions.models.config.config_parser import Node, ModelGraph, ModelBuildError

try:
    import tomllib as toml
except ModuleNotFoundError:
    import pip._vendor.tomli as toml


def get_data_dict(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    with open(config_path, "rb") as f:
        data_dict = toml.load(f)

    if "data" in data_dict:
        return data_dict["data"]

    return data_dict


def add_data_args(mg: ModelGraph, data_dict: dict) -> None:
    """Add info from data_dict to the `data_args` of the Nodes in the ModelGraph.

    Any data in data_dict that doesn't belong to a Node is added to the ModelGraph's
    `data_args`.
    """
    for node in mg.nodes:
        temp = data_dict
        skip = False

        for part in node.name.split("."):
            try:
                temp = temp[part]
            except KeyError:
                skip = True
                break

        if not skip:
            if not hasattr(node, "data_args"):
                node.data_args = temp
            else:
                node.data_args.update(temp)

        # find any data from top level (i.e. not belonging to a specific node)
        node_names = [node.name for node in mg.nodes]
        global_data = {k: v for k, v in data_dict.items() if k not in node_names}

        mg.data_args.update(global_data)



def component_to_data_args_map(mg: ModelGraph) -> dict[str, ChainMap]:
    node_data = {}

    for node in mg.nodes:
        node_data[node.name] = ChainMap(mg.data_args, *(n.data_args for n in nx.descendants(mg.subgraph(), node)), node.data_args)

    return node_data


def get_parent_name_by_type(mg: ModelGraph, node: Node, parent_type: str, exact_match: bool = True) -> str:
        for n in mg.parents(node):
                if n.type == parent_type or (not exact_match and parent_type in n.type):
                    return n.name
        raise ModelBuildError(f"{repr(node.name)} does not have a parent component matching type {parent_type}.")


def get_data(mg: ModelGraph, comp_data: dict | None = None) -> dict[str, ComponentData]:
    comp_data_args = component_to_data_args_map(mg)

    comp_data = comp_data if comp_data is not None else {}

    nodes = [node for node in mg.build_order if node.type != "default"]

    # first pass: create component data objects for nodes without inputs
    for node in nodes:
        # TODO: make a more generic check?
        if node.type in ("tracer", ):
            continue

        cd_type = ComponentData._component_registry[node.type]
        cd_kwargs = {"node": node}

        if node.type in ("flux", "bc", "forward_model") or node.type.endswith("likelihood"):
            cd_kwargs["comp_data_args"] = comp_data_args[node.name]

        comp_data[node.name] = cd_type(**cd_kwargs)

    # second pass: create nodes with inputs, build h matrices
    for node in nodes:

        if node.type == "tracer":
            cd_type = ComponentData._component_registry[node.type]
            cd_kwargs = {"node": node}

            try:
                cd_kwargs["flux_data"] = comp_data[node.inputs[0].name]
            except KeyError as e:
                raise ModelBuildError(f"Tracer node data for {node.name} must be created after data for its flux input.") from e

            comp_data[node.name] = cd_type(**cd_kwargs)

        elif node.type == "sigma":
            likelihood_data = comp_data[get_parent_name_by_type(mg, node, "likelihood", exact_match=False)]

            comp_data[node.name].get_data(likelihood_data)

        elif node.type == "flux":
            forward_data = comp_data[get_parent_name_by_type(mg, node, "forward_model")]
            likelihood_data = comp_data[get_parent_name_by_type(mg, node, "likelihood", exact_match=False)]
            units = likelihood_data.units

            comp_data[node.name].compute_basis(forward_data.mean_fp)
            comp_data[node.name].compute_h_matrix(forward_data.footprints, units=units)

        elif node.type == "bc":
            forward_data = comp_data[get_parent_name_by_type(mg, node, "forward_model")]
            likelihood_data = comp_data[get_parent_name_by_type(mg, node, "likelihood", exact_match=False)]
            units = likelihood_data.units

            comp_data[node.name].compute_h_matrix(forward_data.footprints, units=units)

        elif node.name in comp_data and hasattr(comp_data[node.name], "merge_data"):
            comp_data[node.name].merge_data(comp_data)

    # third pass: align likelihoods and forward models

    # return data
    return comp_data


def make_data_dict(model_graph: ModelGraph, conf: dict, units_dict: dict | None = None, comp_data: dict | None = None) -> tuple[dict, dict]:
    """Make data dict for model graph.

    Returns:
        tuple of data dict for model and dict to ComponentData objects for model
    """
    # get component data from graph; this contains arguments that can be used to find data for components
    add_data_args(model_graph, conf["data"])
    comp_data = comp_data or {}
    comp_data = get_data(model_graph, comp_data)

    combined_data = {}
    units_dict = units_dict or {}

    for node in model_graph.build_order:
        if node.type.endswith("likelihood"):
            likelihood = comp_data[node.name].merged_data
            forward = comp_data[node.inputs[0].name].h_matrix
            species = comp_data[node.name].species
            units = units_dict.get(species)
            combined_data[species] = align_and_merge(likelihood, forward, units=units, output_prefix=species)

    data_dict = defaultdict(dict)
    for node in model_graph.build_order:
        for ds in combined_data.values():
            for da in ds.filter_by_attrs(origin=node.name).values():
                data_dict[node.name][da.attrs["param"]] = da

    return data_dict, comp_data
