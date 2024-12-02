from collections import ChainMap
from pathlib import Path

import networkx as nx

from .config_parser import Node, ModelGraph

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
