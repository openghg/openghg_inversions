from argparse import ArgumentParser
from pathlib import Path

from openghg.objectstore import get_writable_buckets
from openghg.util._user import _add_path_to_config

from openghg_inversions.tutorial.data_helpers import add_test_data


def make_tutorial_store(store: str, store_path: str | Path | None = None) -> None:
    """Make object store with data for basic inversions."""
    if store not in get_writable_buckets():
        store_path = Path(store_path) if store_path is not None else Path.home()
        print(f"Store {store} not found. Creating object store in {store_path} and adding to OpenGHG config.")
        _add_path_to_config(path=store_path, name=store)

    add_test_data(store=store)


def make_country_folder(out_path: str | Path) -> None:
    """Copy country files to location for use in tutorials."""
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("store", type=str, default="inversions_tests", help="Object store to add data to. If this store does not exist, it will be created.")
    parser.add_argument("--store-path", type=str, default=None, help="Path to place store if it does not already exist. Defaults to $HOME.", required=False)

    args = parser.parse_args()

    make_tutorial_store(store=args.store, store_path=args.store_path)
