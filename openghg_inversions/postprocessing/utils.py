from functools import wraps
import inspect
from collections.abc import Callable

import xarray as xr


def add_suffix(suffix: str):
    """Decorator to add suffix to variable names of dataset returned by decorated function.

    For example::

        @add_suffix("abc")
        def some_func():
           ...

    will add "_abc" to the end of all the data variables in the dataset returned by
    some_func. (So this only works if the output of some_func is an xr.Dataset or xr.DataArray.)

    Note: technically, add_suffix creates a new decorator
    each time it is called. This is the decorate function
    that is returned. Then the actual "decoration" is done by
    the decorate function.

    So::

        @add_suffix("mean")
        def calc_mean():
            pass

    is the same as::

        temp = add_suffix("mean")  # get `decorate`

        @temp
        def calc_mean():
            pass

    Args:
        suffix: Suffix to add to variable names.
    """

    def decorate(func):
        @wraps(func)
        def call(*args, **kwargs):
            result = func(*args, **kwargs)
            rename_dict = {dv: str(dv) + "_" + suffix for dv in result.data_vars}
            return result.rename(rename_dict)

        return call

    return decorate


def update_attrs(prefix: str):
    """Decorator to update all attributes of output dataset with prefix.

    Args:
        prefix: prefix to append to attributes; an underscore will be
          inserted after this prefix.

    Raises:
        ValueError: if the decorated function does not return xr.Dataset

    """

    def decorate(func):
        @wraps(func)
        def call(*args, **kwargs):
            try:
                data = next(arg for arg in args if isinstance(arg, xr.Dataset))
            except StopIteration:
                try:
                    data = next(arg for arg in kwargs.values() if isinstance(arg, xr.Dataset))
                except StopIteration:
                    raise ValueError(
                        "`update_attrs` can only decorate functions that accept and return an xr.Dataset."
                    )

            result = func(*args, **kwargs)

            for dv in data.data_vars:
                if dv in result:
                    result[dv].attrs = data[dv].attrs

                    if "long_name" in result[dv].attrs:
                        result[dv].attrs["long_name"] = prefix + "_" + result[dv].attrs["long_name"]

            return result

        return call

    return decorate


def make_replace_names_dict(names: list[str], old: str, new: str) -> dict[str, str]:
    """Make dictionary for renaming data variables by replacing "old" with "new".

    Args:
        names: list of data variable names to update
        old: old string to replace
        new: replacement string for old

    Returns:
        dictionary mapping old names to new names

    """
    return {name: name.replace(old, new) for name in names}


def rename_by_replacement(ds: xr.Dataset, old: str, new: str) -> xr.Dataset:
    """Rename data variables by replacing "old" with "new".

    If the string `old` is contained in a data variable's name, it is replaced
    with `new`.

    Args:
        ds: dataset whose data variables will be renamed
        old: string to replace
        new: replacement string

    Returns:
        dataset with renamed data variables

    """
    rename_dict = make_replace_names_dict(list(ds.data_vars), old, new)
    return ds.rename(rename_dict)


def get_parameters(func: Callable) -> list[str]:
    """Return list of parameters for a function.

    Args:
        func: function to inspect

    Returns:
        list of parameters for the given function

    """
    return list(inspect.signature(func).parameters.keys())
