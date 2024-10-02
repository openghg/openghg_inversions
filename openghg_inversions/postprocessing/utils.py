from functools import wraps


def add_suffix(suffix: str):
    """Decorator to add suffix to variable names of dataset returned by decorated function.

    For example:

    @add_suffix("abc")
    def some_func():
       ...

    will add "_abc" to the end of all the data variables in the dataset returned by
    `some_func`. (So this only works is the output of `some_func` is an xr.Dataset or xr.DataArray.)

    Note: technically, `add_suffix` creates a new decorator
    each time it is called. This is the `decorate` function
    that is returned. Then the actual "decoration" is done by
    the `decorate` function.

    So

    @add_suffix("mean")
    def calc_mean():
        pass

    is the same as

    temp = add_suffix("mean")  # get `decorate`

    @temp
    def calc_mean():
        pass
    """

    def decorate(func):
        @wraps(func)
        def call(*args, **kwargs):
            result = func(*args, **kwargs)
            rename_dict = {dv: str(dv) + "_" + suffix for dv in result.data_vars}
            return result.rename(rename_dict)

        return call

    return decorate


def make_replace_names_dict(names: list[str], old: str, new: str) -> dict[str, str]:
    return {name: name.replace(old, new) for name in names}


def rename_by_replacement(ds: xr.Dataset, old: str, new: str) -> xr.Dataset:
    rename_dict = make_replace_names_dict(list(ds.data_vars), old, new)
    return ds.rename(rename_dict)
