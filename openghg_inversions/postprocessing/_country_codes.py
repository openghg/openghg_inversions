"""
Utilities for converting country names to ISO country codes.
"""

import functools
import json
import re
import string
from pathlib import Path
from typing import Any, Iterable, Literal, Optional


# PREPROCESSING FUNCTIONS
def maybe_umlauts(x: str) -> str:
    """Treat ae, ue, oe as possible umlauts.

    Returns a string that can be used for re.search,
    where the "e" after "ae", "ue", "oe" is optional.
    """
    return re.sub(r"([aou])e", r"\1e?", x, flags=re.IGNORECASE)


def remove_punctuation(x: str, skip_dot: bool = False) -> str:
    """Remove punctuation from x, optionally ignoring '.'"""
    # move - to beginning
    punc_string = "-" + string.punctuation.replace("-", "")

    if skip_dot:
        punc_string = punc_string.replace(".", "")

    punc_pat = rf"[{punc_string}]"
    return re.sub(punc_pat, "", x)


def remove_common_words(x: str) -> str:
    """Remove common words from a string."""
    common_words_pat = r"the|and|&|of"
    return re.sub(common_words_pat, "", x, flags=re.IGNORECASE)


def drop_extra_whitespace(x: str) -> str:
    """Replace contiguous whitespace with a single space."""
    return re.sub(r"\s+", " ", x)


def extend_abbrs(x: str) -> str:
    """Convert abbreviations into a patterns to matches words beginning with the abbreviation."""
    parts = x.split(".")
    if len(parts) == 1:
        return x
    return r" ".join(p + r"[a-zA-Z]*" for p in parts if p)


# CONVERSION FUNCTIONS
@functools.lru_cache
def get_iso3166_codes() -> dict[str, Any]:
    """Load dictionary mapping alpha-2 country codes to other country information."""
    postprocessing_path = Path(__file__).parent
    with open(postprocessing_path / "iso3166.json", "r", encoding="utf8") as f:
        iso3166 = json.load(f)
    return iso3166


def to_string(x: Any) -> str:
    """Convert to string, decoding if x is bytes."""
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def preprocess(x: str) -> str:
    """Preprocess string for matching."""
    x = x.lower()
    x = remove_common_words(x)
    x = remove_punctuation(x)
    x = drop_extra_whitespace(x)
    return x


def preprocess_pat(x: str) -> str:
    """Create a pattern to match against preprocessed strings."""
    x = x.lower()
    x = remove_common_words(x)
    x = remove_punctuation(x, skip_dot=True)
    x = extend_abbrs(x)
    x = maybe_umlauts(x)
    x = drop_extra_whitespace(x)
    return x


def get_country_code(
    x: str, iso3166: Optional[dict[str, dict[str, Any]]] = None, code: Literal["alpha2", "alpha3"] = "alpha3"
) -> str:
    """Get alpha-2 or alpha-3 (default) country code given the name of a country."""
    if iso3166 is None:
        iso3166 = get_iso3166_codes()

    if preprocess(x) in ("ocean", "sea", "land"):
        return x

    # first try to match long names, ignoring "The " at the beginning of a name
    for v in iso3166.values():  # type: ignore
        if x.lower().lstrip("the ") == v["iso_long_name"].lower().lstrip("the "):
            return v[code]

    # next try to match unofficial names
    for v in iso3166.values():  # type: ignore
        if any(x.lower() == name.lower() for name in v["unofficial_names"] + [v["iso_short_name"]]):
            return v[code]

    # next try to match substrings...
    x_pat = preprocess_pat(x)
    for v in iso3166.values():
        names = [v["iso_long_name"], v["iso_short_name"]] + [name for name in v["unofficial_names"]]
        if any(re.search(x_pat, preprocess(name)) for name in names):
            return v[code]

    # if no matches are found, return x
    return x


def get_country_codes(
    country_names: Iterable[Any],
    iso3166: Optional[dict[str, dict[str, Any]]] = None,
    code: Literal["alpha2", "alpha3"] = "alpha3",
) -> list[str]:
    """Return list of country codes from iterable (e.g. list or array) of country names."""
    str_country_names = map(to_string, country_names)
    return list(get_country_code(country_name, iso3166, code) for country_name in str_country_names)
