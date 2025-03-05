"""Utilities for converting country names to ISO country codes."""

from __future__ import annotations

import functools
import json
import re
import string
from pathlib import Path
from typing import Any, Literal, SupportsIndex, overload
from collections import UserList
from collections.abc import Iterable
from typing_extensions import Self


# PREPROCESSING FUNCTIONS
def maybe_umlauts(x: str) -> str:
    """Treat ae, ue, oe as possible umlauts.

    Returns a string that can be used for re.search,
    where the "e" after "ae", "ue", "oe" is optional.

    Args:
        x: string to modify

    Returns:
        regex string where e.g. "ae" will now match "ae" or "a"

    """
    return re.sub(r"([aou])e", r"\1e?", x, flags=re.IGNORECASE)


def remove_punctuation(x: str, skip_dot: bool = False) -> str:
    """Remove punctuation from string.

    Args:
        x: string to remove punctuation from
        skip_dot: if True, do not remove '.' (period/full stop)

    Returns:
        string without punctuation

    """
    # move - to beginning
    punc_string = "-" + string.punctuation.replace("-", "")

    if skip_dot:
        punc_string = punc_string.replace(".", "")

    punc_pat = rf"[{punc_string}]"
    return re.sub(punc_pat, "", x)


def remove_common_words(x: str) -> str:
    """Remove common words from a string."""
    common_words = ["the", "and", "&", "of"]

    # surround by \b to avoid matching substrings
    common_words_pat = r"|".join(rf"\b{word}\b" for word in common_words)

    return re.sub(common_words_pat, r"", x, flags=re.IGNORECASE)


def drop_extra_whitespace(x: str) -> str:
    """Replace contiguous whitespace with a single space."""
    return re.sub(r"\s+", " ", x)


def extend_abbrs(x: str) -> str:
    """Convert abbreviations into a patterns to matches words beginning with the abbreviation.

    For instance, "N. Wales" --> r"N[a-zA-Z]* Wales", which is a pattern that would match "North Wales".

    Special handling is added for "St."
    """
    parts = x.split(".")
    if len(parts) == 1:
        return x

    def convert_abbr(abbr: str) -> str:
        if abbr.lower().strip() == "st":
            return r"[sS]aint"
        return abbr + r"[a-zA-Z]*"

    # add extension pattern to all but last part
    extended_parts = [convert_abbr(p) for p in parts[:-1] if p]
    return r" ".join(extended_parts) + parts[-1]  # last part would typically have its own whitespace


# ISO country code info
@functools.lru_cache
def get_iso3166_codes() -> dict[str, Any]:
    """Load dictionary mapping alpha-2 country codes to other country information."""
    iso_path = Path(__file__).parent / "iso3166.json"
    with iso_path.open(encoding="utf8") as f:
        iso3166 = json.load(f)
    return iso3166


@functools.lru_cache
def iso3to2_dict() -> dict[str, str]:
    """Dict mapping ISO alpha3 codes to alpha2 codes."""
    iso3166 = get_iso3166_codes()
    return {v["alpha3"]: k for k, v in iso3166.items()}


def iso3to2(alpha3: str) -> str:
    return iso3to2_dict().get(alpha3.upper(), alpha3)


# CONVERSION FUNCTIONS
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
    x: str, iso3166: dict[str, dict[str, Any]] | None = None, code: Literal["alpha2", "alpha3"] = "alpha3"
) -> str:
    """Get alpha-2 or alpha-3 (default) country code given the name of a country.

    JSON data on country names and ISO country codes is search to try to match
    the given country name to its alpha-2 or alpha-3 ISO code.

    Args:
        x: name of country
        iso3166: optional dictionary of ISO-3166 data on country codes. (By
          default, data packaged with OpenGHG Inversions is used.)
        code: type of code to return: "alpha-2" for 2 letter codes, and "alpha-3"
          for 3 letter codes.

    Returns:
        Two or three letter country code, if it is found; otherwise, the input
        country string is returned.

    """
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
    iso3166: dict[str, dict[str, Any]] | None = None,
    code: Literal["alpha2", "alpha3"] = "alpha3",
) -> list[str]:
    """Return list of country codes from iterable of country names.

    Args:
        country_names: iterable (e.g. list or array) of country names
        iso3166: optional dictionary of ISO-3166 data on country codes. (By
          default, data packaged with OpenGHG Inversions is used.)
        code: type of code to return: "alpha-2" for 2 letter codes, and "alpha-3"
          for 3 letter codes.
        iso3166:

    Returns:
        list of country codes in same order as input list of country names.
    """
    str_country_names = map(to_string, country_names)
    return list(get_country_code(country_name, iso3166, code) for country_name in str_country_names)


class CountryInfo:
    """Stores country name and ISO alpha2/alpha3 codes.

    The input can be an alpha2 or alpha3 code, or a country name, which will be
    associated (if possible) with an alpha2 code.

    Once created, if a match for the alpha2 code is found, then the input country name, alpha2
    and alpha3 codes, and official ISO long and short names will be available.

    If no match is found, these values all default to the input country name. To check if a match
    was found, use the `.is_recognised` attribute.

    Equality of `CountryInfo` objects is based on the alpha2 code, so no conversions are needed to
    test equality. (This also works if comparing a `CountryInfo` object to a string.)
    """

    def __init__(self, country_name: str | CountryInfo) -> None:
        self.input_name = country_name if isinstance(country_name, str) else country_name.input_name

        iso3166 = get_iso3166_codes()

        if len(self.input_name) == 2 and self.input_name.upper() in iso3166:
            self._alpha2 = self.input_name.upper()
        elif len(self.input_name) == 3 and iso3to2(self.input_name) in iso3166:
            self._alpha2 = iso3to2(self.input_name)
        else:
            alpha2 = get_country_code(self.input_name, code="alpha2")
            self._alpha2 = alpha2 if alpha2 != self.input_name else None

        self.alpha3 = iso3166[self._alpha2]["alpha3"] if self._alpha2 else self.input_name
        self.iso_long_name = iso3166[self._alpha2]["iso_long_name"] if self._alpha2 else self.input_name
        self.iso_short_name = iso3166[self._alpha2]["iso_short_name"] if self._alpha2 else self.input_name

    def __repr__(self) -> str:
        if self.is_recognised:
            return f"CountryInfo({self.iso_short_name}, alpha2='{self.alpha2}')"
        return f"CountryInfo({self.input_name}, alpha2=None)"

    @property
    def alpha2(self) -> str:
        if self._alpha2 is not None:
            return self._alpha2
        return self.input_name

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.alpha2 == CountryInfo(other).alpha2
        elif isinstance(other, self.__class__):
            return self.alpha2 == other.alpha2
        else:
            raise NotImplementedError(f"Cannot compare CountryInfo with type {type(other)}.")

    @property
    def is_recognised(self) -> bool:
        """Return True if ISO codes were found for the given input."""
        return self._alpha2 is not None


class CountryInfoList(UserList):
    """Container class for CountryInfo objects.

    The `country_code` argument sets the behavior for accessing values.
    For instance, if `country_code = "alpha3"`, then when retrieving values by
    indexing or iterating over the CountryInfoList, the values will be the alpha3
    codes.

    This means that the format used to input country names doesn't matter: the output
    format is controlled by `country_code`, and comparisons are done using equality of
    `CountryInfo` objects, so no conversion is required before comparing.

    NOTE: to iterate over underlying `CountryInfo` objects, use the `.data` attribute.
    For instance:
    >>> type(next(iter(country_info_list.data)))
    CountryInfo

    versus
    >>> type(next(iter(country_info_list)))
    str

    """

    def __init__(
        self,
        country_names: Iterable[str] | None = None,
        country_code: Literal["alpha2", "alpha3"] | None = None,
    ) -> None:
        if country_names is None:
            super().__init__()
        else:
            super().__init__(CountryInfo(name) for name in country_names)

        self._country_code = country_code

    @property
    def country_code(self) -> Literal["alpha2", "alpha3"] | None:
        return self._country_code  # type: ignore

    @country_code.setter
    def _(self, value: Literal["alpha2", "alpha3"] | None) -> None:
        self._country_code = value

    def __str__(self) -> str:
        return str(list(self))

    def __setitem__(self, index: SupportsIndex | slice, item: Any) -> None:
        if isinstance(index, SupportsIndex):
            if isinstance(item, str | CountryInfo):
                self.data[index] = CountryInfo(item)
            else:
                raise ValueError("Can only assign `str` or `CountryInfo`.")
        raise ValueError("Cannot assign to slice of CountryInfoList.")

    def insert(self, i: int, item: Any) -> None:
        if isinstance(item, str | CountryInfo):
            self.data.insert(i, CountryInfo(item))
        else:
            raise ValueError("Can only insert `str` or `CountryInfo`.")

    def append(self, item: Any) -> None:
        if isinstance(item, str | CountryInfo):
            self.data.append(CountryInfo(item))
        else:
            raise ValueError("Can only append `str` or `CountryInfo`.")

    def extend(self, other: Iterable) -> None:
        if isinstance(other, type(self)):
            self.data.extend(other.data)
        else:
            self.data.extend(CountryInfo(item) for item in other)

    @overload
    def __getitem__(self, index: SupportsIndex) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: SupportsIndex | slice) -> str | Self:
        """Return alpha2, alpha3, or input name of item at given index, depending on mode."""
        if isinstance(index, SupportsIndex):
            mode = self.country_code or "input_name"  # use "input_name" if country code is None
            return getattr(self.data[index], mode)

        return type(self)(self.data[index], self.country_code)

    def select_by_country_info(self, country_list: CountryInfoList | Iterable[str]) -> CountryInfoList:
        matches = []

        for country in country_list:
            try:
                match_idx = self.index(country)
            except ValueError as e:
                raise ValueError(f"No match for '{country}' in CountryInfoList.") from e
            else:
                matches.append(self[match_idx])

        return type(self)(matches, self.country_code)
