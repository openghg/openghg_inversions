"""Generate Sphinx references from live xarray-validate schemas.

The JSON-compatible description supplements the library's serialization with
dtype alternatives, attribute type names and callback docstrings. It is a
documentation view, not a format for reconstructing executable validators.
"""

from __future__ import annotations

from importlib import import_module
import inspect
import json
from typing import Any, ClassVar

import numpy as np
from xarray_validate import DataArraySchema, DatasetSchema
from xarray_validate.base import BaseSchema

from openghg_inversions._serialization_codecs import _json_value


def _dtype_description(dtype: Any) -> str | list[Any]:
    """Retain generic dtype families and alternatives without narrowing them."""
    if isinstance(dtype, tuple):
        return [_dtype_description(item) for item in dtype]
    if dtype is np.integer:
        return "integer"
    if dtype is np.floating:
        return "floating"
    return np.dtype(dtype).str


def _metadata(value: Any) -> Any:
    """Name Python types explicitly and reuse the existing metadata codec."""
    if isinstance(value, type):
        return value.__qualname__ if value.__module__ == "builtins" else f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, dict):
        return {str(key): _metadata(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_metadata(item) for item in value]
    return _json_value(value)


def describe_schema(schema: BaseSchema) -> dict[str, Any]:
    """Export a schema for documentation without inspecting any scientific data.

    Args:
        schema: Live Dataset, DataArray, coordinate or attribute schema.

    Returns:
        JSON-compatible structural description including callback names and
        docstrings. Missing callback documentation is labelled explicitly.
        The original schema is unchanged; no callbacks are executed.
    """
    description = schema.serialize()
    if isinstance(schema, DataArraySchema) and schema.dtype is not None:
        description["dtype"] = _dtype_description(schema.dtype.dtype)
    for group in ("data_vars", "coords", "attrs"):
        children = getattr(schema, group, None)
        if isinstance(children, dict):
            description[group] = {str(name): describe_schema(child) for name, child in children.items()}
        elif isinstance(children, BaseSchema):
            description[group] = describe_schema(children)
    checks = getattr(schema, "checks", ())
    if checks:
        description["checks"] = [
            {"name": getattr(check, "__qualname__", type(check).__qualname__),
             "description": inspect.getdoc(check) or "No description supplied; see the validation code."}
            for check in checks
        ]
    return _metadata(description)


def _literal(value: Any) -> str:
    """Keep schema metadata on one escaped reStructuredText table line."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return "``" + text.replace("\\", "\\\\").replace("`", "\\`").replace("\n", " ") + "``"


def _constraint(key: str, value: Any) -> str:
    """Present concrete dtype tokens as familiar names and unions as alternatives."""
    if key == "dtype":
        dtypes = value if isinstance(value, list) else [value]
        value = " or ".join(item if item in {"integer", "floating"} else np.dtype(item).name for item in dtypes)
    return f"{key}: {_literal(value)}"


def render_schema(description: dict[str, Any]) -> str:
    """Render an exported schema as tables and colocated relationship rules.

    Args:
        description: Output of :func:`describe_schema`.

    Returns:
        reStructuredText with required/pattern fields, dimension order,
        constraints, extra-field policies and callback descriptions.
    """
    if "data_vars" not in description:
        description = {"data_vars": {"value": description}}
    return _render_schema(description, path="Dataset")


def _render_schema(description: dict[str, Any], *, path: str) -> str:
    """Render nested variable, coordinate and attribute groups at their paths."""
    lines = []
    for group, label in (("data_vars", "Variables"), ("coords", "Coordinates"), ("attrs", "Attributes")):
        container = description if group == "data_vars" else description.get(group, {})
        children = container.get(group, {})
        if not children:
            continue
        lines.extend([
            f".. rubric:: {path}: {label}", "",
            "Extra fields are " + ("allowed." if container.get("allow_extra_keys", True) else "rejected."), "",
            ".. list-table::", "   :header-rows: 1", "   :widths: 30 12 24 34", "",
            "   * - Field", "     - Required", "     - Dimensions", "     - Constraints",
        ])
        for name, child in children.items():
            pattern = (name.startswith("{") and name.endswith("}")) or "*" in name or "?" in name
            required = "Pattern" if pattern else ("Yes" if container.get("require_all_keys", True) else "No")
            dims = child.get("dims")
            if isinstance(dims, dict):
                dimension_text = _literal(dims["dims"]) + " (any order)"
            elif dims is not None:
                dimension_text = "Scalar" if not dims else _literal(dims) + " (ordered)"
            elif group == "attrs":
                dimension_text = "Not applicable"
            else:
                dimension_text = "Unrestricted"
            constraints = [
                _constraint(key, value) for key, value in child.items()
                if key not in {"dims", "coords", "attrs", "checks"} and value is not None
            ]
            lines.extend([
                f"   * - {_literal(name)}", f"     - {required}", f"     - {dimension_text}",
                "     - " + ("; ".join(constraints) or "Unrestricted"),
            ])
        lines.append("")
        if any((name.startswith("{") and name.endswith("}")) or "*" in name or "?" in name for name in children):
            lines.extend(["Pattern entries constrain matching fields without requiring a match.", ""])
        for name, child in children.items():
            lines.append(_render_schema(child, path=f"{path}.{name}"))
    if description.get("checks"):
        lines.extend([f".. rubric:: {path}: Relationship checks", ""])
        for check in description["checks"]:
            lines.extend([
                f"* {_literal(check['name'].rsplit('.', 1)[-1])}: {' '.join(check['description'].split())}", "",
            ])
    return "\n".join(lines)


def load_schema(factory: str, kwargs: dict[str, Any]) -> DatasetSchema | DataArraySchema:
    """Call a documented schema factory, including a contract's class method.

    Args:
        factory: Import path such as ``package.module:Contract.schema``.
        kwargs: Explicit keyword arguments selecting this schema variant.

    Returns:
        The selected live schema, without invoking its validator.

    Raises:
        TypeError: If the factory does not return a Dataset or DataArray schema.
    """
    module, name = factory.split(":", 1)
    target: Any = import_module(module)
    for part in name.split("."):
        target = getattr(target, part)
    schema = target(**kwargs)
    if not isinstance(schema, DatasetSchema | DataArraySchema):
        raise TypeError(f"{factory} must return a DatasetSchema or DataArraySchema.")
    return schema


def setup(app: Any) -> dict[str, bool]:
    """Register the schema-reference directive when loaded by Sphinx.

    Args:
        app: Sphinx application building the reference.

    Returns:
        Parallel-build safety declarations for this stateless extension.
    """
    from docutils import nodes
    from docutils.parsers.rst import directives
    from docutils.statemachine import StringList
    from sphinx.util.docutils import SphinxDirective

    class SchemaReference(SphinxDirective):
        """Insert a reference generated from a trusted documentation factory."""

        required_arguments = 1
        option_spec: ClassVar = {"kwargs": directives.unchanged}

        def run(self) -> list[nodes.Node]:
            """Import the selected schema and parse its generated tables."""
            kwargs = json.loads(self.options.get("kwargs", "{}"))
            schema = load_schema(self.arguments[0], kwargs)
            self.env.note_dependency(inspect.getfile(type(schema)))
            module = import_module(self.arguments[0].split(":", 1)[0])
            self.env.note_dependency(module.__file__)
            source = (
                f"Schema factory: {_literal(self.arguments[0])}.\n\n"
                f"Selected options: {_literal(kwargs)}.\n\n"
                + render_schema(describe_schema(schema))
            )
            container = nodes.container()
            self.state.nested_parse(StringList(source.splitlines(), source=self.get_source_info()[0]), 0, container)
            return list(container.children)

    app.add_directive("schema-reference", SchemaReference)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
