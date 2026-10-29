"""Prompt templates: ``skills/<skill>/prompt.md`` (+ ``prompt.human.md``) with ``{{name}}`` placeholders.

The templates hold the source prompts verbatim; ``{{name}}`` uses the source
f-string variable name. Rendering is strict in both directions: a placeholder
without a value, or a value without a placeholder, is an error, so a template
and its caller cannot drift apart silently. Values are substituted in a single
pass, so report text that itself contains ``{{...}}`` is never re-expanded.

Structured roles (those whose submission is JSON validated by a pydantic schema)
get a generated JSON-output section appended; it is built from the schema's
field names, types and verbatim descriptions, not written by hand.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import BaseModel

HUMAN_MARKER = "---HUMAN---"

_PLACEHOLDER = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")


class PromptError(ValueError):
    """A template is missing, or its placeholders and values do not match."""


def skills_dir() -> Path:
    """The skills directory: ``$TA_SKILLS_DIR`` or ``<repo>/skills`` beside the ``src`` tree."""
    override = os.environ.get("TA_SKILLS_DIR")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2] / "skills"


def load_template(skill: str, human: bool = False) -> str | None:
    """The template text of ``skill``; ``None`` for a missing human template, an error for the system one."""
    path = skills_dir() / skill / ("prompt.human.md" if human else "prompt.md")
    if not path.is_file():
        if human:
            return None
        raise PromptError(f"prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def placeholders(template: str) -> set[str]:
    """The ``{{names}}`` a template uses."""
    return set(_PLACEHOLDER.findall(template))


def render(template: str, values: dict[str, str]) -> str:
    """Substitute every ``{{name}}``; a missing or an unused value is a ``PromptError``."""
    wanted = placeholders(template)
    missing = sorted(wanted - values.keys())
    if missing:
        raise PromptError(f"no value for placeholder(s): {', '.join(missing)}")
    unused = sorted(values.keys() - wanted)
    if unused:
        raise PromptError(f"value(s) without a placeholder in the template: {', '.join(unused)}")
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)


def render_skill(skill: str, values: dict[str, str]) -> tuple[str, str | None]:
    """``(system, human)`` for ``skill``; ``human`` is ``None`` when it has one prompt only.

    ``values`` covers the placeholders of both templates together.
    """
    system = load_template(skill)
    human = load_template(skill, human=True)
    wanted = placeholders(system) | (placeholders(human) if human is not None else set())
    missing = sorted(wanted - values.keys())
    if missing:
        raise PromptError(f"{skill}: no value for placeholder(s): {', '.join(missing)}")
    unused = sorted(values.keys() - wanted)
    if unused:
        raise PromptError(f"{skill}: value(s) without a placeholder: {', '.join(unused)}")
    sub = {k: values[k] for k in placeholders(system)}
    system_text = render(system, sub)
    human_text = None
    if human is not None:
        human_text = render(human, {k: values[k] for k in placeholders(human)})
    return system_text, human_text


def join_parts(system: str, human: str | None) -> str:
    """The full prompt as ``ta run prompt`` prints it: system, then ``---HUMAN---``, then human."""
    if human is None:
        return system
    return f"{system}\n{HUMAN_MARKER}\n{human}"


# --- JSON output section generated from a pydantic schema ---------------------------------


def _resolve(prop: dict, defs: dict) -> dict:
    if "$ref" in prop:
        return {**defs[prop["$ref"].rsplit("/", 1)[-1]], **{k: v for k, v in prop.items() if k != "$ref"}}
    return prop


def _type_label(prop: dict, defs: dict) -> tuple[str, bool]:
    """``(type label, nullable)`` for one JSON-schema property."""
    prop = _resolve(prop, defs)
    nullable = False
    if "anyOf" in prop:
        options = []
        for option in prop["anyOf"]:
            if option.get("type") == "null":
                nullable = True
            else:
                options.append(_type_label(option, defs)[0])
        label = " or ".join(options)
        return label, nullable
    if "enum" in prop:
        return "one of " + " | ".join(f'"{v}"' for v in prop["enum"]), nullable
    kind = prop.get("type", "any")
    label = {"number": "number", "integer": "integer", "string": "string", "boolean": "boolean"}.get(
        kind, str(kind)
    )
    bounds = []
    if "minimum" in prop:
        bounds.append(f">= {prop['minimum']:g}")
    if "maximum" in prop:
        bounds.append(f"<= {prop['maximum']:g}")
    if "minLength" in prop:
        bounds.append("non-empty")
    if bounds:
        label += " (" + ", ".join(bounds) + ")"
    return label, nullable


def json_output_section(schema: type[BaseModel]) -> str:
    """The submission format of a structured role, generated from its pydantic schema."""
    spec = schema.model_json_schema()
    defs = spec.get("$defs", {})
    required = set(spec.get("required", []))
    lines = [
        "## JSON submission",
        "",
        "Write your answer to the submission file as one JSON object: no code fences, no text "
        "outside the object, exactly these keys. The rendered markdown report is produced from "
        "these fields, so put each part of your answer in the field that names it.",
        "",
    ]
    for name, prop in spec["properties"].items():
        label, nullable = _type_label(prop, defs)
        need = "required" if name in required and not nullable else "optional, use null to omit"
        if name in required and nullable:
            need = "required, null allowed"
        description = _resolve(prop, defs).get("description") or prop.get("description", "")
        if description.startswith("Required"):
            # The schema coerces a placeholder to null, but the submission check rejects a missing one.
            need = "required: a submission without a positive number is rejected"
        lines.append(f"- `{name}` ({label}; {need}): {description}")
    return "\n".join(lines)
