# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TENT's configuration, MC_TENT_CONF, as TENT's Config reads it: a JSON
file's path or inline JSON, whose '/'-separated keys are looked up as a
nested path first, then as one flat top-level key (Config::get)."""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

ENV = "MC_TENT_CONF"


def load() -> dict[str, Any] | None:
    """MC_TENT_CONF's settings; None when unset or unreadable (TENT then
    ignores it too)."""
    value = os.environ.get(ENV)
    if not value:
        return None
    try:
        text = Path(value).read_text() if os.path.exists(value) else value
        conf = json.loads(text)
    except (OSError, ValueError):
        return None
    return conf if isinstance(conf, dict) else None


def lookup(
    conf: dict[str, Any],
    key: str,
    accept: Callable[[Any], bool] = lambda value: True,
) -> tuple[bool, Any] | None:
    """Where TENT finds `key`'s value: (True, value) on its nested path,
    (False, value) as a flat key, None if neither holds a non-null value
    `accept` takes (TENT skips one it cannot convert)."""
    node: Any = conf
    for segment in filter(None, key.split("/")):
        if not isinstance(node, dict) or segment not in node:
            node = None
            break
        node = node[segment]
    if node is not None and accept(node):
        return True, node
    flat = conf.get(key)
    if flat is not None and accept(flat):
        return False, flat
    return None


def store(conf: dict[str, Any], key: str, value: Any, nested: bool = True) -> None:
    """Set `key` on its nested path (which TENT reads first), or as a flat
    key."""
    if not nested:
        conf[key] = value
        return
    *parents, last = filter(None, key.split("/"))
    node = conf
    for segment in parents:
        child = node.get(segment)
        if not isinstance(child, dict):
            child = node[segment] = {}
        node = child
    node[last] = value


def save(conf: dict[str, Any]) -> None:
    """Make `conf` this process's MC_TENT_CONF, inline."""
    os.environ[ENV] = json.dumps(conf)
