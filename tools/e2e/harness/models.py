"""A run's model: a Hugging Face id, which the harness fetches.

A simulated server builds its model from the configuration and serves the
tokenizer; it never reads weights. So a model is a repository's small
files: ``<org>/<name>[@<revision>]`` in a run config is fetched into
``models/<org>--<name>/<commit>/`` (not tracked), without its weights. The
directory is named by the commit the revision resolved to, so a run's saved
model path says exactly which files it used.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
HUB = "https://huggingface.co"
MODEL_ID = re.compile(r"^(?P<id>[A-Za-z0-9][\w.-]*/[\w.-]+)(?:@(?P<revision>[\w./-]+))?$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
# Weights and other large artifacts, by suffix: nothing a simulated server reads.
WEIGHTS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".h5", ".msgpack", ".ot", ".tflite",
           ".npz", ".tar", ".zip", ".mlmodel", ".pb")
TIMEOUT = 60


def resolve(value: str, *, fetch: bool = False) -> Path:
    """The fetched copy of the model ``value`` names by its Hugging Face id.

    Without ``fetch`` nothing is downloaded: an id not fetched yet gives the
    directory it would have (a pinned commit's) or its repository's."""
    match = MODEL_ID.match(value)
    if not match:
        raise ValueError(f"model is a Hugging Face id, <org>/<name>[@<revision>], not {value!r}")
    repository, revision = match["id"], match["revision"]
    home = MODELS / repository.replace("/", "--")
    if revision and COMMIT.match(revision):
        if _fetched(home / revision) or not fetch:
            return home / revision
    elif not revision:
        # An unpinned id is whichever commit was fetched for it: runs stay
        # reproducible, and work offline, until the copy is removed.
        fetched = sorted(d for d in home.glob("*") if COMMIT.match(d.name) and _fetched(d))
        if len(fetched) > 1:
            raise ValueError(f"model {repository}: {len(fetched)} commits are fetched ({', '.join(d.name for d in fetched)}); "
                             f"name one as {repository}@<commit>")
        if fetched:
            return fetched[0]
    if not fetch:
        return home / (revision or "main")
    return download(repository, revision or "main")


def _fetched(directory: Path) -> bool:
    return (directory / "config.json").is_file()


def _get(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "e2e-harness"})
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(request, timeout=TIMEOUT)


def download(repository: str, revision: str) -> Path:
    """Fetch the repository's files at ``revision``, weights left out, into
    its commit's directory; the directory."""
    quoted = urllib.parse.quote(revision, safe="")
    try:
        with _get(f"{HUB}/api/models/{repository}/revision/{quoted}") as response:
            info = json.load(response)
        commit = info["sha"]
        target = MODELS / repository.replace("/", "--") / commit
        if _fetched(target):
            return target
        names = [s["rfilename"] for s in info.get("siblings", []) if not s["rfilename"].lower().endswith(WEIGHTS)]
        if "config.json" not in names:
            raise ValueError(f"model {repository}@{revision} has no config.json")
        partial = target.with_name(target.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        for name in names:
            destination = partial / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with _get(f"{HUB}/{repository}/resolve/{commit}/{urllib.parse.quote(name)}") as response, \
                    destination.open("wb") as file:
                shutil.copyfileobj(response, file)
        partial.rename(target)
        return target
    except urllib.error.HTTPError as error:
        hint = " (a gated or private model needs HF_TOKEN)" if error.code in (401, 403) else ""
        raise ValueError(f"cannot fetch model {repository}@{revision} from {HUB}: HTTP {error.code}{hint}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ValueError(f"cannot fetch model {repository}@{revision} from {HUB}: {error}") from None
