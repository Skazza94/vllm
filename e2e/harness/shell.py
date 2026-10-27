"""Running docker and privileged network commands."""

from __future__ import annotations

import subprocess


class CommandError(RuntimeError):
    def __init__(self, command: list[str], code: int, output: str) -> None:
        super().__init__(f"{' '.join(command)} exited with {code}: {output.strip()}")
        self.command, self.code, self.output = command, code, output


def run(command: list[str], *, check: bool = True, input: str | None = None,
        timeout: float | None = 120) -> str:
    """Run ``command``; returns its stdout, raises CommandError on failure."""

    result = subprocess.run(command, input=input, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise CommandError(command, result.returncode, result.stderr or result.stdout)
    return result.stdout


def sudo(*command: str, check: bool = True) -> str:
    """A privileged command in the host's namespaces (password-less sudo)."""

    return run(["sudo", "-n", *command], check=check)


def in_netns(pid: int, *command: str, check: bool = True) -> str:
    """A host command (ip, tc) inside the network namespace of process ``pid``."""

    return sudo("nsenter", "-t", str(pid), "-n", *command, check=check)
