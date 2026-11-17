"""Display argv without executing it or interpreting shell/Python source."""

import re
import shlex
from pathlib import PurePosixPath


def _unwrap(argv):
    """Remove only understood timeout/env prefixes; unknown syntax stays literal."""
    while argv:
        name = PurePosixPath(argv[0]).name
        if name not in {"timeout", "env"}:
            return argv
        index = 1
        while index < len(argv):
            arg = argv[index]
            if arg == "--":
                index += 1
                break
            flags = (
                {"--foreground", "--preserve-status", "--verbose"}
                if name == "timeout"
                else {"-i", "--ignore-environment"}
            )
            values = (
                {"-s", "--signal", "-k", "--kill-after"}
                if name == "timeout"
                else {"-u", "--unset", "-C", "--chdir"}
            )
            if arg in flags:
                index += 1
            elif arg in values:
                if index + 1 >= len(argv) or not argv[index + 1]:
                    return None
                index += 2
            elif any(arg.startswith(option + "=") for option in values if option.startswith("--")):
                if not arg.split("=", 1)[1]:
                    return None
                index += 1
            elif arg.startswith("-"):
                return None
            else:
                break
        if name == "timeout":
            if index >= len(argv) or not re.fullmatch(
                r"(?:\d+(?:\.\d*)?|\.\d+)[smhd]?", argv[index]
            ):
                return None
            index += 1
        else:
            while index < len(argv) and re.match(r"[A-Za-z_][A-Za-z_0-9]*=", argv[index]):
                index += 1
        if index >= len(argv):
            return None
        argv = argv[index:]
    return None


def _short_path(value):
    if value.startswith("/") and value != "/":
        return "…/" + PurePosixPath(value).name
    if value.startswith("--") and "=" in value:
        option, argument = value.split("=", 1)
        return option + "=" + _short_path(argument)
    return value


def command_summary(argv):
    """Keep actual command/arguments recognizable; full argv is a separate detail."""
    command = _unwrap(argv)
    if command is None:
        text = shlex.join(argv)
    else:
        # Shell source and Python -c payloads are opaque, even when they contain paths.
        executable = PurePosixPath(command[0]).name
        if executable in {"sh", "bash", "dash", "zsh", "fish", "csh", "tcsh"}:
            text = shlex.join(command)
        else:
            display = [executable]
            for index, argument in enumerate(command[1:], 1):
                opaque = (
                    re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable)
                    and command[index - 1] == "-c"
                )
                display.append(argument if opaque else _short_path(argument))
            text = shlex.join(display)
    return text if len(text) <= 500 else text[:499] + "…"
