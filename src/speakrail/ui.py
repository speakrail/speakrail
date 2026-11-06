"""Color for the terminal, in the colors of speakrail.com. Only when a person is looking: never in a log or a pipe, or with NO_COLOR.

Everything Speakrail prints goes through paint() line by line, so a message needs no markup and a log file stays plain text. `speakrail start`
paints what it reads from the service's log the same way. These are the 256 colors, which every terminal has.
"""
from __future__ import annotations

import re
import sys

from .config import CFG

RESET = "\033[0m"
PINK, GREEN, YELLOW, RED, GREY, WHITE = "\033[38;5;218m", "\033[38;5;114m", "\033[38;5;179m", "\033[38;5;210m", "\033[38;5;245m", "\033[38;5;255m"
BOLD, UNDER = "\033[1m", "\033[4m"
STATES = {"ready": GREEN, "loading": YELLOW, "starting": YELLOW, "unloading": YELLOW, "stopped": GREY, "failed": RED + BOLD}


def _c(style: str, text: str) -> str:
    return f"{style}{text}{RESET}"


def _state(word: str) -> str:
    return _c(STATES.get(word, ""), word) if word in STATES else word


def _links(text: str) -> str:
    return re.sub(r"(?:https?|wss?)://\S+", lambda m: _c(PINK + UNDER, m[0]), text)


RULES = [
    (r"^(Speakrail is ready)(: )(.*)$", lambda m: _c(GREEN + BOLD, m[1]) + m[2] + _links(m[3])),
    (r"^(Stop it with: )(.*)$", lambda m: _c(GREY, m[1]) + _c(WHITE + BOLD, m[2])),
    (r"^API \(.*$", lambda m: _links(_c(GREY, m[0]).replace(RESET, "") + RESET)),
    (r"^\[([a-z]+/[^\]]+)\] (\w+)(.*)$", lambda m: _c(PINK, f"[{m[1]}]") + " " + _state(m[2]) + _c(GREY, m[3])),                      # an engine's state
    (r"^(ok)(\s+)(.*)$", lambda m: _c(GREEN, m[1]) + m[2] + m[3]),                                                              # doctor
    (r"^(warn)(\s+)(.*)$", lambda m: _c(YELLOW, m[1]) + m[2] + _c(YELLOW, m[3])),
    (r"^(FAIL)(\s+)(.*)$", lambda m: _c(RED + BOLD, m[1]) + m[2] + _c(RED, m[3])),
    (r"^(\s+)(-> .*)$", lambda m: m[1] + _c(YELLOW, m[2])),
    (r"^Ready to run\.$", lambda m: _c(GREEN + BOLD, m[0])),
    (r"^\d+ problems? to fix\.$", lambda m: _c(RED + BOLD, m[0])),
    (r"^(Warning:?)(\s+)(.*)$", lambda m: _c(YELLOW + BOLD, m[1]) + m[2] + _c(YELLOW, m[3])),                                   # sections of status and explain
    (r"^(Error)(\s+)(.*)$", lambda m: _c(RED + BOLD, m[1]) + m[2] + _c(RED, m[3])),
    (r"^(Engines)(\s+)(\w+)(.*)$", lambda m: _c(PINK + BOLD, m[1]) + m[2] + _state(m[3]) + m[4]),
    (r"^(Machine|Config|Memory|Engines|Idle|Benchmark|Installs|Models|Build|Path)(\s+)(.*)$", lambda m: _c(PINK + BOLD, m[1]) + m[2] + m[3]),
    (r"^(Features|Warnings|Not chosen|Licenses)$", lambda m: _c(PINK + BOLD, m[0])),
    (r"^(\s+)(Why:|Tested on)(.*)$", lambda m: _c(GREY, m[0])),
    (r"^(  .{20} )(on|off)(\s+)(.*)$", lambda m: m[1] + _c(GREEN if m[2] == "on" else GREY, m[2]) + m[3] + _c(GREY, m[4])),      # a feature
    (r"^(  .{20} )(\S+)(\s+)(from the config|default|.*:.*)$", lambda m: m[1] + m[2] + m[3] + _c(GREY, m[4])),
    (r"^(  )(\S+/\S+)(\s+)(ready|starting|stopped|failed)(.*)$", lambda m: m[1] + m[2] + m[3] + _state(m[4]) + _c(GREY, m[5])),   # an engine in the status
    (r"^(Running)( .*)$", lambda m: _c(GREEN + BOLD, m[1]) + _links(m[2])),
    (r"^(Already running)(.*)$", lambda m: _c(YELLOW + BOLD, m[1]) + _links(m[2])),
    (r"^Not running.*$", lambda m: _c(YELLOW, m[0])),
    (r"^(Stopped\.?|Loaded\b.*|Offloaded\b.*|Measured:.*|Phrase cache ready.*)$", lambda m: _c(GREEN, m[0])),
    (r"^Speakrail is installed\.$", lambda m: _c(GREEN + BOLD, m[0])),
    (r"^(Measuring|Rendering|Changed:|Building|Setting up|Fetching|\[\d+/\d+\]).*$", lambda m: _c(PINK, m[0])),
    (r"^(Tight on memory|This machine is|Not enough memory|Benchmark failed|Phrase cache not rendered).*$", lambda m: _c(YELLOW, m[0])),
    (r"^(Engine failed|Stopping:|The harness|It stopped|Port \d+|Voice ).*$", lambda m: _c(RED, m[0])),
    (r"^(No session for|Critically short).*$", lambda m: _c(YELLOW, m[0])),
]
RULES = [(re.compile(p), fn) for p, fn in RULES]


def color_on(stream=None) -> bool:
    return not CFG.no_color and (stream or sys.stdout).isatty()


def paint(text: str, on: bool | None = None) -> str:
    if not (color_on() if on is None else on):
        return text
    out = []
    for line in text.split("\n"):
        for rule, fn in RULES:
            if (m := rule.match(line)):
                line = fn(m)
                break
        out.append(line)
    return "\n".join(out)


def say(text: str = "") -> None:
    print(paint(text), flush=True)
