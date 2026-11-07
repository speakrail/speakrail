"""`speakrail doctor`: can this machine run the setup auto mode would choose, and if not, what to do about it."""
from __future__ import annotations

import sys
from pathlib import Path

from . import containers
from .config import CFG
from .contracts import REQUIRED
from .manifest import render, values_for
from .resolver import ResolveError, explain
from .run import missing_voice, paths, setup
from .supervisor import port_free
from .ui import say

OK, WARN, FAIL = "ok", "warn", "FAIL"


def checks(args) -> list[tuple[str, str, str]]:
    """-> (status, what, what to do)"""
    out = []
    home, models = paths()
    out.append((OK if sys.version_info >= (3, 12) else FAIL, f"Python {sys.version_info.major}.{sys.version_info.minor}", "" if sys.version_info >= (3, 12) else "Speakrail needs Python 3.12"))
    try:
        r, _, _ = setup(args)
    except (ResolveError, ValueError) as e:
        out.append((FAIL, "No setup for this machine", str(e)))
        return out
    hw, c = r.hardware, r.config
    out.append((OK, f"Machine: {hw.describe()}", ""))
    out.append((OK, f"Config: {c.name} ({c.status})", ""))
    if hw.free_memory_gib < c.memory_peak_gib:
        out.append((WARN, f"{hw.free_memory_gib:.1f} GB free now, the setup peaks at {c.memory_peak_gib:.1f} GB", "Close other apps first"))
    else:
        out.append((OK, f"{hw.free_memory_gib:.1f} GB free, the setup peaks at {c.memory_peak_gib:.1f} GB", ""))
    missing_models = False
    for role, e in r.engines.items():
        values = values_for(e, home, models, home / "run" / e.slug, r.params.get(e.id))
        problem_docker = containers.problem(gpu=e.needs_gpu_runtime) if e.in_containers else ""
        missing_images = [] if problem_docker else [image for image in {render(p.image, values, e.id) for p in e.processes if p.image} if not containers.image_present(image)]
        if e.in_containers and problem_docker:
            out.append((FAIL, f"{e.id}: can't run its containers", problem_docker))
        elif missing_images:
            out.append((FAIL, f"{e.id}: image {missing_images[0]} isn't on this machine", f"docker pull {missing_images[0]}"))
        elif e.python and not Path(values["python"]).exists():
            out.append((FAIL, f"{e.id}: no environment", f"speakrail engine install {e.id}"))
        else:
            out.append((OK, f"{e.id}: installed", ""))
        for need in e.needs:
            path = render(need, values, e.id)
            if not Path(path).exists():
                missing_models = True
                out.append((FAIL, f"{e.id}: missing {path}", "The installer fetches this"))
        for port in e.ports.values():
            if not port_free(int(port)):
                out.append((FAIL, f"{e.id}: port {port} is in use", "Stop what's using it"))
        for cap in REQUIRED.get(role, ()):
            if cap not in e.capabilities:
                out.append((FAIL, f"{e.id}: doesn't provide {cap}", "This engine can't run with the harness"))
    tok = r.harness.get("tokenizer_dir", CFG.tokenizer_dir)
    if not (Path(tok) / "tokenizer.json").exists():
        missing_models = True
        out.append((FAIL, f"Tokenizer missing: {tok}", "The installer fetches this"))
    else:
        out.append((OK, "Tokenizer found", ""))
    if (voice := r.harness.get("voice")) and "/" in voice and not Path(voice).exists():
        out.append((FAIL, f"Voice missing: {voice}", "Pick another one with --set voice=..."))
    if (problem := missing_voice(r)):
        out.append((FAIL, problem, ""))
    if missing_models and hw.disk_free_gib < c.disk_gib:
        out.append((FAIL, f"{hw.disk_free_gib:.0f} GB disk free, the models need {c.disk_gib:.0f} GB", "Free some disk space"))
    else:
        out.append((OK, f"{hw.disk_free_gib:.0f} GB disk free", ""))
    if not port_free(CFG.ui_port):
        out.append((FAIL, f"UI port {CFG.ui_port} is in use", "Pick another with UI_PORT"))
    for w in r.warnings:
        out.append((WARN, w, ""))
    return out


def main(args) -> int:
    results = checks(args)
    for status, what, fix in results:
        say(f"{status:5} {what}" + (f"\n      -> {fix}" if fix else ""))
    bad = sum(s == FAIL for s, _, _ in results)
    say("\nReady to run." if not bad else f"\n{bad} problem{'s' if bad != 1 else ''} to fix.")
    return 1 if bad else 0


def explain_main(args) -> int:
    try:
        r, configs, engines = setup(args)
        from .run import stored_benchmark
        if (stored := stored_benchmark(r)):
            r, configs, engines = setup(args, benchmark=stored)
    except (ResolveError, ValueError) as e:
        say(e)
        return 2
    say(explain(r))
    return 0


def list_main(args) -> int:
    from .hardware import detect
    from .resolver import _too_big, _unusable, discover_configs
    from .manifest import discover
    hw = detect()
    configs, engines = discover_configs(home=paths()[0]), discover()
    for c in sorted(configs.values(), key=lambda x: (-x.priority, x.name)):
        why = None if hw is None else (_unusable(c, hw, engines) or _too_big(c, hw))
        say(f"{c.name:16} {c.status:13} {'fits here' if hw and not why else 'not here: ' + (why or 'no hardware found'):50} {c.description}")
    return 0
