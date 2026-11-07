"""speakrail run | start | stop | status | logs | load | offload | config | doctor | bench | engine | install | update | rollback | uninstall"""
from __future__ import annotations

import argparse
import asyncio
import signal
import subprocess
import threading

from .contracts.conformance import check
from .contracts.fidelity import compare
from .manifest import ManifestError, discover
from .supervisor import EngineError, Supervisor, install
from .ui import say


def _engine(engines, name):
    if name not in engines:
        raise SystemExit(f"no engine {name!r}; known: {', '.join(sorted(engines)) or '(none)'}")
    return engines[name]


def _params(items) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--param wants key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def cmd_list(engines, _):
    for e in sorted(engines.values(), key=lambda x: x.id):
        where = "here" if e.runs_on() else f"only on {', '.join(e.platforms)}"
        print(f"{e.id:18} {where:18} {e.description}")


def cmd_show(engines, a):
    e = _engine(engines, a.engine)
    print(f"{e.id}: {e.description}\n  dir          {e.dir}\n  capabilities {', '.join(sorted(e.capabilities)) or '-'}\n  platforms    {', '.join(e.platforms)}\n  ports        {e.ports}\n  params       {e.params}")


def cmd_install(engines, a):
    e = _engine(engines, a.engine)
    try:
        env = install(e)
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        print(f"{e.id}: install failed: {err}")
        return 1
    print(f"{e.id}: {'environment at ' + str(env) if env else 'nothing to install'}")


def _say(engine, state, detail):
    say(f"[{engine}] {state}{': ' + detail if detail else ''}")


def cmd_run(engines, a):
    """start one engine in the foreground. Exits non-zero if it fails."""
    e = _engine(engines, a.engine)
    done = threading.Event()
    failed = []

    def on_event(engine, state, detail):
        _say(engine, state, detail)
        if state == "failed":
            failed.append(detail)
            done.set()

    sup = Supervisor(on_event=on_event)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: done.set())
    try:
        sup.start(e, _params(a.param))
        print(f"[{e.id}] {sup.url(e.id)}", flush=True)
        done.wait()
    except EngineError as err:
        print(f"[{e.id}] {err}", flush=True)
        return 1
    finally:
        sup.close()
    return 1 if failed else 0


def _start_for_test(engine, url, params):
    if url:
        return None, url
    sup = Supervisor(on_event=_say)
    sup.start(engine, params)
    return sup, sup.url(engine.id)


def cmd_test(engines, a):
    e = _engine(engines, a.engine)
    sup = None
    try:
        sup, url = _start_for_test(e, a.url, _params(a.param))
        results = asyncio.run(check(e, url))
    except EngineError as err:
        print(f"{e.id}: can't start: {err}")
        return 1
    finally:
        if sup:
            sup.close()
    for r in results:
        print(f"{'PASS' if r.ok else 'FAIL'}  {r.name}{'  (' + r.detail + ')' if r.detail and not r.ok else ''}")
    failed = sum(not r.ok for r in results)
    print(f"{e.id}: {len(results) - failed} of {len(results)} passed")
    return 1 if failed or not results else 0


def cmd_fidelity(engines, a):
    e = _engine(engines, a.engine)
    if e.role != "llm":
        raise SystemExit("fidelity applies to LLM engines")
    sup = None
    try:
        sup, url = _start_for_test(e, a.url, _params(a.param))
        r = asyncio.run(compare(url, a.reference))
    except EngineError as err:
        print(f"{e.id}: can't start: {err}")
        return 1
    finally:
        if sup:
            sup.close()
    print(f"{e.id}: same decision on {r.same} of {r.total}; largest logprob difference {r.max_delta:.3f}, {r.max_delta_likely:.3f} where likely")
    for d in r.differing:
        print(f"  prompt {d['index']}: expected {d['expected']}, got {d['got']}")
    return 0 if r.ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="speakrail")
    top = p.add_subparsers(dest="group", required=True)
    eng = top.add_parser("engine", help="the engines: list, install, run and test them").add_subparsers(dest="cmd", required=True)
    eng.add_parser("list").set_defaults(fn=cmd_list)
    for name, fn, helptext in (("show", cmd_show, "what a manifest declares"), ("install", cmd_install, "create the engine's environment"),
                               ("run", cmd_run, "start one engine in the foreground"), ("test", cmd_test, "conformance checks"),
                               ("fidelity", cmd_fidelity, "compare decisions with a reference")):
        s = eng.add_parser(name, help=helptext)
        s.add_argument("engine")
        s.set_defaults(fn=fn)
        if name in ("run", "test", "fidelity"):
            s.add_argument("--param", action="append", help="key=value, overrides a manifest param")
        if name in ("test", "fidelity"):
            s.add_argument("--url", help="test an engine that is already running")
        if name == "fidelity":
            s.add_argument("--reference", required=True)
    def choices(sp):
        sp.add_argument("--config", help="use this config instead of choosing one")
        sp.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a setting, for example notes=true")
        sp.add_argument("--param", action="append", metavar="ENGINE:KEY=VALUE", help="override an engine param, for example asr/audiocpp:threads=2")
        sp.add_argument("--force", action="store_true", help="try a config that doesn't fit this machine's memory")

    run_p = top.add_parser("run", help="start everything for this machine, in this terminal")
    start_p = top.add_parser("start", help="start everything in the background")
    for sp in (run_p, start_p):
        choices(sp)
        sp.add_argument("--no-bench", action="store_true", help="skip the first-launch benchmark")
    top.add_parser("stop", help="stop what's running, from any terminal")
    top.add_parser("status", help="what's running and how much memory it holds")
    logs_p = top.add_parser("logs", help="show a log")
    logs_p.add_argument("name", nargs="?", help="harness, or an engine like asr-audiocpp; without it, the background service's")
    logs_p.add_argument("-n", dest="lines", type=int, default=40, help="how many lines")
    logs_p.add_argument("-f", dest="follow", action="store_true", help="keep following it")
    top.add_parser("load", help="load the engines again").add_argument("--force", action="store_true", help="even if they may not fit in memory")
    top.add_parser("offload", help="unload the engines until `speakrail load`; the web page stays up").add_argument("--force", action="store_true", help="even if it ends an open session")
    cfg_p = top.add_parser("config", help="what would run here, and why")
    cfg_p.add_argument("action", nargs="?", choices=["explain", "list"], default="explain")
    choices(cfg_p)
    choices(top.add_parser("doctor", help="check that this machine is ready"))
    choices(top.add_parser("bench", help="measure this machine again"))
    inst = top.add_parser("install", help="what install.sh runs once Python is ready")
    inst.add_argument("--yes", action="store_true", help="no questions: the choices it would suggest")
    inst.add_argument("--config", help="install this config instead of picking one from the list")
    inst.add_argument("--force", action="store_true", help="with --config: even if it needs more memory than this machine has")
    inst.add_argument("--models", help="where the models go, instead of <home>/models")
    inst.add_argument("--no-modify-path", action="store_true", help="don't add speakrail to your shell's PATH")
    inst.add_argument("--fresh", action="store_true", help=argparse.SUPPRESS)      # install.sh made the folder just now, so Cancel can take it away
    inst.add_argument("--chosen", action="store_true", help=argparse.SUPPRESS)     # --config is what auto mode chose last time (a rollback), not the user's pick
    upd = top.add_parser("update", help="install the newest version; the one before stays for speakrail rollback")
    upd.add_argument("--yes", action="store_true", help="no questions")
    upd.add_argument("--check", action="store_true", help="only say whether there is a newer version")
    top.add_parser("rollback", help="go back to the version before the last update").add_argument("--yes", action="store_true", help="no questions")
    un = top.add_parser("uninstall", help="remove Speakrail from this machine")
    un.add_argument("--yes", action="store_true", help="no questions")
    un.add_argument("--keep-models", action="store_true", help="leave the downloaded models where they are")
    un.add_argument("--keep-images", action="store_true", help="leave the Docker images")
    a = p.parse_args(argv)
    try:
        if a.group == "engine":
            return a.fn(discover(), a) or 0
        if a.group == "install":
            from . import installer
            return installer.main(a)
        if a.group in ("update", "rollback", "uninstall"):
            from . import lifecycle
            return {"update": lifecycle.update_main, "rollback": lifecycle.rollback_main, "uninstall": lifecycle.uninstall_main}[a.group](a)
        from . import doctor, run, service
        commands = {"start": service.start_main, "stop": service.stop_main, "status": service.status_main, "logs": service.logs_main, "load": service.load_main, "offload": service.offload_main}
        if a.group in commands:
            return commands[a.group](a)
        if a.group == "run":
            return run.main(a)
        if a.group == "doctor":
            return doctor.main(a)
        if a.group == "bench":
            return run.bench_main(a)
        return doctor.list_main(a) if a.action == "list" else doctor.explain_main(a)
    except ManifestError as e:
        print(f"manifest error: {e}")
        return 2
