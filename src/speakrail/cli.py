"""speakrail engine list | show | install | run | test | fidelity"""
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
    print(f"[{engine}] {state}{': ' + detail if detail else ''}", flush=True)


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
    a = p.parse_args(argv)
    try:
        return a.fn(discover(), a) or 0
    except ManifestError as e:
        print(f"manifest error: {e}")
        return 2
