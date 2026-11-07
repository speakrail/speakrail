"""Local tools for the live harness. Declarations come verbatim from the training catalog (data/tools_catalog.json),
and results are shaped like the training scripts' results
(e.g. list_add -> {list, count}, dice_roll -> {dice, total}, stopwatch -> {elapsed_s, laps}), so the model sees what it learned.

  record_note / list_add / todo_add   in-memory per session          counter      inc / dec / reset / get
  stopwatch                           start / stop / lap / read / reset (wall clock)
  get_weather                         Open-Meteo (geocoding + forecast, no key); "home"/"here"/"" = DEFAULT_CITY
  unit_convert                        length, mass, volume, speed, temperature (factor tables)
  calculator                          safe arithmetic (ast: numbers, + - * / // % **, parentheses, a few math functions)
  dice_roll                           NdM(+/-K), e.g. "2d6+1"            get_time     {iso_time, weekday}, optional IANA zone
"""
from __future__ import annotations

import ast
import datetime
import json
import math
import operator
import os
import random
import re
import time
import zoneinfo

import aiohttp

from ..config import CFG

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tools_catalog.json")) as _f:
    CAT = json.load(_f)["tools"]
LOCAL_TOOLS = ["get_weather", "record_note", "list_add", "todo_add", "counter", "stopwatch", "unit_convert", "calculator",
               "dice_roll", "get_time"]
STATEFUL_TOOLS = {"record_note", "list_add", "todo_add", "counter", "stopwatch"}   # side effects: not run on an unadopted spec
DEFAULT_CITY = CFG.home_city
DEFAULT_TZ = CFG.home_timezone
# reset_chat: the session's own tool (not in the training catalog), run by MicroSession, never by the ToolBox. The
# harness enforces the confirmation: see MicroSession._reset_tool
RESET_TOOL = "reset_chat"
RESET_TOOL_SPEC = {"domain": "system", "mode": "sync", "args": {"confirmed?": "boolean", "instructions?": "string"},
                   "arg_descriptions": {"confirmed": "true only after the user said yes to the reset",
                                        "instructions": "the user's standing instructions for the new chat, if they gave any"},
                   "description": "Wipe this whole conversation and start over with an empty context. Optional instructions "
                                  "become the user's standing instructions in the new chat. Always ask the user first, and call "
                                  "with confirmed=true only after they said yes."}

WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog", 51: "light drizzle",
       53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle", 61: "light rain", 63: "rain",
       65: "heavy rain", 66: "freezing rain", 67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
       77: "snow grains", 80: "light rain showers", 81: "rain showers", 82: "heavy rain showers", 85: "snow showers",
       86: "heavy snow showers", 95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail"}

# unit -> (dimension, factor to the base unit); temperature is handled separately
UNITS = {}
def _u(dim, factor, *names):
    for n in names: UNITS[n] = (dim, factor)
_u("len", 1.0, "m", "meter", "meters", "metre", "metres"); _u("len", 1000.0, "km", "kilometer", "kilometers", "kilometre", "kilometres")
_u("len", 0.01, "cm", "centimeter", "centimeters", "centimetre", "centimetres"); _u("len", 0.001, "mm", "millimeter", "millimeters")
_u("len", 1609.344, "mi", "mile", "miles"); _u("len", 0.3048, "ft", "foot", "feet"); _u("len", 0.0254, "in", "inch", "inches")
_u("len", 0.9144, "yd", "yard", "yards")
_u("mass", 1.0, "g", "gram", "grams"); _u("mass", 1000.0, "kg", "kilogram", "kilograms", "kilo", "kilos")
_u("mass", 453.59237, "lb", "lbs", "pound", "pounds"); _u("mass", 28.349523, "oz", "ounce", "ounces"); _u("mass", 0.001, "mg", "milligram", "milligrams")
_u("vol", 1.0, "ml", "milliliter", "milliliters", "millilitre", "millilitres"); _u("vol", 1000.0, "l", "liter", "liters", "litre", "litres")
_u("vol", 236.588, "cup", "cups"); _u("vol", 14.787, "tbsp", "tablespoon", "tablespoons"); _u("vol", 4.929, "tsp", "teaspoon", "teaspoons")
_u("vol", 29.574, "fl oz", "fluid ounce", "fluid ounces"); _u("vol", 3785.41, "gal", "gallon", "gallons"); _u("vol", 473.176, "pint", "pints")
_u("speed", 1.0, "km/h", "kph", "kmh", "kilometers per hour"); _u("speed", 1.609344, "mph", "miles per hour"); _u("speed", 3.6, "m/s", "meters per second")
TEMP = {"c": "C", "celsius": "C", "°c": "C", "f": "F", "fahrenheit": "F", "°f": "F", "k": "K", "kelvin": "K"}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log, "log10": math.log10, "sin": math.sin, "cos": math.cos,
          "tan": math.tan, "exp": math.exp, "floor": math.floor, "ceil": math.ceil}
_CONST = {"pi": math.pi, "e": math.e}


def _calc(node):
    if isinstance(node, ast.Expression): return _calc(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _calc(node.left), _calc(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 100: raise ValueError("exponent too large")
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS: return _OPS[type(node.op)](_calc(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_calc(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONST: return _CONST[node.id]
    raise ValueError("unsupported expression")


def _num(v):
    v = round(float(v), 4)
    return int(v) if v == int(v) else v


class ToolBox:
    def __init__(self, http: aiohttp.ClientSession):
        self.http = http
        self.notes: list[str] = []; self.lists: dict[str, list[str]] = {}; self.todos: list[dict] = []
        self.counters: dict[str, int] = {}; self.watches: dict[str, dict] = {}

    async def call(self, name: str, args: dict) -> dict:
        try:
            return await getattr(self, "t_" + name)(**{k.rstrip("?"): v for k, v in (args or {}).items()})
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}
        except Exception as e:                                           # noqa: BLE001 -- the reply must go on
            return {"error": f"{type(e).__name__}: {e}"}

    async def t_record_note(self, text=""):
        self.notes.append(str(text)); return {"saved": True, "count": len(self.notes)}

    async def t_list_add(self, list="list", item=""):                    # noqa: A002 -- the catalog's argument name
        items = self.lists.setdefault(str(list).lower(), []); items.append(str(item))
        return {"list": str(list), "count": len(items)}

    async def t_todo_add(self, text="", due=""):
        self.todos.append({"text": str(text), "due": str(due or "")}); return {"ok": True, "count": len(self.todos)}

    async def t_counter(self, name="count", action="get"):
        k = str(name).lower(); v = self.counters.get(k, 0)
        v = {"inc": v + 1, "dec": v - 1, "reset": 0}.get(str(action), v)
        self.counters[k] = v; return {"value": v}

    async def t_stopwatch(self, action="read", name="default"):
        w = self.watches.setdefault(str(name or "default").lower(), {"start": None, "acc": 0.0, "laps": []})
        now = time.monotonic()
        el = lambda: w["acc"] + (now - w["start"] if w["start"] is not None else 0.0)
        a = str(action)
        if a == "start":
            if w["start"] is None: w["start"] = now
        elif a == "stop":
            w["acc"] = el(); w["start"] = None
        elif a == "lap":
            w["laps"].append(round(el(), 1))
        elif a == "reset":
            w.update(start=None, acc=0.0, laps=[])
        out = {"elapsed_s": round(el(), 1) if a != "reset" else 0}
        if w["laps"]: out["laps"] = list(w["laps"])
        return out

    async def t_get_weather(self, location=""):
        loc = str(location or "").strip()
        if loc.lower() in ("", "home", "here", "current location", "my location"): loc = DEFAULT_CITY
        to = aiohttp.ClientTimeout(total=4)
        async with self.http.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": loc, "count": 1}, timeout=to) as r:
            g = (await r.json()).get("results") or []
        if not g:
            return {"error": f"no place called {loc}"}
        g = g[0]
        params = {"latitude": g["latitude"], "longitude": g["longitude"], "timezone": "auto", "forecast_days": 1,
                  "current": "temperature_2m,weather_code,wind_speed_10m",
                  "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"}
        async with self.http.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=to) as r:
            d = await r.json()
        cur, day = d.get("current") or {}, d.get("daily") or {}
        first = lambda k: (day.get(k) or [None])[0]
        rain = first("precipitation_probability_max")
        return {"location": g.get("name", loc), "condition": WMO.get(cur.get("weather_code"), "unknown"),
                "temp_c": round(cur.get("temperature_2m", 0)), "wind_kmh": round(cur.get("wind_speed_10m", 0)),
                "forecast": f"{WMO.get(first('weather_code'), 'mixed')} today, high {round(first('temperature_2m_max') or 0)}, "
                            f"low {round(first('temperature_2m_min') or 0)}" + (f", {rain}% chance of rain" if rain is not None else "")}

    async def t_unit_convert(self, value=0, **kw):
        src, dst = str(kw.get("from", "")).strip().lower(), str(kw.get("to", "")).strip().lower()
        v = float(value)
        if src in TEMP and dst in TEMP:
            a, b = TEMP[src], TEMP[dst]
            c = {"C": v, "F": (v - 32) * 5 / 9, "K": v - 273.15}[a]
            out = {"C": c, "F": c * 9 / 5 + 32, "K": c + 273.15}[b]
            return {"value": _num(out), "unit": kw.get("to")}
        if src in UNITS and dst in UNITS and UNITS[src][0] == UNITS[dst][0]:
            return {"value": _num(v * UNITS[src][1] / UNITS[dst][1]), "unit": kw.get("to")}
        if src in UNITS and dst in UNITS and {UNITS[src][0], UNITS[dst][0]} == {"vol", "mass"}:   # cups to grams: as water
            return {"value": _num(v * UNITS[src][1] / UNITS[dst][1]), "unit": kw.get("to"), "note": "assuming water density"}
        return {"error": f"can't convert {kw.get('from')} to {kw.get('to')}"}

    async def t_calculator(self, expression=""):
        e = str(expression).lower().replace("^", "**").replace("×", "*").replace("÷", "/").replace(",", "")
        e = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"(\1/100)*", e)            # 15% of 80
        e = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d(])", r"(\1/100)", e)        # 80 * 15%  (x % y stays modulo)
        e = re.sub(r"(?<=[\d)])\s*x\s*(?=[\d(])", "*", e)                          # 3 x 4
        try:
            v = _calc(ast.parse(e, mode="eval"))
        except Exception:
            return {"error": "parse error"}
        return {"value": _num(v)}

    async def t_dice_roll(self, expr="1d6"):
        m = re.fullmatch(r"\s*(\d*)\s*d\s*(\d+)\s*([+-]\s*\d+)?\s*", str(expr).lower())
        if not m:
            return {"error": f"can't read {expr}"}
        n, sides = int(m.group(1) or 1), int(m.group(2)); mod = int((m.group(3) or "0").replace(" ", ""))
        if not (1 <= n <= 100 and 2 <= sides <= 1000):
            return {"error": "too many dice"}
        dice = [random.randint(1, sides) for _ in range(n)]
        return {"dice": dice, "total": sum(dice) + mod}

    async def t_get_time(self, timezone=""):
        try:
            tz = zoneinfo.ZoneInfo(str(timezone)) if timezone else zoneinfo.ZoneInfo(DEFAULT_TZ)
        except Exception:
            return {"error": f"unknown time zone {timezone}"}
        now = datetime.datetime.now(tz).replace(microsecond=0)
        return {"iso_time": now.isoformat(), "weekday": now.strftime("%A")}
