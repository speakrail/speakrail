"""Tool argument schemas: the catalog shorthand -> the JSON Schema the model sees in its tool declarations
(protocol.py). Pure Python, no dependencies.

Shorthand (catalog `args` values; a trailing '?' on a key = optional):
  string | int | number | bool | date | datetime | time | email | phone | url | currency | money | enum:a|b|c
  list<T> | object{key: T, key?: T, ...} | object (free-form)
Tools may instead carry a ready JSON Schema in `parameters` (imported tools, hand-written complex ones).
"""
from __future__ import annotations
import re

FORMATS = {   # shorthand base type -> JSON Schema (descriptions tell the model the exact format)
    "date": {"type": "string", "format": "date", "description": "YYYY-MM-DD"},
    "datetime": {"type": "string", "format": "date-time", "description": "local time, YYYY-MM-DDTHH:MM"},
    "time": {"type": "string", "description": "HH:MM, 24-hour"},
    "email": {"type": "string", "format": "email"},
    "phone": {"type": "string", "description": "international format, e.g. +14155550123"},
    "url": {"type": "string", "format": "uri"},
    "currency": {"type": "string", "description": "ISO 4217 code, e.g. EUR"},
}
BASE = {"string": "string", "str": "string", "int": "integer", "integer": "integer", "number": "number", "float": "number",
        "double": "number", "bool": "boolean", "boolean": "boolean", "dict": "object", "object": "object", "list": "array",
        "array": "array", "any": "string"}


# ------------------------------------------------------------------ shorthand -> JSON Schema
def _split(s, sep=","):
    """split at top-level separators (not inside <>, {})"""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<{": depth += 1
        elif ch in ">}": depth -= 1
        if ch == sep and depth == 0: out.append(cur); cur = ""
        else: cur += ch
    if cur.strip(): out.append(cur)
    return [x.strip() for x in out]


def parse(t):
    """shorthand type string -> JSON Schema dict (v2 shorthand gives exactly what protocol.py produced before)"""
    t = (t or "string").strip()
    if t.startswith("enum:"): return {"type": "string", "enum": t[5:].split("|")}
    m = re.fullmatch(r"list<(.+)>", t, re.S)
    if m: return {"type": "array", "items": parse(m.group(1))}
    m = re.fullmatch(r"object\s*\{(.*)\}", t, re.S)
    if m: return parse_fields(m.group(1))
    if t == "money": return parse_fields("amount: number, currency: currency")
    if t in FORMATS: return dict(FORMATS[t])
    if t.startswith("object"): return {"type": "object"}
    return {"type": BASE.get(t, "string")}


def parse_fields(body, descriptions=None):
    props, req = {}, []
    for f in _split(body):
        if ":" not in f: k, v = f, "string"                        # bare field name = string
        else: k, v = (x.strip() for x in f.split(":", 1))
        key = k.rstrip("?")
        s = parse(v)
        if descriptions and key in descriptions: s = {"description": descriptions[key], **{a: b for a, b in s.items() if a != "description"}}
        props[key] = s
        if not k.endswith("?"): req.append(key)
    return {"type": "object", "properties": props, **({"required": req} if req else {})}


def parameters(spec):
    """a catalog tool -> its JSON Schema `parameters` object"""
    if isinstance(spec.get("parameters"), dict): return normalize(spec["parameters"])
    props, req, ad = {}, [], spec.get("arg_descriptions") or {}
    for k, v in (spec.get("args") or {}).items():
        key = k.rstrip("?"); s = parse(v)
        if key in ad: s = {"description": ad[key], **{a: b for a, b in s.items() if a != "description"}}
        props[key] = s
        if not k.endswith("?"): req.append(key)
    return {"type": "object", "properties": props, **({"required": req} if req else {})}


def normalize(s, depth=0):
    """a foreign schema (ToolACE 'dict', xLAM 'str, optional', ...) -> plain JSON Schema with standard type names"""
    if not isinstance(s, dict) or depth > 6: return {"type": "string"}
    out = {}
    if s.get("description"): out["description"] = str(s["description"]).strip()
    t = s.get("type")
    if isinstance(t, list): t = next((x for x in t if x != "null"), "string")
    t = BASE.get(str(t).split(",")[0].strip().lower(), None) if t else None
    if t is None: t = "object" if "properties" in s else ("array" if "items" in s else "string")
    out["type"] = t
    if isinstance(s.get("enum"), list) and s["enum"]: out["enum"] = [x for x in s["enum"] if isinstance(x, (str, int, float))]
    if s.get("format") in ("date", "date-time", "email", "uri"): out["format"] = s["format"]
    if t == "array": out["items"] = normalize(s.get("items") or {"type": "string"}, depth + 1)
    if t == "object" and isinstance(s.get("properties"), dict):
        out["properties"] = {k: normalize(v, depth + 1) for k, v in s["properties"].items() if isinstance(k, str)}
        req = [r for r in (s.get("required") or []) if r in out["properties"]] if isinstance(s.get("required"), list) else []
        if req: out["required"] = req
    return out
