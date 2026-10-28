"""Local rule-based planner (free, offline). Muse Spark replaces it in the meta backend.

parse(text, names) -> (calls, unknown_clauses)
  names: {id: [name, ...]} used to find object mentions.
Clauses split on ',', ';', ' then ', ' and then '. "what if ..." / "will ..." -> simulate.
"""
import re

from agent.schema import COLORS

NAMED_G = {"moon": 1.62, "mars": 3.71, "jupiter": 24.8, "earth": 9.81, "zero": 0.0, "no": 0.0}
SPEED_WORDS = {"gently": 0.3, "gentle": 0.3, "softly": 0.3, "slightly": 0.25, "lightly": 0.3,
               "hard": 1.8, "harder": 2.2, "very hard": 3.0, "fast": 1.8}
DIR_WORDS = {"left": "left", "right": "right", "forward": "forward", "forwards": "forward",
             "back": "back", "backward": "back", "backwards": "back", "away": "away",
             "toward me": "toward", "towards me": "toward", "closer": "toward"}


# Synonyms -> canonical words the clause rules know. Order matters (longer phrases first).
SYNONYMS = [
    (r"\bknock (.+?) over\b", r"push \1 hard"), (r"\boff the table\b", "away hard"),
    (r"\b(shove|nudge|slide|kick|poke|bump|tap|flick|hit|bash|send|budge|scoot|swipe)\b", "push"),
    (r"\b(toss|launch|fling|yeet|chuck|hurl|lob|flip)\b", "throw"),
    (r"\blet (.+?) fall\b", r"drop \1"), (r"\b(release|dropping|fall)\b", "drop"),
    (r"\b(get rid of|take away|take out|vanish|disappear|erase|delete|hide|clear)\b", "remove"),
    (r"\b(bring back|unhide|undelete|put back)\b", "restore"),
    (r"\b(start over|reset everything|reset all|restart|put everything back|undo everything)\b", "reset"),
    (r"\b(paint|colou?r|dye|tint|change the colou?r of|recolou?r)\b", "make"),
    (r"\b(enlarge|grow|scale up|upsize)\b", "make bigger"), (r"\b(larger|huge|giant)\b", "bigger"),
    (r"\b(shrink|scale down|downsize)\b", "make smaller"), (r"\b(tiny|mini)\b", "smaller"),
    (r"\b(to the left|leftwards|leftward)\b", "left"), (r"\b(to the right|rightwards|rightward)\b", "right"),
    (r"\b(further|farther|away from me|backwards from me)\b", "away"),
    (r"\b(softly|lightly|a bit|a little|slowly|carefully)\b", "gently"),
    (r"\b(strongly|forcefully|really hard|super hard|violently|quickly)\b", "hard"),
    (r"\b(on to|onto|on top of|atop|above)\b", "on"),
    (r"\b(lunar|moon-like)\b", "moon"), (r"\b(weightless|no gravity|space)\b", "zero gravity"),
    (r"\b(everything|all objects|all of them|all items|every object)\b", "all"),
]
VOCAB = {"push", "drop", "throw", "stack", "make", "remove", "restore", "reset", "measure", "gravity",
         "left", "right", "away", "toward", "forward", "back", "gently", "hard", "bigger", "smaller",
         "slippery", "bouncy", "heavier", "lighter", "moon", "mars", "jupiter", *COLORS}


def _fuzzy_tokens(t, extra):
    """Fix typos: tokens >= 4 letters close to a known word are replaced by it."""
    import difflib
    vocab = sorted(VOCAB | extra)
    out = []
    for w in t.split():
        if len(w) >= 4 and w not in vocab:
            m = difflib.get_close_matches(w, vocab, n=1, cutoff=0.8)
            w = m[0] if m else w
        out.append(w)
    return " ".join(out)


def normalize(text, names):
    t = " " + text.lower().strip().rstrip("?.!") + " "
    name_words = {w for ns in names.values() for n in ns for w in n.split() if len(w) >= 4}
    t = " " + _fuzzy_tokens(t, name_words) + " "
    for pat, rep in SYNONYMS:
        t = re.sub(pat, rep, t)
    return t


def _num(s):
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(cm|mm|m|kg|g)?\b", s)
    if not m:
        return None
    v, u = float(m.group(1)), m.group(2)
    return {"cm": v / 100, "mm": v / 1000, "g": v / 1000}.get(u, v)


def _objects(text, names):
    """Object mentions in order of appearance -> list of ids (longest names first)."""
    hits = []
    for k, ns in names.items():
        for n in sorted(ns, key=len, reverse=True):
            if len(n) < 2 or n.isdigit() and f"object {n}" not in text and f"#{n}" not in text:
                continue
            for m in re.finditer(r"\b" + re.escape(n) + r"\b", text):
                hits.append((m.start(), k))
    seen, out = set(), []
    for _, k in sorted(hits):
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _clause(c, names, last):
    """One clause -> list of calls. `last` = last mentioned object id (for 'it')."""
    objs = _objects(c, names)
    if not objs and re.search(r"\b(it|that|this|this one|that one|them)\b", c) and last is not None:
        objs = [last]
    if not objs and re.search(r"\ball\b", c) and not re.search(r"\breset\b", c):
        # "drop everything": the same clause for every object.
        out = []
        for k in names:
            r = _clause(c.replace(" all ", f" {names[k][0]} "), names, None)
            out += r or []
        return out or None
    o = str(objs[0]) if objs else None

    if re.search(r"\breset\b|start over|put (everything|all) back", c):
        return [{"name": "reset", "args": {}}]
    if re.search(r"\b(list|what objects|which objects|what is on)\b", c):
        return [{"name": "list_objects", "args": {}}]
    if "gravity" in c:
        for w, g in NAMED_G.items():
            if re.search(rf"\b{w}\b", c):
                return [{"name": "set_gravity", "args": {"g": g}}]
        n = _num(c)
        if n is not None:
            return [{"name": "set_gravity", "args": {"g": n}}]
        if re.search(r"double", c):
            return [{"name": "set_gravity", "args": {"g": 19.62}}]
    if o is None:
        return None
    if re.search(r"\b(measure|how big|how tall|how wide|size of|dimensions)\b", c):
        return [{"name": "measure", "args": {"obj": o}}]
    if re.search(r"\b(remove|delete|hide|get rid of)\b", c):
        return [{"name": "remove", "args": {"obj": o}}]
    if re.search(r"\b(restore|bring back|undo)\b", c):
        return [{"name": "restore", "args": {"obj": o}}]
    if re.search(r"\b(stack|put|place|set)\b.*\b(on|onto|on top of)\b", c) and len(objs) >= 2:
        return [{"name": "stack", "args": {"obj": str(objs[0]), "base": str(objs[1])}}]
    if re.search(r"\b(drop|let .* fall)\b", c):
        h = _num(c)
        return [{"name": "drop", "args": {"obj": o, **({"height": h} if h else {})}}]
    if re.search(r"\b(throw|toss|launch|flip)\b", c):
        return [{"name": "throw", "args": {"obj": o}}]
    if re.search(r"\b(push|nudge|shove|slide|knock|move|kick)\b", c):
        d = next((v for w, v in DIR_WORDS.items() if re.search(rf"\b{w}\b", c)), None)
        args = {"obj": o, "direction": d or "away"}
        if len(objs) >= 2 and re.search(r"\b(into|toward|towards|at)\b", c):
            args.update(direction="toward", target=str(objs[1]))
        elif re.search(r"\bknock\b", c) and not d:
            args["speed"] = 1.8
        for w, s in SPEED_WORDS.items():
            if re.search(rf"\b{w}\b", c):
                args["speed"] = s
        return [{"name": "push", "args": args}]
    if re.search(r"\b(make|turn|paint|colou?r)\b", c):
        col = next((k for k in COLORS if re.search(rf"\b{k}\b", c)), None)
        if col:
            return [{"name": "set_color", "args": {"obj": o, "color": col}}]
        m = re.search(r"(\d+(?:\.\d+)?)\s*(x|times)\s*(bigger|larger|smaller)?", c)
        if m:
            f = float(m.group(1))
            return [{"name": "scale", "args": {"obj": o, "factor": 1 / f if m.group(3) == "smaller" else f}}]
        if re.search(r"\b(twice as big|double( the)? size|bigger|larger)\b", c):
            return [{"name": "scale", "args": {"obj": o, "factor": 2.0 if re.search("twice|double", c) else 1.5}}]
        if re.search(r"\b(half( the)? size|smaller|tiny)\b", c):
            return [{"name": "scale", "args": {"obj": o, "factor": 0.5}}]
        if re.search(r"\b(ice|icy|slippery|frictionless)\b", c):
            return [{"name": "set_friction", "args": {"obj": o, "mu": 0.02}}]
        if re.search(r"\b(rubber|grippy|sticky)\b", c):
            return [{"name": "set_friction", "args": {"obj": o, "mu": 1.4}}]
        if re.search(r"\b(bouncy|superball|rubber ball)\b", c):
            return [{"name": "set_bounce", "args": {"obj": o, "e": 0.9}}]
        if re.search(r"\bheav(y|ier)\b", c):
            n = _num(c)
            return [{"name": "set_mass", "args": {"obj": o, "kg": n if n else 5.0}}]
        if re.search(r"\blight(er)?\b|feather", c):
            n = _num(c)
            return [{"name": "set_mass", "args": {"obj": o, "kg": n if n else 0.02}}]
    if re.search(r"\bweigh|mass\b", c) and _num(c) is not None:
        return [{"name": "set_mass", "args": {"obj": o, "kg": _num(c)}}]
    return None


def parse(text, names, default=None):
    """default: object id used for "it" when no object was mentioned before (the GUI selection)."""
    simulate = bool(re.match(r"\s*(what if|what happens if|what would happen if|will|would|predict|does|do)\b",
                             text.lower()))
    t = normalize(text, names)
    t = re.sub(r"^\s*(what if|what happens if|what would happen if|predict|please|can you|could you|"
               r"would you|i want to|i want you to|let's|lets)\s+", " ", t)
    t = re.sub(r"\b(i|we|you)\s+", "", t)
    verbs = r"push|drop|throw|stack|put|place|make|remove|restore|reset|measure|set"
    clauses = [c.strip() for c in re.split(rf",|;|\bthen\b|\band then\b|\balso\b|\band\b(?= (?:{verbs})\b)", t)
               if c and c.strip()]
    calls, unknown, last = [], [], default
    for c in clauses:
        r = _clause(" " + c + " ", names, last)
        if r is None:
            unknown.append(c)
            continue
        calls += r
        for x in r:
            if "obj" in x["args"]:
                last = int(x["args"]["obj"]) if str(x["args"]["obj"]).isdigit() else last
    if unknown:
        unknown = [u + "  (try: push / drop / throw / put X on Y / make X red|bigger|slippery / "
                   "remove / reset / what if ...)" for u in unknown]
    if simulate and calls:
        acts = [c for c in calls if c["name"] in ("push", "drop", "throw", "stack")]
        calls = [{"name": "simulate", "args": {"actions": acts}}] if acts else calls
    return calls, unknown
