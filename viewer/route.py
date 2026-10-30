"""Fast routing of typed questions (no model call). Anything not matched goes to the Muse planner.

One message can ask for several things ("what is this and find me similar ones"): route() returns every mode
that matches, in a fixed order. The object can be the selection or named in the text (find_object).
"""
import re

SIMILAR = re.compile(r"\b(similar|same (one|product|thing)|where (can|do|could|should) (i|we) (buy|get|find)|"
                     r"buy (this|it|one)|find (this|it|one)|(pictures?|photos?|images?) of|"
                     r"show me (some )?(pictures|photos|images)|alternatives?|something like (this|it)|"
                     r"other (brands|options)|find me)\b")
EXPLAIN = re.compile(r"\b(what('?s| is) (this|it|that)( thing)?|what am i looking at|how (do|can|should) (i|you) use|"
                     r"how to use|how does (this|it) work|explain|show me how|tell me (about|more)|"
                     r"what does (this|it) do|what('?s| is) (this|it) for|identify)\b")
GUIDE = re.compile(r"(how (do|can|would|should) (i|we|you)|how to|step by step|teach me|help me|walk me through|"
                   r"show me how|what (can|do|should) (i|we) (make|do|play) with|what (can|should) (i|we) do)\b")
USE_IT = re.compile(r"how (do|can|should) (i|you) use|how to use")
LOOK = re.compile(r"\b(use|switch to|show) (it |this |the )?(as )?(the |a )?(3d |three d )?(model|scan|scanned|original) "
                  r"(look|version|instead)\b|\b(3d|model) look\b|\bback to (the )?(scan|scanned|original)( look)?\b")
MODEL = re.compile(r"\b(3d|three d|3-d)\b.*\b(model|version|file|mesh|copy)\b|\b(make|build|create|download|export|"
                   r"get) (me )?(a |the )?(model|mesh|glb|obj file)\b")


def route(text, has_object):
    """Modes in order: 'explain', 'similar', 'model', 'guide'. [] = ask the Muse planner.
    has_object: an object is selected or named in the text."""
    q = " ".join(text.lower().split())
    modes = []
    if has_object and EXPLAIN.search(q):
        modes.append("explain")
    if has_object and SIMILAR.search(q):
        modes.append("similar")
    if has_object and LOOK.search(q):
        modes.append("look")
    elif has_object and MODEL.search(q):
        modes.append("model")
    if GUIDE.match(q) and not (has_object and USE_IT.match(q)):
        modes.append("guide")
    return modes


def find_object(text, names):
    """Object id whose name/alias appears in the text (longest match wins), or None. names: {id: [names]}."""
    q = " " + " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()) + " "
    best, best_len = None, 0
    for k, ns in names.items():
        for n in ns:
            n = n.lower().strip()
            if len(n) > 3 and not n.isdigit() and f" {n} " in q and len(n) > best_len:
                best, best_len = k, len(n)
    return best
