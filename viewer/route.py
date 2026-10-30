"""Fast routing of typed questions (no model call). Anything not matched goes to the Muse planner."""
import re

SIMILAR = re.compile(r"\b(similar|same (one|product|thing)|where (can|do|could|should) (i|we) (buy|get|find)|"
                     r"buy (this|it|one)|find (this|it|one)|(pictures?|photos?|images?) of (this|it|that)|"
                     r"show me (some )?(pictures|photos|images)|alternatives?|something like (this|it)|"
                     r"other (brands|options))\b")
EXPLAIN = re.compile(r"\b(what('?s| is) (this|it|that)( thing)?|what am i looking at|how (do|can|should) (i|you) use|"
                     r"how to use|how does (this|it) work|explain|show me how|tell me (about|more)|"
                     r"what does (this|it) do|what('?s| is) (this|it) for|identify)\b")
GUIDE = re.compile(r"(how (do|can|would|should) (i|we|you)|how to|step by step|teach me|help me|walk me through|"
                   r"what can i (make|do) with)\b")


def route(text, selected):
    """'similar' | 'explain' | 'guide' | None (= ask the Muse planner). selected: an object is selected."""
    q = " ".join(text.lower().split())
    if selected and SIMILAR.search(q):
        return "similar"
    if selected and EXPLAIN.search(q):
        return "explain"
    if GUIDE.match(q) and not (selected and re.match(r"how (do|can|should) (i|you) use", q)):
        return "guide"
    return None
