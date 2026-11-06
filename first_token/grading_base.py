"""Balanced boxed-answer extraction and Math-Verify parsing, adapted from cues.grader."""
import re

from math_verify import parse


def boxed(text):
    """Last \\boxed{...} with balanced braces, else a trailing 'Final answer:' capture, else None. Scans backwards
    from the last \\boxed{ so a completion that boxes a first attempt, revises, and boxes again returns the final
    answer, and nested LaTeX like \\boxed{\\frac{\\sqrt{2}}{2}} survives."""
    if not text:
        return None
    start = text.rfind("\\boxed{")
    while start != -1:
        i = start + len("\\boxed{")
        depth = 1
        while i < len(text) and depth:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if not depth:
                    return text[start + len("\\boxed{"):i]
            i += 1
        start = text.rfind("\\boxed{", 0, start)      # unbalanced: try an earlier one
    t = re.findall(r"[Ff]inal answer:?\s*(.+)", text)
    return t[-1].strip() if t else None


def extract(text):
    """The answer the text commits to under the prompt's contract, or None."""
    return boxed(text)


def _parse(s):
    try:
        return parse("\\boxed{" + str(s) + "}")
    except Exception:
        return None
