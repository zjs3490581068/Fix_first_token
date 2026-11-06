"""Four predefined English single-token cues, one sample per MATH-500 problem."""
from .common import main

CONDITIONS = [("Solution", " Solution", [12478]), ("Proof", " Proof", [36991]),
              ("Let", " Let", [6771]), ("I", " I", [358])]


if __name__ == "__main__":
    main("experiments.english_tokens", "english_tokens.json", CONDITIONS)
