"""Classify a puzzle prompt into one of 6 categories."""

CATEGORIES = ("bit_manip", "physics", "unit_conv", "cipher", "roman", "algebra")


def categorize(prompt: str) -> str:
    p = prompt.lower()
    if "bit manipulation" in p or "8-bit binary" in p:
        return "bit_manip"
    if "gravitational" in p or "d = 0.5*g*t" in p:
        return "physics"
    if "unit conversion" in p or "convert the following measurement" in p:
        return "unit_conv"
    if "encryption" in p or "decrypt" in p:
        return "cipher"
    if "numeral system" in p:
        return "roman"
    return "algebra"
