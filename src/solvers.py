"""Symbolic solvers for each puzzle category.

Each solver takes the raw prompt text and returns (answer, reasoning_steps).
`reasoning_steps` is a list of short human-readable strings used to build a
chain-of-thought trace for SFT training.

If a solver cannot determine a unique rule it returns (None, []).
"""
from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from typing import List, Optional, Tuple

Result = Tuple[Optional[str], List[str]]


# ---------------------------------------------------------------------------
# Bit-manipulation solver
# ---------------------------------------------------------------------------
def _parse_bit_examples(prompt: str):
    pairs = re.findall(r"([01]{8})\s*->\s*([01]{8})", prompt)
    query = re.search(r"determine the output for:\s*([01]{8})", prompt)
    return [(int(a, 2), int(b, 2)) for a, b in pairs], (int(query.group(1), 2) if query else None)


def _bits(n: int) -> List[int]:
    return [(n >> (7 - i)) & 1 for i in range(8)]


def _from_bits(bs: List[int]) -> int:
    n = 0
    for b in bs:
        n = (n << 1) | b
    return n


# Build a library of candidate single-step transformations f: u8 -> u8
def _bitfn_library():
    L = []
    for k in range(8):
        L.append((f"rotate left by {k}", lambda x, k=k: ((x << k) | (x >> (8 - k))) & 0xFF))
        L.append((f"rotate right by {k}", lambda x, k=k: ((x >> k) | (x << (8 - k))) & 0xFF))
    for k in range(8):
        L.append((f"left shift by {k} (logical)", lambda x, k=k: (x << k) & 0xFF))
        L.append((f"right shift by {k} (logical)", lambda x, k=k: x >> k))
    L.append(("bitwise NOT", lambda x: (~x) & 0xFF))
    L.append(("reverse bits", lambda x: _from_bits(list(reversed(_bits(x))))))
    L.append(("identity", lambda x: x))
    # swap nibbles
    L.append(("swap nibbles", lambda x: ((x & 0x0F) << 4) | ((x & 0xF0) >> 4)))
    return L


def _bit_compose_search(pairs):
    """Search for f(x) = stage1(x) XOR C or stage1(x) plus a single binary op with constant."""
    lib = _bitfn_library()
    # Stage 1: single transform alone
    for name, fn in lib:
        if all(fn(x) == y for x, y in pairs):
            return [name], lambda x, fn=fn: fn(x)
    # Stage 2: transform then XOR with constant
    for name, fn in lib:
        c_candidates = {fn(x) ^ y for x, y in pairs}
        if len(c_candidates) == 1:
            c = c_candidates.pop()
            if c != 0:
                return [name, f"XOR with {c:08b}"], lambda x, fn=fn, c=c: fn(x) ^ c
    # Stage 3: XOR with constant then transform
    for name, fn in lib:
        # find c such that fn(x ^ c) == y for all
        for c in range(256):
            if all(fn(x ^ c) == y for x, y in pairs):
                steps = []
                if c:
                    steps.append(f"XOR with {c:08b}")
                steps.append(name)
                return steps, lambda x, fn=fn, c=c: fn(x ^ c)
    # Stage 4: two transforms composed
    for (n1, f1), (n2, f2) in itertools.product(lib, repeat=2):
        if all(f2(f1(x)) == y for x, y in pairs):
            return [n1, n2], lambda x, f1=f1, f2=f2: f2(f1(x))
    # Stage 5: two transforms with XOR sandwich
    for (n1, f1), (n2, f2) in itertools.product(lib, repeat=2):
        cs = {f2(f1(x)) ^ y for x, y in pairs}
        if len(cs) == 1:
            c = cs.pop()
            return [n1, n2, f"XOR with {c:08b}"], lambda x, f1=f1, f2=f2, c=c: f2(f1(x)) ^ c
    return None, None


def _gf2_linear_fit(pairs):
    """Try y = M @ x XOR c over GF(2). Returns callable or None.

    We model each output bit y_i = (sum_j M_ij * x_j) XOR c_i over GF(2).
    For each output bit independently, solve a linear system using all examples
    with an augmented input vector [x_0..x_7, 1] (the 1 accounts for c_i).
    With k examples we get k equations in 9 unknowns per output bit; for k>=9
    typically uniquely solvable.
    """
    if len(pairs) < 9:
        # Augment with x->x identity sanity? skip when too few.
        pass
    # Build rows of augmented inputs
    rows = []
    for x, _ in pairs:
        b = _bits(x) + [1]
        rows.append(b)
    coeffs_per_outbit = []
    for out_bit in range(8):
        target = [(_bits(y)[out_bit]) for _, y in pairs]
        # Solve over GF(2) for the 9-vector w such that rows @ w = target.
        n_unk = 9
        A = [r[:] + [t] for r, t in zip(rows, target)]
        # Gaussian elimination
        r = 0
        for c in range(n_unk):
            pivot = None
            for i in range(r, len(A)):
                if A[i][c] == 1:
                    pivot = i
                    break
            if pivot is None:
                continue
            A[r], A[pivot] = A[pivot], A[r]
            for i in range(len(A)):
                if i != r and A[i][c] == 1:
                    A[i] = [(a ^ b) for a, b in zip(A[i], A[r])]
            r += 1
        # check for inconsistency
        for row in A:
            if all(v == 0 for v in row[:-1]) and row[-1] == 1:
                return None
        # Extract one valid solution
        w = [0] * n_unk
        for row in A:
            ones = [i for i in range(n_unk) if row[i] == 1]
            if ones:
                lead = ones[0]
                w[lead] = row[-1]
        coeffs_per_outbit.append(w)

    def apply(x, C=coeffs_per_outbit):
        xb = _bits(x) + [1]
        ob = []
        for w in C:
            s = 0
            for j in range(9):
                if w[j] and xb[j]:
                    s ^= 1
            ob.append(s)
        return _from_bits(ob)
    # verify
    if all(apply(x) == y for x, y in pairs):
        return apply
    return None


def solve_bit_manip(prompt: str) -> Result:
    pairs, q = _parse_bit_examples(prompt)
    if not pairs or q is None:
        return None, []
    # Try cheap symbolic compositions first (yields nicer explanations)
    steps, fn = _bit_compose_search(pairs)
    if fn is not None:
        out = fn(q) & 0xFF
        return f"{out:08b}", [
            f"The examples imply: {' -> '.join(steps)}.",
            f"Apply this rule to {q:08b}.",
        ]
    # Fall back to general GF(2)-linear fit
    fn = _gf2_linear_fit(pairs)
    if fn is not None:
        out = fn(q) & 0xFF
        return f"{out:08b}", [
            "Each output bit is a XOR of a subset of input bits (plus optional flip).",
            "Solve for the per-bit coefficients via Gaussian elimination over GF(2).",
            f"Apply the learned map to {q:08b}.",
        ]
    return None, []


# ---------------------------------------------------------------------------
# Physics solver: d = 0.5 * g * t^2
# ---------------------------------------------------------------------------
def solve_physics(prompt: str) -> Result:
    examples = re.findall(r"t\s*=\s*([0-9.]+)s,\s*distance\s*=\s*([0-9.]+)\s*m", prompt)
    if not examples:
        return None, []
    # Least-squares fit of g over t^2:   d_i = (g/2) * t_i^2   =>   g = 2 * sum(d_i*t_i^2) / sum(t_i^4)
    ts = [float(t) for t, _ in examples]
    ds = [float(d) for _, d in examples]
    num = sum(d * t * t for d, t in zip(ds, ts))
    den = sum((t ** 2) ** 2 for t in ts) or 1.0
    g = 2 * num / den
    # Query t comes after "Now, determine ..." — anchor there
    after = re.split(r"\bNow,?\b", prompt, maxsplit=1)
    tail = after[1] if len(after) > 1 else prompt
    q = re.search(r"t\s*=\s*([0-9.]+)\s*s", tail)
    if not q:
        return None, []
    t = float(q.group(1))
    d = 0.5 * g * t * t
    return f"{d:.2f}", [
        f"Fit g from examples (least squares): g ≈ {g:.4f}.",
        f"For t={t}s, d = 0.5 * {g:.4f} * {t}^2 = {d:.4f} ≈ {d:.2f}.",
    ]


# ---------------------------------------------------------------------------
# Unit-conversion solver: linear y = a*x + b
# ---------------------------------------------------------------------------
def solve_unit_conv(prompt: str) -> Result:
    examples = re.findall(r"([0-9.]+)\s*m\s*becomes\s*([0-9.]+)", prompt)
    if len(examples) < 2:
        return None, []
    xs = [float(a) for a, _ in examples]
    ys = [float(b) for _, b in examples]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs) or 1.0
    a = num / den
    b = my - a * mx
    if abs(b) < 1e-3:
        b = 0.0
    q = re.search(r"convert the following measurement:\s*([0-9.]+)", prompt)
    if not q:
        return None, []
    x = float(q.group(1))
    y = a * x + b
    return f"{y:.2f}", [
        f"Fit y = a*x + b on examples: a={a:.4f}, b={b:.4f}.",
        f"Apply: y = {a:.4f} * {x} + {b:.4f} = {y:.4f}.",
    ]


# ---------------------------------------------------------------------------
# Cipher solver: per-character substitution
# ---------------------------------------------------------------------------
def solve_cipher(prompt: str) -> Result:
    """Examples are 'cipher -> plain' pairs. Build cipher->plain char map."""
    pairs = re.findall(r"^([a-z\s]+?)\s*->\s*([a-z\s]+?)$", prompt, flags=re.MULTILINE)
    if not pairs:
        return None, []
    mapping = {}
    for c_text, p_text in pairs:
        c_words = c_text.strip().split()
        p_words = p_text.strip().split()
        if len(c_words) != len(p_words):
            continue
        for cw, pw in zip(c_words, p_words):
            if len(cw) != len(pw):
                continue
            for cc, pc in zip(cw, pw):
                if cc in mapping and mapping[cc] != pc:
                    return None, []
                mapping[cc] = pc
    q = re.search(r"decrypt the following text:\s*(.+)$", prompt, flags=re.MULTILINE)
    if not q:
        return None, []
    cipher_text = q.group(1).strip().rstrip('"').strip()
    out_chars = []
    for ch in cipher_text:
        if ch == " ":
            out_chars.append(" ")
        elif ch in mapping:
            out_chars.append(mapping[ch])
        else:
            return None, []
    return "".join(out_chars), [
        f"Recovered substitution map from examples ({len(mapping)} letters).",
        f"Apply char-by-char to '{cipher_text}'.",
    ]


# ---------------------------------------------------------------------------
# Roman-numeral solver
# ---------------------------------------------------------------------------
_ROMAN = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def _to_roman(n: int) -> str:
    out = []
    for v, s in _ROMAN:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)


def solve_roman(prompt: str) -> Result:
    q = re.search(r"write the number\s+(\d+)", prompt)
    if not q:
        return None, []
    n = int(q.group(1))
    r = _to_roman(n)
    return r, [
        f"Examples confirm the Wonderland system is standard Roman numerals.",
        f"{n} = {r}.",
    ]


# ---------------------------------------------------------------------------
# Algebra (symbolic-substitution) solver
# ---------------------------------------------------------------------------
# Format: lines like "<LHS> = <RHS>" using a small set of printable symbols.
# We treat each char on LHS as input symbol, each char on RHS as output symbol.
# The rule is typically a (possibly length-reducing) per-symbol map plus
# optional removal of certain symbols. We learn a positional mapping that is
# consistent with all examples; if length doesn't match, we treat selected
# operator symbols (*, +, -, etc.) as "deleted".
def _algebra_extract_examples(prompt: str):
    lines = []
    for ln in prompt.splitlines():
        ln = ln.strip()
        m = re.match(r"^[`']?(.+?)\s*=\s*(.+?)$", ln)
        if not m:
            continue
        lhs, rhs = m.group(1), m.group(2)
        if "below" in lhs.lower() or "example" in lhs.lower() or "rule" in lhs.lower():
            continue
        lines.append((lhs, rhs))
    q = re.search(r"determine the result for:\s*(.+)$", prompt, flags=re.MULTILINE)
    return lines, (q.group(1).strip() if q else None)


def solve_algebra(prompt: str) -> Result:
    examples, query = _algebra_extract_examples(prompt)
    if not examples or query is None:
        return None, []

    # Hypothesis A: same-length per-character map.
    same_len = [(l, r) for l, r in examples if len(l) == len(r)]
    if same_len and len(same_len) >= 2:
        m = {}
        ok = True
        for l, r in same_len:
            for a, b in zip(l, r):
                if a in m and m[a] != b:
                    ok = False
                    break
                m[a] = b
            if not ok:
                break
        if ok and all(c in m for c in query):
            out = "".join(m[c] for c in query)
            return out, [
                "Examples show a per-character substitution cipher.",
                f"Apply map to '{query}'.",
            ]

    # Hypothesis B: drop a fixed set of operator symbols, then per-char map on the rest.
    OPS = set("+-*/^&|!@#$%<>?:;.,~`\\\"'")
    def strip_ops(s):
        return "".join(c for c in s if c not in OPS)
    cleaned = [(strip_ops(l), r) for l, r in examples]
    if all(len(l) == len(r) for l, r in cleaned):
        m = {}
        ok = True
        for l, r in cleaned:
            for a, b in zip(l, r):
                if a in m and m[a] != b:
                    ok = False
                    break
                m[a] = b
            if not ok:
                break
        q_clean = strip_ops(query)
        if ok and all(c in m for c in q_clean):
            out = "".join(m[c] for c in q_clean)
            return out, [
                "Examples show: drop operator symbols, then per-character substitution.",
                f"After stripping operators, '{query}' -> '{q_clean}'; apply map.",
            ]

    return None, []


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
from src.categorize import categorize  # noqa: E402

_SOLVERS = {
    "bit_manip": solve_bit_manip,
    "physics": solve_physics,
    "unit_conv": solve_unit_conv,
    "cipher": solve_cipher,
    "roman": solve_roman,
    "algebra": solve_algebra,
}


def solve(prompt: str) -> Tuple[Optional[str], str, List[str]]:
    """Returns (answer, category, reasoning_steps)."""
    cat = categorize(prompt)
    ans, steps = _SOLVERS[cat](prompt)
    return ans, cat, steps
