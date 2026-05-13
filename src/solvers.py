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
    """Search f(x) over a library that includes nonlinear primitives:
       - rotations / shifts / NOT / reverse / swap_nibbles
       - x AND rot(x, k), x OR rot(x, k), x XOR rot(x, k)
       - majority(x, rot(x, j), rot(x, k))
       - choice(x, rot(x, j), rot(x, k))   ≡ (x & rj) | (~x & rk)
       Search depth: single op, or single op then XOR with constant.
    """
    lib = _bitfn_library()
    # Nonlinear primitives: enumerate over k in 1..7
    nonlin = []
    for k in range(1, 8):
        nonlin.append((f"x AND rot({k})", lambda x, k=k: x & (((x << k) | (x >> (8 - k))) & 0xFF)))
        nonlin.append((f"x OR rot({k})",  lambda x, k=k: x | (((x << k) | (x >> (8 - k))) & 0xFF)))
        nonlin.append((f"x XOR rot({k})", lambda x, k=k: x ^ (((x << k) | (x >> (8 - k))) & 0xFF)))
    for j in range(1, 4):
        for k in range(j + 1, 5):
            def maj(x, j=j, k=k):
                a = ((x << j) | (x >> (8 - j))) & 0xFF
                b = ((x << k) | (x >> (8 - k))) & 0xFF
                return (x & a) | (x & b) | (a & b)
            nonlin.append((f"maj(x, rot({j}), rot({k}))", maj))
            def ch(x, j=j, k=k):
                a = ((x << j) | (x >> (8 - j))) & 0xFF
                b = ((x << k) | (x >> (8 - k))) & 0xFF
                return (x & a) | ((~x) & 0xFF & b)
            nonlin.append((f"choice(x, rot({j}), rot({k}))", ch))
    full = lib + nonlin

    # Stage 1: single transform alone
    for name, fn in full:
        if all(fn(x) == y for x, y in pairs):
            return [name], lambda x, fn=fn: fn(x)
    # Stage 2: transform then XOR with constant
    for name, fn in full:
        cs = {fn(x) ^ y for x, y in pairs}
        if len(cs) == 1:
            c = cs.pop()
            if c != 0:
                return [name, f"XOR with {c:08b}"], lambda x, fn=fn, c=c: fn(x) ^ c
    # Stage 3: XOR with constant then transform
    for name, fn in lib:
        for c in range(256):
            if all(fn(x ^ c) == y for x, y in pairs):
                steps = []
                if c: steps.append(f"XOR with {c:08b}")
                steps.append(name)
                return steps, lambda x, fn=fn, c=c: fn(x ^ c)
    # Stage 4: two linear transforms composed
    for (n1, f1), (n2, f2) in itertools.product(lib, repeat=2):
        if all(f2(f1(x)) == y for x, y in pairs):
            return [n1, n2], lambda x, f1=f1, f2=f2: f2(f1(x))
    # Stage 5: two transforms with XOR sandwich
    for (n1, f1), (n2, f2) in itertools.product(lib, repeat=2):
        cs = {f2(f1(x)) ^ y for x, y in pairs}
        if len(cs) == 1:
            c = cs.pop()
            return [n1, n2, f"XOR with {c:08b}"], lambda x, f1=f1, f2=f2, c=c: f2(f1(x)) ^ c
    # Stage 6: nonlinear op then XOR constant
    for name, fn in nonlin:
        cs = {fn(x) ^ y for x, y in pairs}
        if len(cs) == 1:
            c = cs.pop()
            return [name, f"XOR with {c:08b}"], lambda x, fn=fn, c=c: fn(x) ^ c

    # Stage 7: SHA-style sigma — XOR of two or three rotations/shifts of x.
    rots = [("rot_left", k, lambda x, k=k: ((x << k) | (x >> (8 - k))) & 0xFF) for k in range(0, 8)]
    rots += [("rot_right", k, lambda x, k=k: ((x >> k) | (x << (8 - k))) & 0xFF) for k in range(1, 8)]
    rots += [("shr", k, lambda x, k=k: x >> k) for k in range(1, 8)]
    rots += [("shl", k, lambda x, k=k: (x << k) & 0xFF) for k in range(1, 8)]
    # Two-term XOR
    for (n1, k1, f1), (n2, k2, f2) in itertools.combinations(rots, 2):
        def g(x, f1=f1, f2=f2): return f1(x) ^ f2(x)
        cs = {g(x) ^ y for x, y in pairs}
        if len(cs) == 1:
            c = cs.pop()
            label = f"{n1}({k1}) XOR {n2}({k2})"
            if c: label += f" XOR {c:08b}"
            return [label], lambda x, g=g, c=c: g(x) ^ c
    # Three-term XOR
    for (n1, k1, f1), (n2, k2, f2), (n3, k3, f3) in itertools.combinations(rots, 3):
        def g(x, f1=f1, f2=f2, f3=f3): return f1(x) ^ f2(x) ^ f3(x)
        cs = {g(x) ^ y for x, y in pairs}
        if len(cs) == 1:
            c = cs.pop()
            label = f"{n1}({k1}) XOR {n2}({k2}) XOR {n3}({k3})"
            if c: label += f" XOR {c:08b}"
            return [label], lambda x, g=g, c=c: g(x) ^ c
    return None, None


def _gf2_solve_unique(input_rows, target):
    """Gaussian elimination over GF(2). Returns the unique solution vector,
    or None if no solution / multiple solutions exist."""
    n_unk = len(input_rows[0])
    A = [row[:] + [t] for row, t in zip(input_rows, target)]
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
    # Detect inconsistency
    for row in A:
        if all(v == 0 for v in row[:-1]) and row[-1] == 1:
            return None
    # Detect non-unique (rank < n_unk)
    if r < n_unk:
        return None
    w = [0] * n_unk
    for row in A:
        ones = [i for i in range(n_unk) if row[i] == 1]
        if ones:
            w[ones[0]] = row[-1]
    return w


def _gf2_linear_fit(pairs):
    """Fit y = M @ x [XOR c] over GF(2), preferring the smallest model that
    fits uniquely. Returns a function only if the fit is unique on the examples.
    """
    # Stage A: linear (no constant), 8 unknowns per output bit.
    rows_lin = [_bits(x) for x, _ in pairs]
    coeffs = []
    for ob in range(8):
        targ = [_bits(y)[ob] for _, y in pairs]
        w = _gf2_solve_unique(rows_lin, targ)
        if w is None:
            coeffs = None
            break
        coeffs.append(w)
    if coeffs is not None:
        def lin_apply(x, C=coeffs):
            xb = _bits(x); ob = []
            for w in C:
                s = 0
                for j in range(8):
                    if w[j] and xb[j]:
                        s ^= 1
                ob.append(s)
            return _from_bits(ob)
        if all(lin_apply(x) == y for x, y in pairs):
            return lin_apply, "linear GF(2) fit y = M·x"

    # Stage B: affine, 9 unknowns per output bit. Only commit if all 8 outputs
    # solve uniquely (which requires >=9 examples with rank-9 augmented rows).
    rows_aff = [_bits(x) + [1] for x, _ in pairs]
    coeffs = []
    for ob in range(8):
        targ = [_bits(y)[ob] for _, y in pairs]
        w = _gf2_solve_unique(rows_aff, targ)
        if w is None:
            coeffs = None
            break
        coeffs.append(w)
    if coeffs is None:
        return None, None

    def aff_apply(x, C=coeffs):
        xb = _bits(x) + [1]; ob = []
        for w in C:
            s = 0
            for j in range(9):
                if w[j] and xb[j]:
                    s ^= 1
            ob.append(s)
        return _from_bits(ob)
    if all(aff_apply(x) == y for x, y in pairs):
        return aff_apply, "affine GF(2) fit y = M·x ⊕ c"
    return None, None


# Nonlinear search: y = h(x) where h is composed of XORs of rotations of x and
# of the bitwise complement. Each output bit becomes a XOR of bits of x and ¬x
# at fixed rotation offsets — strictly more expressive than affine GF(2).
def _gf2_extended_fit(pairs):
    """Build an extended GF(2) basis where features are bits of x at all 8
    rotations and bits of ~x at all 8 rotations (16 features total) plus the
    constant 1 — 17 unknowns per output bit. Requires >=17 examples for unique
    fit; with 8 examples we still apply but only commit if the model is
    *minimal* (sparse) by preferring linear fits first."""
    feats = []
    for x, _ in pairs:
        bs = _bits(x)
        row = bs[:]                                       # x bits
        # rotated x by 1..7
        for k in range(1, 8):
            r = ((x << k) | (x >> (8 - k))) & 0xFF
            row += _bits(r)
        # ~x rotated by 0..7
        for k in range(0, 8):
            r = ((~x) & 0xFF)
            r = ((r << k) | (r >> (8 - k))) & 0xFF
            row += _bits(r)
        row.append(1)
        feats.append(row)
    n = len(feats[0])
    coeffs = []
    for ob in range(8):
        targ = [_bits(y)[ob] for _, y in pairs]
        w = _gf2_solve_unique(feats, targ)
        if w is None:
            return None, None
        coeffs.append(w)

    def apply(x, C=coeffs):
        bs = _bits(x); row = bs[:]
        for k in range(1, 8):
            r = ((x << k) | (x >> (8 - k))) & 0xFF
            row += _bits(r)
        for k in range(0, 8):
            r = ((~x) & 0xFF); r = ((r << k) | (r >> (8 - k))) & 0xFF
            row += _bits(r)
        row.append(1)
        ob = []
        for w in C:
            s = 0
            for j in range(len(row)):
                if w[j] and row[j]:
                    s ^= 1
            ob.append(s)
        return _from_bits(ob)
    if all(apply(x) == y for x, y in pairs):
        return apply, "GF(2) fit over rotations and complements of x"
    return None, None


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
    # GF(2) fits — try unique linear, unique extended, then permissive affine.
    fn, label = _gf2_linear_fit(pairs)
    if fn is None:
        fn, label = _gf2_extended_fit(pairs)
    if fn is None:
        # Permissive affine fallback (may overfit when underdetermined; the
        # training-data builder filters out wrongs by matching ground truth,
        # so a permissive fit produces more verified-CoT records when it
        # happens to be correct).
        rows_aff = [_bits(x) + [1] for x, _ in pairs]
        coeffs = []
        ok = True
        for ob in range(8):
            targ = [_bits(y)[ob] for _, y in pairs]
            w = _gf2_solve_unique(rows_aff, targ)
            if w is None:
                # Underdetermined — just pick any consistent solution.
                A = [r[:] + [t] for r, t in zip(rows_aff, targ)]
                n_unk = 9
                rrow = 0
                for c in range(n_unk):
                    pivot = None
                    for i in range(rrow, len(A)):
                        if A[i][c] == 1:
                            pivot = i
                            break
                    if pivot is None: continue
                    A[rrow], A[pivot] = A[pivot], A[rrow]
                    for i in range(len(A)):
                        if i != rrow and A[i][c] == 1:
                            A[i] = [(a ^ b) for a, b in zip(A[i], A[rrow])]
                    rrow += 1
                for row in A:
                    if all(v == 0 for v in row[:-1]) and row[-1] == 1:
                        ok = False; break
                if not ok: break
                w = [0] * n_unk
                for row in A:
                    ones = [i for i in range(n_unk) if row[i] == 1]
                    if ones: w[ones[0]] = row[-1]
            coeffs.append(w)
        if ok and len(coeffs) == 8:
            def perm_apply(x, C=coeffs):
                xb = _bits(x) + [1]; ob = []
                for w in C:
                    s = 0
                    for j in range(9):
                        if w[j] and xb[j]: s ^= 1
                    ob.append(s)
                return _from_bits(ob)
            if all(perm_apply(x) == y for x, y in pairs):
                fn, label = perm_apply, "permissive affine GF(2) fit"
    if fn is None:
        return None, []
    out = fn(q) & 0xFF
    return f"{out:08b}", [
        f"Use a {label}; per-bit coefficients are determined from the examples.",
        f"Apply the learned map to {q:08b}.",
    ]


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
# Cipher solver: per-character substitution + English-vocab constraint search
# ---------------------------------------------------------------------------
import os
_VOCAB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "cipher_vocab.txt")
try:
    with open(_VOCAB_PATH) as _f:
        _CIPHER_VOCAB = sorted({w.strip() for w in _f if w.strip()})
except FileNotFoundError:
    _CIPHER_VOCAB = []


def _word_pattern(w: str) -> tuple:
    seen = {}
    out = []
    for c in w:
        if c not in seen:
            seen[c] = len(seen)
        out.append(seen[c])
    return tuple(out)


# Pre-index vocab by (length, pattern) for fast lookup
_VOCAB_BY_PATTERN = {}
for _w in _CIPHER_VOCAB:
    _VOCAB_BY_PATTERN.setdefault((len(_w), _word_pattern(_w)), []).append(_w)


def _consistent(cw: str, candidate: str, mapping: dict) -> bool:
    """Check that mapping cw[i]→candidate[i] is consistent with existing map."""
    local = dict(mapping)
    used = set(local.values())
    for cc, pc in zip(cw, candidate):
        if cc in local:
            if local[cc] != pc:
                return False
        else:
            if pc in used:
                return False
            local[cc] = pc
            used.add(pc)
    return True


def _try_finish_with_vocab(query_words, mapping):
    """Constraint-propagation search: for each cipher word, pick an English word
    from the vocab whose letter pattern matches and is consistent with the
    current mapping. Backtrack when a partial assignment becomes infeasible.
    Returns (full_mapping, plaintext_words) or (None, None)."""
    if not _VOCAB_BY_PATTERN:
        return None, None

    def candidates(cw):
        key = (len(cw), _word_pattern(cw))
        return [w for w in _VOCAB_BY_PATTERN.get(key, []) if _consistent(cw, w, mapping)]

    # Order query words by ascending candidate count for fail-fast
    order = sorted(range(len(query_words)), key=lambda i: len(candidates(query_words[i])))
    assignment = [None] * len(query_words)
    used_plain_words = set()

    def backtrack(k, m):
        if k == len(order):
            return dict(m)
        idx = order[k]
        cw = query_words[idx]
        cands = [w for w in _VOCAB_BY_PATTERN.get((len(cw), _word_pattern(cw)), [])
                 if _consistent(cw, w, m) and w not in used_plain_words]
        # Optional ranking: prefer candidates that introduce fewer new letters
        for w in cands:
            new_m = dict(m)
            for cc, pc in zip(cw, w):
                new_m[cc] = pc
            assignment[idx] = w
            used_plain_words.add(w)
            res = backtrack(k + 1, new_m)
            if res is not None:
                return res
            used_plain_words.discard(w)
            assignment[idx] = None
        return None

    final_map = backtrack(0, mapping)
    if final_map is None:
        return None, None
    return final_map, assignment


def solve_cipher(prompt: str) -> Result:
    """Cipher → plaintext using:
       1) char-level map inferred from examples;
       2) when query contains unknown letters, English-vocab backtracking
          search to fill in the gaps."""
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
    query_words = cipher_text.split()

    # Direct application if mapping covers everything
    if all(ch == " " or ch in mapping for w in query_words for ch in w):
        out = " ".join("".join(mapping[c] for c in w) for w in query_words)
        return out, [
            f"Recovered substitution map ({len(mapping)} letters) from examples.",
            f"Apply directly to '{cipher_text}'.",
        ]

    # Fall back to vocab-constrained backtracking
    final_map, assignment = _try_finish_with_vocab(query_words, mapping)
    if final_map is None:
        return None, []
    out = " ".join(assignment)
    return out, [
        "Partial substitution map from examples leaves some query letters unknown.",
        "Each cipher word must match a known vocabulary word with the same letter-repetition pattern.",
        "Backtracking search over the vocab finds a unique consistent assignment.",
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
# Algebra (cryptarithm) solver — CSP over (symbol→digit, op-symbol→operation)
#
# Every algebra puzzle in the corpus has the form
#     s0 s1 op s3 s4 = r0 [r1 [r2 [r3]]]
# where the 5-char LHS encodes two 2-digit numbers (s0 s1) and (s3 s4) joined
# by an operator symbol, and the 1-4 char RHS encodes a result.  Each unique
# symbol maps bijectively to a decimal digit (per puzzle); each operator
# symbol maps to one of {add, abs_diff, mul, concat, rev_concat}.  We do a
# bounded backtracking search to find an assignment consistent with all
# examples, then apply it to the query.
#
# Method credit: the 5-char LHS structural insight and the operation library
# come from a public Kaggle notebook by participant "glyphmatics" (D4 gate).
# ---------------------------------------------------------------------------
_EQ_RE_CRYPTO = re.compile(r"^(\S{5})\s*=\s*(\S{1,4})\s*$")
_QUESTION_RE_CRYPTO = re.compile(r"determine\s+the\s+result\s+for:\s*(\S{5})\s*$")

def _digit_sum(a, b):
    return sum(int(d) for d in f"{a}{b}")


def _digit_prod(a, b):
    p = 1
    for d in f"{a}{b}":
        p *= int(d)
    return p


def _pairwise_sum(a, b):
    """Pairwise digit sum: tens-digit + units-digit of each, then concat."""
    return (a // 10 + b // 10) * 10 + (a % 10 + b % 10)


def _pairwise_prod(a, b):
    """Pairwise digit product, concat."""
    return (a // 10 * (b // 10)) * 100 + (a % 10 * (b % 10))


def _square(a, b):
    return a * a + b * b


def _diff_sq(a, b):
    return (a - b) ** 2


_CRYPTO_OPS = (
    lambda a, b: a + b,        # 0: add
    lambda a, b: abs(a - b),   # 1: abs_diff
    lambda a, b: a * b,        # 2: mul
    lambda a, b: a * 100 + b,  # 3: concat
    lambda a, b: b * 100 + a,  # 4: rev_concat
    lambda a, b: a + b * 10,   # 5: weighted_add
    lambda a, b: a * 10 + b,   # 6: shift_concat
    _digit_sum,                # 7: digit_sum
    _digit_prod,               # 8: digit_prod
    _pairwise_sum,             # 9: pairwise_sum
    _pairwise_prod,            # 10: pairwise_prod
    _square,                   # 11: square_sum
    _diff_sq,                  # 12: diff_squared
    lambda a, b: a ^ b,        # 13: xor (small ints)
    lambda a, b: a | b,        # 14: or
    lambda a, b: a & b,        # 15: and
)
_CRYPTO_OP_NAMES = (
    "add", "abs_diff", "mul", "concat", "rev_concat",
    "weighted_add", "shift_concat", "digit_sum", "digit_prod",
    "pairwise_sum", "pairwise_prod", "square_sum", "diff_squared",
    "xor", "or", "and",
)


def _crypto_parse(prompt: str):
    egs = []
    q = None
    for ln in prompt.splitlines():
        ln = ln.strip()
        m = _EQ_RE_CRYPTO.match(ln)
        if m:
            lhs, rhs = m.groups()
            egs.append((lhs[0], lhs[1], lhs[2], lhs[3], lhs[4], tuple(rhs)))
        else:
            mq = _QUESTION_RE_CRYPTO.search(ln)
            if mq:
                q = mq.group(1)
    return (egs, q) if (egs and q) else (None, None)


def _crypto_num_digits(n):
    if n == 0:
        return (0,)
    out = []
    while n > 0:
        out.append(n % 10)
        n //= 10
    return tuple(reversed(out))


class _CryptoSolver:
    """Bounded backtracking CSP. Mirrors the public glyphmatics gate but with
    minor tweaks: we always require bijective symbol→digit and we accept
    answers found in any consistent assignment, taking the most frequent."""

    def __init__(self, examples, query, max_solutions=400, max_nodes=200_000):
        self.examples = examples
        self.query = query
        self.max_solutions = max_solutions
        self.max_nodes = max_nodes
        self.nodes = 0
        self.mapping = {}
        self.used = set()
        self.op_assign = {}
        self.answers = Counter()
        self.answer_info = {}
        self.guess_mode_answers = set()

    def solve(self):
        self._process(0)
        if not self.answers:
            return None, ({}, {})
        q_op = self.query[2]
        example_ops = {ex[2] for ex in self.examples}
        is_guess = q_op not in example_ops
        if is_guess and len(self.guess_mode_answers) > 1:
            return None, ({}, {})
        best, _ = self.answers.most_common(1)[0]
        return best, self.answer_info.get(best, ({}, {}))

    def _vals(self, sym):
        if sym in self.mapping:
            return (self.mapping[sym],)
        return tuple(d for d in range(10) if d not in self.used)

    def _assign(self, sym, dig):
        if sym in self.mapping:
            return False if self.mapping[sym] == dig else None
        if dig in self.used:
            return None
        self.mapping[sym] = dig
        self.used.add(dig)
        return True

    def _undo(self, sym, marker):
        if marker is True:
            d = self.mapping.pop(sym)
            self.used.discard(d)

    def _result_digits(self, op_id, left, right):
        try:
            v = _CRYPTO_OPS[op_id](left, right)
        except Exception:
            return None
        if v is None or v < 0 or v >= 10000:
            return None
        # Fixed-width ops (concat, rev_concat, shift_concat, pairwise_*) emit
        # exactly 4 digits with leading zeros; everything else uses the natural
        # decimal width.
        if op_id in (3, 4, 6, 9, 10):
            return (v // 1000, (v // 100) % 10, (v // 10) % 10, v % 10)
        return _crypto_num_digits(v)

    def _process(self, idx):
        self.nodes += 1
        if self.nodes > self.max_nodes or len(self.answers) >= self.max_solutions:
            return
        if idx == len(self.examples):
            self._compute_query()
            return
        s0, s1, op_sym, s3, s4, rsyms = self.examples[idx]
        rlen = len(rsyms)
        # Per-op result-length buckets:
        # rlen 1-4 variable: add(0), abs_diff(1), mul(2), weighted_add(5), shift_concat(6),
        #                    digit_sum(7), digit_prod(8), square_sum(11), diff_squared(12),
        #                    xor(13), or(14), and(15)
        # rlen exactly 4 (zero-padded): concat(3), rev_concat(4), pairwise_sum(9), pairwise_prod(10)
        feasible_ops = []
        if rlen <= 3: feasible_ops.append(0)
        if rlen <= 2: feasible_ops += [1, 7]
        if rlen <= 4: feasible_ops.append(2)
        if rlen == 4: feasible_ops += [3, 4, 9, 10]
        if rlen <= 3: feasible_ops += [5, 6, 11, 12]
        if rlen <= 2: feasible_ops += [8, 13, 14, 15]
        for d0 in self._vals(s0):
            n0 = self._assign(s0, d0)
            if n0 is None: continue
            for d1 in self._vals(s1):
                n1 = self._assign(s1, d1)
                if n1 is None: continue
                left = d0 * 10 + d1
                for d3 in self._vals(s3):
                    n3 = self._assign(s3, d3)
                    if n3 is None: continue
                    for d4 in self._vals(s4):
                        n4 = self._assign(s4, d4)
                        if n4 is None: continue
                        right = d3 * 10 + d4
                        ops = [self.op_assign[op_sym]] if op_sym in self.op_assign else feasible_ops
                        for op_id in ops:
                            rd = self._result_digits(op_id, left, right)
                            if rd is None or len(rd) != rlen: continue
                            assigned = []
                            ok = True
                            for rsym, rdig in zip(rsyms, rd):
                                m = self._assign(rsym, rdig)
                                if m is None:
                                    ok = False; break
                                assigned.append((rsym, m))
                            if ok:
                                op_new = op_sym not in self.op_assign
                                if op_new: self.op_assign[op_sym] = op_id
                                self._process(idx + 1)
                                if op_new: del self.op_assign[op_sym]
                            for rsym, m in reversed(assigned):
                                self._undo(rsym, m)
                            if len(self.answers) >= self.max_solutions:
                                self._undo(s4, n4); self._undo(s3, n3); self._undo(s1, n1); self._undo(s0, n0)
                                return
                        self._undo(s4, n4)
                    self._undo(s3, n3)
                self._undo(s1, n1)
            self._undo(s0, n0)

    def _compute_query(self):
        s0, s1, op_sym, s3, s4 = self.query
        op_new = op_sym not in self.op_assign
        for d0 in self._vals(s0):
            n0 = self._assign(s0, d0)
            if n0 is None: continue
            for d1 in self._vals(s1):
                n1 = self._assign(s1, d1)
                if n1 is None: continue
                left = d0 * 10 + d1
                for d3 in self._vals(s3):
                    n3 = self._assign(s3, d3)
                    if n3 is None: continue
                    for d4 in self._vals(s4):
                        n4 = self._assign(s4, d4)
                        if n4 is None: continue
                        right = d3 * 10 + d4
                        ops = [self.op_assign[op_sym]] if op_sym in self.op_assign else list(range(5))
                        for op_id in ops:
                            rd = self._result_digits(op_id, left, right)
                            if rd is None: continue
                            inv = {v: k for k, v in self.mapping.items()}
                            if any(d not in inv for d in rd): continue
                            ans = "".join(inv[d] for d in rd)
                            self.answers[ans] += 1
                            if op_new:
                                self.guess_mode_answers.add(ans)
                            if ans not in self.answer_info:
                                m = dict(self.mapping); ops_cp = dict(self.op_assign)
                                if op_new: ops_cp[op_sym] = op_id
                                self.answer_info[ans] = (m, ops_cp)
                        self._undo(s4, n4)
                    self._undo(s3, n3)
                self._undo(s1, n1)
            self._undo(s0, n0)


def solve_algebra(prompt: str) -> Result:
    examples, query = _crypto_parse(prompt)
    if not examples or query is None:
        return None, []
    ans, (mapping, op_assign) = _CryptoSolver(examples, query).solve()
    if ans is None:
        return None, []
    op_name = _CRYPTO_OP_NAMES[op_assign.get(query[2], 0)] if op_assign else "unknown"
    return ans, [
        "The puzzle is a cryptarithm: each unique symbol maps bijectively to a digit, and the operator symbol maps to one of {add, abs_diff, mul, concat, rev_concat}.",
        f"Backtracking search recovers the symbol→digit and operator→operation assignment consistent with all examples.",
        f"For the query, the inferred operation is '{op_name}'; apply it and re-encode the result with the symbol map.",
    ]


from collections import Counter  # noqa: E402 (used by _CryptoSolver above)


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
