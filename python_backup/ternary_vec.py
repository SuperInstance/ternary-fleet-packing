"""
ternary_vec.py — Trit vector operations for fleet use.

All functions operate on list[int] of {-1, 0, +1} representing balanced
ternary trits.  The operations mirror c-ternary.h conventions:

    AND  = min(a, b)   — strongest negative / "avoid" dominates
    OR   = max(a, b)   — strongest positive / "choose" dominates
    NEG  = -x          — flip polarity
    SUM  = Σ a[i]      — integer sum
    DOT  = Σ a[i]*b[i] — dot product
    NORM = Σ |a[i]|    — L1 norm (count non-zero trits)
    DIST = positions where a[i] != b[i] — Hamming-like distance
"""

from typing import List


def _check_trit(val: int, label: str = "value") -> None:
    """Raise ValueError if val is not a valid trit {-1, 0, +1}."""
    if val not in (-1, 0, 1):
        raise ValueError(
            f"Invalid trit {label}={val}, expected -1, 0, or +1"
        )


def _check_vec(vec: List[int], label: str = "vec") -> None:
    """Validate all elements of a vector are valid trits."""
    for i, v in enumerate(vec):
        _check_trit(v, f"{label}[{i}]")


def _check_same_len(a: List[int], b: List[int]) -> None:
    if len(a) != len(b):
        raise ValueError(
            f"Vector length mismatch: len(a)={len(a)} != len(b)={len(b)}"
        )


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def ct_and_vec(a: List[int], b: List[int]) -> List[int]:
    """Element-wise ternary AND (min).

    For each position:  result = min(a, b)  in the ordering -1 < 0 < +1.
    """
    _check_vec(a, "a")
    _check_vec(b, "b")
    _check_same_len(a, b)
    return [min(ai, bi) for ai, bi in zip(a, b)]


def ct_or_vec(a: List[int], b: List[int]) -> List[int]:
    """Element-wise ternary OR (max).

    For each position:  result = max(a, b)  in the ordering -1 < 0 < +1.
    """
    _check_vec(a, "a")
    _check_vec(b, "b")
    _check_same_len(a, b)
    return [max(ai, bi) for ai, bi in zip(a, b)]


def ct_neg(vec: List[int]) -> List[int]:
    """Negate each trit: -1 -> +1, 0 -> 0, +1 -> -1."""
    _check_vec(vec)
    return [-v for v in vec]


def ct_sum(vec: List[int]) -> int:
    """Integer sum of all trits."""
    _check_vec(vec)
    return sum(vec)


def ct_dot(a: List[int], b: List[int]) -> int:
    """Dot product sum(a[i] * b[i])."""
    _check_vec(a, "a")
    _check_vec(b, "b")
    _check_same_len(a, b)
    return sum(ai * bi for ai, bi in zip(a, b))


def ct_norm(vec: List[int]) -> int:
    """L1 norm sum(|a[i]|) -- number of non-zero positions for {-1,0,+1}."""
    _check_vec(vec)
    return sum(abs(x) for x in vec)


def ct_distance(a: List[int], b: List[int]) -> int:
    """Hamming-like distance: count of positions where trits differ."""
    _check_vec(a, "a")
    _check_vec(b, "b")
    _check_same_len(a, b)
    return sum(1 for ai, bi in zip(a, b) if ai != bi)
