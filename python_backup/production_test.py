#!/usr/bin/env python3
"""
production_test.py — Comprehensive test suite for
ternary_vec.py and i2i_pack.py.
"""

import math
import struct
import sys
import unittest
from typing import List

from ternary_vec import (
    ct_and_vec,
    ct_or_vec,
    ct_neg,
    ct_sum,
    ct_dot,
    ct_norm,
    ct_distance,
)
from i2i_pack import (
    pack_trits,
    unpack_trits,
    trit_byte_to_vec,
    vec_to_trit_byte,
    PROTOCOL_VERSION,
    HEADER_SIZE,
    TRITS_PER_BYTE,
)


# =========================================================================
# Tests: ternary_vec.py
# =========================================================================

class TestTernaryVecValidation(unittest.TestCase):
    """Input validation and edge cases."""

    def test_empty_vectors(self):
        self.assertEqual(ct_and_vec([], []), [])
        self.assertEqual(ct_or_vec([], []), [])
        self.assertEqual(ct_neg([]), [])
        self.assertEqual(ct_sum([]), 0)
        self.assertEqual(ct_dot([], []), 0)
        self.assertEqual(ct_norm([]), 0)
        self.assertEqual(ct_distance([], []), 0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            ct_and_vec([1], [1, 0])
        with self.assertRaises(ValueError):
            ct_or_vec([1, 0], [1])
        with self.assertRaises(ValueError):
            ct_dot([1, 0, -1], [1, 0])
        with self.assertRaises(ValueError):
            ct_distance([1], [0, -1])

    def test_invalid_trit_values(self):
        for fn in (ct_sum, ct_norm, ct_neg):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(ValueError):
                    fn([0, 2])       # 2 is only valid for i2i_pack
                with self.assertRaises(ValueError):
                    fn([-2])
                with self.assertRaises(ValueError):
                    fn([99])

    def test_invalid_trit_values_binary(self):
        for fn in (ct_and_vec, ct_or_vec, ct_dot, ct_distance):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(ValueError):
                    fn([0, 3], [0, 0])
                with self.assertRaises(ValueError):
                    fn([0, 0], [-2, 0])


class TestTernaryVecAnd(unittest.TestCase):
    """ct_and_vec = element-wise min."""

    def test_same(self):
        self.assertEqual(
            ct_and_vec([-1, -1, 0, 0, 1, 1], [-1, 1, -1, 1, -1, 1]),
            [-1, -1, -1, 0, -1, 1],
        )

    def test_all_pairs(self):
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                expected = min(a, b)
                with self.subTest(a=a, b=b):
                    self.assertEqual(ct_and_vec([a], [b]), [expected])

    def test_identity(self):
        vec = [-1, 0, 1, -1, 0, 1]
        # and with all +1 should produce the original (min(x, +1) = x)
        self.assertEqual(ct_and_vec(vec, [1]*len(vec)), vec)

    def test_zero_annihilator(self):
        vec = [-1, 0, 1, -1, 0, 1]
        # and with all 0 should zero out anything positive, but keep -1
        self.assertEqual(ct_and_vec(vec, [0]*len(vec)), [-1, 0, 0, -1, 0, 0])

    def test_single_element(self):
        self.assertEqual(ct_and_vec([-1], [1]), [-1])
        self.assertEqual(ct_and_vec([1], [1]), [1])
        self.assertEqual(ct_and_vec([0], [1]), [0])


class TestTernaryVecOr(unittest.TestCase):
    """ct_or_vec = element-wise max."""

    def test_same(self):
        self.assertEqual(
            ct_or_vec([-1, -1, 0, 0, 1, 1], [-1, 1, -1, 1, -1, 1]),
            [-1, 1, 0, 1, 1, 1],
        )

    def test_all_pairs(self):
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                expected = max(a, b)
                with self.subTest(a=a, b=b):
                    self.assertEqual(ct_or_vec([a], [b]), [expected])

    def test_identity(self):
        vec = [-1, 0, 1, -1, 0, 1]
        # or with all -1 should produce the original (max(x, -1) = x)
        self.assertEqual(ct_or_vec(vec, [-1]*len(vec)), vec)

    def test_one_annihilator(self):
        vec = [-1, 0, 1, -1, 0, 1]
        # or with all +1 should produce all +1
        self.assertEqual(ct_or_vec(vec, [1]*len(vec)), [1]*len(vec))


class TestTernaryVecNeg(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(ct_neg([-1, 0, 1]), [1, 0, -1])

    def test_double_negation(self):
        vec = [-1, 0, 1, -1, 0, 1, -1, 1, 0]
        self.assertEqual(ct_neg(ct_neg(vec)), vec)

    def test_single(self):
        self.assertEqual(ct_neg([-1]), [1])
        self.assertEqual(ct_neg([0]), [0])
        self.assertEqual(ct_neg([1]), [-1])


class TestTernaryVecSum(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(ct_sum([-1, 0, 1, -1, 0, 1]), 0)

    def test_all_positive(self):
        self.assertEqual(ct_sum([1, 1, 1]), 3)

    def test_all_negative(self):
        self.assertEqual(ct_sum([-1, -1, -1]), -3)

    def test_mixed(self):
        self.assertEqual(ct_sum([-1, -1, 1]), -1)
        self.assertEqual(ct_sum([1, 1, -1]), 1)

    def test_zeros(self):
        self.assertEqual(ct_sum([0, 0, 0]), 0)

    def test_single(self):
        self.assertEqual(ct_sum([-1]), -1)
        self.assertEqual(ct_sum([0]), 0)
        self.assertEqual(ct_sum([1]), 1)


class TestTernaryVecDot(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(ct_dot([-1, 0, 1], [-1, 0, 1]), 2)
        # (-1)*(-1) + 0*0 + 1*1 = 1 + 0 + 1 = 2

    def test_orthogonal(self):
        self.assertEqual(ct_dot([-1, 0, 1], [1, 0, -1]), -2)
        # (-1)*1 + 0*0 + 1*(-1) = -1 + 0 + -1 = -2

    def test_zero_vec(self):
        vec = [-1, 0, 1]
        self.assertEqual(ct_dot(vec, [0, 0, 0]), 0)

    def test_opposite(self):
        a = [-1, 1, 0, -1, 1]
        b = [1, -1, 0, 1, -1]
        # sum of: -1 + -1 + 0 + -1 + -1 = -4
        self.assertEqual(ct_dot(a, b), -4)

    def test_same_vec(self):
        vec = [-1, 0, 1, -1, 0, 1]
        # sum of squares: 1+0+1+1+0+1 = 4
        self.assertEqual(ct_dot(vec, vec), 4)


class TestTernaryVecNorm(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(ct_norm([-1, 0, 1, -1, 0, 1]), 4)

    def test_all_nonzero(self):
        self.assertEqual(ct_norm([-1, 1, -1, 1, -1, 1]), 6)

    def test_all_zero(self):
        self.assertEqual(ct_norm([0, 0, 0]), 0)

    def test_single(self):
        self.assertEqual(ct_norm([-1]), 1)
        self.assertEqual(ct_norm([0]), 0)
        self.assertEqual(ct_norm([1]), 1)


class TestTernaryVecDistance(unittest.TestCase):

    def test_same(self):
        vec = [-1, 0, 1, -1, 0, 1]
        self.assertEqual(ct_distance(vec, vec), 0)

    def test_opposite(self):
        self.assertEqual(ct_distance([-1, 0, 1], [1, 0, -1]), 2)

    def test_all_different(self):
        self.assertEqual(ct_distance([-1, -1, -1], [1, 1, 1]), 3)

    def test_partial(self):
        self.assertEqual(ct_distance([-1, 0, 1], [-1, 0, 0]), 1)

    def test_empty(self):
        self.assertEqual(ct_distance([], []), 0)


# =========================================================================
# Tests: i2i_pack.py
# =========================================================================

class TestI2IBitLevel(unittest.TestCase):
    """Low-level byte <-> trit packing."""

    def test_vec_to_trit_byte_all_avoid(self):
        # 4 trits all -1: codes 0,0,0,0 -> byte 0
        self.assertEqual(vec_to_trit_byte([-1, -1, -1, -1]), 0)

    def test_vec_to_trit_byte_all_unknown(self):
        # 4 trits all 0: codes 1,1,1,1 -> bits (01)(01)(01)(01) = 0x55
        self.assertEqual(vec_to_trit_byte([0, 0, 0, 0]), 0b01010101)

    def test_vec_to_trit_byte_all_choose(self):
        # 4 trits all +1: codes 2,2,2,2 -> bits (10)(10)(10)(10) = 0xAA
        self.assertEqual(vec_to_trit_byte([1, 1, 1, 1]), 0b10101010)

    def test_vec_to_trit_byte_all_reserved(self):
        # 4 trits all 2: codes 3,3,3,3 -> bits (11)(11)(11)(11) = 0xFF
        self.assertEqual(vec_to_trit_byte([2, 2, 2, 2]), 0xFF)

    def test_roundtrip_byte(self):
        """Byte -> trits -> byte should roundtrip."""
        for b in range(256):
            trits = trit_byte_to_vec(b)
            self.assertEqual(len(trits), 4)
            reconstructed = vec_to_trit_byte(trits)
            self.assertEqual(reconstructed, b, f"Failed at byte 0x{b:02x}")

    def test_specific_trit_order(self):
        """LSB trit first: trit[0] in bits 0-1, trit[1] in bits 2-3, etc."""
        # Set each trit position individually
        for pos in range(4):
            # Place -1 at position pos, all others 0
            trits = [0, 0, 0, 0]
            trits[pos] = -1
            b = vec_to_trit_byte(trits)
            # Only the 2-bit field at 2*pos should be set
            self.assertEqual((b >> (pos * 2)) & 0x03, 0)
            # Other fields should be 01 (code for 0)
            for other in range(4):
                if other != pos:
                    self.assertEqual((b >> (other * 2)) & 0x03, 1)

    def test_trit_byte_to_vec_counts(self):
        """Each byte should produce exactly 4 trits."""
        for b in range(256):
            trits = trit_byte_to_vec(b)
            self.assertEqual(len(trits), 4)


class TestI2IPackRoundtrip(unittest.TestCase):
    """Full pack/unpack roundtrips."""

    def _roundtrip(self, trits: List[int]):
        packed = pack_trits(trits)
        version, unpacked = unpack_trits(packed)
        self.assertEqual(version, PROTOCOL_VERSION)
        self.assertEqual(unpacked, trits)

    def test_empty(self):
        packed = pack_trits([])
        version, unpacked = unpack_trits(packed)
        self.assertEqual(version, PROTOCOL_VERSION)
        self.assertEqual(unpacked, [])

    def test_single_trit(self):
        for t in (-1, 0, 1):
            with self.subTest(trit=t):
                self._roundtrip([t])

    def test_two_trits(self):
        self._roundtrip([-1, 0])
        self._roundtrip([0, 1])
        self._roundtrip([1, -1])
        self._roundtrip([-1, 1])

    def test_three_trits(self):
        self._roundtrip([-1, 0, 1])

    def test_four_trits_exactly_one_byte(self):
        self._roundtrip([-1, 0, 1, -1])
        self._roundtrip([0, 1, -1, 0])
        self._roundtrip([1, -1, 0, 1])

    def test_five_trits_crosses_byte_boundary(self):
        self._roundtrip([-1, 0, 1, -1, 0])

    def test_seven_trits(self):
        self._roundtrip([1, 1, 1, -1, -1, -1, 0])

    def test_thirty_two_trits(self):
        """The canonical I2I bottle size: 32 trits = 64 bits = 8 bytes payload."""
        trits = [
            -1, 0, 1, -1, 0, 1, -1, 0,
            1, -1, 0, 1, -1, 0, 1, -1,
            0, 1, -1, 0, 1, -1, 0, 1,
            -1, 0, 1, -1, 0, 1, -1, 0,
        ]
        self._roundtrip(trits)

    def test_large_vector(self):
        trits = [-1, 0, 1] * 100  # 300 trits
        self._roundtrip(trits)

    def test_all_avoid(self):
        self._roundtrip([-1] * 50)

    def test_all_unknown(self):
        self._roundtrip([0] * 50)

    def test_all_choose(self):
        self._roundtrip([1] * 50)

    def test_alternating(self):
        n = 64
        trits = [(-1 if i % 2 == 0 else 1) for i in range(n)]
        self._roundtrip(trits)

    def test_includes_reserved(self):
        """The packing format supports trit value 2 (reserved)."""
        self._roundtrip([-1, 0, 1, 2])
        self._roundtrip([2, 2, 2, 2, 2, 2, 2, 2])


class TestI2IPackStructure(unittest.TestCase):
    """Verify the wire format structure."""

    def test_version_byte(self):
        packed = pack_trits([-1, 0, 1])
        self.assertEqual(packed[0], PROTOCOL_VERSION)

    def test_trit_count_field(self):
        n = 42
        trits = [0] * n
        packed = pack_trits(trits)
        count = struct.unpack_from("<I", packed, 1)[0]
        self.assertEqual(count, n)

    def test_payload_size(self):
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 32, 100]:
            trits = [0] * n
            packed = pack_trits(trits)
            expected_payload = math.ceil(n / TRITS_PER_BYTE)
            expected_total = HEADER_SIZE + expected_payload
            self.assertEqual(len(packed), expected_total,
                             f"n={n}: expected {expected_total} bytes, got {len(packed)}")

    def test_all_bytes_zero_for_avoid(self):
        """All -1 trits should produce a zero-filled payload."""
        packed = pack_trits([-1] * 32)
        for i in range(HEADER_SIZE, len(packed)):
            self.assertEqual(packed[i], 0,
                             f"byte {i} should be 0x00 for all-avoid trits")


class TestI2IPackErrors(unittest.TestCase):
    """Error handling for unpack."""

    def test_data_too_short(self):
        with self.assertRaises(ValueError):
            unpack_trits(b"\x01\x00\x00\x00")  # only 5 bytes, need 5+ceil(0/4)=5

    def test_data_truncated_payload(self):
        """Header says 8 trits (needs 2 payload bytes) but only 1 payload byte provided."""
        # version=1, n=8 (8 trits), then only 1 payload byte
        data = struct.pack("<BI", 1, 8) + b"\x00"
        with self.assertRaises(ValueError):
            unpack_trits(data)

    def test_large_count_truncated(self):
        """Header claims large count but payload too short."""
        data = struct.pack("<BI", 1, 1000) + b"\x00" * 10
        with self.assertRaises(ValueError):
            unpack_trits(data)


# =========================================================================
# Integration: combined vec ops + i2i packing
# =========================================================================

class TestIntegration(unittest.TestCase):
    """Combined pipeline: ternary ops followed by pack/unpack."""

    def test_and_then_pack(self):
        a = [-1, 0, 1, -1, 0, 1, -1, 1]
        b = [1, -1, 0, -1, 1, 0, 0, 0]
        result = ct_and_vec(a, b)
        # manual: min(-1,1)=-1, min(0,-1)=-1, min(1,0)=0,
        #         min(-1,-1)=-1, min(0,1)=0, min(1,0)=0,
        #         min(-1,0)=-1, min(1,0)=0
        expected = [-1, -1, 0, -1, 0, 0, -1, 0]
        self.assertEqual(result, expected)

        packed = pack_trits(result)
        version, unpacked = unpack_trits(packed)
        self.assertEqual(version, 1)
        self.assertEqual(unpacked, expected)

    def test_or_then_pack(self):
        a = [-1, 0, 1, -1, 0, 1]
        b = [1, -1, 0, -1, 1, 0]
        result = ct_or_vec(a, b)
        # manual: max(-1,1)=1, max(0,-1)=0, max(1,0)=1,
        #         max(-1,-1)=-1, max(0,1)=1, max(1,0)=1
        expected = [1, 0, 1, -1, 1, 1]
        self.assertEqual(result, expected)

        packed = pack_trits(result)
        _, unpacked = unpack_trits(packed)
        self.assertEqual(unpacked, expected)

    def test_pack_then_negate_then_pack(self):
        original = [-1, 0, 1, -1, 0, 1, 0, 0, 1, -1]
        # pack -> unpack -> negate -> pack -> unpack
        _, trits = unpack_trits(pack_trits(original))
        negated = ct_neg(trits)
        _, reconstructed = unpack_trits(pack_trits(negated))
        self.assertEqual(reconstructed, [1, 0, -1, 1, 0, -1, 0, 0, -1, 1])

    def test_dot_after_pack_unpack(self):
        trits = [-1, 0, 1, -1, 0, 1, -1, 1, 0, -1, 1, 0]
        _, restored = unpack_trits(pack_trits(trits))
        self.assertEqual(ct_dot(trits, restored), ct_norm(trits))
        self.assertEqual(ct_distance(trits, restored), 0)


# =========================================================================
# Entry point
# =========================================================================

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
