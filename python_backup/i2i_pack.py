"""
i2i_pack.py — Encode/decode trit vectors for I2I bottle transport.

Packing scheme
--------------
Each trit is encoded as 2 bits, enabling 4 trits per byte:

    -1 (avoid)   -> 0b00
     0 (unknown) -> 0b01
    +1 (choose)  -> 0b10
     2 (reserved)-> 0b11

Within a byte, trit-0 occupies the lowest 2 bits (bits 0-1),
trit-1 is at bits 2-3, trit-2 at bits 4-5, trit-3 at bits 6-7.

Wire format
-----------
  [version:1B] [trit_count:4B LE] [payload:ceil(n_trits/4)B]

  - version:     protocol version, currently 1
  - trit_count:  number of trits in the vector (little-endian uint32)
  - payload:     packed trit bytes
"""

import struct
from math import ceil
from typing import List, Tuple

# Mapping trit -> 2-bit code (trit value 0..3, with -1 mapped to index 0)
_TRIT_TO_BITS = [-1, 0, 1, 2]  # index -1->0, 0->1, 1->2, 2->3
_BITS_TO_TRIT = [-1,  0,  1,  2]  # 0->-1, 1->0, 2->+1, 3->2

# Protocol constants
PROTOCOL_VERSION = 1
HEADER_SIZE = 5  # 1 (version) + 4 (trit count)
TRITS_PER_BYTE = 4


def _trit_to_2bit(trit: int) -> int:
    """Convert a trit (-1, 0, +1) to its 2-bit encoding (0, 1, 2)."""
    if trit == -1:
        return 0
    if trit == 0:
        return 1
    if trit == 1:
        return 2
    if trit == 2:
        return 3
    raise ValueError(f"Invalid trit value {trit}, expected -1, 0, +1, or 2")


def _bits_to_trit(bits: int) -> int:
    """Convert a 2-bit value (0-3) back to a trit (-1, 0, +1, 2)."""
    if bits == 0:
        return -1
    if bits == 1:
        return 0
    if bits == 2:
        return 1
    if bits == 3:
        return 2
    raise ValueError(f"Invalid 2-bit code {bits}, expected 0-3")


def trit_byte_to_vec(b: int) -> List[int]:
    """Unpack one byte to a list of 4 trits (LSB trit first)."""
    trits = []
    for shift in range(0, 8, 2):
        code = (b >> shift) & 0x03
        trits.append(_bits_to_trit(code))
    return trits


def vec_to_trit_byte(trits: List[int]) -> int:
    """Pack 4 trits into one byte (LSB trit first)."""
    b = 0
    for i, t in enumerate(trits):
        code = _trit_to_2bit(t)
        b |= (code & 0x03) << (i * 2)
    return b


def pack_trits(trits: List[int]) -> bytes:
    """Pack a list of trits into the I2I wire format.

    Args:
        trits: List of trit values (-1, 0, +1).

    Returns:
        bytes in wire format: [version][trit_count LE][payload].
    """
    n = len(trits)
    payload_len = ceil(n / TRITS_PER_BYTE)
    payload = bytearray(payload_len)

    for i, t in enumerate(trits):
        byte_idx = i // TRITS_PER_BYTE
        bit_shift = (i % TRITS_PER_BYTE) * 2
        code = _trit_to_2bit(t)
        payload[byte_idx] |= (code & 0x03) << bit_shift

    header = struct.pack("<BI", PROTOCOL_VERSION, n)
    return header + bytes(payload)


def unpack_trits(data: bytes) -> Tuple[int, List[int]]:
    """Unpack I2I wire format back into a protocol version and trit list.

    Args:
        data: Raw bytes in I2I wire format.

    Returns:
        Tuple of (protocol_version, trits).

    Raises:
        ValueError: If data is too short or version is unsupported.
    """
    if len(data) < HEADER_SIZE:
        raise ValueError(
            f"Data too short: {len(data)} bytes, minimum {HEADER_SIZE}"
        )

    version = data[0]
    n = struct.unpack_from("<I", data, 1)[0]
    payload_len = ceil(n / TRITS_PER_BYTE)
    expected_total = HEADER_SIZE + payload_len

    if len(data) < expected_total:
        raise ValueError(
            f"Data too short for {n} trits: got {len(data)} bytes, "
            f"expected at least {expected_total}"
        )

    payload = data[HEADER_SIZE:HEADER_SIZE + payload_len]
    trits = []
    for i in range(n):
        byte_idx = i // TRITS_PER_BYTE
        bit_shift = (i % TRITS_PER_BYTE) * 2
        code = (payload[byte_idx] >> bit_shift) & 0x03
        trits.append(_bits_to_trit(code))

    return version, trits
