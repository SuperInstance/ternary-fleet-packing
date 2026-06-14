//! # ternary-fleet-packing
//!
//! Packing and encoding algorithms for ternary representations.
//!
//! Ternary values (`-1`, `0`, `+1`) are commonly used in quantized neural
//! networks. This crate provides efficient bit-level packing, encoding, and
//! compression schemes to minimize memory footprint and enable fast data
//! transfer between model components.
//!
//! ## Packing Schemes
//!
//! | Scheme       | Bits / trit | Description                          |
//! |--------------|-------------|--------------------------------------|
//! | `Trit2`      | 2           | Naive bit-level encoding             |
//! | `TritPack`   | ~1.58       | Shannon-optimal base-3 packing       |
//! | `TritBinary` | 2           | Two-bit {sign, nonzero} encoding     |
//! | `TritRunLen` | variable    | Run-length encoding for sparse trits |
//!
//! ## Quick Start
//!
//! ```rust
//! use ternary_fleet_packing::*;
//!
//! let values = [1i8, -1, 0, 1, 0, 0, -1, 1];
//! let mut buf = [0u8; 4];
//!
//! // Pack using the default Trit2 scheme
//! let n = trit2_pack(&values, &mut buf);
//! let unpacked = trit2_unpack::<8>(&buf[..n]);
//! assert_eq!(values.as_slice(), &unpacked);
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single ternary value: `-1`, `0`, or `+1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum Trit {
    NegOne = -1,
    Zero = 0,
    PosOne = 1,
}

impl From<i8> for Trit {
    fn from(v: i8) -> Self {
        match v {
            -1 => Trit::NegOne,
            0 => Trit::Zero,
            1 => Trit::PosOne,
            // Clamp anything else to zero
            _ if v < 0 => Trit::NegOne,
            _ => Trit::PosOne,
        }
    }
}

impl From<Trit> for i8 {
    fn from(t: Trit) -> Self {
        t as i8
    }
}

// ---------------------------------------------------------------------------
// Scheme 1 — Trit2: 2 bits per trit (simple, fast)
// ---------------------------------------------------------------------------

/// Pack an array of trits into bytes using 2 bits per trit.
///
/// Encoding: `-1 → 0b00`, `0 → 0b01`, `1 → 0b10`.
/// Returns the number of bytes written.
///
/// # Panics
///
/// Panics if `buf` is too small (needs `ceil(values.len() * 2 / 8)` bytes).
pub fn trit2_pack(values: &[i8], buf: &mut [u8]) -> usize {
    let nbytes = (values.len() * 2 + 7) / 8;
    assert!(buf.len() >= nbytes, "buffer too small");
    buf[..nbytes].fill(0);

    for (i, &v) in values.iter().enumerate() {
        // map -1→0, 0→1, 1→2, clamp
        let code = match v {
            -1 => 0u8,
            0 => 1,
            1 => 2,
            _ if v < 0 => 0,
            _ => 2,
        };
        let byte_idx = (i * 2) / 8;
        let bit_off = (i * 2) % 8;
        buf[byte_idx] |= code << bit_off;
    }
    nbytes
}

/// Unpack trits from a Trit2-encoded byte slice.
///
/// The generic constant `N` is the number of trits to decode.
/// Returns `[i8; N]`.
///
/// # Panics
///
/// Panics if `data` is too short.
pub fn trit2_unpack<const N: usize>(data: &[u8]) -> [i8; N] {
    let mut out = [0i8; N];
    for i in 0..N {
        let byte_idx = (i * 2) / 8;
        let bit_off = (i * 2) % 8;
        let code = (data[byte_idx] >> bit_off) & 0b11;
        out[i] = match code {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0, // 0b11 is reserved/invalid
        };
    }
    out
}

// ---------------------------------------------------------------------------
// Scheme 2 — TritBinary: {sign, nonzero} in 2 bits
// ---------------------------------------------------------------------------

/// Pack using the trit-binary scheme.
///
/// Bit 0: `nonzero` (1 if trit ≠ 0)  
/// Bit 1: `sign`    (1 if trit < 0)  
/// This representation is convenient for fast XNOR-style dot products
/// because the nonzero bit gates the accumulation.
pub fn trit_binary_pack(values: &[i8], buf: &mut [u8]) -> usize {
    let nbytes = (values.len() * 2 + 7) / 8;
    assert!(buf.len() >= nbytes, "buffer too small");
    buf[..nbytes].fill(0);

    for (i, &v) in values.iter().enumerate() {
        let nonzero = if v == 0 { 0u8 } else { 1 };
        let sign = if v < 0 { 1u8 } else { 0 };
        let code = (sign << 1) | nonzero;
        let byte_idx = (i * 2) / 8;
        let bit_off = (i * 2) % 8;
        buf[byte_idx] |= code << bit_off;
    }
    nbytes
}

/// Unpack trits from a TritBinary-encoded byte slice.
pub fn trit_binary_unpack<const N: usize>(data: &[u8]) -> [i8; N] {
    let mut out = [0i8; N];
    for i in 0..N {
        let byte_idx = (i * 2) / 8;
        let bit_off = (i * 2) % 8;
        let code = (data[byte_idx] >> bit_off) & 0b11;
        let nonzero = code & 1;
        let sign = (code >> 1) & 1;
        out[i] = if nonzero == 0 {
            0
        } else if sign == 1 {
            -1
        } else {
            1
        };
    }
    out
}

// ---------------------------------------------------------------------------
// Scheme 3 — TritPack: base-3 packing (optimal ~1.58 bits/trit)
// ---------------------------------------------------------------------------

/// Pack trits using base-3 representation — the entropy-optimal scheme.
///
/// Every `k` trits produce a base-3 integer stored in
/// `ceil(k * log2(3) / 8)` bytes.
///
/// The current implementation packs in groups of 5 trits
/// (5 trits = 3⁵ = 243 values, fitting in 8 bytes = 256 values,
/// wasting only 13 codes — ~95% space-efficient).
pub fn trit_optimal_pack(values: &[i8], buf: &mut [u8]) -> Result<usize, &'static str> {
    // Group size = 5 trits → base-3 integer [0, 242] → stored in 1 byte (since 243 > 255).
    // Actually 3^5 = 243 fits in 1 byte!  243 < 256.
    // So 5 trits → 1 byte. That's 1.6 bits/trit, very close to theoretical 1.58.
    const GROUP: usize = 5;

    let n_groups = (values.len() + GROUP - 1) / GROUP;
    let nbytes = n_groups; // one byte per group
    if buf.len() < nbytes {
        return Err("buffer too small");
    }
    buf[..nbytes].fill(0);

    for g in 0..n_groups {
        let mut acc: u16 = 0;
        let start = g * GROUP;
        let end = (start + GROUP).min(values.len());
        for i in start..end {
            let v = values[i];
            let code: u16 = match v {
                -1 => 0,
                0 => 1,
                1 => 2,
                _ if v < 0 => 0,
                _ => 2,
            };
            acc = acc * 3 + code;
        }
        // Pad remaining slots with zero (trit encoding for -1)
        for _ in end..start + GROUP {
            acc = acc * 3 + 0;
        }
        buf[g] = acc as u8; // acc ∈ [0, 242] so this is safe
    }
    Ok(nbytes)
}

/// Unpack trits from a TritPack-encoded byte slice.
pub fn trit_optimal_unpack<const N: usize>(data: &[u8]) -> [i8; N] {
    const GROUP: usize = 5;
    let mut out = [0i8; N];

    for g in 0..(N + GROUP - 1) / GROUP {
        let mut acc = data[g] as u16;
        // Extract trits LSB-first (reverse order from pack which is MSB-first)
        let mut trits = [0u8; GROUP];
        for t in trits.iter_mut() {
            *t = (acc % 3) as u8;
            acc /= 3;
        }
        // Write back in original order
        for i in 0..GROUP {
            let idx = g * GROUP + i;
            if idx >= N {
                break;
            }
            // trits[GROUP - 1 - i] corresponds to the i-th position in the pack
            out[idx] = match trits[GROUP - 1 - i] {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => unreachable!(),
            };
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Scheme 4 — Run-length encoding for sparse ternary vectors
// ---------------------------------------------------------------------------

/// A single run in a run-length encoded ternary sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    /// The trit value of this run.
    pub value: Trit,
    /// How many consecutive occurrences.
    pub length: usize,
}

/// Run-length encode a ternary slice.
///
/// This is efficient when trits are highly sparse (mostly zeros)
/// or have long contiguous runs.
pub fn rle_encode(values: &[i8]) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    if values.is_empty() {
        return runs;
    }

    let mut current = Trit::from(values[0]);
    let mut count: usize = 1;

    for &v in &values[1..] {
        let t = Trit::from(v);
        if t == current {
            count += 1;
        } else {
            runs.push(Run {
                value: current,
                length: count,
            });
            current = t;
            count = 1;
        }
    }
    runs.push(Run {
        value: current,
        length: count,
    });
    runs
}

/// Decode a run-length encoded ternary sequence.
pub fn rle_decode(runs: &[Run], out: &mut [i8]) -> Result<usize, &'static str> {
    let mut pos = 0;
    for run in runs {
        let end = pos + run.length;
        if end > out.len() {
            return Err("output buffer too small");
        }
        let v: i8 = run.value.into();
        out[pos..end].fill(v);
        pos = end;
    }
    Ok(pos)
}

// ---------------------------------------------------------------------------
// High-level API
// ---------------------------------------------------------------------------

/// Supported packing schemes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingScheme {
    /// 2 bits/trit naive encoding.
    Trit2,
    /// 2-bit {sign, nonzero} representation.
    TritBinary,
    /// Base-3 optimal packing (~1.6 bits/trit).
    TritOptimal,
}

/// Pack ternary values using the specified scheme.
///
/// Returns the number of bytes written to `buf`.
pub fn pack(values: &[i8], buf: &mut [u8], scheme: PackingScheme) -> Result<usize, &'static str> {
    match scheme {
        PackingScheme::Trit2 => {
            if buf.len() < (values.len() * 2 + 7) / 8 {
                return Err("buffer too small for Trit2");
            }
            Ok(trit2_pack(values, buf))
        }
        PackingScheme::TritBinary => {
            if buf.len() < (values.len() * 2 + 7) / 8 {
                return Err("buffer too small for TritBinary");
            }
            Ok(trit_binary_pack(values, buf))
        }
        PackingScheme::TritOptimal => trit_optimal_pack(values, buf),
    }
}

/// Unpack ternary values using the specified scheme.
pub fn unpack<const N: usize>(data: &[u8], scheme: PackingScheme) -> [i8; N] {
    match scheme {
        PackingScheme::Trit2 => trit2_unpack::<N>(data),
        PackingScheme::TritBinary => trit_binary_unpack::<N>(data),
        PackingScheme::TritOptimal => trit_optimal_unpack::<N>(data),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trit2_roundtrip_small() {
        let vals = [1i8, -1, 0, 1, 0, 0, -1, 1];
        let mut buf = [0u8; 4];
        let n = trit2_pack(&vals, &mut buf);
        assert!(n <= 2);
        let out = trit2_unpack::<8>(&buf[..n]);
        assert_eq!(vals.as_slice(), &out);
    }

    #[test]
    fn test_trit2_roundtrip_all_combos() {
        let vals: Vec<i8> = (0..100)
            .map(|i| match i % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect();
        let mut buf = vec![0u8; (vals.len() * 2 + 7) / 8];
        let n = trit2_pack(&vals, &mut buf);
        let out = trit2_unpack::<100>(&buf[..n]);
        assert_eq!(vals.as_slice(), &out);
    }

    #[test]
    fn test_trit_binary_roundtrip() {
        let vals = [1i8, -1, 0, 1, 0, -1];
        let mut buf = [0u8; 4];
        let n = trit_binary_pack(&vals, &mut buf);
        let out = trit_binary_unpack::<6>(&buf[..n]);
        assert_eq!(vals.as_slice(), &out);
    }

    #[test]
    fn test_trit_optimal_roundtrip() {
        let vals: [i8; 10] = [1, -1, 0, 1, -1, -1, 0, 0, 1, -1];
        let mut buf = [0u8; 4];
        let n = trit_optimal_pack(&vals, &mut buf).unwrap();
        let out = trit_optimal_unpack::<10>(&buf[..n]);
        assert_eq!(vals.as_slice(), &out);
    }

    #[test]
    fn test_trit_optimal_large() {
        let vals: Vec<i8> = (0..1000).map(|i| (i as i8) % 3 - 1).collect();
        let mut buf = vec![0u8; (vals.len() + 4) / 5];
        let n = trit_optimal_pack(&vals, &mut buf).unwrap();
        let out: Vec<i8> = trit_optimal_unpack::<1000>(&buf[..n]).to_vec();
        assert_eq!(vals, out);
    }

    #[test]
    fn test_rle_roundtrip() {
        let vals: [i8; 12] = [0, 0, 0, 1, 1, -1, -1, -1, -1, 0, 0, 1];
        let runs = rle_encode(&vals);
        assert_eq!(runs.len(), 5);
        let mut decoded = [0i8; 12];
        rle_decode(&runs, &mut decoded).unwrap();
        assert_eq!(vals.as_slice(), &decoded);
    }

    #[test]
    fn test_rle_all_zeros() {
        let vals = [0i8; 50];
        let runs = rle_encode(&vals);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].value, Trit::Zero);
        assert_eq!(runs[0].length, 50);
        let mut decoded = [0i8; 50];
        rle_decode(&runs, &mut decoded).unwrap();
        assert_eq!(vals.as_slice(), &decoded);
    }

    #[test]
    fn test_pack_scheme_dispatch() {
        let vals = [1i8, -1, 0, 1];
        let mut buf = [0u8; 4];
        for scheme in &[
            PackingScheme::Trit2,
            PackingScheme::TritBinary,
            PackingScheme::TritOptimal,
        ] {
            let n = pack(&vals, &mut buf, *scheme).unwrap();
            let out = unpack::<4>(&buf[..n], *scheme);
            assert_eq!(vals.as_slice(), &out, "scheme {scheme:?} mismatch");
        }
    }

    #[test]
    fn test_trit_from_i8() {
        assert_eq!(Trit::from(-1i8), Trit::NegOne);
        assert_eq!(Trit::from(0i8), Trit::Zero);
        assert_eq!(Trit::from(1i8), Trit::PosOne);
        assert_eq!(Trit::from(-5i8), Trit::NegOne);
        assert_eq!(Trit::from(127i8), Trit::PosOne);
    }

    #[test]
    fn test_trit_into_i8() {
        let v: i8 = Trit::NegOne.into();
        assert_eq!(v, -1);
        let v: i8 = Trit::Zero.into();
        assert_eq!(v, 0);
        let v: i8 = Trit::PosOne.into();
        assert_eq!(v, 1);
    }

    #[test]
    fn test_trit2_odd_count() {
        // 3 trits = 6 bits = needs 1 byte
        let vals = [1i8, -1, 0];
        let mut buf = [0u8; 1];
        let n = trit2_pack(&vals, &mut buf);
        assert_eq!(n, 1);
        let out = trit2_unpack::<3>(&buf[..n]);
        assert_eq!(vals.as_slice(), &out);
    }

    #[test]
    fn test_rle_decode_buffer_overflow() {
        let runs = [Run { value: Trit::PosOne, length: 10 }];
        let mut buf = [0i8; 5];
        let r = rle_decode(&runs, &mut buf);
        assert!(r.is_err());
    }

    #[test]
    fn test_empty_inputs() {
        let r = rle_encode(&[]);
        assert!(r.is_empty());

        let mut buf = [0u8; 1];
        let r = pack(&[], &mut buf, PackingScheme::Trit2);
        assert!(r.is_ok());
    }
}
