use ternary_fleet_packing::*;

#[test]
fn debug_optimal() {
    // 1000 elements, divisible by 5
    let vals: Vec<i8> = (0..1000).map(|i| (i as i8) % 3 - 1).collect();
    let mut buf = vec![0u8; (vals.len() + 4) / 5];
    let n = trit_optimal_pack(&vals, &mut buf).unwrap();
    assert_eq!(n, 200, "n should be 200");
    let out = trit_optimal_unpack::<1000>(&buf[..n]);
    for i in 0..1000 {
        assert_eq!(vals[i], out[i], "Mismatch at {i}: got {} expected {}", out[i], vals[i]);
    }
}
