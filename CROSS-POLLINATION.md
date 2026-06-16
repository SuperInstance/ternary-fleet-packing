# CROSS-POLLINATION.md — ternary-fleet-packing

> **Conservation Law Connection:** Packing density directly affects η

## Role in the Conservation Law

`ternary-fleet-packing` optimizes how fleet components are packed into deployable
binaries. Packing efficiency directly impacts η (overhead):

- **Dense packing** → smaller binary → faster cold start → lower η (initialization)
- **Sparse packing** → modular deployment → higher γ per component but more routing η
- **Optimal packing** → minimize total η → maximize available γ under the C budget

The conservation law predicts that optimal packing density for n components is
approximately 1 − δ(n), leaving δ(n) headroom for runtime coordination overhead.

## delta-clt Verification Results

The delta-clt dependency graph simulation models the trade-off:
- At n=50, edge_density=0.3: γ/C ≈ 97% (dense packing works at scale)
- At n=10, edge_density=0.3: γ/C ≈ 91% (sparse is proportionally costly)

This means packing should prioritize density for large fleets (n≥50) and accept
sparse overhead for small fleets where modularity aids debugging.

## Cross-Repo Connections

### → ternary-fleet
Packs the sub-crates of ternary-fleet into deployment artifacts.

**Shared:** Same component set. Packing reads the fleet dependency graph.
**Different:** `fleet` defines components; `packing` optimizes their deployment form.

### → superinstance-protocol
`superinstance-protocol` defines the wire format for fleet communication.
Packed components communicate via the protocol. The protocol's MessagePack payload
encoding affects packing — shared serialization reduces η.

**Shared:** Both optimize fleet-level data density.
**Different:** `packing` is binary deployment; `protocol` is runtime communication.

### → conservation-languages
`conservation-languages` studies how different language implementations affect
conservation law performance. Packing is language-agnostic but benefits from
language-specific optimizations (e.g., Rust's zero-cost abstractions pack tighter).

**Shared:** Both study implementation efficiency under the conservation law.
**Different:** `packing` is about binary layout; `languages` is about source language choice.

## Fleet Position

```
┌───────────────────────────────────────────────┐
│  ternary-fleet-packing — THE η OPTIMIZER       │
│                                                │
│  Fleet components ──► PACKER ──► Binary        │
│                          │                     │
│  Optimize: minimize binary η (cold start,      │
│            memory, routing overhead)            │
│  Constraint: γ quality must not degrade        │
│  Target density: ≈ 1 − δ(n)                    │
│                                                │
│  Input: ternary-fleet (components)             │
│  Output: deployable artifacts                  │
│  Commutes with: superinstance-protocol         │
└───────────────────────────────────────────────┘
```

