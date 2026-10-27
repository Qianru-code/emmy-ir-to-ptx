# Submission form text

## Tech stack

Python 3.12+, Emmy 0.3.11 (typed IR loading), custom PTX 8.0 / sm_89 code generation, NumPy, Requests, and the official GPU runner.

## Approach

I implemented a Python compiler that loads Emmy Kernel IR using Graph.from_dict and emits PTX directly. The lowering does not invoke Emmy's CUDA renderer, emmy compile, nvcc, or another compiler. AI coding assistance was used during development and debugging.

The compiler reads the kernel name from the IR and argument order from the sibling meta.json. It recursively lowers index expressions to integer instructions, computes row-major byte addresses, decodes Tile thread coordinates, and inserts work-item bounds checks. Multi-value loads and stores are emitted as consecutive scalar accesses.

Loops and conditionals become labels and branches. Accumulators are initialized at the appropriate loop scope using their reduction axes and seed flags. Optional rescaled bases and the IR's combine statements preserve online softmax state updates. WarpShuffle and TreeHalve provide warp-local and shared-memory reduction.

For matrix multiplication, the compiler supports cp.async copies, lane offsets, commit/wait groups, barriers, and aligned shared-memory allocations. Allocations above 48 KiB use the dynamic shared-memory pool reserved by launch metadata. Atomic float32 addition is supported where requested.

The saved official-runner results show 15/15 public problems passing. Two supplemental NumPy-checked cases also pass with changed dimensions, subtraction, excess threads, non-divisible reduction tails, and reordered arguments. Hidden-set results are not available. The current scope is static shapes and float32 buffers; unsupported IR features are rejected. The focus is correctness rather than tensor-core or performance optimization.

## Video title

Emmy IR to PTX Compiler | RiftStack Emmy Hackathon 2026

## Video description

A walkthrough of a Python compiler that translates typed Emmy Kernel IR directly into PTX. The demonstration covers thread mapping, memory accesses, reductions, asynchronous copies, shared-memory allocation, and recorded public-suite validation. AI coding assistance was used during development. Public tests: 15/15 passed; hidden-set results are not available.
