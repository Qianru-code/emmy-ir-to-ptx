# Architecture

## Input contract

`main()` reads the IR path from `argv[1]`, deserializes it with Emmy's `Graph.from_dict`, and requires exactly one KernelOp in the graph. It reads the sibling `meta.json` and selects the matching kernel entry. `Compiler.compile()` produces PTX with the IR entry name and metadata-defined parameter order.

## Lowering structure

`Compiler` maintains maps of value names to virtual registers, buffers to base addresses, and buffers to static shapes. `reg()` allocates virtual float32, 32-bit integer, 64-bit address, or predicate registers. ptxas performs physical register allocation.

`expr()` recursively emits integer arithmetic, comparisons, logical operations, and integer ternary selection. `address()` flattens row-major indices and converts element offsets to byte offsets. The implementation assumes four-byte float32 elements.

`body()` dispatches on typed statement classes. Tile decodes a linear global thread index into axis coordinates and guards excess work items. Grouped raster axes are remapped when present. Load and Write scalarize multi-value accesses into consecutive elements; atomic writes emit floating-point reduction-add instructions.

Arithmetic supports add/sum, subtract, multiply/prod, divide, maximum/amax, minimum, copy, negative, abs, sqrt, reciprocal, rsqrt, relu, exp, and log. Exponential and logarithm use base-two approximate instructions with a base conversion. Other operations are rejected.

Loops use explicit labels and conditional branches. `seed_loop()` places accumulator identities before the loop associated with the accumulator's reduction axes, respecting the loop seed flag. Accum uses its optional `base` as the left operand when present, supporting rescaled streaming state.

WarpShuffle emits XOR butterflies and evaluates the IR's combine statements. TreeHalve loads state pairs from shared memory, evaluates the same combine statements, stores partials, synchronizes, and broadcasts the final state.

## Shared memory and transport

A prepass computes aligned offsets for Smem declarations. Allocations above 48 KiB use an external dynamic shared-memory pool and require sufficient `smem_bytes` in the launch metadata; smaller allocations are declared statically. Buffer addresses use the computed pool offsets.

CpAsyncCopy combines the shared-memory base index with `lane_index` when supplied. It emits asynchronous global-to-shared copies; CpAsyncCommit and CpAsyncWait preserve group ordering. Sync lowers to block or warp barriers. The compiler follows the schedule supplied in the IR rather than selecting a new tiling strategy.

## Limits

Static shapes and float32 buffers are required. Scalar launch arguments, auxiliary threads, non-NONE swizzles, TreeHalve with an inner body, tensor-core IR, and unimplemented statements/expressions are outside the supported scope. There is no autotuning, liveness analysis, or general optimization pass. Numerical correctness is evaluated within each problem's tolerance.
