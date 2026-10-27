# Emmy IR to PTX Compiler

A Python compiler for the RiftStack Emmy Hackathon 2026. It walks typed Emmy Kernel IR and emits PTX directly, preserving the input schedule, memory accesses, control flow, and reduction semantics.

**Recorded results: 15/15 public problems passed on the official GPU runner.** Two additional tests also passed with changed dimensions, subtraction, excess threads, non-divisible reduction tails, and reordered parameters. These are public and supplemental results, not hidden-set results. See [validation](docs/VALIDATION.md).

## Setup

Requires Python 3.12 or newer. The recorded environment used Python 3.13.9 and Emmy 0.3.11. A local CUDA installation or GPU is not required to emit PTX or use the hosted runner.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Compile

```sh
python compile.py problems/p01_vector_add/k_add_pointwise.ir.json > out.ptx
```

The evaluation interface is exactly:

```sh
python compile.py path/to/kernel.ir.json > out.ptx
```

Keep the supplied `meta.json` in the same directory as the IR file. The kernel entry name comes from the IR; argument order comes from the matching entry in `meta.json`. Standard output contains PTX only. Errors go to standard error.

## Verify on the official runner

```sh
python submit.py p01_vector_add --compiler compile.py
python verify.py
python tests/variants.py
```

These commands require internet access and send generated PTX and test inputs to the organizer's GPU runner. They test the compiler; they do not submit a hackathon entry. `verify.py` writes `verification.json`. After a transient network failure, use `python verify.py --retry-failed` to rerun failed entries while retaining successful results.

## Implementation

- Typed IR loading with `Graph.from_dict`; no use of Emmy's renderer, `emmy compile`, nvcc, or another compiler for IR-to-PTX translation.
- Recursive integer expressions and row-major buffer addressing.
- Tile coordinate decoding, bounds checks, and grouped raster mapping.
- Elementwise floating-point operations; consecutive scalar accesses for multi-value loads and stores.
- Explicit loop and conditional lowering, accumulator initialization, and rescaled accumulator bases.
- Warp shuffles and shared-memory tree reduction using the IR's combine statements.
- Static or dynamic shared memory, asynchronous copies with lane offsets, commit/wait operations, and barriers.
- Atomic float32 addition where requested by the IR; output zeroing remains the launcher's responsibility.

See [architecture](docs/ARCHITECTURE.md) for details.

## Scope

The current implementation targets static shapes and float32 buffers, emitting PTX 8.0 for `sm_89`. It is focused on correctness, with no tensor-core or systematic performance optimization. Unsupported buffer types, shared-memory swizzles, auxiliary threads, and TreeHalve with an `inner` body are rejected. Supported arithmetic operations are listed in the architecture document. Public-suite success does not guarantee hidden-set coverage.

## Files

| File | Purpose |
| --- | --- |
| `compile.py` | Original IR-to-PTX implementation |
| `requirements.txt` | Python dependencies |
| `verify.py` | Public-suite runner and JSON report |
| `tests/variants.py` | Supplemental tests with independent NumPy expectations |
| `verification.json`, `variants.log` | Recorded successful test results |
| `submit.py`, `problems/`, `examples/`, `tools/` | Organizer-provided test tooling and fixtures |
| `TASK.md`, `MVP.md`, `docs/API.md` | Organizer-provided specifications |

## Provenance and AI assistance

The official task repository is [cloudrift-ai/emmy-hackathon-2026](https://github.com/cloudrift-ai/emmy-hackathon-2026), based on upstream commit `ec8cde1943f88efea3827a95bda6085553d2d407`. The organizer's original README is preserved in `docs/UPSTREAM_README.md`; its relative links refer to the original repository root.

AI coding assistance was used to develop and debug the compiler and prepare documentation. The participant's video explains the implementation in their own voice. The IR-to-PTX lowering is implemented in this source tree; test fixtures and the runner client come from the organizer.
