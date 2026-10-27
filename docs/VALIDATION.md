# Validation

These results were recorded on the official runner. The report timestamp is the latest report update; `--retry-failed` may retain earlier successful entries. Kernel timings in logs are indicative single-run measurements, not benchmark claims.

- Saved report: `verification.json`.
- Supplemental evidence: `variants.log`.
- Recorded report update (UTC): `2026-10-25T22:53:15.371478+00:00`.
- Compiler SHA-256 at packaging: `9f9ccdbbb341dc62cd49302fd626e1571bc93b5ed77a3caa781662b8012d2dcc`.

| Public problem | Recorded result |
| --- | --- |
| a01_attention | PASS |
| a02_attention_128 | PASS |
| f01_rmsnorm | PASS |
| f02_row_softmax | PASS |
| f03_l2norm | PASS |
| m01_matmul_512 | PASS |
| m02_matmul_rect | PASS |
| p00_scalar_add | PASS |
| p01_vector_add | PASS |
| p02_scale_shift | PASS |
| p03_relu_mul | PASS |
| r01_array_sum | PASS |
| r02_big_sum | PASS |
| r03_mean | PASS |
| r04_row_sum | PASS |

## Supplemental cases

1. 37 Tile cells / 74 float32 elements, subtraction, 64 launched threads, and output-first parameter order. Maximum absolute error: 0.
2. A 7 x 1003 row sum, non-divisible strided tails, and output-first parameter order. Maximum absolute error: approximately 9.537e-06.

Expected results are computed independently with NumPy. These tests exercise two extra cases; they do not establish exhaustive hidden-set coverage.

## Reproduce

```sh
python verify.py
python tests/variants.py
```

The hosted runner must be reachable. A network or service failure is distinct from a compiler correctness failure. Local compilation alone checks that PTX is emitted; it does not establish that ptxas accepts it or that GPU results are correct.
