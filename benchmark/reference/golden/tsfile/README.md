# TSFile goldens (P0.2a)

Regenerate: `./scripts/export_tsfile_golden_java.sh` from repo root (Java uses Maven `org.apache.tsfile:tsfile:2.0.2`).

**Commit** `manifest.tsv` and all `*_default.bin` files under this directory so CI can run (workflow `.github/workflows/cpp-tsfile-goldens.yml`).

Verify locally: `cmake -S cpp -B build -DENABLE_TSFILE_GOLDEN_TESTS=ON && cmake --build build --target verify_tsfile_goldens tsfile_p02_golden_test` then run `verify_tsfile_goldens` and `tsfile_p02_golden_test` (paths are compiled in via `TSFILE_GOLDEN_DIR`, or pass this directory as argv).

Column fixture: 64 values — INT32 `10000 + i*3`, FLOAT `1.0f + i*0.03125f` (see `cpp/tests/tsfile_p02_golden_test.cpp`).

| Filename |
|----------|
| `TS_2DIFF_INT32_default.bin`, `TS_2DIFF_FLOAT_default.bin` |
| `RLE_INT32_default.bin`, `RLE_FLOAT_default.bin` |
| `GORILLA_INT32_default.bin`, `GORILLA_FLOAT_default.bin` |
| `CHIMP_INT32_default.bin`, `CHIMP_FLOAT_default.bin` |
| `SPRINTZ_INT32_default.bin`, `SPRINTZ_FLOAT_default.bin` |
| `RLBE_INT32_default.bin`, `RLBE_FLOAT_default.bin` |

`manifest.tsv` lists `sha256` and byte lengths. **RLBE FLOAT** bytes match **`FloatRLBE` in tsfile 2.0.2** (IEEE `fsub` deltas into the bit stream); vendored `FloatRLBE.java` under `external/...` may still show an older `extends IntRLBE` pattern — treat **Maven + these goldens** as the contract.
