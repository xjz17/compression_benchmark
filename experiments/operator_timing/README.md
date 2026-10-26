# Operator timing runner

The timing runner measures encode/decode paths on caller-provided typed column
inputs and verifies byte-exact reconstruction. It does not contain historical
results or remote-machine launch state.

Use `python run_timing_shard.py --help`, `python finalize_timing.py --help`, and
`python smoke_test.py --help` for the current command-line contracts. Before a
larger campaign, run the smoke test with the production driver and catalog:

```bash
python smoke_test.py --driver /path/to/driver --catalog /path/to/catalog.json
```
