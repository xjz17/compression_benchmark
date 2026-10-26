# Algorithm-by-operator composition runner

This harness evaluates compositions that are actually executed together. It
keeps algorithm-only, operator-only, and observed pipeline measurements
separate; independent stage optima are not treated as an observed joint optimum.

The CMake target builds the bundle CLI, while `source/shard_runner.py` manages
sharded execution. The route builder under `algorithm_routes/` resolves the
supported algorithm backends.

Run the Python tests from this directory before launching a campaign:

```bash
python -m unittest discover -s tests -v
python algorithm_routes/test_routes.py
```
