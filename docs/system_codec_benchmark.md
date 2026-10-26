# TsFile and InfluxDB system-codec benchmark

The runner `scripts/run_system_codec_smallest5.sh` benchmarks the five smallest source CSV files, measured by bytes, from the canonical 19-dataset paper pool. It fails if any of the 19 inputs is missing.

Input conversion keeps only columns whose every row is numeric and non-NaN. Incomplete columns are excluded because stock InfluxDB TSM Gorilla reserves NaN as its stream terminator. The shared `SCDBIN01` file is column-major, so TsFile and InfluxDB receive identical row and value bits. Conversion fails if a selected dataset has no complete numeric column.

Each codec receives three warm-up runs and ten measured runs. Codec order rotates on every repetition to reduce fixed-order drift. Reported write time includes system encoding, physical file write, index finalization, close, and durability sync. Read time includes opening and fully decoding the physical file. Every value is compared by raw IEEE bits. InfluxDB block CRCs are also verified. A run is invalid unless every row reports `hash_ok=true`.

TsFile baselines are BOS, Sub-column, Plain, Gorilla v1/v2, Chimp, RLBE, and Camel, all with the outer compressor disabled. Sprintz is excluded because the v2.4.0 double path failed the bit-exact smoke check.

InfluxDB uses the native v2.9.1 TSM file layer, not Parquet. The runner writes real `.tsm` files through `TSMWriter`, including block CRCs and the native index, and reads them through the mmap-backed `TSMReader`, explicitly verifying each CRC before decode. Its float-value codecs are BOS, Sub-column, and the stock Gorilla implementation. `INTEGER_DELTA_ADAPTIVE_BITCAST` routes the exact IEEE 754 bits through InfluxDB's native integer `delta + ZigZag` codec, which adaptively chooses Simple8b, RLE, or uncompressed storage. It is labeled as an adapter rather than a stock float codec. TSM has no dictionary value encoding: strings use Snappy, booleans use bit packing, integers and timestamps use adaptive delta/Simple8b/RLE, and floats use Gorilla by default. The native adaptive timestamp codec is held constant for every InfluxDB float-value comparison.

Timings are intended for comparisons among codecs inside the same system. TsFile and InfluxDB have different file and query contracts, so their absolute times should not be presented as a direct database-versus-database ranking. InfluxDB now exposes `storage-tsm-float-encoding=gorilla|bos|subcolumn`; the setting controls both normal cache snapshots and float blocks rewritten by compaction, while readers continue to dispatch from each block header. The benchmark still measures the TSM file layer directly so codec comparisons do not include WAL, indexing, or query-planner work. Reads immediately follow writes and therefore represent warm-page-cache full scans, not cold-cache latency.

The three source repositories must be clean before the runner starts. The output directory retains the complete raw CSV, validated summary, source and converted-input SHA-256 values, exact command, environment and tool versions, pinned commits, and dirty-state flags. The summarizer requires the exact five datasets in the input manifest, all expected codecs, and unique iterations 0 through 9 for every dataset/codec group.

After the five-dataset run succeeds, generate the requested four-panel PNG and PDF with:

```bash
python3 scripts/plot_system_codec_four_panel.py \
  /path/to/system_codec_summary.csv \
  /path/to/system_codec_read_write_four_panel
```

The four panels are TsFile write, TsFile read, InfluxDB TSM write, and InfluxDB TSM read. The plotter requires five complete, bit-validated datasets by default and refuses an incomplete matrix.
