#!/usr/bin/env python3
"""
Prepare curated master CSVs from raw datasets in ``backend/dataset_temp/``.

Output root: ``E:/xjz/compression_benchmark/backend_dataset_csv`` when that directory
exists, else ``backend/dataset_csv/`` (see ``scripts/dataset_csv_source.py``).

Design notes:
- Pass ``--dataset <name>`` to run one known dataset handler.
- If ``--dataset`` is omitted, this script exits without processing.
- Only explicitly supported dataset names are handled.

Current supported datasets:
- ``City-temp``, ``Food-prices``, ``SSD-bench``, ``World-City``, ``poi``, ``vehicle-charge``
  (under ``公开数据集/``), ``Bar-Crawl``:
  copy source CSV row-by-row with LF on every line (overwrite).
- ``Air-pressure-1min`` / ``Air-pressure-30min``:
  merge ``公开数据集/Air-pressure/NEON_pressure-air/<folder>/*{1min|30min}*.csv`` in folder order
  -> ``backend/dataset_csv/Air-pressure-{1min|30min}.csv`` (one header, LF on every line).
- ``Dewpoint-temp-1min`` / ``Dewpoint-temp-30min``:
  merge ``公开数据集/Dewpoint-temp/NEON_rel-humidity-buoy/<folder>/*{1min|30min}*.csv`` in folder order
  -> ``backend/dataset_csv/Dewpoint-temp-{1min|30min}.csv`` (one header, LF on every line).
- ``PM10-dust-30minutes`` / ``PM10-dust-60minutes``:
  merge ``公开数据集/PM10-dust/NEON_size-dust-particulate/<folder>/*{30_minutes|60_minutes}*.csv`` in folder order
  -> ``backend/dataset_csv/PM10-dust-{30minutes|60minutes}.csv`` (one header, LF on every line).
- ``IR-bio-temp-1minute`` / ``IR-bio-temp-30minute``:
  merge ``公开数据集/IR-bio-temp/NEON_temp-bio/<folder>/*{1_minute|30_minute}*.csv`` in folder/file order
  -> ``backend/dataset_csv/IR-bio-temp-{1minute|30minute}.csv`` (one header, LF on every line).
- ``UCR``:
  ``backend/dataset_temp/UCR_Anomaly_FullData/*_UCR_*.txt`` (all ``NNN_UCR_*.txt`` shards)
  -> ``backend/dataset_csv/UCR.csv`` (wide CSV; column headers are txt basenames without ``.txt``).
  Each shard is one column: one decimal per line, or whitespace-separated values on a single line.
- ``Bar-Crawl``:
  ``backend/dataset_temp/Leco/all_accelerometer_data_pids_13.csv``
  -> ``backend/dataset_csv/Bar-Crawl.csv``.
- ``Bird-migration``:
  ``backend/dataset_temp/migration_original.csv`` (Movebank animal tracking; all fields quoted in source)
  -> ``backend/dataset_csv/Bird-migration.csv`` (comma-separated, unquoted fields, LF).
- ``books``:
  ``backend/dataset_temp/Leco/books_200M_uint32`` (8-byte LE header + uint32 payload)
  -> ``backend/dataset_csv/books.csv`` (one decimal value per line, LF; overwrite).
- ``fb``:
  ``backend/dataset_temp/Leco/fb_200M_uint64`` (8-byte LE header + uint64 payload)
  -> ``backend/dataset_csv/fb.csv`` (one decimal value per line; uint64 bit pattern
  reinterpreted as signed int64 when above ``2**63-1``, LF; overwrite).
- ``wiki_ts``:
  ``backend/dataset_temp/Leco/wiki_ts_200M_uint64`` (8-byte LE header + uint64 payload)
  -> ``backend/dataset_csv/wiki_ts.csv`` (one decimal value per line, LF; overwrite).
- ``linear``:
  ``backend/dataset_temp/Leco/linear_200M_uint32.txt``
  -> ``backend/dataset_csv/linear.csv`` (one decimal value per line, LF; overwrite).
- ``normal``:
  ``backend/dataset_temp/Leco/normal_200M_uint32.txt``
  -> ``backend/dataset_csv/normal.csv`` (one decimal value per line, LF; overwrite).
- ``realtor``:
  ``backend/dataset_temp/Leco/realtor-data.zip.csv``
  -> ``backend/dataset_csv/realtor.csv`` (copy row-by-row with LF; overwrite).
- ``rating``:
  ``backend/dataset_temp/Leco/rating.csv`` (quoted header/fields in source)
  -> ``backend/dataset_csv/rating.csv`` (comma-separated, unquoted fields, LF).
- ``Comments``:
  ``backend/dataset_temp/Leco/Comments.csv``
  -> ``backend/dataset_csv/Comments.csv`` (copy row-by-row with LF; overwrite).
- **Leco/data** (``backend/dataset_temp/Leco/data/<file>`` -> ``backend/dataset_csv/<stem>.csv``;
  comma-separated, LF). Keys match filename stems (``1year`` from ``1year.arff``,
  ``adult.data`` from ``adult.data.csv``, etc.): ``1year``, ``DSL-StrongPasswordData``,
  ``Rail_Insurance_Claims``, ``adult`` (merge ``adult.data.csv`` + ``adult_test.csv`` +
  deduped ``adult_train.csv`` rows -> ``adult.csv``; sources must all exist under Leco/data),
  ``agaricus-lepiota.data``, ``bank`` (``bank_train.csv`` + ``bank_train_target.csv`` -> ``bank.csv``),
  ``credit_scoring_sample``, ``credit_scoring_test``, ``credit_scoring_train``,
  ``ehresp_2014``, ``flight_delays_test``, ``flight_delays_train``, ``hmeq``,
  ``medium_posts``, ``mlbootcamp5_train``, ``prolongation_service_train``, ``scores``,
  ``spam``, ``spooky_authors``, ``spooky_writer_sample_submission``, ``spooky_writer_test``,
  ``spooky_writer_train``, ``vgsales``, ``video_games_sales``, ``weights_heights``,
  ``winequality-white``.
- ``vehicle-charge``:
  ``backend/dataset_temp/公开数据集/vehicle-charge/station_data_dataverse.csv``
  -> ``backend/dataset_csv/vehicle-charge.csv``.
- ``NYC_Yellow_Taxi_Trip``:
  merge ``公开数据集/NYC_Yellow_Taxi_Trip/*.csv`` in filename order
  -> ``backend/dataset_csv/NYC_Yellow_Taxi_Trip.csv`` (one header, LF on every line;
  headers compared case-insensitively when shards differ only by casing).
- ``solar_wind``:
  ``backend/dataset_temp/公开数据集/msg/solar_wind.csv``
  -> ``backend/dataset_csv/solar_wind.csv`` (copy row-by-row with LF; overwrite).
- ``Climate_Weather``:
  merge ``公开数据集/Climate_Weather/{central_west,north,northeast,south,southeast}.csv``
  in that order -> ``backend/dataset_csv/Climate_Weather.csv`` (one header, LF on every line).
- ``meteoblue``:
  merge ``backend/dataset_temp/meteoblue/*.csv`` in filename order (skip 9 metadata rows
  per file; row 10 = header) -> ``backend/dataset_csv/meteoblue.csv`` (comma-separated, LF).
- ``Stocks-Germany``:
  ``backend/dataset_temp/公开数据集/Stocks-Germany-sample.txt``
  -> ``backend/dataset_csv/Stocks-Germany.csv`` (copy row-by-row with LF; no header; overwrite).
- ``HOUSE``:
  ``backend/dataset_temp/公开数据集/HOUSE/household_power_consumption.txt`` (semicolon-separated;
  pandas ``dtype=str``; lone ``?`` cells -> empty; comma ``HOUSE.csv``, LF).
- **Public BI Benchmark** (``PublicBIbenchmark/<name>/``; explicit ``*.csv.bz2`` shards per dataset;
  pipe-delimited, no header; pandas ``dtype=str`` -> comma CSV with ``column_0``… header).
  Shards and options are defined in ``_PUBLICBI_DATASET_SOURCES`` (includes ``Arade``,
  ``Bimbo``, ``CMSprovider``, ``CityMaxCapita``, ``CommonGovernment``, ``Corporations``,
  ``Eixo``, ``Euro2016``, ``Food``, ``Generico``, ``HashTags``, ``Hatred``, ``IGlocations1``,
  ``IGlocations2``, ``Medicare1``–``Medicare3``, ``MedPayment1``/``MedPayment2``, ``Motos``,
  ``MulheresMil``, ``NYC``, ``PanCreactomy1``/``PanCreactomy2``, ``Physicians``, ``Provider``,
  ``RealEstate1``/``RealEstate2``, ``Redfin1``–``Redfin4``, ``Rentabilidad``, ``Romance``,
  ``SalariesFrance``, ``TableroSistemaPenal`` (``_1`` only), ``Taxpayer``, ``Telco``,
  ``TrainsUK1`` (``_2``–``_4``), ``TrainsUK2``, ``MLB`` (``MLB_1`` only),
  ``Uberlandia``, ``USCensus``, ``Wins`` (``Wins_1`` only), ``YaleLanguages``).
- **HPC time-series observational CSV** (``backend/dataset_temp/hpc_ts_obs_csv/<stem>.csv``;
  comma-separated, usually no header; scientific notation -> plain decimal with at most 19
  significant digits; LF):
  ``acs_wht_f32`` (7500 columns), ``astro_mhd_f64``, ``astro_pt_f64``, ``h3d_temp_f32``,
  ``hdr_night_f32`` (16384 columns), ``hst_wfc3_ir_f32`` (2417 columns),
  ``hst_wfc3_uvis_f32`` (5110 columns), ``tpcds_web_f32`` (15 columns), ``tpch_order_f64``,
  ``ts_gas_f32``, ``turbulence_f32``, ``wave_f32``, ``wesad_chest_f64`` (8 columns),
  ``g24_78_usb2_f32``, ``hdr_palermo_f32``, ``jane_street_f64``, ``jw_mirimage_f32``,
  ``miranda3d_f32``, ``num_brain_f64``, ``num_control_f64``, ``phone_gyro_f64``,
  ``rsim_f32``, ``spain_gas_price_f64``, ``spitzer_irac_f32``, ``tpcds_catalog_f32``,
  ``tpcds_store_f32``, ``tpch_lineitem_f32``, ``tpcxbb_store_f64``, ``tpcxbb_web_f64``.
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from dataset_csv_source import dataset_csv_root  # noqa: E402

DATASET_TEMP = BACKEND / "dataset_temp"
_PUBLIC_DATASETS_ROOT = DATASET_TEMP / "公开数据集"
METEOBLUE_ROOT = DATASET_TEMP / "meteoblue"
METEOBLUE_METADATA_ROWS = 9
_HPC_TS_OBS_ROOT = DATASET_TEMP / "hpc_ts_obs_csv"
_HPC_MAX_SIG_DIGITS = 19


def _dataset_csv_out(filename: str) -> Path:
    """Preferred master CSV write path (external root when available)."""
    return dataset_csv_root(BACKEND) / filename


def _copy_csv_lines_with_lf(src: Path, dst: Path, *, encoding: str = "utf-8") -> Path:
    if not src.is_file():
        raise FileNotFoundError(f"Missing source CSV: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding=encoding, errors="replace", newline="") as handle, dst.open(
        "w", encoding="utf-8", newline="\n"
    ) as out:
        for line in handle:
            out.write(line.rstrip("\r\n") + "\n")
    return dst


def _public_bi_bz2_column_widths(sources: list[Path]) -> dict[Path, int]:
    """Preview each shard (first row) to learn pipe-delimited column counts."""
    import pandas as pd

    widths: dict[Path, int] = {}
    for src in sources:
        sample = pd.read_csv(
            src,
            sep="|",
            header=None,
            dtype=str,
            nrows=1,
            keep_default_na=False,
            compression="infer",
            quoting=csv.QUOTE_NONE,
        )
        widths[src] = len(sample.columns)
    return widths


def _export_pipe_bz2_sources_to_comma_csv(
    sources: list[Path],
    dst: Path,
    *,
    chunksize: int = 500_000,
    pad_to_max_columns: bool = False,
) -> Path:
    """Read pipe-delimited bz2 rows as strings; merge files; write comma-separated CSV.

    Public BI Tableau exports embed ``"`` inside text fields; disable CSV quoting so
    ``|`` remains the only delimiter (see FastLanes ``issue_1_list`` datasets).
    """
    import pandas as pd

    if not sources:
        raise FileNotFoundError("No bz2 sources provided")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shard_widths = _public_bi_bz2_column_widths(sources)
    unique_widths = set(shard_widths.values())
    target_cols: int | None = None
    if len(unique_widths) == 1:
        target_cols = next(iter(unique_widths))
    elif pad_to_max_columns:
        target_cols = max(unique_widths)
        print(
            "Note: Public BI shards have different column counts "
            f"{sorted(unique_widths)}; padding rows to {target_cols} columns.",
            file=sys.stderr,
        )
    wrote_any = False
    for src in sources:
        if not src.is_file():
            raise FileNotFoundError(f"Missing source file: {src}")
        first_chunk_in_file = True
        for chunk in pd.read_csv(
            src,
            sep="|",
            header=None,
            dtype=str,
            keep_default_na=False,
            chunksize=chunksize,
            compression="infer",
            quoting=csv.QUOTE_NONE,
        ):
            ncols = len(chunk.columns)
            if target_cols is None:
                raise ValueError(
                    "Column count mismatch across Public BI shards: "
                    f"{ {p.name: w for p, w in shard_widths.items()} }. "
                    "Pass pad_to_max_columns=True for datasets such as Wins."
                )
            if ncols != target_cols:
                if ncols > target_cols:
                    raise ValueError(
                        f"Column count mismatch in {src}: expected {target_cols}, got {ncols}"
                    )
                chunk.columns = list(range(ncols))
                chunk = chunk.reindex(columns=list(range(target_cols)), fill_value="")
            chunk.columns = [f"column_{i}" for i in range(target_cols)]
            write_header = not wrote_any and first_chunk_in_file
            chunk.to_csv(
                dst,
                mode="w" if not wrote_any else "a",
                index=False,
                header=write_header,
                lineterminator="\n",
            )
            wrote_any = True
            first_chunk_in_file = False
    if not wrote_any:
        raise ValueError(f"No rows found in sources: {sources}")
    return dst


def _export_semicolon_txt_to_comma_csv(
    src: Path,
    dst: Path,
    *,
    chunksize: int = 500_000,
    empty_cell_literals: tuple[str, ...] = (),
) -> Path:
    """Read semicolon-delimited text as strings; write comma-separated CSV with LF."""
    import pandas as pd

    if not src.is_file():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    wrote_any = False
    for chunk in pd.read_csv(
        src,
        sep=";",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    ):
        for literal in empty_cell_literals:
            chunk = chunk.where(chunk != literal, "")
        chunk.to_csv(
            dst,
            mode="w" if not wrote_any else "a",
            index=False,
            header=not wrote_any,
            lineterminator="\n",
        )
        wrote_any = True
    if not wrote_any:
        raise ValueError(f"No rows found in {src}")
    return dst


def _export_quoted_csv_to_comma_csv(
    src: Path, dst: Path, *, encoding: str = "utf-8"
) -> Path:
    """Read standard quoted CSV; write comma-separated output without quoting fields."""
    if not src.is_file():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding=encoding, errors="replace", newline="") as handle, dst.open(
        "w", encoding="utf-8", newline="\n"
    ) as out:
        reader = csv.reader(handle)
        writer = csv.writer(out, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        wrote_any = False
        for row in reader:
            writer.writerow(row)
            wrote_any = True
    if not wrote_any:
        raise ValueError(f"No rows found in {src}")
    return dst


_PUBLICBI_ROOT = DATASET_TEMP / "PublicBIbenchmark"


def _resolve_public_bi_bz2_sources(
    dataset_name: str, filenames: list[str]
) -> list[Path]:
    folder = _PUBLICBI_ROOT / dataset_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing Public BI folder: {folder}")
    sources: list[Path] = []
    for name in filenames:
        path = folder / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing Public BI source file: {path}")
        sources.append(path)
    return sources


def _handle_public_bi_benchmark(
    dataset_name: str,
    filenames: list[str],
    *,
    pad_to_max_columns: bool = False,
) -> Path:
    sources = _resolve_public_bi_bz2_sources(dataset_name, filenames)
    return _export_pipe_bz2_sources_to_comma_csv(
        sources,
        _dataset_csv_out(f"{dataset_name}.csv"),
        pad_to_max_columns=pad_to_max_columns,
    )


# dataset_name -> (explicit shard filenames, pad_to_max_columns)
_PUBLICBI_DATASET_SOURCES: dict[str, tuple[list[str], bool]] = {
    "Arade": (["Arade_1.csv.bz2"], False),
    "Bimbo": (["Bimbo_1.csv.bz2"], False),
    "CMSprovider": (["CMSprovider_1.csv.bz2", "CMSprovider_2.csv.bz2"], False),
    "CityMaxCapita": (["CityMaxCapita_1.csv.bz2"], False),
    "CommonGovernment": (["CommonGovernment_1.csv.bz2"], False),
    "Corporations": (["Corporations_1.csv.bz2"], False),
    "Eixo": (["Eixo_1.csv.bz2"], False),
    "Euro2016": (["Euro2016_1.csv.bz2"], False),
    "Food": (["Food_1.csv.bz2"], False),
    "Generico": (["Generico_1.csv.bz2"], False),
    "HashTags": (["HashTags_1.csv.bz2"], False),
    "Hatred": (["Hatred_1.csv.bz2"], False),
    "IGlocations1": (["IGlocations1_1.csv.bz2"], False),
    "IGlocations2": (["IGlocations2_1.csv.bz2", "IGlocations2_2.csv.bz2"], False),
    "Medicare1": (["Medicare1_1.csv.bz2", "Medicare1_2.csv.bz2"], False),
    "Medicare2": (["Medicare2_1.csv.bz2", "Medicare2_2.csv.bz2"], False),
    "Medicare3": (["Medicare3_1.csv.bz2"], False),
    "MedPayment1": (["MedPayment1_1.csv.bz2"], False),
    "MedPayment2": (["MedPayment2_1.csv.bz2"], False),
    "Motos": (["Motos_1.csv.bz2", "Motos_2.csv.bz2"], False),
    "MulheresMil": (["MulheresMil_1.csv.bz2"], False),
    "NYC": (["NYC_1.csv.bz2", "NYC_2.csv.bz2"], False),
    "PanCreactomy1": (["PanCreactomy1_1.csv.bz2"], False),
    "PanCreactomy2": (["PanCreactomy2_1.csv.bz2", "PanCreactomy2_2.csv.bz2"], False),
    "Physicians": (["Physicians_1.csv.bz2"], False),
    "Provider": (["Provider_1.csv.bz2"], False),
    "RealEstate1": (["RealEstate1_1.csv.bz2"], False),
    "RealEstate2": (["RealEstate2_1.csv.bz2"], False),
    "Redfin1": (["Redfin1_1.csv.bz2"], False),
    "Redfin2": (["Redfin2_1.csv.bz2"], False),
    "Redfin3": (["Redfin3_1.csv.bz2"], False),
    "Redfin4": (["Redfin4_1.csv.bz2"], False),
    "Rentabilidad": (["Rentabilidad_1.csv.bz2"], False),
    "Romance": (["Romance_1.csv.bz2", "Romance_2.csv.bz2"], False),
    "SalariesFrance": (["SalariesFrance_1.csv.bz2"], False),
    "TableroSistemaPenal": (["TableroSistemaPenal_1.csv.bz2"], False),
    "Taxpayer": (["Taxpayer_1.csv.bz2"], False),
    "Telco": (["Telco_1.csv.bz2"], False),
    "TrainsUK1": (
        ["TrainsUK1_2.csv.bz2", "TrainsUK1_3.csv.bz2", "TrainsUK1_4.csv.bz2"],
        False,
    ),
    "TrainsUK2": (["TrainsUK2_1.csv.bz2", "TrainsUK2_2.csv.bz2"], False),
    "MLB": (["MLB_1.csv.bz2"], False),
    "Uberlandia": (["Uberlandia_1.csv.bz2"], False),
    "USCensus": (["USCensus_1.csv.bz2"], False),
    "Wins": (["Wins_1.csv.bz2"], False),
    "YaleLanguages": (["YaleLanguages_1.csv.bz2", "YaleLanguages_2.csv.bz2", "YaleLanguages_3.csv.bz2", "YaleLanguages_4.csv.bz2", "YaleLanguages_5.csv.bz2"], False),
}


def _public_bi_dataset_handler(dataset_name: str):
    def _handler() -> Path:
        filenames, pad = _PUBLICBI_DATASET_SOURCES[dataset_name]
        return _handle_public_bi_benchmark(
            dataset_name, filenames, pad_to_max_columns=pad
        )

    return _handler


_NEON_PRESSURE_AIR_ROOT = _PUBLIC_DATASETS_ROOT / "Air-pressure" / "NEON_pressure-air"
_NEON_DEWPOINT_TEMP_ROOT = _PUBLIC_DATASETS_ROOT / "Dewpoint-temp" / "NEON_rel-humidity-buoy"
_NEON_PM10_DUST_ROOT = _PUBLIC_DATASETS_ROOT / "PM10-dust" / "NEON_size-dust-particulate"
_NEON_IR_BIO_TEMP_ROOT = _PUBLIC_DATASETS_ROOT / "IR-bio-temp" / "NEON_temp-bio"


def _list_neon_interval_csvs(
    root: Path, interval_token: str, *, one_per_folder: bool = True
) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing directory: {root}")
    sources: list[Path] = []
    for folder in sorted(root.iterdir(), key=lambda p: p.name):
        if not folder.is_dir():
            continue
        matches = sorted(
            path
            for path in folder.iterdir()
            if path.is_file()
            and interval_token in path.name
            and path.suffix.lower() == ".csv"
        )
        if one_per_folder:
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one *{interval_token}*.csv in {folder}, found {len(matches)}"
                )
            sources.append(matches[0])
        elif not matches:
            raise FileNotFoundError(
                f"Expected at least one *{interval_token}*.csv in {folder}, found 0"
            )
        else:
            sources.extend(matches)
    if not sources:
        raise FileNotFoundError(
            f"No {interval_token} CSV files under {root}"
        )
    return sources


def _merge_csv_files_lines_with_lf(
    sources: list[Path],
    dst: Path,
    *,
    header_compare: str = "exact",
) -> Path:
    """Merge CSV shards; write comma-separated rows with LF line endings."""
    if header_compare not in ("exact", "casefold"):
        raise ValueError(f"Unsupported header_compare: {header_compare!r}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    header: str | None = None

    def _headers_match(expected: str, actual: str) -> bool:
        if header_compare == "casefold":
            return expected.casefold() == actual.casefold()
        return expected == actual

    with dst.open("w", encoding="utf-8", newline="\n") as out:
        for src in sources:
            with src.open("r", encoding="utf-8", newline="") as handle:
                for line_no, line in enumerate(handle):
                    text = line.rstrip("\r\n")
                    if line_no == 0:
                        if header is None:
                            header = text
                            out.write(text + "\n")
                        elif not _headers_match(header, text):
                            raise ValueError(
                                f"Header mismatch in {src}: expected {header!r}, got {text!r}"
                            )
                        continue
                    out.write(text + "\n")
    return dst


def _list_csv_files_in_dir(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing directory: {root}")
    sources = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    )
    if not sources:
        raise FileNotFoundError(f"No CSV files under {root}")
    return sources


def _resolve_named_csv_sources(root: Path, names: tuple[str, ...]) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing directory: {root}")
    sources: list[Path] = []
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing source CSV: {path}")
        sources.append(path)
    return sources


_CLIMATE_WEATHER_CSVS = (
    "central_west.csv",
    "north.csv",
    "northeast.csv",
    "south.csv",
    "southeast.csv",
)


def _handle_neon_interval(
    root: Path, interval_token: str, output_stem: str, *, one_per_folder: bool = True
) -> Path:
    sources = _list_neon_interval_csvs(
        root, interval_token, one_per_folder=one_per_folder
    )
    return _merge_csv_files_lines_with_lf(
        sources,
        _dataset_csv_out(f"{output_stem}.csv"),
    )


def _handle_air_pressure_1min() -> Path:
    return _handle_neon_interval(_NEON_PRESSURE_AIR_ROOT, "1min", "Air-pressure-1min")


def _handle_air_pressure_30min() -> Path:
    return _handle_neon_interval(_NEON_PRESSURE_AIR_ROOT, "30min", "Air-pressure-30min")


def _handle_dew_point_temp_1min() -> Path:
    return _handle_neon_interval(
        _NEON_DEWPOINT_TEMP_ROOT, "1min", "Dewpoint-temp-1min"
    )


def _handle_dew_point_temp_30min() -> Path:
    return _handle_neon_interval(
        _NEON_DEWPOINT_TEMP_ROOT, "30min", "Dewpoint-temp-30min"
    )


def _handle_pm10_dust_30minutes() -> Path:
    return _handle_neon_interval(
        _NEON_PM10_DUST_ROOT, "30_minutes", "PM10-dust-30minutes"
    )


def _handle_pm10_dust_60minutes() -> Path:
    return _handle_neon_interval(
        _NEON_PM10_DUST_ROOT, "60_minutes", "PM10-dust-60minutes"
    )


def _handle_ir_bio_temp_1minute() -> Path:
    return _handle_neon_interval(
        _NEON_IR_BIO_TEMP_ROOT,
        "1_minute",
        "IR-bio-temp-1minute",
        one_per_folder=False,
    )


def _handle_ir_bio_temp_30minute() -> Path:
    return _handle_neon_interval(
        _NEON_IR_BIO_TEMP_ROOT,
        "30_minute",
        "IR-bio-temp-30minute",
        one_per_folder=False,
    )


def _handle_city_temp() -> Path:
    return _copy_csv_lines_with_lf(
        _PUBLIC_DATASETS_ROOT / "City-temp" / "city_temperature.csv",
        _dataset_csv_out("City-temp.csv"),
    )


def _handle_house() -> Path:
    return _export_semicolon_txt_to_comma_csv(
        _PUBLIC_DATASETS_ROOT / "HOUSE" / "household_power_consumption.txt",
        _dataset_csv_out("HOUSE.csv"),
        empty_cell_literals=("?",),
    )


def _handle_vehicle_charge() -> Path:
    return _copy_csv_lines_with_lf(
        _PUBLIC_DATASETS_ROOT / "vehicle-charge" / "station_data_dataverse.csv",
        _dataset_csv_out("vehicle-charge.csv"),
    )


def _handle_nyc_yellow_taxi_trip() -> Path:
    root = _PUBLIC_DATASETS_ROOT / "NYC_Yellow_Taxi_Trip"
    sources = _list_csv_files_in_dir(root)
    return _merge_csv_files_lines_with_lf(
        sources,
        _dataset_csv_out("NYC_Yellow_Taxi_Trip.csv"),
        header_compare="casefold",
    )


def _handle_climate_weather() -> Path:
    root = _PUBLIC_DATASETS_ROOT / "Climate_Weather"
    sources = _resolve_named_csv_sources(root, _CLIMATE_WEATHER_CSVS)
    return _merge_csv_files_lines_with_lf(
        sources,
        _dataset_csv_out("Climate_Weather.csv"),
    )


def _handle_food_prices() -> Path:
    food_dir = _PUBLIC_DATASETS_ROOT / "Food-prices"
    # Upstream file naming appears with both spellings in different snapshots.
    candidates = (
        food_dir / "wfpvan_foodprices.csv",
        food_dir / "wfpvam_foodprices.csv",
    )
    src = next((p for p in candidates if p.is_file()), candidates[0])
    dst = _dataset_csv_out("Food-prices.csv")
    if not src.is_file():
        raise FileNotFoundError(
            f"Missing source CSV. Tried: {', '.join(str(p) for p in candidates)}"
        )
    return _copy_csv_lines_with_lf(src, dst)


def _handle_ssd_bench() -> Path:
    ssd_dir = _PUBLIC_DATASETS_ROOT / "SD-bench"
    candidates = (
        ssd_dir / "SSD_HDD_benchmarks_v9.csv",
        ssd_dir / "SSD_HDD_benchmarks.csv",
    )
    src = next((p for p in candidates if p.is_file()), candidates[0])
    dst = _dataset_csv_out("SSD-bench.csv")
    if not src.is_file():
        raise FileNotFoundError(
            f"Missing source CSV. Tried: {', '.join(str(p) for p in candidates)}"
        )
    return _copy_csv_lines_with_lf(src, dst)


def _handle_stocks_germany() -> Path:
    return _copy_csv_lines_with_lf(
        _PUBLIC_DATASETS_ROOT / "Stocks-Germany-sample.txt",
        _dataset_csv_out("Stocks-Germany.csv"),
    )


_UCR_TXT_PATTERN = re.compile(r"^(\d{3})_UCR_.*\.txt$", re.IGNORECASE)


def _decimal_plain_string(line: str) -> str:
    text = line.strip()
    if not text:
        return ""
    try:
        return format(Decimal(text), "f")
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal literal: {text!r}") from exc


def _hpc_decimal_to_plain_string(value: Decimal) -> str:
    """Render a finite Decimal as plain text (no exponent) from its coefficient tuple."""
    if value.is_zero():
        return "0"
    sign = "-" if value < 0 else ""
    magnitude = value.copy_abs()
    coeff = magnitude.as_tuple()
    digits = "".join(str(d) for d in coeff.digits)
    exp = coeff.exponent
    if exp >= 0:
        plain = digits + ("0" * exp)
    else:
        pos = len(digits) + exp
        if pos <= 0:
            plain = "0." + ("0" * (-pos)) + digits
        else:
            plain = digits[:pos] + "." + digits[pos:]
    if "." in plain:
        plain = plain.rstrip("0").rstrip(".") or "0"
    return sign + plain


def format_hpc_numeric_cell(text: str) -> str:
    """Convert one CSV cell to plain decimal with at most 19 significant digits."""
    raw = text.strip()
    if not raw:
        return raw
    if raw in ("0", "-0", "+0"):
        return "0"
    has_sci = "e" in raw.lower()
    if not has_sci:
        unsigned = raw.lstrip("+-")
        if unsigned.replace(".", "", 1).isdigit():
            sig = unsigned.lstrip("0").replace(".", "", 1).lstrip("0")
            if not sig:
                return "0"
            if len(sig) <= _HPC_MAX_SIG_DIGITS:
                return raw
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return text
    if not value.is_finite():
        return raw
    with localcontext() as ctx:
        ctx.prec = _HPC_MAX_SIG_DIGITS
        ctx.rounding = ROUND_HALF_EVEN
        rounded = +value
    return _hpc_decimal_to_plain_string(rounded)


def _export_hpc_ts_obs_csv(src: Path, dst: Path) -> Path:
    """Headerless HPC comma CSV -> ``column_0``… comma CSV; normalize numeric cells."""
    if not src.is_file():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    column_count: int | None = None
    with src.open("r", encoding="utf-8", errors="replace", newline="") as handle, dst.open(
        "w", encoding="utf-8", newline="\n"
    ) as out:
        reader = csv.reader(handle)
        writer = csv.writer(out, lineterminator="\n")
        for row in reader:
            if not row:
                continue
            if column_count is None:
                column_count = len(row)
                writer.writerow([f"column_{i}" for i in range(column_count)])
            elif len(row) != column_count:
                raise ValueError(
                    f"Column count mismatch in {src} at data row {row_count + 1}: "
                    f"expected {column_count}, got {len(row)}"
                )
            writer.writerow([format_hpc_numeric_cell(cell) for cell in row])
            row_count += 1
    if column_count is None or row_count == 0:
        raise ValueError(f"No rows found in {src}")
    return dst


def _handle_hpc_ts_obs_csv(stem: str) -> Path:
    return _export_hpc_ts_obs_csv(
        _HPC_TS_OBS_ROOT / f"{stem}.csv",
        _dataset_csv_out(f"{stem}.csv"),
    )


def _handle_acs_wht_f32() -> Path:
    return _handle_hpc_ts_obs_csv("acs_wht_f32")


def _handle_astro_mhd_f64() -> Path:
    return _handle_hpc_ts_obs_csv("astro_mhd_f64")


def _handle_astro_pt_f64() -> Path:
    return _handle_hpc_ts_obs_csv("astro_pt_f64")


def _handle_h3d_temp_f32() -> Path:
    return _handle_hpc_ts_obs_csv("h3d_temp_f32")


def _handle_hdr_night_f32() -> Path:
    return _handle_hpc_ts_obs_csv("hdr_night_f32")


def _handle_hst_wfc3_ir_f32() -> Path:
    return _handle_hpc_ts_obs_csv("hst_wfc3_ir_f32")


def _handle_hst_wfc3_uvis_f32() -> Path:
    return _handle_hpc_ts_obs_csv("hst_wfc3_uvis_f32")


def _handle_tpcds_web_f32() -> Path:
    return _handle_hpc_ts_obs_csv("tpcds_web_f32")


def _handle_tpch_order_f64() -> Path:
    return _handle_hpc_ts_obs_csv("tpch_order_f64")


def _handle_ts_gas_f32() -> Path:
    return _handle_hpc_ts_obs_csv("ts_gas_f32")


def _handle_turbulence_f32() -> Path:
    return _handle_hpc_ts_obs_csv("turbulence_f32")


def _handle_wave_f32() -> Path:
    return _handle_hpc_ts_obs_csv("wave_f32")


def _handle_wesad_chest_f64() -> Path:
    return _handle_hpc_ts_obs_csv("wesad_chest_f64")


def _handle_g24_78_usb2_f32() -> Path:
    return _handle_hpc_ts_obs_csv("g24_78_usb2_f32")


def _handle_hdr_palermo_f32() -> Path:
    return _handle_hpc_ts_obs_csv("hdr_palermo_f32")


def _handle_jane_street_f64() -> Path:
    return _handle_hpc_ts_obs_csv("jane_street_f64")


def _handle_jw_mirimage_f32() -> Path:
    return _handle_hpc_ts_obs_csv("jw_mirimage_f32")


def _handle_miranda3d_f32() -> Path:
    return _handle_hpc_ts_obs_csv("miranda3d_f32")


def _handle_num_brain_f64() -> Path:
    return _handle_hpc_ts_obs_csv("num_brain_f64")


def _handle_num_control_f64() -> Path:
    return _handle_hpc_ts_obs_csv("num_control_f64")


def _handle_phone_gyro_f64() -> Path:
    return _handle_hpc_ts_obs_csv("phone_gyro_f64")


def _handle_rsim_f32() -> Path:
    return _handle_hpc_ts_obs_csv("rsim_f32")


def _handle_spain_gas_price_f64() -> Path:
    return _handle_hpc_ts_obs_csv("spain_gas_price_f64")


def _handle_spitzer_irac_f32() -> Path:
    return _handle_hpc_ts_obs_csv("spitzer_irac_f32")


def _handle_tpcds_catalog_f32() -> Path:
    return _handle_hpc_ts_obs_csv("tpcds_catalog_f32")


def _handle_tpcds_store_f32() -> Path:
    return _handle_hpc_ts_obs_csv("tpcds_store_f32")


def _handle_tpch_lineitem_f32() -> Path:
    return _handle_hpc_ts_obs_csv("tpch_lineitem_f32")


def _handle_tpcxbb_store_f64() -> Path:
    return _handle_hpc_ts_obs_csv("tpcxbb_store_f64")


def _handle_tpcxbb_web_f64() -> Path:
    return _handle_hpc_ts_obs_csv("tpcxbb_web_f64")


def _list_ucr_source_files(ucr_dir: Path) -> list[Path]:
    indexed: list[tuple[int, Path]] = []
    unmatched: list[Path] = []
    for path in ucr_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".txt":
            continue
        match = _UCR_TXT_PATTERN.match(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
        else:
            unmatched.append(path)
    if unmatched:
        sample = ", ".join(p.name for p in sorted(unmatched)[:10])
        suffix = " …" if len(unmatched) > 10 else ""
        raise FileNotFoundError(
            f"Unexpected txt files in {ucr_dir} (expected NNN_UCR_*.txt): {sample}{suffix}"
        )
    if not indexed:
        raise FileNotFoundError(
            f"No UCR txt files matching {_UCR_TXT_PATTERN.pattern} under {ucr_dir}"
        )
    indexed.sort(key=lambda item: item[0])
    return [path for _, path in indexed]


def _read_ucr_column(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        lines = [raw.rstrip("\r\n") for raw in handle]
    non_blank = [line for line in lines if line.strip()]
    if len(non_blank) == 1:
        values: list[str] = []
        for token_no, token in enumerate(non_blank[0].split(), start=1):
            try:
                values.append(_decimal_plain_string(token))
            except ValueError as exc:
                raise ValueError(f"{path.name} token {token_no}: {exc}") from exc
        return values

    values = []
    for line_no, raw in enumerate(lines, start=1):
        if not raw.strip():
            values.append("")
            continue
        try:
            values.append(_decimal_plain_string(raw))
        except ValueError as exc:
            raise ValueError(f"{path.name} line {line_no}: {exc}") from exc
    return values


def _handle_bar_crawl() -> Path:
    return _copy_csv_lines_with_lf(
        DATASET_TEMP / "Leco" / "all_accelerometer_data_pids_13.csv",
        _dataset_csv_out("Bar-Crawl.csv"),
    )


def _handle_bird_migration() -> Path:
    return _export_quoted_csv_to_comma_csv(
        DATASET_TEMP / "migration_original.csv",
        _dataset_csv_out("Bird-migration.csv"),
    )


def _handle_realtor() -> Path:
    return _copy_csv_lines_with_lf(
        DATASET_TEMP / "Leco" / "realtor-data.zip.csv",
        _dataset_csv_out("realtor.csv"),
    )


def _handle_rating() -> Path:
    return _export_quoted_csv_to_comma_csv(
        DATASET_TEMP / "Leco" / "rating.csv",
        _dataset_csv_out("rating.csv"),
    )


def _handle_comments() -> Path:
    return _copy_csv_lines_with_lf(
        DATASET_TEMP / "Leco" / "Comments.csv",
        _dataset_csv_out("Comments.csv"),
    )


LECO_DATA_ROOT = DATASET_TEMP / "Leco" / "data"


def _leco_data_src(filename: str) -> Path:
    return LECO_DATA_ROOT / filename


def _leco_data_dst(stem: str) -> Path:
    return _dataset_csv_out(f"{stem}.csv")


def _export_delimited_to_comma_csv(
    src: Path,
    dst: Path,
    *,
    sep: str,
    header: str | int | None = "infer",
    names: list[str] | None = None,
    chunksize: int = 500_000,
    encoding: str = "utf-8",
) -> Path:
    """Read delimited text as strings; write comma-separated CSV with LF."""
    import pandas as pd

    if not src.is_file():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    read_kwargs: dict = {
        "sep": sep,
        "dtype": str,
        "keep_default_na": False,
        "chunksize": chunksize,
        "encoding": encoding,
    }
    if names is not None:
        read_kwargs["names"] = names
        read_kwargs["header"] = 0
    elif header is not None:
        read_kwargs["header"] = header
    wrote_any = False
    for chunk in pd.read_csv(src, **read_kwargs):
        chunk.to_csv(
            dst,
            mode="w" if not wrote_any else "a",
            index=False,
            header=not wrote_any,
            lineterminator="\n",
        )
        wrote_any = True
    if not wrote_any:
        raise ValueError(f"No rows found in {src}")
    return dst


def _export_headerless_comma_to_comma_csv(src: Path, dst: Path) -> Path:
    """Headerless comma-separated rows -> ``column_0``… comma CSV with LF."""
    if not src.is_file():
        raise FileNotFoundError(f"Missing source file: {src}")
    with src.open("r", encoding="utf-8", newline="") as handle:
        first = next(csv.reader(handle), None)
    if not first:
        raise ValueError(f"No rows found in {src}")
    names = [f"column_{i}" for i in range(len(first))]
    return _export_delimited_to_comma_csv(
        src, dst, sep=",", header=None, names=names
    )


def _export_arff_to_comma_csv(src: Path, dst: Path) -> Path:
    """Parse Weka ARFF ``@attribute`` / ``@data``; write comma-separated CSV with LF."""
    if not src.is_file():
        raise FileNotFoundError(f"Missing source file: {src}")
    attrs: list[str] = []
    rows: list[list[str]] = []
    in_data = False
    with src.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            lower = line.lower()
            if lower.startswith("@attribute"):
                parts = line.split()
                if len(parts) >= 2:
                    attrs.append(parts[1])
                continue
            if lower == "@data":
                in_data = True
                continue
            if lower.startswith("@"):
                continue
            if in_data:
                rows.append([cell.strip() for cell in line.split(",")])
    if not attrs:
        raise ValueError(f"No @attribute rows found in {src}")
    if not rows:
        raise ValueError(f"No @data rows found in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="\n") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(attrs)
        writer.writerows(rows)
    return dst


def _export_single_column_values_to_comma_csv(
    src: Path, dst: Path, *, column_name: str = "column_0"
) -> Path:
    """One value per line (no header) -> single-column comma CSV with LF."""
    if not src.is_file():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    wrote_any = False
    with src.open("r", encoding="utf-8", newline="") as handle, dst.open(
        "w", encoding="utf-8", newline="\n"
    ) as out:
        writer = csv.writer(out, lineterminator="\n")
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if not wrote_any:
                writer.writerow([column_name])
                wrote_any = True
            writer.writerow([text])
    if not wrote_any:
        raise ValueError(f"No rows found in {src}")
    return dst


def _handle_1year() -> Path:
    return _export_arff_to_comma_csv(
        _leco_data_src("1year.arff"), _leco_data_dst("1year")
    )


def _handle_dsl_strong_password_data() -> Path:
    return _copy_csv_lines_with_lf(
        _leco_data_src("DSL-StrongPasswordData.csv"),
        _leco_data_dst("DSL-StrongPasswordData"),
    )


def _handle_rail_insurance_claims() -> Path:
    return _copy_csv_lines_with_lf(
        _leco_data_src("Rail_Insurance_Claims.csv"),
        _leco_data_dst("Rail_Insurance_Claims"),
    )


def _handle_adult() -> Path:
    """Merge UCI Adult train/test sources into one comma-separated ``adult.csv``."""
    import pandas as pd

    data_src = _leco_data_src("adult.data.csv")
    test_src = _leco_data_src("adult_test.csv")
    train_src = _leco_data_src("adult_train.csv")
    for path in (data_src, test_src, train_src):
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file: {path}")

    standard_columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary",
    ]
    rename_map = {
        "Age": "age",
        "Workclass": "workclass",
        "Education": "education",
        "Education_Num": "education-num",
        "Martial_Status": "marital-status",
        "Occupation": "occupation",
        "Relationship": "relationship",
        "Race": "race",
        "Sex": "sex",
        "Capital_Gain": "capital-gain",
        "Capital_Loss": "capital-loss",
        "Hours_per_week": "hours-per-week",
        "Country": "native-country",
        "Target": "salary",
    }

    def normalize_row(values: list[str]) -> tuple[str, ...] | None:
        cells = [str(value).strip() for value in values]
        if len(cells) != len(standard_columns):
            return None
        if not cells[0].isdigit():
            return None
        salary = cells[-1]
        if salary.endswith(".") and salary[:-1] in ("<=50K", ">50K"):
            cells[-1] = salary[:-1]
        for idx, cell in enumerate(cells):
            if cell == "":
                cells[idx] = "?"
        return tuple(cells)

    def read_data_rows() -> list[list[str]]:
        with data_src.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if [cell.strip() for cell in header] != standard_columns:
                raise ValueError(
                    f"Unexpected adult.data header in {data_src}: {header!r}"
                )
            return [list(row) for row in reader]

    def read_delimited_rows(src: Path) -> list[list[str]]:
        frame = pd.read_csv(
            src,
            sep=";",
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )
        frame = frame.rename(columns=rename_map)
        missing = [name for name in standard_columns if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns in {src}: {missing}")
        frame = frame[standard_columns]
        return frame.values.tolist()

    merged_rows: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for src_rows in (
        read_data_rows(),
        read_delimited_rows(test_src),
        read_delimited_rows(train_src),
    ):
        for row in src_rows:
            normalized = normalize_row(row)
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            merged_rows.append(list(normalized))

    if not merged_rows:
        raise ValueError("No adult rows found after merge")

    dst = _dataset_csv_out("adult.csv")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(standard_columns)
        writer.writerows(merged_rows)
    return dst


def _handle_agaricus_lepiota_data() -> Path:
    return _export_headerless_comma_to_comma_csv(
        _leco_data_src("agaricus-lepiota.data.txt"),
        _leco_data_dst("agaricus-lepiota.data"),
    )


def _handle_bank() -> Path:
    """Join bank marketing features and labels into one comma-separated ``bank.csv``."""
    train_src = _leco_data_src("bank_train.csv")
    target_src = _leco_data_src("bank_train_target.csv")
    for path in (train_src, target_src):
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file: {path}")

    with train_src.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"No rows found in {train_src}") from exc
        train_rows = list(reader)

    targets: list[str] = []
    with target_src.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                targets.append(text)

    if not train_rows:
        raise ValueError(f"No feature rows found in {train_src}")
    if not targets:
        raise ValueError(f"No label rows found in {target_src}")
    if len(train_rows) != len(targets):
        raise ValueError(
            f"Row count mismatch: {train_src} has {len(train_rows)} rows, "
            f"{target_src} has {len(targets)} labels"
        )

    out_header = header + ["y"]
    expected_cols = len(header)
    merged_rows: list[list[str]] = []
    for index, (row, target) in enumerate(zip(train_rows, targets), start=1):
        if len(row) != expected_cols:
            raise ValueError(
                f"Unexpected column count in {train_src} row {index}: "
                f"expected {expected_cols}, got {len(row)}"
            )
        merged_rows.append([*row, target])

    dst = _dataset_csv_out("bank.csv")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(out_header)
        writer.writerows(merged_rows)
    return dst


def _handle_credit_scoring_sample() -> Path:
    return _export_delimited_to_comma_csv(
        _leco_data_src("credit_scoring_sample.csv"),
        _leco_data_dst("credit_scoring_sample"),
        sep=";",
    )


def _handle_credit_scoring_test() -> Path:
    return _copy_csv_lines_with_lf(
        _leco_data_src("credit_scoring_test.csv"),
        _leco_data_dst("credit_scoring_test"),
    )


def _handle_credit_scoring_train() -> Path:
    return _copy_csv_lines_with_lf(
        _leco_data_src("credit_scoring_train.csv"),
        _leco_data_dst("credit_scoring_train"),
    )


def _handle_ehresp_2014() -> Path:
    return _copy_csv_lines_with_lf(
        _leco_data_src("ehresp_2014.csv"), _leco_data_dst("ehresp_2014")
    )


def _handle_flight_delays_test() -> Path:
    return _copy_csv_lines_with_lf(
        _leco_data_src("flight_delays_test.csv"),
        _leco_data_dst("flight_delays_test"),
    )


def _handle_flight_delays_train() -> Path:
    return _copy_csv_lines_with_lf(
        _leco_data_src("flight_delays_train.csv"),
        _leco_data_dst("flight_delays_train"),
    )


def _handle_hmeq() -> Path:
    return _copy_csv_lines_with_lf(
        _leco_data_src("hmeq.csv"), _leco_data_dst("hmeq")
    )


def _handle_medium_posts() -> Path:
    return _export_delimited_to_comma_csv(
        _leco_data_src("medium_posts.csv"),
        _leco_data_dst("medium_posts"),
        sep="\t",
    )


def _handle_mlbootcamp5_train() -> Path:
    return _export_delimited_to_comma_csv(
        _leco_data_src("mlbootcamp5_train.csv"),
        _leco_data_dst("mlbootcamp5_train"),
        sep=";",
    )


def _handle_prolongation_service_train() -> Path:
    return _export_delimited_to_comma_csv(
        _leco_data_src("prolongation_service_train.csv"),
        _leco_data_dst("prolongation_service_train"),
        sep="\t",
        encoding="iso-8859-1",
    )


def _handle_scores() -> Path:
    return _copy_csv_lines_with_lf(
        _leco_data_src("scores.csv"), _leco_data_dst("scores")
    )


def _handle_spam() -> Path:
    return _export_quoted_csv_to_comma_csv(
        _leco_data_src("spam.csv"),
        _leco_data_dst("spam"),
        encoding="latin-1",
    )


def _handle_spooky_authors() -> Path:
    return _export_quoted_csv_to_comma_csv(
        _leco_data_src("spooky_authors.csv"), _leco_data_dst("spooky_authors")
    )


def _handle_spooky_writer_sample_submission() -> Path:
    return _export_quoted_csv_to_comma_csv(
        _leco_data_src("spooky_writer_sample_submission.csv"),
        _leco_data_dst("spooky_writer_sample_submission"),
    )


def _handle_spooky_writer_test() -> Path:
    return _export_quoted_csv_to_comma_csv(
        _leco_data_src("spooky_writer_test.csv"),
        _leco_data_dst("spooky_writer_test"),
    )


def _handle_spooky_writer_train() -> Path:
    return _export_quoted_csv_to_comma_csv(
        _leco_data_src("spooky_writer_train.csv"),
        _leco_data_dst("spooky_writer_train"),
    )


def _handle_vgsales() -> Path:
    return _copy_csv_lines_with_lf(
        _leco_data_src("vgsales.csv"), _leco_data_dst("vgsales")
    )


def _handle_video_games_sales() -> Path:
    return _copy_csv_lines_with_lf(
        _leco_data_src("video_games_sales.csv"),
        _leco_data_dst("video_games_sales"),
    )


def _handle_weights_heights() -> Path:
    return _copy_csv_lines_with_lf(
        _leco_data_src("weights_heights.csv"),
        _leco_data_dst("weights_heights"),
    )


def _handle_winequality_white() -> Path:
    return _export_delimited_to_comma_csv(
        _leco_data_src("winequality-white.csv"),
        _leco_data_dst("winequality-white"),
        sep=";",
    )


_LECO_BINARY_HEADER_BYTES = 8
_LECO_BINARY_CHUNK_VALUES = 1_000_000


def _uint64_to_signed_int64(value: int) -> int:
    """Reinterpret uint64 bits as signed int64 (two's complement wrap)."""
    return struct.unpack("<q", struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF))[0]


def _export_leco_binary_uint_csv(
    src: Path,
    dst: Path,
    *,
    value_bytes: int,
    struct_code: str,
    type_label: str,
    reinterpret_as_signed_int64: bool = False,
) -> Path:
    if not src.is_file():
        raise FileNotFoundError(f"Missing source file: {src}")

    file_size = src.stat().st_size
    payload_bytes = file_size - _LECO_BINARY_HEADER_BYTES
    if payload_bytes < 0 or payload_bytes % value_bytes != 0:
        raise ValueError(
            f"Unexpected {src.name} payload size: {file_size} bytes "
            f"(expected 8-byte header plus a multiple of {value_bytes})"
        )

    expected_count = payload_bytes // value_bytes
    dst.parent.mkdir(parents=True, exist_ok=True)
    chunk_bytes = _LECO_BINARY_CHUNK_VALUES * value_bytes
    written = 0

    with src.open("rb") as handle, dst.open("w", encoding="utf-8", newline="\n") as out:
        header = handle.read(_LECO_BINARY_HEADER_BYTES)
        if len(header) != _LECO_BINARY_HEADER_BYTES:
            raise ValueError(f"Missing {_LECO_BINARY_HEADER_BYTES}-byte header in {src}")
        (declared_count,) = struct.unpack("<Q", header)
        if declared_count != expected_count:
            raise ValueError(
                f"Header count {declared_count} does not match payload "
                f"({expected_count} {type_label} values)"
            )

        while True:
            raw = handle.read(chunk_bytes)
            if not raw:
                break
            if len(raw) % value_bytes != 0:
                raise ValueError(f"Truncated {type_label} payload in {src}")
            count = len(raw) // value_bytes
            values = struct.unpack(f"<{count}{struct_code}", raw)
            if reinterpret_as_signed_int64:
                if struct_code != "Q":
                    raise ValueError(
                        "reinterpret_as_signed_int64 requires uint64 struct_code 'Q'"
                    )
                formatted = (str(_uint64_to_signed_int64(value)) for value in values)
            else:
                formatted = (str(value) for value in values)
            block = "\n".join(formatted)
            if written:
                out.write("\n")
            out.write(block)
            written += count

        if written:
            out.write("\n")

    if written != expected_count:
        raise ValueError(
            f"Wrote {written} values, expected {expected_count} from {src}"
        )
    return dst


def _handle_books() -> Path:
    return _export_leco_binary_uint_csv(
        DATASET_TEMP / "Leco" / "books_200M_uint32",
        _dataset_csv_out("books.csv"),
        value_bytes=4,
        struct_code="I",
        type_label="uint32",
    )


def _handle_fb() -> Path:
    return _export_leco_binary_uint_csv(
        DATASET_TEMP / "Leco" / "fb_200M_uint64",
        _dataset_csv_out("fb.csv"),
        value_bytes=8,
        struct_code="Q",
        type_label="uint64",
        reinterpret_as_signed_int64=True,
    )


def _handle_wiki_ts() -> Path:
    return _export_leco_binary_uint_csv(
        DATASET_TEMP / "Leco" / "wiki_ts_200M_uint64",
        _dataset_csv_out("wiki_ts.csv"),
        value_bytes=8,
        struct_code="Q",
        type_label="uint64",
    )


def _handle_linear() -> Path:
    return _copy_csv_lines_with_lf(
        DATASET_TEMP / "Leco" / "linear_200M_uint32.txt",
        _dataset_csv_out("linear.csv"),
    )


def _handle_normal() -> Path:
    return _copy_csv_lines_with_lf(
        DATASET_TEMP / "Leco" / "normal_200M_uint32.txt",
        _dataset_csv_out("normal.csv"),
    )


def _handle_world_city() -> Path:
    return _copy_csv_lines_with_lf(
        _PUBLIC_DATASETS_ROOT / "City-lat" / "worldcities.csv",
        _dataset_csv_out("World-City.csv"),
    )


def _handle_poi() -> Path:
    return _copy_csv_lines_with_lf(
        _PUBLIC_DATASETS_ROOT / "poi" / "poi.csv",
        _dataset_csv_out("poi.csv"),
    )


def _handle_solar_wind() -> Path:
    return _copy_csv_lines_with_lf(
        _PUBLIC_DATASETS_ROOT / "msg" / "solar_wind.csv",
        _dataset_csv_out("solar_wind.csv"),
    )


def _merge_meteoblue_csv_sources(sources: list[Path], dst: Path) -> Path:
    """Skip meteoblue metadata rows; merge shards into one header + data CSV with LF."""
    if not sources:
        raise FileNotFoundError("No meteoblue CSV sources provided")
    dst.parent.mkdir(parents=True, exist_ok=True)
    header: str | None = None
    data_rows = 0

    with dst.open("w", encoding="utf-8", newline="\n") as out:
        for src in sources:
            if not src.is_file():
                raise FileNotFoundError(f"Missing source CSV: {src}")
            with src.open("r", encoding="utf-8", newline="") as handle:
                for line_no, line in enumerate(handle):
                    text = line.rstrip("\r\n")
                    if line_no < METEOBLUE_METADATA_ROWS:
                        continue
                    if line_no == METEOBLUE_METADATA_ROWS:
                        if header is None:
                            header = text
                            out.write(text + "\n")
                        elif header != text:
                            raise ValueError(
                                f"Header mismatch in {src}: expected {header!r}, got {text!r}"
                            )
                        continue
                    out.write(text + "\n")
                    data_rows += 1

    if header is None:
        raise ValueError(f"Missing header row after metadata in {sources[0]}")
    if data_rows == 0:
        raise ValueError(f"No data rows found in meteoblue sources: {sources}")
    return dst


def _handle_meteoblue() -> Path:
    sources = _list_csv_files_in_dir(METEOBLUE_ROOT)
    return _merge_meteoblue_csv_sources(sources, _dataset_csv_out("meteoblue.csv"))


def _handle_ucr() -> Path:
    ucr_dir = DATASET_TEMP / "UCR_Anomaly_FullData"
    if not ucr_dir.is_dir():
        raise FileNotFoundError(f"Missing UCR source directory: {ucr_dir}")

    source_files = _list_ucr_source_files(ucr_dir)
    columns: list[tuple[str, list[str]]] = []
    max_rows = 0
    for path in source_files:
        col_values = _read_ucr_column(path)
        max_rows = max(max_rows, len(col_values))
        columns.append((path.stem, col_values))

    dst = _dataset_csv_out("UCR.csv")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="\n") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow([header for header, _ in columns])
        for row_idx in range(max_rows):
            writer.writerow(
                [
                    col_values[row_idx] if row_idx < len(col_values) else ""
                    for _, col_values in columns
                ]
            )
    return dst


HANDLERS = {
    "1year": _handle_1year,
    "DSL-StrongPasswordData": _handle_dsl_strong_password_data,
    "Rail_Insurance_Claims": _handle_rail_insurance_claims,
    "adult": _handle_adult,
    "agaricus-lepiota.data": _handle_agaricus_lepiota_data,
    "bank": _handle_bank,
    "credit_scoring_sample": _handle_credit_scoring_sample,
    "credit_scoring_test": _handle_credit_scoring_test,
    "credit_scoring_train": _handle_credit_scoring_train,
    "ehresp_2014": _handle_ehresp_2014,
    "flight_delays_test": _handle_flight_delays_test,
    "flight_delays_train": _handle_flight_delays_train,
    "hmeq": _handle_hmeq,
    "medium_posts": _handle_medium_posts,
    "mlbootcamp5_train": _handle_mlbootcamp5_train,
    "prolongation_service_train": _handle_prolongation_service_train,
    "scores": _handle_scores,
    "spam": _handle_spam,
    "spooky_authors": _handle_spooky_authors,
    "spooky_writer_sample_submission": _handle_spooky_writer_sample_submission,
    "spooky_writer_test": _handle_spooky_writer_test,
    "spooky_writer_train": _handle_spooky_writer_train,
    "vgsales": _handle_vgsales,
    "video_games_sales": _handle_video_games_sales,
    "weights_heights": _handle_weights_heights,
    "winequality-white": _handle_winequality_white,
    "Air-pressure-1min": _handle_air_pressure_1min,
    "Air-pressure-30min": _handle_air_pressure_30min,
    "Dewpoint-temp-1min": _handle_dew_point_temp_1min,
    "Dewpoint-temp-30min": _handle_dew_point_temp_30min,
    "PM10-dust-30minutes": _handle_pm10_dust_30minutes,
    "PM10-dust-60minutes": _handle_pm10_dust_60minutes,
    "IR-bio-temp-1minute": _handle_ir_bio_temp_1minute,
    "IR-bio-temp-30minute": _handle_ir_bio_temp_30minute,
    "Bar-Crawl": _handle_bar_crawl,
    "Bird-migration": _handle_bird_migration,
    "City-temp": _handle_city_temp,
    "Climate_Weather": _handle_climate_weather,
    "Comments": _handle_comments,
    "Food-prices": _handle_food_prices,
    "HOUSE": _handle_house,
    "SSD-bench": _handle_ssd_bench,
    "Stocks-Germany": _handle_stocks_germany,
    "UCR": _handle_ucr,
    "vehicle-charge": _handle_vehicle_charge,
    "World-City": _handle_world_city,
    "books": _handle_books,
    "fb": _handle_fb,
    "linear": _handle_linear,
    "meteoblue": _handle_meteoblue,
    "normal": _handle_normal,
    "NYC_Yellow_Taxi_Trip": _handle_nyc_yellow_taxi_trip,
    "poi": _handle_poi,
    "realtor": _handle_realtor,
    "rating": _handle_rating,
    "solar_wind": _handle_solar_wind,
    "wiki_ts": _handle_wiki_ts,
    "acs_wht_f32": _handle_acs_wht_f32,
    "astro_mhd_f64": _handle_astro_mhd_f64,
    "astro_pt_f64": _handle_astro_pt_f64,
    "h3d_temp_f32": _handle_h3d_temp_f32,
    "hdr_night_f32": _handle_hdr_night_f32,
    "hst_wfc3_ir_f32": _handle_hst_wfc3_ir_f32,
    "hst_wfc3_uvis_f32": _handle_hst_wfc3_uvis_f32,
    "tpcds_web_f32": _handle_tpcds_web_f32,
    "tpch_order_f64": _handle_tpch_order_f64,
    "ts_gas_f32": _handle_ts_gas_f32,
    "turbulence_f32": _handle_turbulence_f32,
    "wave_f32": _handle_wave_f32,
    "wesad_chest_f64": _handle_wesad_chest_f64,
    "g24_78_usb2_f32": _handle_g24_78_usb2_f32,
    "hdr_palermo_f32": _handle_hdr_palermo_f32,
    "jane_street_f64": _handle_jane_street_f64,
    "jw_mirimage_f32": _handle_jw_mirimage_f32,
    "miranda3d_f32": _handle_miranda3d_f32,
    "num_brain_f64": _handle_num_brain_f64,
    "num_control_f64": _handle_num_control_f64,
    "phone_gyro_f64": _handle_phone_gyro_f64,
    "rsim_f32": _handle_rsim_f32,
    "spain_gas_price_f64": _handle_spain_gas_price_f64,
    "spitzer_irac_f32": _handle_spitzer_irac_f32,
    "tpcds_catalog_f32": _handle_tpcds_catalog_f32,
    "tpcds_store_f32": _handle_tpcds_store_f32,
    "tpch_lineitem_f32": _handle_tpch_lineitem_f32,
    "tpcxbb_store_f64": _handle_tpcxbb_store_f64,
    "tpcxbb_web_f64": _handle_tpcxbb_web_f64,
}
HANDLERS.update(
    {name: _public_bi_dataset_handler(name) for name in _PUBLICBI_DATASET_SOURCES}
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        metavar="NAME",
        help="Dataset name to process",
    )
    args = parser.parse_args()

    dataset = (args.dataset or "").strip()
    if not dataset:
        print("No --dataset provided. Nothing to do.")
        return 0

    handler = HANDLERS.get(dataset)
    if handler is None:
        known = ", ".join(sorted(HANDLERS))
        print(
            f"Unsupported dataset: {dataset!r}. Supported: {known}",
            file=sys.stderr,
        )
        return 1

    out = handler()
    print(f"Prepared dataset_csv CSV ({dataset_csv_root(BACKEND)}): {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
