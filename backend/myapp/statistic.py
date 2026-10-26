import matplotlib
matplotlib.use('Agg')

import math
import struct
from collections import Counter
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_INTEGER_LOGICAL_TYPES = frozenset(
    {'integer', 'int', 'long', 'timestamp', 'datetime', 'date', 'time'}
)
_FLOAT_LOGICAL_TYPES = frozenset({'float', 'double'})


def float_to_bits(f):
    """Legacy 32-bit float bits (kept for callers outside catalog charts)."""
    return struct.unpack('I', struct.pack('f', float(f)))[0]


def bits_to_float(bits):
    return struct.unpack('f', struct.pack('I', bits))[0]


def is_integer_logical_type(logical_type: str) -> bool:
    return str(logical_type or '').strip().lower() in _INTEGER_LOGICAL_TYPES


def is_float_logical_type(logical_type: str) -> bool:
    return not is_integer_logical_type(logical_type)


def value_to_bits64(value, *, as_float: bool) -> int:
    if as_float:
        return struct.unpack('<Q', struct.pack('<d', float(value)))[0]
    try:
        iv = int(value)
    except (TypeError, ValueError):
        iv = int(float(value))
    return iv & 0xFFFFFFFFFFFFFFFF


def count_trailing_zeros64(bits: int) -> int:
    bits &= 0xFFFFFFFFFFFFFFFF
    if bits == 0:
        return 64
    return (bits & -bits).bit_length() - 1


def count_leading_zeros64(bits: int) -> int:
    bits &= 0xFFFFFFFFFFFFFFFF
    if bits == 0:
        return 64
    return 64 - bits.bit_length()


def count_trailing_zeros(bits: int) -> int:
    return count_trailing_zeros64(bits)


def count_leading_zeros(bits: int) -> int:
    return count_leading_zeros64(bits)


def compute_log(x):
    """Compute ceil(log2(x+1)); 0 when x <= 0."""
    return math.ceil(math.log2(x + 1)) if x > 0 else 0


def _is_finite_number(value) -> bool:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(num)


def byte_entropy(arr: list):
    byte_arr = b''.join(struct.pack('<d', float(x)) for x in arr if _is_finite_number(x))
    if not byte_arr:
        return 0.0
    counts = Counter(byte_arr)
    total = len(byte_arr)
    entropy = -sum((count/total) * math.log2(count/total)
                   for count in counts.values() if count > 0)
    return entropy


def value_entropy(arr: list) -> float:
    """Shannon entropy (bits) over distinct numeric values in ``arr``."""
    if not arr:
        return 0.0
    counts = Counter(float(x) for x in arr)
    total = len(arr)
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count > 0
    )


def _finite_statistic_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def statistic_row_byte_entropy(row: dict) -> float | None:
    """Legacy ``entropy`` column is treated as byte entropy."""
    for key in ('byte_entropy', 'entropy'):
        num = _finite_statistic_number(row.get(key))
        if num is not None:
            return num
    return None


def statistic_row_value_entropy(row: dict) -> float | None:
    return _finite_statistic_number(row.get('value_entropy'))


def _statistic_row_match_key(row: dict) -> tuple[int | None, str]:
    raw_idx = row.get('column_index')
    try:
        col_idx = int(raw_idx)
    except (TypeError, ValueError):
        col_idx = None
    name = str(row.get('column_name') or '').strip()
    return col_idx, name


def _baseline_row_lookup(rows: list[dict]) -> dict[tuple[int | None, str], dict]:
    out: dict[tuple[int | None, str], dict] = {}
    for row in rows:
        key = _statistic_row_match_key(row)
        if key[1] or key[0] is not None:
            out[key] = row
    return out


def enrich_statistic_rows_for_api(
    rows: list[dict],
    *,
    baseline_rows: list[dict] | None = None,
) -> list[dict]:
    """Expose byte/value entropy fields; optional deltas when ``baseline_rows`` is set."""
    baseline_by_key = _baseline_row_lookup(baseline_rows or [])
    enriched: list[dict] = []
    for row in rows:
        item = dict(row)
        byte_e = statistic_row_byte_entropy(row)
        value_e = statistic_row_value_entropy(row)
        item['byte_entropy'] = byte_e
        item['value_entropy'] = value_e
        if byte_e is not None:
            item['entropy'] = byte_e
        if baseline_rows is not None:
            baseline = baseline_by_key.get(_statistic_row_match_key(row))
            base_byte = statistic_row_byte_entropy(baseline) if baseline else None
            base_value = statistic_row_value_entropy(baseline) if baseline else None
            item['byte_entropy_change'] = (
                float(byte_e - base_byte)
                if byte_e is not None and base_byte is not None
                else None
            )
            item['value_entropy_change'] = (
                float(value_e - base_value)
                if value_e is not None and base_value is not None
                else None
            )
        enriched.append(item)
    return enriched


def repeat_words(data: list, limit=8):
    finite_count = sum(1 for value in data if _is_finite_number(value))
    if finite_count == 0:
        return 0.0
    covered = 0
    key = None
    run_length = 0
    for val in data:
        if not _is_finite_number(val):
            if run_length >= limit:
                covered += run_length
            key = None
            run_length = 0
        elif key is not None and val == key:
            run_length += 1
        else:
            if run_length >= limit:
                covered += run_length
            key = val
            run_length = 1
    if run_length >= limit:
        covered += run_length
    return float(covered) / finite_count


def sortedness(data: list):
    count = 0
    adjacent_finite_pairs = 0
    for left, right in zip(data, data[1:]):
        if not _is_finite_number(left) or not _is_finite_number(right):
            continue
        adjacent_finite_pairs += 1
        if left <= right:
            count += 1
    return count / adjacent_finite_pairs if adjacent_finite_pairs else 0.0


def scaling(data: list) -> list:
    n = 0
    for value in data:
        if '.' in str(value):
            decimal_part = str(value).split('.')[1]
            if len(decimal_part) > n:
                n = len(decimal_part)

    int_list = []
    for num_str in data:
        num_scaled = num_str * (10 ** n)
        num_int = int(num_scaled)
        int_list.append(num_int)

    min_int = min(int_list)
    for i in range(len(int_list)):
        int_list[i] = int_list[i] - min_int

    return int_list


def bit_widths_for_values(column_data: list, logical_type: str) -> list[int]:
    """Integer columns: bit width from 64-bit magnitude; float: scaling then bit width."""
    if is_integer_logical_type(logical_type):
        widths: list[int] = []
        for raw in column_data:
            if not _is_finite_number(raw):
                continue
            try:
                mag = abs(int(raw) if not isinstance(raw, float) else int(float(raw)))
            except (TypeError, ValueError, OverflowError):
                continue
            widths.append(compute_log(mag))
        return widths

    finite = [x for x in column_data if _is_finite_number(x)]
    if not finite:
        return []
    return [compute_log(v) for v in scaling(finite)]


def xor_zero_distributions_for_values(
    column_data: list,
) -> tuple[list[int], list[int]]:
    """Leading/trailing zero counts from consecutive IEEE754 float64 XOR (``tolist`` order)."""
    if len(column_data) < 2:
        return [], []
    leading: list[int] = []
    trailing: list[int] = []
    pre_value_b = value_to_bits64(column_data[0], as_float=True)
    for i in range(1, len(column_data)):
        b_value = value_to_bits64(column_data[i], as_float=True)
        xor_value = pre_value_b ^ b_value
        leading.append(count_leading_zeros64(xor_value))
        trailing.append(count_trailing_zeros64(xor_value))
        pre_value_b = b_value
    return leading, trailing


def distribution_rows_from_counts(value_counts: dict[int, int]) -> list[dict[str, float | int]]:
    """Histogram bins as JSON-serializable rows (sorted by category)."""
    total = sum(int(v) for v in value_counts.values())
    if total <= 0:
        return []
    rows: list[dict[str, float | int]] = []
    for category, count in sorted((int(k), int(v)) for k, v in value_counts.items()):
        rows.append(
            {
                'category': category,
                'count': count,
                'percentage': float(count) / float(total) * 100.0,
            }
        )
    return rows


def bit_width_distribution_data(log_values: list[int]) -> list[dict[str, float | int]]:
    value_counts: dict[int, int] = defaultdict(int)
    for val in log_values:
        value_counts[int(val)] += 1
    return distribution_rows_from_counts(value_counts)


def xor_zero_distribution_data(zero_counts: dict[int, int]) -> list[dict[str, float | int]]:
    normalized = {int(k): int(v) for k, v in zero_counts.items()}
    return distribution_rows_from_counts(normalized)


def column_bit_width_distribution_data(
    df: pd.DataFrame,
    logical_type: str,
) -> list[dict[str, float | int]]:
    log_values: list[int] = []
    for column in df.columns:
        series = pd.to_numeric(df[column], errors='coerce')
        finite_series = series[np.isfinite(series)]
        if finite_series.empty:
            continue
        log_values.extend(bit_widths_for_values(finite_series.tolist(), logical_type))
    return bit_width_distribution_data(log_values)


def column_xor_zero_distribution_data(
    df: pd.DataFrame,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    leading_all: list[int] = []
    trailing_all: list[int] = []
    for column in df.columns:
        column_data = df[column].tolist()
        leading, trailing = xor_zero_distributions_for_values(column_data)
        leading_all.extend(leading)
        trailing_all.extend(trailing)

    leading_counts: dict[int, int] = defaultdict(int)
    trailing_counts: dict[int, int] = defaultdict(int)
    for val in leading_all:
        leading_counts[int(val)] += 1
    for val in trailing_all:
        trailing_counts[int(val)] += 1
    return (
        xor_zero_distribution_data(leading_counts),
        xor_zero_distribution_data(trailing_counts),
    )


def _plot_bit_width_distribution(log_values: list[int], output_path: str, *, font_size: int = 18) -> None:
    rows = bit_width_distribution_data(log_values)
    if not rows:
        plt.figure(figsize=(10, 6))
        plt.xlabel('Bit width', fontsize=font_size)
        plt.ylabel('Frequency (%)', fontsize=font_size)
        plt.xticks(np.arange(0, 65, 5), fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.savefig(output_path)
        plt.close()
        return

    x_values = [int(row['category']) for row in rows]
    y_values = [float(row['percentage']) for row in rows]

    plt.figure(figsize=(10, 6))
    plt.bar(x_values, y_values)
    plt.xlabel('Bit width', fontsize=font_size)
    plt.ylabel('Frequency (%)', fontsize=font_size)
    plt.xticks(np.arange(0, 65, 5), fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.savefig(output_path)
    plt.close()


def _plot_xor_zero_distribution(
    zero_counts: dict[int, int],
    *,
    output_path: str,
    xlabel: str,
    font_size: int = 18,
) -> None:
    rows = xor_zero_distribution_data(zero_counts)
    if not rows:
        plt.figure(figsize=(10, 6))
        plt.xlabel(xlabel, fontsize=font_size)
        plt.ylabel('Frequency (%)', fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.savefig(output_path)
        plt.close()
        return

    all_x_values = [int(row['category']) for row in rows]
    y_values = [float(row['percentage']) for row in rows]

    plt.figure(figsize=(10, 6))
    plt.bar(all_x_values, y_values, color='blue', alpha=0.7)
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel('Frequency (%)', fontsize=font_size)
    if len(all_x_values) > 10:
        max_value = max(all_x_values)
        show_labels = list(range(0, max_value + 1, 4))
        plt.xticks(show_labels, fontsize=font_size)
    else:
        plt.xticks(all_x_values, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.savefig(output_path)
    plt.close()


def statistic_bit_width_distribution(
    df: pd.DataFrame,
    output_path: str,
    logical_type: str,
    font_size: int = 18,
) -> None:
    log_values: list[int] = []
    for column in df.columns:
        series = pd.to_numeric(df[column], errors='coerce')
        finite_series = series[np.isfinite(series)]
        if finite_series.empty:
            continue
        log_values.extend(bit_widths_for_values(finite_series.tolist(), logical_type))
    _plot_bit_width_distribution(log_values, output_path, font_size=font_size)


def statistic_xor_distribution(
    df: pd.DataFrame,
    leading_output_path: str,
    trailing_output_path: str,
    logical_type: str,
    font_size: int = 18,
) -> None:
    leading_all: list[int] = []
    trailing_all: list[int] = []
    for column in df.columns:
        column_data = df[column].tolist()
        leading, trailing = xor_zero_distributions_for_values(column_data)
        leading_all.extend(leading)
        trailing_all.extend(trailing)

    leading_counts = defaultdict(int)
    trailing_counts = defaultdict(int)
    for val in leading_all:
        leading_counts[int(val)] += 1
    for val in trailing_all:
        trailing_counts[int(val)] += 1

    _plot_xor_zero_distribution(
        leading_counts,
        output_path=leading_output_path,
        xlabel='Number of Leading Zeros',
        font_size=font_size,
    )
    _plot_xor_zero_distribution(
        trailing_counts,
        output_path=trailing_output_path,
        xlabel='Number of Trailing Zeros',
        font_size=font_size,
    )


def statistic_scaling(df: pd.DataFrame, output_path: str, font_size: int = 18):
    """Backward-compatible wrapper (float scaling bit-width chart)."""
    statistic_bit_width_distribution(df, output_path, 'float', font_size=font_size)


def statistic_xor(df: pd.DataFrame, leading_output_path: str, trailing_output_path: str, font_size: int = 18):
    """Backward-compatible wrapper (64-bit XOR charts, float logical type)."""
    statistic_xor_distribution(
        df,
        leading_output_path,
        trailing_output_path,
        'float',
        font_size=font_size,
    )


def statistic_df(
    df: pd.DataFrame,
    output_path: str,
    *,
    column_index_by_name: dict[str, int] | None = None,
):
    results = []

    for seq_idx, column in enumerate(df.columns):
        series = pd.to_numeric(df[column], errors='coerce')
        finite_series = series[np.isfinite(series)]
        if len(finite_series) < 2:
            continue
        column_data = finite_series.tolist()
        sequence_with_gaps = series.tolist()

        column_index = seq_idx
        if column_index_by_name is not None:
            try:
                column_index = int(column_index_by_name.get(str(column), seq_idx))
            except (TypeError, ValueError):
                column_index = seq_idx

        results.append({
            'column_index': column_index,
            'column_name': column,
            'mean': float(np.nanmean(column_data)),
            'standard_deviation': float(np.nanstd(column_data)),
            'spread': float(np.nanmax(column_data) - np.nanmin(column_data)),
            'entropy': float(byte_entropy(column_data)),
            'value_entropy': float(value_entropy(column_data)),
            'repeat': float(repeat_words(sequence_with_gaps)),
            'sortedness': float(sortedness(sequence_with_gaps)),
        })

    pd.DataFrame(results).to_csv(output_path, index=False, lineterminator='\n')
