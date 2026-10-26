import time

import gzip
import lzma

from .benchmark_loss import accumulate_byte_loss
from .benchmark_wire import column_original_size, pack_float64_le
from .native_lz_family import compress_native_lz_family, decompress_native_lz_family


def test_common_compression(data: bytes, algorithm: str):
    loss_acc = [0.0, 0.0]
    native_key = {
        'brotli': 'brotli',
        'lz4-framed': 'lz4-framed',
        'lz4': 'lz4-framed',
        'snappy-framed': 'snappy-framed',
        'snappy': 'snappy-framed',
    }.get(algorithm)

    start_time = int(time.perf_counter() * 1e9)
    if native_key is None:
        if algorithm not in {'gz', 'gzip'}:
            raise KeyError(algorithm)
        compressed_data = gzip.compress(data)
    else:
        compressed_data = compress_native_lz_family(data, native_key)
    compression_time = int(time.perf_counter() * 1e9) - start_time

    start_time = int(time.perf_counter() * 1e9)
    if native_key is None:
        decompressed_data = gzip.decompress(compressed_data)
    else:
        decompressed_data = decompress_native_lz_family(compressed_data, native_key, len(data))
    decompression_time = int(time.perf_counter() * 1e9) - start_time

    accumulate_byte_loss(data, decompressed_data, loss_acc)

    return [
        len(data),
        len(compressed_data),
        compression_time,
        decompression_time,
        loss_acc[0],
        loss_acc[1],
    ]


def test_fpzip_compression(data: list):
    loss_acc = [0.0, 0.0]
    payload = pack_float64_le([float(x) for x in data])
    start_time = int(time.perf_counter() * 1e9)
    compressed_data = compress_native_lz_family(payload, "fpzip")
    compression_time = int(time.perf_counter() * 1e9) - start_time

    start_time = int(time.perf_counter() * 1e9)
    decompressed_data = decompress_native_lz_family(compressed_data, "fpzip", len(payload))
    decompression_time = int(time.perf_counter() * 1e9) - start_time

    accumulate_byte_loss(payload, decompressed_data, loss_acc)

    return [
        column_original_size(len(data), "float"),
        len(compressed_data),
        compression_time,
        decompression_time,
        loss_acc[0],
        loss_acc[1],
    ]


def test_lzma_compression(data: bytes, level: int = 0):
    # lzma2
    filters = [{'id': lzma.FILTER_LZMA2, 'preset': level}]
    compressor = lzma.LZMACompressor(format=lzma.FORMAT_XZ, filters=filters)
    loss_acc = [0.0, 0.0]

    start_time = int(time.perf_counter() * 1e9)
    compressed_data = compressor.compress(data)
    compressed_data += compressor.flush()
    compression_time = int(time.perf_counter() * 1e9) - start_time

    decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    start_time = int(time.perf_counter() * 1e9)
    decompressed_data = decompressor.decompress(compressed_data)
    decompression_time = int(time.perf_counter() * 1e9) - start_time

    accumulate_byte_loss(data, decompressed_data, loss_acc)

    return [
        len(data),
        len(compressed_data),
        compression_time,
        decompression_time,
        loss_acc[0],
        loss_acc[1],
    ]


def test_zstd_compression(data: bytes, level: int = 3):
    loss_acc = [0.0, 0.0]
    algorithm = f"zstd_level_{int(level)}"

    start_time = int(time.perf_counter() * 1e9)
    compressed_data = compress_native_lz_family(data, algorithm)
    compression_time = int(time.perf_counter() * 1e9) - start_time

    start_time = int(time.perf_counter() * 1e9)
    decompressed_data = decompress_native_lz_family(compressed_data, algorithm, len(data))
    decompression_time = int(time.perf_counter() * 1e9) - start_time

    accumulate_byte_loss(data, decompressed_data, loss_acc)

    return [
        len(data),
        len(compressed_data),
        compression_time,
        decompression_time,
        loss_acc[0],
        loss_acc[1],
    ]


def main():
    pass


if __name__ == '__main__':
    main()
