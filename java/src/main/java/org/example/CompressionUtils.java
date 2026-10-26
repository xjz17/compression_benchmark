package org.example;

import com.aayushatharva.brotli4j.Brotli4jLoader;
import com.aayushatharva.brotli4j.decoder.DecoderJNI;
import com.aayushatharva.brotli4j.decoder.DirectDecompress;
import org.apache.commons.compress.compressors.*;
import org.apache.commons.compress.compressors.zstandard.ZstdCompressorInputStream;
import org.apache.commons.compress.compressors.zstandard.ZstdCompressorOutputStream;
import org.apache.tsfile.encoding.decoder.Decoder;
import org.apache.tsfile.encoding.encoder.Encoder;
import org.apache.tsfile.encoding.encoder.TSEncodingBuilder;
import org.apache.tsfile.enums.TSDataType;
import org.apache.tsfile.file.metadata.enums.TSEncoding;
import org.tukaani.xz.LZMA2Options;
import org.tukaani.xz.XZInputStream;
import org.tukaani.xz.XZOutputStream;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CompressionUtils {

    private static int clampTs2DiffMaxPointForDouble(List<?> data, int requestedMaxPoint) {
        int normalized = Math.max(0, requestedMaxPoint);
        if (data == null || data.isEmpty()) {
            return normalized;
        }

        double maxAbs = 0.0;
        for (Object value : data) {
            if (!(value instanceof Number)) {
                continue;
            }
            double v = ((Number) value).doubleValue();
            if (!Double.isFinite(v)) {
                continue;
            }
            double abs = Math.abs(v);
            if (abs > maxAbs) {
                maxAbs = abs;
            }
        }

        if (maxAbs <= 0.0) {
            return normalized;
        }
        double ratio = (double) Long.MAX_VALUE / maxAbs;
        if (Double.isNaN(ratio) || ratio <= 0.0) {
            return 0;
        }
        if (Double.isInfinite(ratio)) {
            return normalized;
        }

        int safeUpper = (int) Math.floor(Math.log10(ratio));
        safeUpper = Math.max(0, safeUpper);
        return Math.min(normalized, safeUpper);
    }

    public static void testCommonCompression(
            byte[] data,
            String algo,
            long[] result)
            throws IOException {
        testCommonCompression(data, algo, result, null);
    }

    public static void testCommonCompression(
            byte[] data,
            String algo,
            long[] result,
            double[] lossAcc)
            throws IOException {
        String algorithm = algo.toLowerCase().trim();

        byte[] compressedData;

        long startTime = System.nanoTime();

        try (ByteArrayOutputStream byteOut = new ByteArrayOutputStream();
                CompressorOutputStream<?> compressorOut = new CompressorStreamFactory()
                        .createCompressorOutputStream(algorithm, byteOut)) {

            compressorOut.write(data);
            compressorOut.flush();
            compressorOut.close();
            compressedData = byteOut.toByteArray();
        } catch (CompressorException e) {
            throw new IOException("Unsupported compression algorithm: " + algorithm, e);
        }

        long endTime = System.nanoTime();
        long compressionTime = endTime - startTime;

        startTime = System.nanoTime();
        byte[] decompressedData;
        try (ByteArrayInputStream byteIn = new ByteArrayInputStream(compressedData);
                CompressorInputStream compressorIn = new CompressorStreamFactory()
                        .createCompressorInputStream(algorithm, byteIn)) {

            ByteArrayOutputStream byteOut = new ByteArrayOutputStream();
            byte[] buffer = new byte[1024];
            int len;
            while ((len = compressorIn.read(buffer)) != -1) {
                byteOut.write(buffer, 0, len);
            }
            decompressedData = byteOut.toByteArray();
        } catch (CompressorException e) {
            throw new IOException("Unsupported compression algorithm: " + algorithm, e);
        }

        endTime = System.nanoTime();
        long decompressionTime = endTime - startTime;

        result[0] = data.length;
        result[1] = compressedData.length;
        result[2] = compressionTime;
        result[3] = decompressionTime;
        BenchmarkLoss.accumulateByteLoss(data, decompressedData, lossAcc);

    }

    public static void testLZMACompression(
            byte[] data,
            int lzmaLevel,
            long[] result)
            throws IOException {
        testLZMACompression(data, lzmaLevel, result, null);
    }

    public static void testLZMACompression(
            byte[] data,
            int lzmaLevel,
            long[] result,
            double[] lossAcc)
            throws IOException {
        byte[] compressedData;

        long startTime = System.nanoTime();

        ByteArrayOutputStream baos = new ByteArrayOutputStream();

        LZMA2Options options = new LZMA2Options();
        options.setPreset(lzmaLevel);

        try (XZOutputStream lzmaOS = new XZOutputStream(baos, options)) {
            lzmaOS.write(data);
        }

        compressedData = baos.toByteArray();
        long endTime = System.nanoTime();
        long compressionTime = endTime - startTime;

        startTime = System.nanoTime();
        ByteArrayInputStream bais = new ByteArrayInputStream(compressedData);
        ByteArrayOutputStream decompressedBaos = new ByteArrayOutputStream();
        try (XZInputStream lzmaIS = new XZInputStream(bais)) {
            byte[] buffer = new byte[1024];
            int len;
            while ((len = lzmaIS.read(buffer)) != -1) {
                decompressedBaos.write(buffer, 0, len);
            }
        }
        byte[] decompressedData = decompressedBaos.toByteArray();
        endTime = System.nanoTime();
        long decompressionTime = endTime - startTime;

        result[0] = data.length;
        result[1] = compressedData.length;
        result[2] = compressionTime;
        result[3] = decompressionTime;
        BenchmarkLoss.accumulateByteLoss(data, decompressedData, lossAcc);
    }

    public static void testZstdCompression(
            byte[] data,
            int zstdLevel,
            long[] result)
            throws IOException {
        testZstdCompression(data, zstdLevel, result, null);
    }

    public static void testZstdCompression(
            byte[] data,
            int zstdLevel,
            long[] result,
            double[] lossAcc)
            throws IOException {
        byte[] compressedData;

        long startTime = System.nanoTime();

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ZstdCompressorOutputStream zstdOut = new ZstdCompressorOutputStream(baos, zstdLevel)) {
            zstdOut.write(data);
            zstdOut.flush();
        }
        compressedData = baos.toByteArray();

        long endTime = System.nanoTime();
        long compressionTime = endTime - startTime;

        startTime = System.nanoTime();
        byte[] decompressedData;

        ByteArrayInputStream bais = new ByteArrayInputStream(compressedData);
        try (ZstdCompressorInputStream zstdIn = new ZstdCompressorInputStream(bais)) {
            ByteArrayOutputStream decompressedBaos = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int len;
            while ((len = zstdIn.read(buffer)) != -1) {
                decompressedBaos.write(buffer, 0, len);
            }
            decompressedData = decompressedBaos.toByteArray();
        }

        endTime = System.nanoTime();
        long decompressionTime = endTime - startTime;

        result[0] = data.length;
        result[1] = compressedData.length;
        result[2] = compressionTime;
        result[3] = decompressionTime;
        BenchmarkLoss.accumulateByteLoss(data, decompressedData, lossAcc);
    }

    /** Order-0 rANS entropy coding (see {@link AnsEntropyRans}). */
    public static void testAnsEntropyCompression(byte[] data, long[] result) {
        testAnsEntropyCompression(data, result, null);
    }

    public static void testAnsEntropyCompression(byte[] data, long[] result, double[] lossAcc) {
        AnsEntropyRans.bench(data, result, lossAcc);
    }

    /** Byte Huffman coding (see {@link HuffmanBytes}). */
    public static void testHuffmanCompression(byte[] data, long[] result) {
        testHuffmanCompression(data, result, null);
    }

    public static void testHuffmanCompression(byte[] data, long[] result, double[] lossAcc) {
        HuffmanBytes.bench(data, result, lossAcc);
    }

    /** Static order-0 range coding (see {@link ArithmeticBytes}). */
    public static void testArithmeticCompression(byte[] data, long[] result) {
        testArithmeticCompression(data, result, null);
    }

    public static void testArithmeticCompression(byte[] data, long[] result, double[] lossAcc) {
        ArithmeticBytes.bench(data, result, lossAcc);
    }

    public static void testBrotliCompression(
            byte[] data,
            long[] result)
            throws IOException {
        testBrotliCompression(data, result, (double[]) null);
    }

    public static void testBrotliCompression(
            byte[] data,
            long[] result,
            double[] lossAcc)
            throws IOException {
        if (!Brotli4jLoader.isAvailable()) {
            throw new IOException("Brotli4j native library is not available");
        }
        byte[] compressedData;

        long startTime = System.nanoTime();

        try {
            compressedData = com.aayushatharva.brotli4j.encoder.Encoder.compress(data);
        } catch (Exception e) {
            throw new IOException("Brotli compression failed", e);
        }

        long endTime = System.nanoTime();
        long compressionTime = endTime - startTime;

        startTime = System.nanoTime();
        byte[] decompressedData;
        try {
            DirectDecompress decompressor = com.aayushatharva.brotli4j.decoder.Decoder.decompress(compressedData);
            if (decompressor.getResultStatus() != DecoderJNI.Status.DONE) {
                throw new IOException("Brotli decompression failed with status: " + decompressor.getResultStatus());
            }
            decompressedData = decompressor.getDecompressedData();
        } catch (Exception e) {
            throw new IOException("Brotli decompression failed", e);
        }

        endTime = System.nanoTime();
        long decompressionTime = endTime - startTime;

        result[0] = data.length;
        result[1] = compressedData.length;
        result[2] = compressionTime;
        result[3] = decompressionTime;
        BenchmarkLoss.accumulateByteLoss(data, decompressedData, lossAcc);
    }

    public static void testBrotliCompression(
            byte[] data,
            long[] result,
            String outputPath)
            throws IOException {
        testBrotliCompression(data, result, outputPath, null);
    }

    public static void testBrotliCompression(
            byte[] data,
            long[] result,
            String outputPath,
            double[] lossAcc)
            throws IOException {
        if (!Brotli4jLoader.isAvailable()) {
            throw new IOException("Brotli4j native library is not available");
        }
        byte[] compressedData;

        long startTime = System.nanoTime();

        try {
            compressedData = com.aayushatharva.brotli4j.encoder.Encoder.compress(data);
        } catch (Exception e) {
            throw new IOException("Brotli compression failed", e);
        }

        // 写入压缩数据到文件
        try (FileOutputStream fos = new FileOutputStream(outputPath, true)) {
            fos.write(compressedData);
        } catch (IOException e) {
            throw new IOException("Failed to write compressed data to file", e);
        }

        long endTime = System.nanoTime();
        long compressionTime = endTime - startTime;

        startTime = System.nanoTime();
        byte[] decompressedData;
        try {
            DirectDecompress decompressor = com.aayushatharva.brotli4j.decoder.Decoder.decompress(compressedData);
            if (decompressor.getResultStatus() != DecoderJNI.Status.DONE) {
                throw new IOException("Brotli decompression failed with status: " + decompressor.getResultStatus());
            }
            decompressedData = decompressor.getDecompressedData();
        } catch (Exception e) {
            throw new IOException("Brotli decompression failed", e);
        }

        endTime = System.nanoTime();
        long decompressionTime = endTime - startTime;

        result[0] = data.length;
        result[1] = compressedData.length;
        result[2] = compressionTime;
        result[3] = decompressionTime;
        BenchmarkLoss.accumulateByteLoss(data, decompressedData, lossAcc);
    }

    public static <T> void testEncoding(
            TSDataType dataType,
            TSEncoding encoding,
            List<T> data,
            long[] result) throws IOException {
        testEncoding(dataType, encoding, data, 0, result, null);
    }

    public static <T> void testEncoding(
            TSDataType dataType,
            TSEncoding encoding,
            List<T> data,
            long[] result,
            double[] lossAcc) throws IOException {
        if (dataType == TSDataType.DOUBLE
                && (encoding == TSEncoding.RLE
                        || encoding == TSEncoding.SPRINTZ
                        || encoding == TSEncoding.RLBE)) {
            java.util.ArrayList<Integer> rawLimbs = new java.util.ArrayList<>(data.size() * 2);
            for (T value : data) {
                long bits = Double.doubleToRawLongBits((Double) value);
                rawLimbs.add((int) bits);
                rawLimbs.add((int) (bits >>> 32));
            }
            double[] limbLoss = new double[2];
            testEncoding(TSDataType.INT32, encoding, rawLimbs, 0, result, limbLoss);
            if (limbLoss[0] == 0.0) {
                if (lossAcc != null) lossAcc[1] += data.size();
                return;
            }
            long fallbackEncodeStart = System.nanoTime();
            ByteBuffer raw = ByteBuffer.allocate(1 + data.size() * Double.BYTES);
            raw.put((byte) 0);
            for (T value : data) raw.putLong(Double.doubleToRawLongBits((Double) value));
            byte[] fallback = raw.array();
            long fallbackEncodeEnd = System.nanoTime();
            long fallbackDecodeStart = System.nanoTime();
            ByteBuffer decodedRaw = ByteBuffer.wrap(fallback);
            decodedRaw.get();
            double[] decoded = new double[data.size()];
            for (int i = 0; i < decoded.length; i++) decoded[i] = Double.longBitsToDouble(decodedRaw.getLong());
            long fallbackDecodeEnd = System.nanoTime();
            result[0] = (long) data.size() * Double.BYTES;
            result[1] = fallback.length;
            result[2] += fallbackEncodeEnd - fallbackEncodeStart;
            result[3] += fallbackDecodeEnd - fallbackDecodeStart;
            @SuppressWarnings("unchecked")
            List<Double> doubleData = (List<Double>) data;
            BenchmarkLoss.accumulateDoubleLoss(doubleData, decoded, decoded.length, lossAcc);
            return;
        }
        testEncoding(dataType, encoding, data, 0, result, lossAcc);
    }

    public static <T> void testEncoding(
            TSDataType dataType,
            TSEncoding encoding,
            List<T> data,
            int maxPointNumber,
            long[] result) throws IOException {
        testEncoding(dataType, encoding, data, maxPointNumber, result, null);
    }

    public static <T> void testEncoding(
            TSDataType dataType,
            TSEncoding encoding,
            List<T> data,
            int maxPointNumber,
            long[] result,
            double[] lossAcc) throws IOException {
        if (encoding == TSEncoding.SPRINTZ && dataType == TSDataType.INT64) {
            @SuppressWarnings("unchecked")
            List<Long> longData = (List<Long>) data;
            long startTime = System.nanoTime();
            byte[] encodedData = Sprintz64Codec.encodeLongs(longData);
            long endTime = System.nanoTime();
            long compressionTime = endTime - startTime;

            startTime = System.nanoTime();
            long[] decoded = Sprintz64Codec.decodeLongs(encodedData);
            endTime = System.nanoTime();

            result[0] = (long) data.size() * BenchmarkWireFormat.INT_BYTES;
            result[1] = encodedData.length;
            result[2] = compressionTime;
            result[3] = endTime - startTime;
            BenchmarkLoss.accumulateLongLoss(longData, decoded, decoded.length, lossAcc);
            return;
        }
        if (encoding == TSEncoding.SPRINTZ && dataType == TSDataType.DOUBLE) {
            @SuppressWarnings("unchecked")
            List<Double> doubleData = (List<Double>) data;
            java.util.ArrayList<Long> bits = new java.util.ArrayList<>(doubleData.size());
            for (Double value : doubleData) {
                bits.add(Double.doubleToRawLongBits(value == null ? 0.0 : value));
            }
            long startTime = System.nanoTime();
            byte[] encodedData = Sprintz64Codec.encodeLongs(bits);
            long endTime = System.nanoTime();
            long compressionTime = endTime - startTime;

            startTime = System.nanoTime();
            long[] decodedBits = Sprintz64Codec.decodeLongs(encodedData);
            double[] decoded = new double[decodedBits.length];
            for (int i = 0; i < decodedBits.length; i++) {
                decoded[i] = Double.longBitsToDouble(decodedBits[i]);
            }
            endTime = System.nanoTime();

            result[0] = (long) data.size() * BenchmarkWireFormat.FLOAT_BYTES;
            result[1] = encodedData.length;
            result[2] = compressionTime;
            result[3] = endTime - startTime;
            BenchmarkLoss.accumulateDoubleLoss(doubleData, decoded, decoded.length, lossAcc);
            return;
        }
        if (dataType == TSDataType.DOUBLE
                && (encoding == TSEncoding.RLE
                        || encoding == TSEncoding.SPRINTZ
                        || encoding == TSEncoding.RLBE)) {
            testEncoding(dataType, encoding, data, result, lossAcc);
            return;
        }
        TSEncodingBuilder builder = TSEncodingBuilder.getEncodingBuilder(encoding);
        if (dataType == TSDataType.FLOAT || dataType == TSDataType.DOUBLE) {
            int effectiveMaxPoint = Math.max(0, maxPointNumber);
            // Avoid TS_2DIFF double fallback-to-round(v) on long overflow.
            if (encoding == TSEncoding.TS_2DIFF && dataType == TSDataType.DOUBLE) {
                effectiveMaxPoint = clampTs2DiffMaxPointForDouble(data, effectiveMaxPoint);
            }
            Map<String, String> props = new HashMap<>();
            props.put(Encoder.MAX_POINT_NUMBER, String.valueOf(effectiveMaxPoint));
            builder.initFromProps(props);
        }
        Encoder encoder = builder.getEncoder(dataType);

        long startTime = System.nanoTime();

        ByteArrayOutputStream baos = new ByteArrayOutputStream();

        for (T value : data) {
            if (dataType == TSDataType.INT32) {
                encoder.encode((Integer) value, baos);
            } else if (dataType == TSDataType.INT64) {
                encoder.encode((Long) value, baos);
            } else if (dataType == TSDataType.FLOAT) {
                encoder.encode((Float) value, baos);
            } else if (dataType == TSDataType.DOUBLE) {
                encoder.encode((Double) value, baos);
            } else {
                throw new IllegalArgumentException("Unsupported data type: " + dataType);
            }
        }
        encoder.flush(baos);

        byte[] encodedData = baos.toByteArray();

        long endTime = System.nanoTime();
        long compressionTime = endTime - startTime;

        startTime = System.nanoTime();

        ByteBuffer buffer = ByteBuffer.wrap(encodedData);

        Decoder decoder = Decoder.getDecoderByType(encoding, dataType);

        int dataSize = data.size();
        if (dataType == TSDataType.INT32) {
            int[] decodedArr = new int[dataSize];
            int idx = 0;
            while (decoder.hasNext(buffer) && idx < dataSize) {
                decodedArr[idx++] = decoder.readInt(buffer);
            }
            if (lossAcc != null && lossAcc.length >= 2) {
                for (int i = 0; i < Math.min(dataSize, idx); i++) {
                    int o = (Integer) data.get(i);
                    int d = decodedArr[i];
                    if (o != d) lossAcc[0] += 1.0;
                }
                lossAcc[0] += Math.abs(dataSize - idx);
                lossAcc[1] += Math.max(dataSize, idx);
            }
        } else if (dataType == TSDataType.INT64) {
            long[] decodedArr = new long[dataSize];
            int idx = 0;
            while (decoder.hasNext(buffer) && idx < dataSize) {
                decodedArr[idx++] = decoder.readLong(buffer);
            }
            @SuppressWarnings("unchecked")
            List<Long> longData = (List<Long>) data;
            BenchmarkLoss.accumulateLongLoss(longData, decodedArr, idx, lossAcc);
        } else if (dataType == TSDataType.FLOAT) {
            float[] decodedArr = new float[dataSize];
            int idx = 0;
            while (decoder.hasNext(buffer) && idx < dataSize) {
                decodedArr[idx++] = decoder.readFloat(buffer);
            }
            if (lossAcc != null && lossAcc.length >= 2) {
                for (int i = 0; i < Math.min(dataSize, idx); i++) {
                    float o = (Float) data.get(i);
                    float d = decodedArr[i];
                    if (Float.floatToRawIntBits(o) != Float.floatToRawIntBits(d)) lossAcc[0] += 1.0;
                }
                lossAcc[0] += Math.abs(dataSize - idx);
                lossAcc[1] += Math.max(dataSize, idx);
            }
        } else if (dataType == TSDataType.DOUBLE) {
            double[] decodedArr = new double[dataSize];
            int idx = 0;
            while (decoder.hasNext(buffer) && idx < dataSize) {
                decodedArr[idx++] = decoder.readDouble(buffer);
            }
            @SuppressWarnings("unchecked")
            List<Double> doubleData = (List<Double>) data;
            BenchmarkLoss.accumulateDoubleLoss(doubleData, decodedArr, idx, lossAcc);
        } else {
            throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }

        endTime = System.nanoTime();
        long decompressionTime = endTime - startTime;

        int valueBytes =
                dataType == TSDataType.INT32 || dataType == TSDataType.FLOAT
                        ? (dataType == TSDataType.INT32 ? Integer.BYTES : Float.BYTES)
                        : (dataType == TSDataType.INT64 ? BenchmarkWireFormat.INT_BYTES
                                : BenchmarkWireFormat.FLOAT_BYTES);
        result[0] = (long) data.size() * valueBytes;
        result[1] = encodedData.length;
        result[2] = compressionTime;
        result[3] = decompressionTime;
    }
}
