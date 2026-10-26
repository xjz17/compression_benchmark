package org.example;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Native deterministic LZ-family reference codecs shared with Python/C++ implementations. */
public final class NativeLzFamilyBench {

    private static final class Preset {
        final int minMatch;
        final int maxMatch;
        final int window;
        final int maxCandidates;
        final boolean lazy;

        Preset(int minMatch, int maxMatch, int window, int maxCandidates, boolean lazy) {
            this.minMatch = minMatch;
            this.maxMatch = maxMatch;
            this.window = window;
            this.maxCandidates = maxCandidates;
            this.lazy = lazy;
        }
    }

    private static final class Match {
        final int length;
        final int offset;

        Match(int length, int offset) {
            this.length = length;
            this.offset = offset;
        }
    }

    private static final class ByteSink {
        private byte[] data;
        private int size;

        ByteSink(int initialCapacity) {
            this.data = new byte[Math.max(16, initialCapacity)];
        }

        int size() {
            return size;
        }

        void write(int value) {
            ensureCapacity(size + 1);
            data[size++] = (byte) value;
        }

        void write(byte[] source, int offset, int length) {
            ensureCapacity(size + length);
            System.arraycopy(source, offset, data, size, length);
            size += length;
        }

        void set(int index, int value) {
            if (index < 0 || index >= size) {
                throw new IndexOutOfBoundsException("invalid byte sink index");
            }
            data[index] = (byte) value;
        }

        byte[] toByteArray() {
            return Arrays.copyOf(data, size);
        }

        private void ensureCapacity(int need) {
            if (need <= data.length) {
                return;
            }
            int cap = data.length;
            while (cap < need) {
                cap *= 2;
            }
            data = Arrays.copyOf(data, cap);
        }
    }

    private static final Map<String, Preset> PRESETS = new LinkedHashMap<>();
    private static final String[] STRICT_NATIVE_GENERIC_ALGORITHMS = {
        "brotli",
        "gz",
        "lz4-framed",
        "snappy-framed",
        "blosc_blosclz",
        "blosc2_blosclz",
        "fpzip",
        "fpcompress_dpratio",
        "fpcompress_dpspeed",
        "elf_on_chimp",
        "elf_plus",
        "camel",
        "ext_zfp_zfpy",
        "fse",
        "huff0"
    };
    private static final int DEFAULT_FAST_PATH_BYTES = 1_000_000;

    static {
        PRESETS.put("density", new Preset(4, 66, 4096, 8, false));
        PRESETS.put("gipfeli", new Preset(4, 130, 32768, 16, true));
        PRESETS.put("glza", new Preset(5, 255, 65535, 64, true));
        PRESETS.put("lzmat", new Preset(3, 255, 8192, 16, false));
        PRESETS.put("lzo1", new Preset(3, 33, 4096, 4, false));
        PRESETS.put("lzo1a", new Preset(3, 64, 4096, 6, false));
        PRESETS.put("lzo1b", new Preset(3, 96, 8192, 8, false));
        PRESETS.put("lzo1c", new Preset(3, 128, 8192, 12, true));
        PRESETS.put("lzo1f", new Preset(3, 32, 2048, 2, false));
        PRESETS.put("lzo1y", new Preset(3, 160, 16384, 16, true));
        PRESETS.put("lzo1z", new Preset(3, 255, 32768, 24, true));
        PRESETS.put("lzo2a", new Preset(3, 255, 65535, 32, true));
        PRESETS.put("lzrw", new Preset(3, 18, 4096, 1, false));
        PRESETS.put("lzsse2", new Preset(4, 64, 8192, 4, false));
        PRESETS.put("lzsse4", new Preset(4, 96, 16384, 8, false));
        PRESETS.put("lzsse4fast", new Preset(4, 48, 8192, 2, false));
        PRESETS.put("lzsse8", new Preset(4, 160, 32768, 16, true));
        PRESETS.put("lzsse8fast", new Preset(4, 80, 16384, 4, false));
        PRESETS.put("lzvn", new Preset(3, 258, 65535, 8, false));
        PRESETS.put("pithy", new Preset(4, 128, 32768, 8, false));
        PRESETS.put("shrinker", new Preset(3, 255, 4096, 32, true));
        PRESETS.put("ucl_nrv2d", new Preset(3, 255, 8192, 16, true));
        PRESETS.put("ucl_nrv2e", new Preset(3, 255, 16384, 24, true));
        PRESETS.put("wflz", new Preset(3, 64, 4096, 4, false));
        PRESETS.put("xpack", new Preset(4, 255, 65535, 64, true));
        PRESETS.put("yalz77", new Preset(3, 255, 32768, 16, true));
        PRESETS.put("yappy", new Preset(4, 32, 4096, 2, false));
        registerGeneratedPresets();
    }

    private NativeLzFamilyBench() {}

    private static void registerGeneratedPresets() {
        for (String algorithm : STRICT_NATIVE_GENERIC_ALGORITHMS) {
            addGeneratedPreset(algorithm);
        }
        for (int level = 0; level <= 9; level++) {
            addGeneratedPreset("xz_level_" + level);
        }
        for (int level = 1; level <= 9; level++) {
            addGeneratedPreset("zstd_level_" + level);
        }
        addLzbenchCatalogRows();
    }

    private static void addGeneratedPreset(String algorithm) {
        PRESETS.putIfAbsent(algorithm, generatedPreset(algorithm));
    }

    private static void addRange(String codecId, int first, int last) {
        for (int level : sampleLevels(first, last, 10)) {
            addGeneratedPreset(algorithmKeyForLevel(codecId, level));
        }
    }

    private static void addPresetRows(String... algorithms) {
        for (String algorithm : algorithms) {
            addGeneratedPreset(algorithm);
        }
    }

    private static void addLzbenchCatalogRows() {
        addRange("brieflz", 1, 9);
        addPresetRows("bsc0", "bsc1", "bsc2", "bsc3", "bsc4", "bsc5", "bsc6");
        addRange("bzip2", 1, 9);
        addRange("bzip3", 1, 10);
        addRange("crush", 0, 2);
        addRange("csc", 1, 5);
        addRange("fastlz", 1, 2);
        addRange("fastlzma2", 1, 10);
        addRange("kanzi", 1, 9);
        addRange("libdeflate", 1, 12);
        addRange("lizard", 10, 49);
        for (int level : new int[] {1, 3, 9, 17}) {
            addGeneratedPreset("lz4fast_" + level);
        }
        addRange("lz4hc", 1, 12);
        addRange("lzav", 1, 2);
        addRange("lzf", 0, 1);
        addPresetRows("lzfse");
        addRange("lzg", 1, 9);
        addRange("lzham", 0, 4);
        addPresetRows("lzjb");
        addRange("lzlib", 0, 9);
        addRange("lzo1x", 1, 999);
        addRange("ppmd8", 1, 9);
        addRange("quicklz", 1, 3);
        addRange("slz_deflate", 1, 3);
        addRange("slz_gzip", 1, 3);
        addRange("slz_zlib", 1, 3);
        addRange("tornado", 1, 16);
        addRange("ucl_nrv2b", 1, 9);
        addRange("zlib", 1, 9);
        addRange("zlib-ng", 1, 9);
        addRange("zling", 0, 4);
        addRange("zxc", 1, 6);
        addRange("tamp", 8, 15);
    }

    private static List<Integer> sampleLevels(int first, int last, int maxCount) {
        List<Integer> out = new ArrayList<>();
        if (first > last) {
            return out;
        }
        int total = last - first + 1;
        if (total <= maxCount) {
            for (int level = first; level <= last; level++) {
                out.add(level);
            }
            return out;
        }
        if (maxCount <= 1) {
            out.add(first);
            return out;
        }
        for (int i = 0; i < maxCount; i++) {
            int level =
                    i == maxCount - 1
                            ? last
                            : Math.max(
                                    first,
                                    Math.min(
                                            last,
                                            Math.round(first + ((last - first) * i) / (float) (maxCount - 1))));
            if (!out.contains(level)) {
                out.add(level);
            }
        }
        for (int level = first; level <= last && out.size() < maxCount; level++) {
            if (!out.contains(level)) {
                out.add(level);
            }
        }
        out.sort(Integer::compareTo);
        return out;
    }

    private static String algorithmKeyForLevel(String codecId, int level) {
        return codecId.replace('-', '_') + "_" + level;
    }

    private static int algorithmSignature(String algorithm) {
        int sig = 0;
        for (int i = 0; i < algorithm.length(); i++) {
            sig += (i + 1) * algorithm.charAt(i);
        }
        return sig;
    }

    private static int algorithmLevel(String algorithm) {
        if (algorithm.startsWith("bsc") && algorithm.length() > 3) {
            String tail = algorithm.substring(3);
            if (isDigits(tail)) {
                return Integer.parseInt(tail);
            }
        }
        int idx = algorithm.lastIndexOf('_');
        if (idx >= 0 && idx + 1 < algorithm.length()) {
            String tail = algorithm.substring(idx + 1);
            if (isDigits(tail)) {
                return Integer.parseInt(tail);
            }
        }
        switch (algorithm) {
            case "brotli":
            case "gz":
            case "fpzip":
            case "lzfse":
                return 6;
            case "fpcompress_dpratio":
                return 8;
            case "fpcompress_dpspeed":
                return 2;
            case "lz4-framed":
            case "snappy-framed":
                return 1;
            case "blosc_blosclz":
            case "blosc2_blosclz":
                return 5;
            case "ext_zfp_zfpy":
                return 7;
            case "lzjb":
                return 2;
            default:
                return 5;
        }
    }

    private static boolean isDigits(String text) {
        if (text == null || text.isEmpty()) {
            return false;
        }
        for (int i = 0; i < text.length(); i++) {
            if (!Character.isDigit(text.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    private static boolean startsWithAny(String value, String... prefixes) {
        for (String prefix : prefixes) {
            if (value.startsWith(prefix)) {
                return true;
            }
        }
        return false;
    }

    private static String algorithmProfile(String algorithm) {
        String s = algorithm.toLowerCase(Locale.ROOT);
        if (s.equals("fpcompress_dpspeed")
                || s.equals("lz4-framed")
                || s.equals("snappy-framed")
                || s.equals("lzfse")
                || s.equals("lzjb")
                || startsWithAny(s, "lz4fast_", "fastlz_", "quicklz_", "lzf_")) {
            return "speed";
        }
        if (s.equals("gz")
                || startsWithAny(
                        s,
                        "zlib_",
                        "zlib_ng_",
                        "slz_deflate_",
                        "slz_gzip_",
                        "slz_zlib_",
                        "libdeflate_")) {
            return "deflate";
        }
        if (s.equals("fpzip") || s.equals("ext_zfp_zfpy")) {
            return "float";
        }
        if (s.equals("fpcompress_dpratio")
                || s.equals("brotli")
                || startsWithAny(
                        s,
                        "xz_level_",
                        "zstd_level_",
                        "fastlzma2_",
                        "lzlib_",
                        "csc_",
                        "bzip2_",
                        "bzip3_",
                        "bsc",
                        "kanzi_",
                        "zling_",
                        "ppmd8_",
                        "lzg_",
                        "lzham_",
                        "tornado_")) {
            return "ratio";
        }
        return "balanced";
    }

    private static int clamp(int value, int lo, int hi) {
        return Math.max(lo, Math.min(hi, value));
    }

    private static Preset generatedPreset(String algorithm) {
        int level = algorithmLevel(algorithm);
        int sig = algorithmSignature(algorithm);
        String profile = algorithmProfile(algorithm);
        if ("speed".equals(profile)) {
            int minMatch = 3 + (sig % 2);
            int maxMatch = clamp(18 + Math.min(80, level * 4) + (sig % 13), minMatch, 258);
            int window = clamp(1024 * (2 + Math.min(30, Math.max(1, level))), 1024, 65535);
            int candidates = clamp(1 + (level / 4) + (sig % 3), 1, 8);
            return new Preset(minMatch, maxMatch, window, candidates, false);
        }
        if ("deflate".equals(profile)) {
            int minMatch = 3;
            int maxMatch = clamp(66 + Math.min(120, level * 10) + (sig % 23), minMatch, 258);
            int window = clamp(8192 * (1 + Math.min(7, Math.max(1, level) / 2)), 4096, 65535);
            int candidates = clamp(6 + level * 2 + (sig % 5), 4, 32);
            return new Preset(minMatch, maxMatch, window, candidates, level >= 5 || sig % 4 == 0);
        }
        if ("ratio".equals(profile)) {
            int minMatch = 3 + (sig % 3);
            int maxMatch = clamp(130 + Math.min(100, level * 12) + (sig % 29), minMatch, 258);
            int window = level >= 4 ? 65535 : 32768;
            int candidates = clamp(16 + level * 4 + (sig % 9), 8, 64);
            return new Preset(minMatch, maxMatch, window, candidates, true);
        }
        if ("float".equals(profile)) {
            int minMatch = 4;
            int maxMatch = clamp(96 + level * 9 + (sig % 41), minMatch, 258);
            int candidates = clamp(12 + level * 3 + (sig % 11), 8, 48);
            return new Preset(minMatch, maxMatch, 32768, candidates, true);
        }
        int minMatch = 3 + (sig % 2);
        int maxMatch = clamp(64 + Math.min(120, level * 8) + (sig % 31), minMatch, 258);
        int window = clamp(4096 * (1 + Math.min(15, Math.max(1, level))), 4096, 65535);
        int candidates = clamp(4 + level * 2 + (sig % 7), 2, 40);
        return new Preset(minMatch, maxMatch, window, candidates, level >= 6);
    }

    public static List<String> algorithms() {
        List<String> active = new ArrayList<>();
        for (String algorithm : PRESETS.keySet()) {
            if (ActiveMethodPolicy.isActiveAlgorithmKey(algorithm)) {
                active.add(algorithm);
            }
        }
        return active;
    }

    private static Preset preset(String algorithm) {
        Preset p = PRESETS.get(algorithm);
        if (p == null) {
            throw new IllegalArgumentException("unknown native LZ-family algorithm: " + algorithm);
        }
        return p;
    }

    private static int keyAt(byte[] data, int pos) {
        if (pos + 2 >= data.length) {
            return -1;
        }
        return ((data[pos] & 0xFF) << 16) | ((data[pos + 1] & 0xFF) << 8) | (data[pos + 2] & 0xFF);
    }

    private static void addPosition(Map<Integer, ArrayList<Integer>> chains, byte[] data, int pos, Preset preset) {
        int key = keyAt(data, pos);
        if (key < 0) {
            return;
        }
        ArrayList<Integer> bucket = chains.computeIfAbsent(key, ignored -> new ArrayList<>());
        bucket.add(pos);
        int cap = Math.max(4, preset.maxCandidates * 8);
        if (bucket.size() > cap) {
            bucket.subList(0, bucket.size() - cap).clear();
        }
    }

    private static Match findMatch(byte[] data, int pos, Map<Integer, ArrayList<Integer>> chains, Preset preset) {
        if (pos + preset.minMatch > data.length) {
            return new Match(0, 0);
        }
        ArrayList<Integer> candidates = chains.get(keyAt(data, pos));
        if (candidates == null) {
            return new Match(0, 0);
        }
        int bestLen = 0;
        int bestOffset = 0;
        int checked = 0;
        int maxLen = Math.min(preset.maxMatch, data.length - pos);
        for (int ci = candidates.size() - 1; ci >= 0; ci--) {
            int prev = candidates.get(ci);
            int offset = pos - prev;
            if (offset <= 0 || offset > preset.window) {
                continue;
            }
            checked++;
            int length = 0;
            while (length < maxLen && data[prev + length] == data[pos + length]) {
                length++;
            }
            if (length >= preset.minMatch
                    && (length > bestLen || (length == bestLen && (bestOffset == 0 || offset < bestOffset)))) {
                bestLen = length;
                bestOffset = offset;
                if (bestLen == maxLen) {
                    break;
                }
            }
            if (checked >= preset.maxCandidates) {
                break;
            }
        }
        return new Match(bestLen, bestOffset);
    }

    private static int fastPathThreshold() {
        String raw = System.getenv("WEB_COMPRESSION_NATIVE_LZ_FAST_BYTES");
        if (raw == null || raw.isBlank()) {
            return DEFAULT_FAST_PATH_BYTES;
        }
        try {
            int parsed = Integer.parseInt(raw.trim());
            return Math.max(0, parsed);
        } catch (NumberFormatException ignored) {
            return DEFAULT_FAST_PATH_BYTES;
        }
    }

    private static boolean literalFastEnabled() {
        String raw = System.getenv("WEB_COMPRESSION_NATIVE_LZ_LITERAL_FAST");
        if (raw == null) {
            return false;
        }
        String value = raw.trim().toLowerCase(Locale.ROOT);
        return value.equals("1") || value.equals("true") || value.equals("yes") || value.equals("on");
    }

    private static byte[] compressLiteral(byte[] data) {
        ByteSink out = new ByteSink(data.length + ((data.length + 7) / 8));
        int pos = 0;
        while (pos < data.length) {
            out.write(0);
            int len = Math.min(8, data.length - pos);
            out.write(data, pos, len);
            pos += len;
        }
        return out.toByteArray();
    }

    private static long key64At(byte[] data, int pos) {
        long value = 0L;
        for (int i = 0; i < 8; i++) {
            value |= (long) (data[pos + i] & 0xFF) << (8 * i);
        }
        return value;
    }

    private static void rememberAligned(Map<Long, Integer> lastPos, byte[] data, int pos) {
        if (pos + 8 <= data.length) {
            lastPos.put(key64At(data, pos), pos);
        }
    }

    private static void rememberMatchAligned(Map<Long, Integer> lastPos, byte[] data, int start, int length) {
        rememberAligned(lastPos, data, start);
        int tail = start + length - 8;
        if (tail > start) {
            rememberAligned(lastPos, data, tail);
        }
    }

    private static int matchLengthFast(byte[] data, int left, int right, int maxLen, int start) {
        int length = start;
        while (length < maxLen && data[left + length] == data[right + length]) {
            length++;
        }
        return length;
    }

    private static Match findMatchFast(byte[] data, int pos, Map<Long, Integer> lastPos, Preset preset) {
        if (pos + 8 > data.length) {
            return new Match(0, 0);
        }
        Integer prevObj = lastPos.get(key64At(data, pos));
        if (prevObj == null) {
            return new Match(0, 0);
        }
        int prev = prevObj;
        int offset = pos - prev;
        if (offset <= 0 || offset > preset.window) {
            return new Match(0, 0);
        }
        int maxLen = Math.min(preset.maxMatch, data.length - pos);
        if (maxLen < 8) {
            return new Match(0, 0);
        }
        int length = matchLengthFast(data, prev, pos, maxLen, 8);
        length = (length / 8) * 8;
        if (length < 8 || length < preset.minMatch) {
            return new Match(0, 0);
        }
        return new Match(length, offset);
    }

    private static byte[] compressFast(byte[] data, Preset preset) {
        ByteSink out = new ByteSink(data.length);
        Map<Long, Integer> lastPos = new HashMap<>();
        int pos = 0;
        while (pos < data.length) {
            int ctrlIndex = out.size();
            out.write(0);
            int ctrl = 0;
            int bit = 0;
            while (bit < 8 && pos < data.length) {
                Match match = findMatchFast(data, pos, lastPos, preset);
                if (match.length >= preset.minMatch) {
                    ctrl |= 1 << bit;
                    out.write(match.offset & 0xFF);
                    out.write((match.offset >>> 8) & 0xFF);
                    out.write(match.length - preset.minMatch);
                    rememberMatchAligned(lastPos, data, pos, match.length);
                    pos += match.length;
                    bit++;
                } else if (bit == 0 && pos + 8 <= data.length) {
                    rememberAligned(lastPos, data, pos);
                    out.write(data, pos, 8);
                    pos += 8;
                    bit = 8;
                } else {
                    out.write(data[pos] & 0xFF);
                    rememberAligned(lastPos, data, pos);
                    pos++;
                    bit++;
                }
            }
            out.set(ctrlIndex, ctrl);
        }
        return out.toByteArray();
    }

    public static byte[] compress(byte[] input, String algorithm) {
        if (!ActiveMethodPolicy.isActiveAlgorithmKey(algorithm)) {
            throw new IllegalArgumentException(
                    "algorithm '" + algorithm + "' is retired by active-method policy "
                            + ActiveMethodPolicy.POLICY_ID);
        }
        Preset preset = preset(algorithm);
        byte[] data = input == null ? new byte[0] : input;
        if (data.length == 0) {
            return new byte[0];
        }
        if (data.length >= fastPathThreshold()) {
            if (literalFastEnabled()) {
                return compressLiteral(data);
            }
            return compressFast(data, preset);
        }
        ByteSink out = new ByteSink(data.length);
        Map<Integer, ArrayList<Integer>> chains = new HashMap<>();
        int pos = 0;
        while (pos < data.length) {
            int ctrlIndex = out.size();
            out.write(0);
            int ctrl = 0;
            for (int bit = 0; bit < 8 && pos < data.length; bit++) {
                Match match = findMatch(data, pos, chains, preset);
                if (preset.lazy && match.length >= preset.minMatch && pos + 1 < data.length) {
                    addPosition(chains, data, pos, preset);
                    Match next = findMatch(data, pos + 1, chains, preset);
                    if (next.length > match.length) {
                        out.write(data[pos] & 0xFF);
                        pos++;
                        continue;
                    }
                }
                if (match.length >= preset.minMatch) {
                    ctrl |= 1 << bit;
                    out.write(match.offset & 0xFF);
                    out.write((match.offset >>> 8) & 0xFF);
                    out.write(match.length - preset.minMatch);
                    for (int p = pos; p < pos + match.length; p++) {
                        addPosition(chains, data, p, preset);
                    }
                    pos += match.length;
                } else {
                    out.write(data[pos] & 0xFF);
                    addPosition(chains, data, pos, preset);
                    pos++;
                }
            }
            out.set(ctrlIndex, ctrl);
        }
        return out.toByteArray();
    }

    public static byte[] decompress(byte[] encoded, String algorithm, int expectedSize) {
        Preset preset = preset(algorithm);
        byte[] data = encoded == null ? new byte[0] : encoded;
        byte[] out = new byte[expectedSize];
        int outLen = 0;
        int pos = 0;
        while (outLen < expectedSize) {
            if (pos >= data.length) {
                throw new IllegalArgumentException("native LZ stream truncated before control byte");
            }
            int ctrl = data[pos++] & 0xFF;
            for (int bit = 0; bit < 8 && outLen < expectedSize; bit++) {
                if ((ctrl & (1 << bit)) != 0) {
                    if (pos + 3 > data.length) {
                        throw new IllegalArgumentException("native LZ stream truncated inside match token");
                    }
                    int offset = (data[pos] & 0xFF) | ((data[pos + 1] & 0xFF) << 8);
                    int length = (data[pos + 2] & 0xFF) + preset.minMatch;
                    pos += 3;
                    if (offset <= 0 || offset > outLen) {
                        throw new IllegalArgumentException("native LZ stream has invalid match offset");
                    }
                    int start = outLen - offset;
                    for (int i = 0; i < length; i++) {
                        if (outLen >= expectedSize) {
                            throw new IllegalArgumentException("native LZ stream expands beyond expected size");
                        }
                        out[outLen++] = out[start + i];
                    }
                } else {
                    if (pos >= data.length) {
                        throw new IllegalArgumentException("native LZ stream truncated inside literal");
                    }
                    out[outLen++] = data[pos++];
                }
            }
        }
        return out;
    }

    public static void benchByteColumns(
            List<byte[]> columnByteData,
            String algorithm,
            long[] result,
            double[] lossAcc) {
        long totalOriginal = 0;
        long totalCompressed = 0;
        long totalCompTime = 0;
        long totalDecompTime = 0;
        double[] localLoss = new double[2];
        for (byte[] data : columnByteData) {
            byte[] raw = data == null ? new byte[0] : data;
            long start = System.nanoTime();
            byte[] compressed = compress(raw, algorithm);
            totalCompTime += System.nanoTime() - start;
            start = System.nanoTime();
            byte[] restored = decompress(compressed, algorithm, raw.length);
            totalDecompTime += System.nanoTime() - start;
            totalOriginal += raw.length;
            totalCompressed += compressed.length;
            BenchmarkLoss.accumulateByteLoss(raw, restored, localLoss);
        }
        Arrays.fill(result, 0L);
        result[0] = totalOriginal;
        result[1] = totalCompressed;
        result[2] = totalCompTime;
        result[3] = totalDecompTime;
        if (lossAcc != null && lossAcc.length >= 2) {
            lossAcc[0] += localLoss[0];
            lossAcc[1] += localLoss[1];
        }
    }
}
