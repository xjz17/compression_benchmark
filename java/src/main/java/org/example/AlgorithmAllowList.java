package org.example;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.Set;

/** Allowlist / incremental-skip helpers for partial staticdata benchmark runs. */
public final class AlgorithmAllowList {
    /** Incremental append mode: skip algorithms already present in report / env. */
    public static final String INCREMENTAL_SKIP_ENV_KEY = "WEB_COMPRESSION_SKIP_ALGORITHMS";
    /** When non-empty, only these algorithm row names are benchmarked (incremental skip ignored). */
    public static final String ALLOWLIST_ENV_KEY = "WEB_COMPRESSION_RUN_ALGORITHMS";

    private AlgorithmAllowList() {
    }

    public static Set<String> fromAllowlistEnv() {
        String raw = System.getenv(ALLOWLIST_ENV_KEY);
        if (raw == null || raw.trim().isEmpty()) {
            raw = System.getProperty(ALLOWLIST_ENV_KEY, "");
        }
        return parseCsvList(raw);
    }

    public static boolean hasAllowlist() {
        return !fromAllowlistEnv().isEmpty();
    }

    /**
     * Whether {@code algorithm} should be benchmarked given optional incremental-skip / allowlist
     * filters. Allowlist ({@link #ALLOWLIST_ENV_KEY}) takes precedence over incremental skip.
     */
    public static boolean shouldRun(Set<String> incrementalSkip, String algorithm) {
        return !shouldSkip(incrementalSkip, algorithm);
    }

    /** True when the algorithm should be skipped (not benchmarked / not flushed). */
    public static boolean shouldSkip(Set<String> incrementalSkip, String algorithm) {
        if (!ActiveMethodPolicy.isActiveAlgorithmKey(algorithm)) {
            return true;
        }
        Set<String> allowlist = fromAllowlistEnv();
        if (!allowlist.isEmpty()) {
            return algorithm == null || !allowlist.contains(algorithm);
        }
        return incrementalSkip != null && algorithm != null && incrementalSkip.contains(algorithm);
    }

    /**
     * Heuristic for {@link ResearchLzbenchCliBench}: when allowlist is active, skip the expensive
     * lzbench subprocess unless at least one requested row name plausibly maps to the lzbench
     * catalog (exact membership is resolved in Python).
     */
    public static boolean allowlistMayWantLzbench(Set<String> incrementalSkip) {
        if (!hasAllowlist()) {
            return true;
        }
        for (String algo : fromAllowlistEnv()) {
            if (shouldSkip(incrementalSkip, algo)) {
                continue;
            }
            if (mightBeLzbenchCatalogRow(algo)) {
                return true;
            }
        }
        return false;
    }

    private static boolean mightBeLzbenchCatalogRow(String algo) {
        if (algo == null || algo.isBlank()) {
            return false;
        }
        if (algo.startsWith("lzbench_")) {
            return !isKnownNonLzbenchRow(algo.substring("lzbench_".length()));
        }
        if (isKnownNonLzbenchRow(algo)) {
            return false;
        }
        if ("lzfse".equals(algo) || "lzjb".equals(algo)) {
            return true;
        }
        if (algo.matches("bsc[0-2]")) {
            return true;
        }
        // Catalog keys are typically codec_level, e.g. brieflz_1, zstd_9, slz_deflate_3.
        return algo.matches("^[a-z][a-z0-9_]*(?:_[0-9]+|_[0-9]+p[0-9]+)$");
    }

    private static boolean isKnownNonLzbenchRow(String algo) {
        if (NativeLzFamilyBench.algorithms().contains(algo)) {
            return true;
        }
        if (algo.equals("ts_2diff_bos")
                || algo.equals("TS_2DIFF")
                || algo.equals("RLE")
                || algo.equals("GORILLA")
                || algo.equals("CHIMP")
                || algo.equals("CHIMP128")
                || algo.equals("SPRINTZ")
                || algo.equals("RLBE")
                || algo.equals("GZIP")
                || algo.equals("SNAPPY")
                || algo.equals("LZ4")
                || algo.equals("BROTLI")
                || algo.equals("brotli")
                || algo.equals("ans")
                || algo.equals("huffman")
                || algo.equals("arithmetic")
                || algo.equals("fpzip")
                || algo.equals("fastpfor")
                || algo.equals("fpc")
                || algo.equals("pfpc")
                || algo.equals("Pongo")
                || algo.equals("most_minimal")
                || algo.equals("corad_tricklet")
                || algo.equals("streamvbyte_delta1")
                || algo.equals("simple8b")
                || algo.equals("leco")
                || algo.equals("alp")
                || algo.equals("elf_on_chimp")
                || algo.equals("elf_plus")
                || algo.equals("camel")
                || algo.equals("blosc_blosclz")
                || algo.equals("blosc2_blosclz")
                || algo.equals("ext_cmix")
                || algo.equals("buff")
                || algo.equals("buff_simd")
                || algo.equals("buff_kernel_native")
                || algo.equals("subcolumn")
                || algo.equals("grouping")
                || algo.equals("reger")
                || algo.equals("gz")
                || algo.equals("snappy-framed")
                || algo.equals("lz4-framed")
                || algo.equals(ColumnarEncodings.DELTA_BINARY_PACKED_ALGORITHM)) {
            return true;
        }
        String lower = algo.toLowerCase();
        return lower.startsWith("col_")
                || lower.startsWith("delta")
                || lower.startsWith("sz_")
                || lower.startsWith("sz3_")
                || lower.startsWith("serf_")
                || lower.startsWith("spdp_")
                || lower.startsWith("ext_")
                || lower.startsWith("machete_")
                || lower.startsWith("simpiece_")
                || lower.startsWith("fpcompress_")
                || lower.startsWith("ext_zfp_")
                || lower.startsWith("xz_level_")
                || lower.startsWith("lzma_level_")
                || lower.startsWith("zstd_level_");
    }

    public static Set<String> fromCsvReport(Path reportPath) {
        Set<String> out = new LinkedHashSet<>();
        if (reportPath == null || !Files.isRegularFile(reportPath)) {
            return out;
        }
        try (Reader reader = Files.newBufferedReader(reportPath, StandardCharsets.UTF_8);
             CSVParser parser =
                     CSVFormat.DEFAULT.builder()
                             .setHeader()
                             .setSkipHeaderRecord(true)
                             .setIgnoreEmptyLines(true)
                             .setTrim(true)
                             .build()
                             .parse(reader)) {
            for (CSVRecord rec : parser) {
                if (rec.size() == 0) {
                    continue;
                }
                String a = rec.isMapped("algorithm") ? rec.get("algorithm") : null;
                if (a == null && rec.size() > 1) {
                    a = rec.get(1);
                }
                if (a == null) {
                    continue;
                }
                String t = a.trim();
                if (!t.isEmpty()) {
                    out.add(t);
                }
            }
        } catch (IOException ignored) {
            // best-effort: treat unreadable report as empty incremental skip set
        }
        return out;
    }

    public static Set<String> fromIncrementalSkipEnv() {
        String raw = System.getenv(INCREMENTAL_SKIP_ENV_KEY);
        if (raw == null || raw.trim().isEmpty()) {
            raw = System.getProperty(INCREMENTAL_SKIP_ENV_KEY, "");
        }
        return parseCsvList(raw);
    }

    public static String toEnvValue(Set<String> algorithms) {
        if (algorithms == null || algorithms.isEmpty()) {
            return "";
        }
        return String.join(",", algorithms);
    }

    private static Set<String> parseCsvList(String raw) {
        Set<String> out = new LinkedHashSet<>();
        if (raw == null || raw.trim().isEmpty()) {
            return out;
        }
        for (String p : raw.split(",")) {
            String t = p == null ? "" : p.trim();
            if (!t.isEmpty()) {
                out.add(t);
            }
        }
        return out;
    }
}
