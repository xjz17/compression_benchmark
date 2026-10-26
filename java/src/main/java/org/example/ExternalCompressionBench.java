package org.example;

import me.lemire.integercompression.Composition;
import me.lemire.integercompression.FastPFOR;
import me.lemire.integercompression.IntWrapper;
import me.lemire.integercompression.IntegerCODEC;
import me.lemire.integercompression.VariableByte;
import org.apache.commons.csv.CSVPrinter;
import org.apache.tsfile.enums.TSDataType;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Research codecs for Ranking / staticdata CSVs (readable {@code algorithm} names).
 * IoTDB-native encodings (e.g. {@code CHIMP}) stay separate from paper Chimp32 here.
 */
public final class ExternalCompressionBench {

    /**
     * Loads Elf and its bundled {@code gr.aueb.delorean.chimp} copies before delegating to the parent,
     * so an unrelated Chimp copy on the application classpath cannot supply an older
     * {@code OutputBitStream} missing {@code getBuffer()}.
     */
    private static final class ElfIsolatingLoader extends URLClassLoader {
        ElfIsolatingLoader(URL[] urls, ClassLoader parent) {
            super(urls, parent);
        }

        @Override
        protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
            synchronized (getClassLoadingLock(name)) {
                Class<?> c = findLoadedClass(name);
                if (c != null) {
                    return c;
                }
                boolean elfOrBundledChimp = name.startsWith("org.urbcomp.startdb.compress.elf.")
                        || name.startsWith("gr.aueb.delorean.chimp.");
                if (elfOrBundledChimp) {
                    c = findClass(name);
                    if (resolve) {
                        resolveClass(c);
                    }
                    return c;
                }
                return super.loadClass(name, resolve);
            }
        }
    }

    /** Child-first for Camel JAR + its Maven deps so Guava/HBase on the benchmark classpath do not win. */
    private static final class CamelIsolatingLoader extends URLClassLoader {
        CamelIsolatingLoader(URL[] urls, ClassLoader parent) {
            super(urls, parent);
        }

        @Override
        protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
            synchronized (getClassLoadingLock(name)) {
                Class<?> c = findLoadedClass(name);
                if (c != null) {
                    return c;
                }
                if (name.startsWith("java.")
                        || name.startsWith("javax.")
                        || name.startsWith("jdk.")
                        || name.startsWith("sun.")
                        || name.startsWith("com.sun.")) {
                    return super.loadClass(name, resolve);
                }
                try {
                    c = findClass(name);
                    if (resolve) {
                        resolveClass(c);
                    }
                    return c;
                } catch (ClassNotFoundException ignored) {
                    return super.loadClass(name, resolve);
                }
            }
        }
    }

    private ExternalCompressionBench() {
    }

    /**
     * Elf JAR for isolated Elf benches. Default: {@code external/compression/elf/target/start-compress-1.0-SNAPSHOT.jar}
     * (build submodule; run {@code mvn dependency:copy-dependencies} there for fastutil). No copy into
     * {@code backend/lib/} required. Override with {@code -Dweb.compression.elf.jar=} or
     * {@code COMPRESSION_BENCHMARK_REPO_ROOT}.
     */
    static File resolveElfJarForIsolatedBench() {
        String prop = System.getProperty("web.compression.elf.jar");
        if (prop != null && !prop.isEmpty()) {
            File f = new File(prop);
            if (f.isFile()) {
                return f;
            }
        }
        String repo = System.getenv("COMPRESSION_BENCHMARK_REPO_ROOT");
        if (repo != null && !repo.isEmpty()) {
            Path fromRepo =
                    Path.of(repo)
                            .resolve("external/compression/elf/target/start-compress-1.0-SNAPSHOT.jar");
            if (Files.isRegularFile(fromRepo)) {
                return fromRepo.toFile();
            }
            Path backendLib =
                    Path.of(repo).resolve("backend/lib/start-compress-1.0-SNAPSHOT.jar");
            if (Files.isRegularFile(backendLib)) {
                return backendLib.toFile();
            }
        }
        Path ud = Path.of(System.getProperty("user.dir", ".")).toAbsolutePath().normalize();
        Path[] candidates = new Path[]{
                ud.resolve("external/compression/elf/target/start-compress-1.0-SNAPSHOT.jar"),
                ud.resolve("../external/compression/elf/target/start-compress-1.0-SNAPSHOT.jar"),
                ud.resolve("../../external/compression/elf/target/start-compress-1.0-SNAPSHOT.jar"),
        };
        for (Path p : candidates) {
            if (Files.isRegularFile(p)) {
                return p.toFile();
            }
        }
        return null;
    }

    private static URL[] elfClasspathUrls(File elfJar) throws IOException {
        List<URL> urls = new ArrayList<>();
        urls.add(elfJar.toURI().toURL());
        Path depDir = elfJar.toPath().getParent().resolve("dependency");
        if (Files.isDirectory(depDir)) {
            try (DirectoryStream<Path> ds = Files.newDirectoryStream(depDir, "*.jar")) {
                for (Path p : ds) {
                    urls.add(p.toUri().toURL());
                }
            }
        }
        return urls.toArray(new URL[0]);
    }

    /**
     * Camel (SIGMOD) project JAR. Override with {@code -Dweb.compression.camel.jar=...}.
     * Run {@code mvn -DskipTests package} under {@code external/compression/camel} so {@code target/dependency/*.jar} exists.
     */
    static File resolveCamelJarForBench() {
        String prop = System.getProperty("web.compression.camel.jar");
        if (prop != null && !prop.isEmpty()) {
            File f = new File(prop);
            if (f.isFile()) {
                return f;
            }
        }
        Path ud = Path.of(System.getProperty("user.dir", ".")).toAbsolutePath().normalize();
        Path[] candidates = new Path[]{
                ud.resolve("external/compression/camel/target/start-compress-1.0-SNAPSHOT.jar"),
                ud.resolve("../external/compression/camel/target/start-compress-1.0-SNAPSHOT.jar"),
                ud.resolve("../../external/compression/camel/target/start-compress-1.0-SNAPSHOT.jar"),
        };
        for (Path p : candidates) {
            if (Files.isRegularFile(p)) {
                return p.toFile();
            }
        }
        return null;
    }

    static List<URL> camelClasspathUrls() throws IOException {
        File jar = resolveCamelJarForBench();
        if (jar == null) {
            return Collections.emptyList();
        }
        List<URL> urls = new ArrayList<>();
        urls.add(jar.toURI().toURL());
        Path depDir = jar.toPath().getParent().resolve("dependency");
        if (Files.isDirectory(depDir)) {
            try (DirectoryStream<Path> ds = Files.newDirectoryStream(depDir, "*.jar")) {
                for (Path p : ds) {
                    urls.add(p.toUri().toURL());
                }
            }
        }
        return urls;
    }

    /** Merge research / streamvbyte rows into {@code totals} (patch benchmark). */
    public static void accumulateRows(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> intCols,
            ArrayList<ArrayList<Double>> floatCols,
            ArrayList<Integer> floatMaxPoints,
            BenchmarkTotals totals,
            Set<String> skipAlgorithms)
            throws IOException {
        Set<String> codecSkipAlgorithms = skipAlgorithms;
        final boolean nativeFastPFOR = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_FASTPFOR"));
        final boolean nativeSpdp = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_SPDP"));
        final boolean nativeMost = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_MOST"));
        final boolean nativeMachete = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_MACHETE"));
        final boolean nativeCorad = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_CORAD"));
        final boolean nativeSimpieceFloat =
                "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_SIMPIECE_FLOAT"));
        if (!nativeFastPFOR
                && tsType == TSDataType.INT64
                && intCols != null
                && !intCols.isEmpty()
                && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "fastpfor")) {
            long[] tFp = new long[4];
            double[] lossFp = new double[2];
            benchJavaFastPFORLong(intCols, tFp, lossFp);
            totals.merge("fastpfor", tFp, lossFp[0], lossFp[1]);
            codecSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            codecSkipAlgorithms.add("fastpfor");
        }
        if (!nativeSpdp
                && tsType == TSDataType.INT64
                && intCols != null
                && !intCols.isEmpty()) {
            for (int spdpLevel : SpdpFloat.levelsToBench()) {
                String spdpRow = SpdpFloat.rowName(spdpLevel);
                if (AlgorithmAllowList.shouldSkip(skipAlgorithms, spdpRow)) {
                    continue;
                }
                long[] tSp = new long[4];
                double[] lossSp = new double[2];
                SpdpFloat.benchSpdpLong(intCols, spdpLevel, tSp, lossSp);
                totals.merge(spdpRow, tSp, lossSp[0], lossSp[1]);
                if (codecSkipAlgorithms == skipAlgorithms) {
                    codecSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
                }
                codecSkipAlgorithms.add(spdpRow);
            }
        }
        if (tsType == TSDataType.INT64
                && intCols != null
                && !intCols.isEmpty()
                && !AlgorithmAllowList.shouldSkip(skipAlgorithms, FpcDouble.ROW_NAME)) {
            long[] tFpc = new long[4];
            double[] lossFpc = new double[2];
            FpcDouble.benchFpcLong(intCols, tFpc, lossFpc);
            totals.merge(FpcDouble.ROW_NAME, tFpc, lossFpc[0], lossFpc[1]);
            if (codecSkipAlgorithms == skipAlgorithms) {
                codecSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            }
            codecSkipAlgorithms.add(FpcDouble.ROW_NAME);
        }
        if (tsType == TSDataType.INT64
                && intCols != null
                && !intCols.isEmpty()
                && !AlgorithmAllowList.shouldSkip(skipAlgorithms, PfpcDouble.ROW_NAME)) {
            long[] tPfpc = new long[4];
            double[] lossPfpc = new double[2];
            PfpcDouble.benchPfpcLong(intCols, tPfpc, lossPfpc);
            totals.merge(PfpcDouble.ROW_NAME, tPfpc, lossPfpc[0], lossPfpc[1]);
            if (codecSkipAlgorithms == skipAlgorithms) {
                codecSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            }
            codecSkipAlgorithms.add(PfpcDouble.ROW_NAME);
        }
        if (!nativeMost
                && tsType == TSDataType.INT64
                && intCols != null
                && !intCols.isEmpty()
                && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "most_minimal")) {
            long[] tMo = new long[4];
            double[] lossMo = new double[2];
            MostFloatMinimal.benchMostLongMinimal(intCols, tMo, lossMo);
            totals.merge("most_minimal", tMo, lossMo[0], lossMo[1]);
            if (codecSkipAlgorithms == skipAlgorithms) {
                codecSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            }
            codecSkipAlgorithms.add("most_minimal");
        }
        if (!nativeMachete
                && tsType == TSDataType.INT64
                && intCols != null
                && !intCols.isEmpty()) {
            final String[] macheteRows = {
                "machete_rel_1e-1",
                "machete_rel_1e-2",
                "machete_rel_1e-3",
                "machete_rel_1e-4",
            };
            final double[] macheteRels = {1e-1, 1e-2, 1e-3, 1e-4};
            for (int mi = 0; mi < macheteRels.length; mi++) {
                if (AlgorithmAllowList.shouldSkip(skipAlgorithms, macheteRows[mi])) {
                    continue;
                }
                long[] tMa = new long[4];
                double[] lossMa = new double[2];
                MacheteFloatRel.benchMacheteLongRel(intCols, macheteRels[mi], tMa, lossMa);
                totals.merge(macheteRows[mi], tMa, lossMa[0], lossMa[1]);
                if (codecSkipAlgorithms == skipAlgorithms) {
                    codecSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
                }
                codecSkipAlgorithms.add(macheteRows[mi]);
            }
        }
        if (!nativeCorad
                && tsType == TSDataType.INT64
                && intCols != null
                && !intCols.isEmpty()
                && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "corad_tricklet")) {
            long[] tCo = new long[4];
            double[] lossCo = new double[2];
            CoradTrickletFloat.benchCoradTrickletLong(intCols, tCo, lossCo);
            totals.merge("corad_tricklet", tCo, lossCo[0], lossCo[1]);
            if (codecSkipAlgorithms == skipAlgorithms) {
                codecSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            }
            codecSkipAlgorithms.add("corad_tricklet");
        }
        if (!nativeSimpieceFloat
                && tsType == TSDataType.INT64
                && intCols != null
                && !intCols.isEmpty()) {
            long[] tS0 = new long[4];
            double[] lossS0 = new double[2];
            if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "simpiece_relRange_0p005")
                    && benchSimPieceLong(intCols, 0.005, tS0, lossS0)) {
                totals.merge("simpiece_relRange_0p005", tS0, lossS0[0], lossS0[1]);
                if (codecSkipAlgorithms == skipAlgorithms) {
                    codecSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
                }
                codecSkipAlgorithms.add("simpiece_relRange_0p005");
            }
            long[] tS1 = new long[4];
            double[] lossS1 = new double[2];
            if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "simpiece_relRange_0p01")
                    && benchSimPieceLong(intCols, 0.01, tS1, lossS1)) {
                totals.merge("simpiece_relRange_0p01", tS1, lossS1[0], lossS1[1]);
                if (codecSkipAlgorithms == skipAlgorithms) {
                    codecSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
                }
                codecSkipAlgorithms.add("simpiece_relRange_0p01");
            }
        }
        final boolean nativeStreamVByte =
                "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_STREAMVBYTE"));
        if (!nativeStreamVByte
                && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "streamvbyte_delta1")) {
            long[] tSv = new long[4];
            double[] lossSv = new double[2];
            if (tsType == TSDataType.INT64 && intCols != null && !intCols.isEmpty()) {
                StreamVByteDelta1.benchStreamVByteDelta1Long(intCols, tSv, lossSv);
            } else if (tsType == TSDataType.DOUBLE && floatCols != null && !floatCols.isEmpty()) {
                ArrayList<ArrayList<Long>> scaled =
                        Ts2DiffBosImprove.scaleDoubleColumnsToLongLists(floatCols, floatMaxPoints);
                if (!scaled.isEmpty()) {
                    StreamVByteDelta1.benchStreamVByteDelta1Long(scaled, tSv, lossSv);
                }
            }
            totals.merge("streamvbyte_delta1", tSv, lossSv[0], lossSv[1]);
        }
        ArrayList<ArrayList<Double>> codecCols =
                BenchmarkColumns.doubleColumnsForCodecBench(tsType, intCols, floatCols);
        if (!codecCols.isEmpty()) {
            List<Integer> caps =
                    tsType == TSDataType.DOUBLE ? floatMaxPoints : null;
            accumulateDoubleCodecResearchRows(codecCols, totals, caps, codecSkipAlgorithms);
        }
    }

    public static void printRows(
            TSDataType tsType,
            ArrayList<ArrayList<Long>> intCols,
            ArrayList<ArrayList<Double>> floatCols,
            CSVPrinter csvWriter) throws IOException {
        final Set<String> skipAlgorithms = AlgorithmAllowList.fromIncrementalSkipEnv();
        Set<String> codecSkipAlgorithms = skipAlgorithms;
        final boolean nativeFastPFOR = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_FASTPFOR"));
        final boolean nativeSpdp = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_SPDP"));
        final boolean nativeMost = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_MOST"));
        final boolean nativeMachete = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_MACHETE"));
        final boolean nativeCorad = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_CORAD"));
        if (!nativeFastPFOR
                && tsType == TSDataType.INT64
                && intCols != null
                && !intCols.isEmpty()
                && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "fastpfor")) {
            long[] tFp = new long[4];
            benchJavaFastPFORLong(intCols, tFp);
            if (tFp[0] > 0) {
                csvWriter.printRecord("fastpfor", tFp[0], tFp[1], tFp[2], tFp[3]);
            }
            codecSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            codecSkipAlgorithms.add("fastpfor");
        }
        if (!nativeSpdp
                && tsType == TSDataType.INT64
                && intCols != null
                && !intCols.isEmpty()) {
            for (int spdpLevel : SpdpFloat.levelsToBench()) {
                String spdpRow = SpdpFloat.rowName(spdpLevel);
                if (AlgorithmAllowList.shouldSkip(skipAlgorithms, spdpRow)) {
                    continue;
                }
                long[] tSp = new long[4];
                SpdpFloat.benchSpdpLong(intCols, spdpLevel, tSp);
                if (tSp[0] > 0) {
                    csvWriter.printRecord(spdpRow, tSp[0], tSp[1], tSp[2], tSp[3]);
                }
                if (codecSkipAlgorithms == skipAlgorithms) {
                    codecSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
                }
                codecSkipAlgorithms.add(spdpRow);
            }
        }
        if (tsType == TSDataType.INT64
                && intCols != null
                && !intCols.isEmpty()
                && !AlgorithmAllowList.shouldSkip(skipAlgorithms, FpcDouble.ROW_NAME)) {
            long[] tFpc = new long[4];
            FpcDouble.benchFpcLong(intCols, tFpc);
            if (tFpc[0] > 0) {
                csvWriter.printRecord(FpcDouble.ROW_NAME, tFpc[0], tFpc[1], tFpc[2], tFpc[3]);
            }
            if (codecSkipAlgorithms == skipAlgorithms) {
                codecSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            }
            codecSkipAlgorithms.add(FpcDouble.ROW_NAME);
        }
        if (tsType == TSDataType.INT64
                && intCols != null
                && !intCols.isEmpty()
                && !AlgorithmAllowList.shouldSkip(skipAlgorithms, PfpcDouble.ROW_NAME)) {
            long[] tPfpc = new long[4];
            PfpcDouble.benchPfpcLong(intCols, tPfpc);
            if (tPfpc[0] > 0) {
                csvWriter.printRecord(PfpcDouble.ROW_NAME, tPfpc[0], tPfpc[1], tPfpc[2], tPfpc[3]);
            }
            if (codecSkipAlgorithms == skipAlgorithms) {
                codecSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            }
            codecSkipAlgorithms.add(PfpcDouble.ROW_NAME);
        }
        if (!nativeMost
                && tsType == TSDataType.INT64
                && intCols != null
                && !intCols.isEmpty()
                && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "most_minimal")) {
            long[] tMo = new long[4];
            MostFloatMinimal.benchMostLongMinimal(intCols, tMo);
            if (tMo[0] > 0) {
                csvWriter.printRecord("most_minimal", tMo[0], tMo[1], tMo[2], tMo[3]);
            }
            if (codecSkipAlgorithms == skipAlgorithms) {
                codecSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            }
            codecSkipAlgorithms.add("most_minimal");
        }
        if (!nativeMachete
                && tsType == TSDataType.INT64
                && intCols != null
                && !intCols.isEmpty()) {
            final String[] macheteRows = {
                "machete_rel_1e-1",
                "machete_rel_1e-2",
                "machete_rel_1e-3",
                "machete_rel_1e-4",
            };
            final double[] macheteRels = {1e-1, 1e-2, 1e-3, 1e-4};
            for (int mi = 0; mi < macheteRels.length; mi++) {
                if (AlgorithmAllowList.shouldSkip(skipAlgorithms, macheteRows[mi])) {
                    continue;
                }
                long[] tMa = new long[4];
                MacheteFloatRel.benchMacheteLongRel(intCols, macheteRels[mi], tMa);
                if (tMa[0] > 0) {
                    csvWriter.printRecord(macheteRows[mi], tMa[0], tMa[1], tMa[2], tMa[3]);
                }
                if (codecSkipAlgorithms == skipAlgorithms) {
                    codecSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
                }
                codecSkipAlgorithms.add(macheteRows[mi]);
            }
        }
        if (!nativeCorad
                && tsType == TSDataType.INT64
                && intCols != null
                && !intCols.isEmpty()
                && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "corad_tricklet")) {
            long[] tCo = new long[4];
            CoradTrickletFloat.benchCoradTrickletLong(intCols, tCo);
            if (tCo[0] > 0) {
                csvWriter.printRecord("corad_tricklet", tCo[0], tCo[1], tCo[2], tCo[3]);
            }
            if (codecSkipAlgorithms == skipAlgorithms) {
                codecSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            }
            codecSkipAlgorithms.add("corad_tricklet");
        }
        final boolean nativeStreamVByte =
                "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_STREAMVBYTE"));
        if (!nativeStreamVByte
                && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "streamvbyte_delta1")) {
            long[] tSv = new long[4];
            if (tsType == TSDataType.INT64 && intCols != null && !intCols.isEmpty()) {
                StreamVByteDelta1.benchStreamVByteDelta1Long(intCols, tSv);
            } else if (tsType == TSDataType.DOUBLE && floatCols != null && !floatCols.isEmpty()) {
                ArrayList<ArrayList<Long>> scaled =
                        Ts2DiffBosImprove.scaleDoubleColumnsToLongLists(floatCols);
                if (!scaled.isEmpty()) {
                    StreamVByteDelta1.benchStreamVByteDelta1Long(scaled, tSv);
                }
            }
            if (tSv[0] > 0) {
                csvWriter.printRecord(
                        "streamvbyte_delta1", tSv[0], tSv[1], tSv[2], tSv[3]);
            }
        }
        ArrayList<ArrayList<Double>> codecCols =
                BenchmarkColumns.doubleColumnsForCodecBench(tsType, intCols, floatCols);
        if (!codecCols.isEmpty()) {
            printDoubleCodecResearchRows(codecCols, csvWriter, codecSkipAlgorithms);
        }
    }

    static void accumulateDoubleCodecResearchRows(
            ArrayList<ArrayList<Double>> columns,
            BenchmarkTotals totals,
            List<Integer> maxPointPerColumn,
            Set<String> skipAlgorithms)
            throws IOException {
        runDoubleCodecResearchRows(columns, totals, null, skipAlgorithms, maxPointPerColumn);
    }

    static void printDoubleCodecResearchRows(
            ArrayList<ArrayList<Double>> columns,
            CSVPrinter csvWriter,
            Set<String> skipAlgorithms) throws IOException {
        runDoubleCodecResearchRows(columns, null, csvWriter, skipAlgorithms, null);
    }

    private static void runDoubleCodecResearchRows(
            ArrayList<ArrayList<Double>> columns,
            BenchmarkTotals totals,
            CSVPrinter csvWriter,
            Set<String> skipAlgorithms,
            List<Integer> maxPointPerColumn)
            throws IOException {
        final boolean nativeElfOnChimpFloat =
                "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_ELF_ON_CHIMP_FLOAT"));
        final boolean nativeElfPlusFloat = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_ELF_PLUS_FLOAT"));
        final boolean nativeCamelFloat = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_CAMEL_FLOAT"));
        if (!nativeElfOnChimpFloat && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "elf_on_chimp")) {
            long[] tElfChimp = new long[4];
            double[] lossElfChimp = new double[2];
            if (benchElfVariant64Double(
                            columns,
                            "org.urbcomp.startdb.compress.elf.compressor.ElfOnChimpCompressor",
                            "org.urbcomp.startdb.compress.elf.decompressor.ElfOnChimpDecompressor",
                            "chimp",
                            tElfChimp,
                            lossElfChimp)) {
                recordCodecBenchRow("elf_on_chimp", tElfChimp, lossElfChimp, skipAlgorithms, totals, csvWriter);
            }
        }
        if (!nativeElfPlusFloat && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "elf_plus")) {
            long[] tElfPlus = new long[4];
            double[] lossElfPlus = new double[2];
            if (benchElfVariant64Double(
                            columns,
                            "org.urbcomp.startdb.compress.elf.compressor.ElfCompressor",
                            "org.urbcomp.startdb.compress.elf.decompressor.ElfDecompressor",
                            "xorCompressor",
                            tElfPlus,
                            lossElfPlus)) {
                recordCodecBenchRow("elf_plus", tElfPlus, lossElfPlus, skipAlgorithms, totals, csvWriter);
            }
        }
        if (!nativeCamelFloat && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "camel")) {
            long[] tCamel = new long[4];
            double[] lossCamel = new double[2];
            if (CamelDoubleNative.benchDoubleColumns(columns, tCamel, lossCamel)) {
                recordCodecBenchRow("camel", tCamel, lossCamel, skipAlgorithms, totals, csvWriter);
            }
        }
        final boolean nativeSimple8b = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_SIMPLE8B"));
        final boolean nativeLeCoFloat = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_LECO_FLOAT"));
        final boolean nativeAlpFloat = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_ALP_FLOAT"));
        final boolean nativeFastPFOR = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_FASTPFOR"));
        final boolean nativeSpdp = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_SPDP"));
        final boolean nativeMost = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_MOST"));
        final boolean nativeMachete = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_MACHETE"));
        final boolean nativeCorad = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_CORAD"));
        final boolean nativeSimpieceFloat =
                "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_SIMPIECE_FLOAT"));
        if (!nativeSpdp) {
            for (int spdpLevel : SpdpFloat.levelsToBench()) {
                String spdpRow = SpdpFloat.rowName(spdpLevel);
                if (AlgorithmAllowList.shouldSkip(skipAlgorithms, spdpRow)) {
                    continue;
                }
                long[] tSp = new long[4];
                double[] lossSp = new double[2];
                SpdpFloat.benchSpdpDouble(columns, spdpLevel, tSp, lossSp);
                recordCodecBenchRow(spdpRow, tSp, lossSp, skipAlgorithms, totals, csvWriter);
            }
        }
        if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, FpcDouble.ROW_NAME)) {
            long[] tFpc = new long[4];
            double[] lossFpc = new double[2];
            FpcDouble.benchFpcDouble(columns, tFpc, lossFpc);
            recordCodecBenchRow(FpcDouble.ROW_NAME, tFpc, lossFpc, skipAlgorithms, totals, csvWriter);
        }
        if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, PfpcDouble.ROW_NAME)) {
            long[] tPfpc = new long[4];
            double[] lossPfpc = new double[2];
            PfpcDouble.benchPfpcDouble(columns, tPfpc, lossPfpc);
            recordCodecBenchRow(PfpcDouble.ROW_NAME, tPfpc, lossPfpc, skipAlgorithms, totals, csvWriter);
        }
        if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, PongoDouble.ROW_NAME)) {
            long[] tPongo = new long[4];
            double[] lossPongo = new double[2];
            PongoDouble.bench(columns, tPongo, lossPongo);
            recordCodecBenchRow(PongoDouble.ROW_NAME, tPongo, lossPongo, skipAlgorithms, totals, csvWriter);
        }
        if (!nativeSimple8b && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "simple8b")) {
            long[] tS8f = new long[4];
            double[] lossS8 = new double[2];
            Simple8bIntColumn.benchSimple8bDouble(columns, maxPointPerColumn, tS8f, lossS8);
            recordCodecBenchRow("simple8b", tS8f, lossS8, skipAlgorithms, totals, csvWriter);
        }
        if (!nativeLeCoFloat && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "leco")) {
            long[] tLeCo = new long[4];
            double[] lossLeCo = new double[2];
            benchLeCoDoubleScaledIntegersFastPFor(columns, maxPointPerColumn, tLeCo, lossLeCo);
            recordCodecBenchRow("leco", tLeCo, lossLeCo, skipAlgorithms, totals, csvWriter);
        }
        if (!nativeAlpFloat && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "alp")) {
            long[] tAlp = new long[4];
            double[] lossAlp = new double[2];
            AlpDoubleBench.benchAlpDouble(columns, tAlp, lossAlp);
            recordCodecBenchRow("alp", tAlp, lossAlp, skipAlgorithms, totals, csvWriter);
        }
        if (!nativeFastPFOR && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "fastpfor")) {
            long[] tFp = new long[4];
            double[] lossFp = new double[2];
            benchJavaFastPFORDouble(columns, tFp, lossFp);
            recordCodecBenchRow("fastpfor", tFp, lossFp, skipAlgorithms, totals, csvWriter);
        }
        if (!nativeMost && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "most_minimal")) {
            long[] tMo = new long[4];
            double[] lossMo = new double[2];
            MostFloatMinimal.benchMostDoubleMinimal(columns, tMo, lossMo);
            recordCodecBenchRow("most_minimal", tMo, lossMo, skipAlgorithms, totals, csvWriter);
        }
        if (!nativeMachete) {
            final String[] macheteRows = {
                "machete_rel_1e-1",
                "machete_rel_1e-2",
                "machete_rel_1e-3",
                "machete_rel_1e-4",
            };
            final double[] macheteRels = {1e-1, 1e-2, 1e-3, 1e-4};
            for (int mi = 0; mi < macheteRels.length; mi++) {
                if (AlgorithmAllowList.shouldSkip(skipAlgorithms, macheteRows[mi])) {
                    continue;
                }
                long[] tMa = new long[4];
                double[] lossMa = new double[2];
                MacheteFloatRel.benchMacheteDoubleRel(columns, macheteRels[mi], tMa, lossMa);
                recordCodecBenchRow(macheteRows[mi], tMa, lossMa, skipAlgorithms, totals, csvWriter);
            }
        }
        if (!nativeCorad && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "corad_tricklet")) {
            long[] tCo = new long[4];
            double[] lossCo = new double[2];
            CoradTrickletFloat.benchCoradTrickletDouble(columns, tCo, lossCo);
            recordCodecBenchRow("corad_tricklet", tCo, lossCo, skipAlgorithms, totals, csvWriter);
        }
        if (!nativeSimpieceFloat) {
            long[] tS0 = new long[4];
            double[] lossS0 = new double[2];
            if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "simpiece_relRange_0p005")
                    && benchSimPieceDouble(columns, 0.005, tS0, lossS0)) {
                recordCodecBenchRow("simpiece_relRange_0p005", tS0, lossS0, skipAlgorithms, totals, csvWriter);
            }
            long[] tS1 = new long[4];
            double[] lossS1 = new double[2];
            if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "simpiece_relRange_0p01")
                    && benchSimPieceDouble(columns, 0.01, tS1, lossS1)) {
                recordCodecBenchRow("simpiece_relRange_0p01", tS1, lossS1, skipAlgorithms, totals, csvWriter);
            }
        }
        ArrayList<ArrayList<Float>> legacy = new ArrayList<>(columns.size());
        for (ArrayList<Double> col : columns) {
            ArrayList<Float> fc = new ArrayList<>(col.size());
            for (Double d : col) {
                fc.add(d == null ? Float.NaN : d.floatValue());
            }
            legacy.add(fc);
        }
        Set<String> floatSkipAlgorithms = skipAlgorithms;
        if (!nativeAlpFloat && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "alp")) {
            floatSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            floatSkipAlgorithms.add("alp");
        }
        if (!nativeFastPFOR && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "fastpfor")) {
            if (floatSkipAlgorithms == skipAlgorithms) {
                floatSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            }
            floatSkipAlgorithms.add("fastpfor");
        }
        if (!nativeSpdp) {
            for (int spdpLevel : SpdpFloat.levelsToBench()) {
                String spdpRow = SpdpFloat.rowName(spdpLevel);
                if (AlgorithmAllowList.shouldSkip(skipAlgorithms, spdpRow)) {
                    continue;
                }
                if (floatSkipAlgorithms == skipAlgorithms) {
                    floatSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
                }
                floatSkipAlgorithms.add(spdpRow);
            }
        }
        if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, FpcDouble.ROW_NAME)) {
            if (floatSkipAlgorithms == skipAlgorithms) {
                floatSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            }
            floatSkipAlgorithms.add(FpcDouble.ROW_NAME);
        }
        if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, PfpcDouble.ROW_NAME)) {
            if (floatSkipAlgorithms == skipAlgorithms) {
                floatSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            }
            floatSkipAlgorithms.add(PfpcDouble.ROW_NAME);
        }
        if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, PongoDouble.ROW_NAME)) {
            if (floatSkipAlgorithms == skipAlgorithms) {
                floatSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            }
            floatSkipAlgorithms.add(PongoDouble.ROW_NAME);
        }
        if (!nativeMost && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "most_minimal")) {
            if (floatSkipAlgorithms == skipAlgorithms) {
                floatSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            }
            floatSkipAlgorithms.add("most_minimal");
        }
        if (!nativeMachete) {
            final String[] macheteRows = {
                "machete_rel_1e-1",
                "machete_rel_1e-2",
                "machete_rel_1e-3",
                "machete_rel_1e-4",
            };
            boolean needSkipMachete = false;
            for (String row : macheteRows) {
                if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, row)) {
                    needSkipMachete = true;
                    break;
                }
            }
            if (needSkipMachete) {
                if (floatSkipAlgorithms == skipAlgorithms) {
                    floatSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
                }
                for (String row : macheteRows) {
                    if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, row)) {
                        floatSkipAlgorithms.add(row);
                    }
                }
            }
        }
        if (!nativeCorad && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "corad_tricklet")) {
            if (floatSkipAlgorithms == skipAlgorithms) {
                floatSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
            }
            floatSkipAlgorithms.add("corad_tricklet");
        }
        if (!nativeSimpieceFloat) {
            boolean needSkipSimpiece =
                    !AlgorithmAllowList.shouldSkip(skipAlgorithms, "simpiece_relRange_0p005")
                            || !AlgorithmAllowList.shouldSkip(skipAlgorithms, "simpiece_relRange_0p01");
            if (needSkipSimpiece) {
                if (floatSkipAlgorithms == skipAlgorithms) {
                    floatSkipAlgorithms = new LinkedHashSet<>(skipAlgorithms);
                }
                if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "simpiece_relRange_0p005")) {
                    floatSkipAlgorithms.add("simpiece_relRange_0p005");
                }
                if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "simpiece_relRange_0p01")) {
                    floatSkipAlgorithms.add("simpiece_relRange_0p01");
                }
            }
        }
        runFloatCodecResearchRows(
                legacy, totals, csvWriter, floatSkipAlgorithms, BenchmarkWireFormat.FLOAT_BYTES);
    }

    private static void recordCodecBenchRow(
            String rowName,
            long[] result,
            Set<String> skipAlgorithms,
            BenchmarkTotals totals,
            CSVPrinter csvWriter)
            throws IOException {
        recordCodecBenchRow(rowName, result, null, skipAlgorithms, totals, csvWriter);
    }

    private static void recordCodecBenchRow(
            String rowName,
            long[] result,
            double[] lossAcc,
            Set<String> skipAlgorithms,
            BenchmarkTotals totals,
            CSVPrinter csvWriter)
            throws IOException {
        if (result == null
                || result.length < 4
                || result[0] <= 0
                || AlgorithmAllowList.shouldSkip(skipAlgorithms, rowName)) {
            return;
        }
        double lossErr = lossAcc != null && lossAcc.length >= 2 ? lossAcc[0] : 0.0;
        double lossRef = lossAcc != null && lossAcc.length >= 2 ? lossAcc[1] : 0.0;
        if (totals != null) {
            totals.merge(rowName, result, lossErr, lossRef);
        } else if (csvWriter != null) {
            csvWriter.printRecord(rowName, result[0], result[1], result[2], result[3]);
        }
    }

    /**
     * Float-targeted research codecs. On int benchmark runs, {@code columns} are int32 values cast to
     * float (zero fractional part).
     */
    static void printFloatCodecResearchRows(
            ArrayList<ArrayList<Float>> columns,
            CSVPrinter csvWriter,
            Set<String> skipAlgorithms) throws IOException {
        printFloatCodecResearchRows(columns, csvWriter, skipAlgorithms, Float.BYTES);
    }

    static void printFloatCodecResearchRows(
            ArrayList<ArrayList<Float>> columns,
            CSVPrinter csvWriter,
            Set<String> skipAlgorithms,
            int floatValueBytes) throws IOException {
        runFloatCodecResearchRows(columns, null, csvWriter, skipAlgorithms, floatValueBytes);
    }

    private static void runFloatCodecResearchRows(
            ArrayList<ArrayList<Float>> columns,
            BenchmarkTotals totals,
            CSVPrinter csvWriter,
            Set<String> skipAlgorithms,
            int floatValueBytes) throws IOException {
        final boolean nativeFastPFOR = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_FASTPFOR"));
        final boolean nativeLeCoFloat = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_LECO_FLOAT"));
        final boolean nativeAlpFloat = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_ALP_FLOAT"));
        final boolean nativeElfOnChimpFloat =
                "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_ELF_ON_CHIMP_FLOAT"));
        final boolean nativeElfPlusFloat = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_ELF_PLUS_FLOAT"));
        final boolean nativeCamelFloat = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_CAMEL_FLOAT"));
        final boolean nativeSimpieceFloat =
                "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_SIMPIECE_FLOAT"));
        final boolean nativeSimple8b = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_SIMPLE8B"));
        final boolean nativeSpdp = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_SPDP"));
        final boolean nativeMost = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_MOST"));
        final boolean nativeMachete = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_MACHETE"));
        final boolean nativeCorad = "1".equals(System.getenv("COMPRESSION_BENCHMARK_CPP_NATIVE_CORAD"));

        if (!nativeSpdp) {
            for (int spdpLevel : SpdpFloat.levelsToBench()) {
                String spdpRow = SpdpFloat.rowName(spdpLevel);
                if (AlgorithmAllowList.shouldSkip(skipAlgorithms, spdpRow)) {
                    continue;
                }
                long[] tSp = new long[4];
                double[] lossSp = new double[2];
                SpdpFloat.benchSpdpFloat(columns, spdpLevel, tSp, floatValueBytes, lossSp);
                recordCodecBenchRow(spdpRow, tSp, lossSp, skipAlgorithms, totals, csvWriter);
            }
        }
        if (!nativeMost && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "most_minimal")) {
            long[] tMo = new long[4];
            double[] lossMo = new double[2];
            MostFloatMinimal.benchMostFloatMinimal(columns, tMo, lossMo);
            recordCodecBenchRow("most_minimal", tMo, lossMo, skipAlgorithms, totals, csvWriter);
        }
        if (!nativeMachete) {
            final String[] macheteRows = {
                "machete_rel_1e-1",
                "machete_rel_1e-2",
                "machete_rel_1e-3",
                "machete_rel_1e-4",
            };
            final double[] macheteRels = {1e-1, 1e-2, 1e-3, 1e-4};
            for (int mi = 0; mi < macheteRels.length; mi++) {
                if (AlgorithmAllowList.shouldSkip(skipAlgorithms, macheteRows[mi])) {
                    continue;
                }
                long[] tMa = new long[4];
                double[] lossMa = new double[2];
                MacheteFloatRel.benchMacheteFloatRel(columns, macheteRels[mi], tMa, lossMa);
                recordCodecBenchRow(macheteRows[mi], tMa, lossMa, skipAlgorithms, totals, csvWriter);
            }
        }
        if (!nativeCorad && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "corad_tricklet")) {
            long[] tCo = new long[4];
            double[] lossCo = new double[2];
            CoradTrickletFloat.benchCoradTrickletFloat(columns, tCo, lossCo);
            recordCodecBenchRow("corad_tricklet", tCo, lossCo, skipAlgorithms, totals, csvWriter);
        }
        // Paper chimp32 row omitted from benchmark CSVs (TsFile "CHIMP" still reported separately).
        if (!nativeFastPFOR && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "fastpfor")) {
            long[] tFp = new long[4];
            double[] lossFp = new double[2];
            benchJavaFastPFORFloat(columns, tFp, lossFp);
            recordCodecBenchRow("fastpfor", tFp, lossFp, skipAlgorithms, totals, csvWriter);
        }
        if (!nativeAlpFloat && !AlgorithmAllowList.shouldSkip(skipAlgorithms, "alp")) {
            long[] tAlp = new long[4];
            double[] lossAlp = new double[2];
            AlpFloatBench.benchAlpFloat(columns, tAlp, lossAlp);
            recordCodecBenchRow("alp", tAlp, lossAlp, skipAlgorithms, totals, csvWriter);
        }
        if (!nativeSimpieceFloat) {
            long[] tS0 = new long[4];
            double[] lossS0 = new double[2];
            if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "simpiece_relRange_0p005")
                    && benchSimPieceFloat(columns, 0.005, tS0, lossS0)) {
                recordCodecBenchRow(
                        "simpiece_relRange_0p005", tS0, lossS0, skipAlgorithms, totals, csvWriter);
            }
            long[] tS1 = new long[4];
            double[] lossS1 = new double[2];
            if (!AlgorithmAllowList.shouldSkip(skipAlgorithms, "simpiece_relRange_0p01")
                    && benchSimPieceFloat(columns, 0.01, tS1, lossS1)) {
                recordCodecBenchRow(
                        "simpiece_relRange_0p01", tS1, lossS1, skipAlgorithms, totals, csvWriter);
            }
        }
    }

    static void benchJavaFastPFOR(List<ArrayList<Integer>> columns, long[] out) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        IntegerCODEC codec = new Composition(new FastPFOR(), new VariableByte());
        try {
            for (List<Integer> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                int[] data = new int[n];
                for (int i = 0; i < n; i++) {
                    data[i] = col.get(i);
                }
                int[] compressed = new int[Math.max(n * 2, n + 4096)];
                IntWrapper inputOffset = new IntWrapper(0);
                IntWrapper outputOffset = new IntWrapper(0);
                long t0 = System.nanoTime();
                codec.compress(data, inputOffset, n, compressed, outputOffset);
                long t1 = System.nanoTime();
                compressed = Arrays.copyOf(compressed, outputOffset.get());
                int[] recovered = new int[n];
                IntWrapper uIn = new IntWrapper(0);
                IntWrapper uOut = new IntWrapper(0);
                long t2 = System.nanoTime();
                codec.uncompress(compressed, uIn, compressed.length, recovered, uOut);
                long t3 = System.nanoTime();
                if (uOut.get() != n || !Arrays.equals(data, recovered)) {
                    continue;
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, false);
                comp += (long) compressed.length * 4L;
                ct += t1 - t0;
                dt += t3 - t2;
            }
        } catch (Throwable ignored) {
            out[0] = 0;
            return;
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
    }

    static void benchJavaFastPFORFloat(List<ArrayList<Float>> columns, long[] out) {
        benchJavaFastPFORFloat(columns, out, null);
    }

    static void benchJavaFastPFORFloat(List<ArrayList<Float>> columns, long[] out, double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        IntegerCODEC codec = new Composition(new FastPFOR(), new VariableByte());
        try {
            for (List<Float> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                int[] data = new int[n];
                for (int i = 0; i < n; i++) {
                    data[i] = Float.floatToRawIntBits(col.get(i));
                }
                int[] compressed = new int[Math.max(n * 2, n + 4096)];
                IntWrapper inputOffset = new IntWrapper(0);
                IntWrapper outputOffset = new IntWrapper(0);
                long t0 = System.nanoTime();
                codec.compress(data, inputOffset, n, compressed, outputOffset);
                long t1 = System.nanoTime();
                compressed = Arrays.copyOf(compressed, outputOffset.get());
                int[] recovered = new int[n];
                IntWrapper uIn = new IntWrapper(0);
                IntWrapper uOut = new IntWrapper(0);
                long t2 = System.nanoTime();
                codec.uncompress(compressed, uIn, compressed.length, recovered, uOut);
                long t3 = System.nanoTime();
                if (uOut.get() != n) {
                    continue;
                }
                boolean ok = true;
                for (int i = 0; i < n; i++) {
                    if (recovered[i] != data[i]) {
                        ok = false;
                        break;
                    }
                }
                if (!ok) {
                    continue;
                }
                if (lossAcc != null) {
                    List<Double> origVals = new ArrayList<>(n);
                    List<Double> decVals = new ArrayList<>(n);
                    for (int i = 0; i < n; i++) {
                        origVals.add((double) col.get(i));
                        decVals.add((double) Float.intBitsToFloat(recovered[i]));
                    }
                    BenchmarkLoss.accumulateDoubleListLoss(origVals, decVals, lossAcc);
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += (long) compressed.length * 4L;
                ct += t1 - t0;
                dt += t3 - t2;
            }
        } catch (Throwable ignored) {
            out[0] = 0;
            return;
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
    }

    static void benchJavaFastPFORDouble(List<ArrayList<Double>> columns, long[] out) {
        benchJavaFastPFORDouble(columns, out, null);
    }

    static void benchJavaFastPFORDouble(List<ArrayList<Double>> columns, long[] out, double[] lossAcc) {
        ArrayList<ArrayList<Long>> bitsColumns = new ArrayList<>(columns.size());
        for (List<Double> col : columns) {
            ArrayList<Long> bits = new ArrayList<>(col.size());
            for (Double dv : col) {
                bits.add(Double.doubleToRawLongBits(dv == null ? Double.NaN : dv));
            }
            bitsColumns.add(bits);
        }
        benchJavaFastPFORLong(bitsColumns, out, lossAcc);
    }

    static void benchJavaFastPFORLong(List<? extends List<Long>> columns, long[] out) {
        benchJavaFastPFORLong(columns, out, null);
    }

    static void benchJavaFastPFORLong(List<? extends List<Long>> columns, long[] out, double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        try {
            for (List<Long> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                long[] data = new long[n];
                for (int i = 0; i < n; i++) {
                    Long lv = col.get(i);
                    data[i] = lv == null ? 0L : lv;
                }
                long t0 = System.nanoTime();
                byte[] payload = Int64ColumnCodec.fastPForEncodeInt64(data, n);
                long t1 = System.nanoTime();
                long[] recovered = Int64ColumnCodec.fastPForDecodeInt64(payload, n);
                long t2 = System.nanoTime();
                if (recovered.length != n) {
                    continue;
                }
                boolean ok = true;
                for (int i = 0; i < n; i++) {
                    if (recovered[i] != data[i]) {
                        ok = false;
                        break;
                    }
                }
                if (!ok) {
                    continue;
                }
                if (lossAcc != null) {
                    BenchmarkLoss.accumulateLongArrayLoss(data, recovered, n, lossAcc);
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += payload.length;
                ct += t1 - t0;
                dt += t2 - t1;
            }
        } catch (Throwable ignored) {
            out[0] = 0;
            return;
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
    }

    /** Max decimal scale tried when converting float to fixed-point int (lossless round-trip). */
    private static final int LECO_FLOAT_MAX_DECIMAL_SCALE = 15;

    /** Max decimal scale tried when converting double to fixed-point int (lossless round-trip). */
    private static final int LECO_DOUBLE_MAX_DECIMAL_SCALE = 15;

    /**
     * Smallest {@code s >= 0} with {@code (float)(round(v·10^s)/10^s) == v} (bit-identical), or {@code -1}.
     */
    static int minDecimalScaleExactFloat(float v) {
        if (Float.isNaN(v) || Float.isInfinite(v)) {
            return -1;
        }
        int bits = Float.floatToRawIntBits(v);
        for (int s = 0; s <= LECO_FLOAT_MAX_DECIMAL_SCALE; s++) {
            double pow = Math.pow(10.0, s);
            long m = Math.round((double) v * pow);
            float r = (float) ((double) m / pow);
            if (Float.floatToRawIntBits(r) == bits) {
                return s;
            }
        }
        return -1;
    }

    /**
     * Smallest {@code s >= 0} with {@code (double)(round(v·10^s)/10^s) == v} (bit-identical), or {@code -1}.
     */
    static int minDecimalScaleExactDouble(double v) {
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            return -1;
        }
        // +0.0 and -0.0: decimal round-trip via scaled int cannot preserve -0.0 bits; quantize stores raw bits.
        if (v == 0.0) {
            return 0;
        }
        long bits = Double.doubleToRawLongBits(v);
        for (int s = 0; s <= LECO_DOUBLE_MAX_DECIMAL_SCALE; s++) {
            double pow = Math.pow(10.0, s);
            long m = Math.round(v * pow);
            double r = ((double) m) / pow;
            if (Double.doubleToRawLongBits(r) == bits) {
                return s;
            }
        }
        return -1;
    }

    static int maxLosslessDecimalScaleDoubleColumn(List<? extends Double> col) {
        int colScale = 0;
        for (Double dv : col) {
            if (dv == null) {
                continue;
            }
            int s = minDecimalScaleExactDouble(dv);
            if (s < 0) {
                return -1;
            }
            if (s > colScale) {
                colScale = s;
            }
        }
        return colScale;
    }

    static int effectiveDecimalScaleForColumn(int autoScale, Integer maxPointCap) {
        if (autoScale < 0) {
            return -1;
        }
        if (maxPointCap == null || maxPointCap < 0) {
            return autoScale;
        }
        return Math.min(autoScale, maxPointCap);
    }

    static int[] quantizeDoubleColumnToInt32(List<? extends Double> col, int colScale) {
        long[] wide = Int64ColumnCodec.quantizeDoubleColumn(col, colScale);
        if (wide == null) {
            return null;
        }
        int[] data = new int[wide.length];
        for (int i = 0; i < wide.length; i++) {
            long m = wide[i];
            if (m < Integer.MIN_VALUE || m > Integer.MAX_VALUE) {
                return null;
            }
            data[i] = (int) m;
        }
        return data;
    }

    /**
     * LeCo-style double path: lossless decimal scaling on IEEE-754 doubles, quantize to {@code int64},
     * then FastPFOR+VariableByte on lo/hi int32 limbs (no int32 narrowing of scaled magnitudes).
     */
    static void benchLeCoDoubleScaledIntegersFastPFor(List<ArrayList<Double>> columns, long[] out) {
        benchLeCoDoubleScaledIntegersFastPFor(columns, null, out);
    }

    static void benchLeCoDoubleScaledIntegersFastPFor(
            List<ArrayList<Double>> columns, List<Integer> maxPointPerColumn, long[] out) {
        benchLeCoDoubleScaledIntegersFastPFor(columns, maxPointPerColumn, out, null);
    }

    static void benchLeCoDoubleScaledIntegersFastPFor(
            List<ArrayList<Double>> columns,
            List<Integer> maxPointPerColumn,
            long[] out,
            double[] lossAcc) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        try {
            for (int ci = 0; ci < columns.size(); ci++) {
                List<Double> col = columns.get(ci);
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                long[] data = new long[n];
                for (int i = 0; i < n; i++) {
                    Double value = col.get(i);
                    data[i] = Double.doubleToRawLongBits(value == null ? Double.NaN : value);
                }
                long t0 = System.nanoTime();
                byte[] payload = Int64ColumnCodec.fastPForEncodeInt64(data, n);
                long t1 = System.nanoTime();
                long t2 = System.nanoTime();
                long[] recovered = Int64ColumnCodec.fastPForDecodeInt64(payload, n);
                long t3 = System.nanoTime();
                if (!Arrays.equals(data, recovered)) {
                    continue;
                }
                if (lossAcc != null) {
                    double[] origArr = new double[n];
                    double[] decArr = new double[n];
                    for (int i = 0; i < n; i++) {
                        Double dv = col.get(i);
                        origArr[i] = dv == null ? Double.NaN : dv;
                        decArr[i] = Double.longBitsToDouble(recovered[i]);
                    }
                    BenchmarkLoss.accumulateDoubleArrayLoss(origArr, decArr, n, lossAcc);
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += payload.length;
                ct += t1 - t0;
                dt += t3 - t2;
            }
        } catch (Throwable ex) {
            if ("1".equals(System.getenv("COMPRESSION_BENCHMARK_LECO_DEBUG"))) {
                ex.printStackTrace(System.err);
            }
            out[0] = 0;
            return;
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
    }

    /**
     * LeCo-style float path: per column, use {@code 10^p} with {@code p} = max per-value minimal decimal scale
     * (lossless), quantize to {@code int}, then FastPFOR+VariableByte (same integer codec as {@link
     * #benchJavaFastPFOR}). Compressed size includes 4 bytes per column for {@code p}. Native LeCo C++ codecs are
     * not invoked.
     */
    static void benchLeCoFloatScaledIntegersFastPFor(List<ArrayList<Float>> columns, long[] out) {
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        IntegerCODEC codec = new Composition(new FastPFOR(), new VariableByte());
        try {
            for (List<Float> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                int colScale = 0;
                boolean scaleOk = true;
                for (float v : col) {
                    int s = minDecimalScaleExactFloat(v);
                    if (s < 0) {
                        scaleOk = false;
                        break;
                    }
                    if (s > colScale) {
                        colScale = s;
                    }
                }
                if (!scaleOk) {
                    continue;
                }
                double pow = Math.pow(10.0, colScale);
                int[] data = new int[n];
                for (int i = 0; i < n; i++) {
                    float v = col.get(i);
                    long m = Math.round((double) v * pow);
                    if (m < Integer.MIN_VALUE || m > Integer.MAX_VALUE) {
                        data = null;
                        break;
                    }
                    int im = (int) m;
                    float r = (float) ((double) im / pow);
                    if (Float.floatToRawIntBits(r) != Float.floatToRawIntBits(v)) {
                        data = null;
                        break;
                    }
                    data[i] = im;
                }
                if (data == null) {
                    continue;
                }
                int[] buf = new int[Math.max(n * 2, n + 4096)];
                IntWrapper inputOffset = new IntWrapper(0);
                IntWrapper outputOffset = new IntWrapper(0);
                long t0 = System.nanoTime();
                codec.compress(data, inputOffset, n, buf, outputOffset);
                long t1 = System.nanoTime();
                int[] compressed = Arrays.copyOf(buf, outputOffset.get());
                int[] recovered = new int[n];
                IntWrapper uIn = new IntWrapper(0);
                IntWrapper uOut = new IntWrapper(0);
                long t2 = System.nanoTime();
                codec.uncompress(compressed, uIn, compressed.length, recovered, uOut);
                long t3 = System.nanoTime();
                if (uOut.get() != n) {
                    continue;
                }
                boolean ok = true;
                for (int i = 0; i < n; i++) {
                    float r = (float) ((double) recovered[i] / pow);
                    if (Float.floatToRawIntBits(r) != Float.floatToRawIntBits(col.get(i))) {
                        ok = false;
                        break;
                    }
                }
                if (!ok) {
                    continue;
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += (long) compressed.length * 4L + 4L;
                ct += t1 - t0;
                dt += t3 - t2;
            }
        } catch (Throwable ignored) {
            out[0] = 0;
            return;
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
    }

    /**
     * Chimp32 paper codec. {@code intColumns}: feed each int as {@code Float.intBitsToFloat}
     * so {@link gr.aueb.delorean.chimp.Chimp32#close()} EOS matches the float decompressor.
     */
    static boolean benchChimp32Paper(List<?> columns, boolean intColumns, long[] out) {
        final Class<?> chimpCl;
        try {
            chimpCl = Class.forName("gr.aueb.delorean.chimp.Chimp32");
        } catch (ClassNotFoundException e) {
            return false;
        }
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        try {
            Class<?> decCl = Class.forName("gr.aueb.delorean.chimp.ChimpDecompressor32");
            Method addValue = chimpCl.getMethod("addValue", float.class);
            Method close = chimpCl.getMethod("close");
            Method getOut = chimpCl.getMethod("getOut");
            Constructor<?> chimpCtor = chimpCl.getConstructor();
            Constructor<?> decCtor = decCl.getConstructor(byte[].class);
            Method readPair = decCl.getMethod("readPair");
            // Do not reflect on OutputBitStream (loads fastutil-dependent method signatures);
            // Chimp32.size tracks the same bit length as writtenBits().
            Method getSize = chimpCl.getMethod("getSize");

            for (Object colObj : columns) {
                List<?> col = (List<?>) colObj;
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                Object compressor = chimpCtor.newInstance();
                long t0 = System.nanoTime();
                if (intColumns) {
                    @SuppressWarnings("unchecked")
                    List<Integer> icol = (List<Integer>) col;
                    for (int v : icol) {
                        addValue.invoke(compressor, Float.intBitsToFloat(v));
                    }
                } else {
                    @SuppressWarnings("unchecked")
                    List<Float> fcol = (List<Float>) col;
                    for (float f : fcol) {
                        addValue.invoke(compressor, f);
                    }
                }
                close.invoke(compressor);
                long t1 = System.nanoTime();
                int bits = (Integer) getSize.invoke(compressor);
                int compBytes = (int) (((long) bits + 7L) / 8L);
                byte[] rawBuf = (byte[]) getOut.invoke(compressor);
                byte[] compressed = Arrays.copyOf(rawBuf, compBytes);

                long t2 = System.nanoTime();
                Object decompressor = decCtor.newInstance((Object) compressed);
                int count = 0;
                while (true) {
                    Object val = readPair.invoke(decompressor);
                    if (val == null) {
                        break;
                    }
                    Method getFloat = val.getClass().getMethod("getFloatValue");
                    float got = (Float) getFloat.invoke(val);
                    if (count >= n) {
                        count = -1;
                        break;
                    }
                    if (intColumns) {
                        @SuppressWarnings("unchecked")
                        List<Integer> icol = (List<Integer>) col;
                        int exp = icol.get(count);
                        if (Float.floatToIntBits(got) != exp) {
                            count = -1;
                            break;
                        }
                    } else {
                        @SuppressWarnings("unchecked")
                        List<Float> fcol = (List<Float>) col;
                        float exp = fcol.get(count);
                        if (Float.floatToIntBits(got) != Float.floatToIntBits(exp)) {
                            count = -1;
                            break;
                        }
                    }
                    count++;
                }
                long t3 = System.nanoTime();
                if (count != n) {
                    continue;
                }
                orig += BenchmarkWireFormat.columnOriginalSize(n, true);
                comp += compBytes;
                ct += t1 - t0;
                dt += t3 - t2;
            }
        } catch (Throwable ignored) {
            out[0] = 0;
            return true;
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
        return true;
    }

    /** Official 64-bit Chimp codec from the upstream Elf/Chimp artifact. */
    static boolean benchChimpDouble(List<ArrayList<Double>> columns, long[] out) {
        File elfJar = resolveElfJarForIsolatedBench();
        if (elfJar == null) {
            return false;
        }
        final URL[] urls;
        try {
            urls = elfClasspathUrls(elfJar);
        } catch (IOException e) {
            return false;
        }
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        URLClassLoader cl = new ElfIsolatingLoader(urls, ClassLoader.getSystemClassLoader());
        try {
            Class<?> encCl = Class.forName("gr.aueb.delorean.chimp.Chimp", true, cl);
            Class<?> decCl = Class.forName("gr.aueb.delorean.chimp.ChimpDecompressor", true, cl);
            Constructor<?> encCtor = encCl.getConstructor();
            Constructor<?> decCtor = decCl.getConstructor(byte[].class);
            Method addValue = encCl.getMethod("addValue", double.class);
            Method close = encCl.getMethod("close");
            Method getOut = encCl.getMethod("getOut");
            Method getSize = encCl.getMethod("getSize");
            Method readValue = decCl.getMethod("readValue");
            final int chunkPoints = 1000;

            for (List<Double> col : columns) {
                for (int start = 0; start < col.size(); start += chunkPoints) {
                    int end = Math.min(col.size(), start + chunkPoints);
                    Object compressor = encCtor.newInstance();
                    long t0 = System.nanoTime();
                    for (int i = start; i < end; i++) {
                        addValue.invoke(compressor, col.get(i).doubleValue());
                    }
                    close.invoke(compressor);
                    long t1 = System.nanoTime();
                    int bits = ((Number) getSize.invoke(compressor)).intValue();
                    int compBytes = (int) (((long) bits + 7L) / 8L);
                    byte[] compressed = (byte[]) getOut.invoke(compressor);
                    if (compBytes <= 0 || compBytes > compressed.length) {
                        return false;
                    }

                    Object decompressor = decCtor.newInstance((Object) compressed);
                    long t2 = System.nanoTime();
                    for (int i = start; i < end; i++) {
                        Double got = (Double) readValue.invoke(decompressor);
                        if (got == null
                                || Double.doubleToRawLongBits(got)
                                        != Double.doubleToRawLongBits(col.get(i))) {
                            return false;
                        }
                    }
                    if (readValue.invoke(decompressor) != null) {
                        return false;
                    }
                    long t3 = System.nanoTime();
                    orig += BenchmarkWireFormat.columnOriginalSize(end - start, true);
                    comp += compBytes;
                    ct += t1 - t0;
                    dt += t3 - t2;
                }
            }
        } catch (Throwable failure) {
            if ("1".equals(System.getenv("COMPRESSION_BENCHMARK_ELF_DEBUG"))) {
                failure.printStackTrace(System.err);
            }
            return false;
        } finally {
            try {
                cl.close();
            } catch (IOException ignoredClose) {
                // ignore
            }
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
        return orig > 0;
    }

    /**
     * 32-bit float Elf family: erasing stage plus either Chimp XOR ({@code ElfOnChimpCompressor32}) or
     * Elf+ XOR ({@code ElfCompressor32} / {@code ElfXORCompressor32}). Isolated {@link URLClassLoader} loads
     * bundled {@code gr.aueb.delorean.chimp} from {@code start-compress-1.0-SNAPSHOT.jar}.
     *
     * @param xorBackendField private field on the compressor holding the Chimp32 / ElfXORCompressor32 (must expose {@code getOutputStream()}).
     */
    /** Chimp {@code OutputBitStream} buffer is ~10 KiB; 2048-value chunks can overflow on some columns. */
    private static final int ELF_XOR_CHUNK_POINTS = 1600;

    private static final class ElfChunkMetrics {
        long orig;
        long comp;
        long ct;
        long dt;
    }

    private static boolean elfDoubleChunkRoundTrip(
            List<Double> col,
            int start,
            int end,
            Constructor<?> encCtor,
            Constructor<?> decCtor,
            Method addValue,
            Method close,
            Method getBytes,
            Method decompress,
            Field xorField,
            ElfChunkMetrics metrics,
            double[] lossAcc)
            throws ReflectiveOperationException {
        int chunkLen = end - start;
        if (chunkLen <= 0) {
            return true;
        }
        Object compressor = encCtor.newInstance();
        long t0 = System.nanoTime();
        for (int i = start; i < end; i++) {
            addValue.invoke(compressor, col.get(i));
        }
        close.invoke(compressor);
        long t1 = System.nanoTime();
        Object xorBackend = xorField.get(compressor);
        Method getOutputStream = xorBackend.getClass().getMethod("getOutputStream");
        Object bitStream = getOutputStream.invoke(xorBackend);
        Method writtenBitsM = bitStream.getClass().getMethod("writtenBits");
        long wb = ((Number) writtenBitsM.invoke(bitStream)).longValue();
        int compBytes = (int) ((wb + 7L) / 8L);
        byte[] rawBuf = (byte[]) getBytes.invoke(compressor);
        if (compBytes > rawBuf.length) {
            return false;
        }
        byte[] cbytes = Arrays.copyOf(rawBuf, compBytes);

        long t2 = System.nanoTime();
        Object decompressor = decCtor.newInstance((Object) cbytes);
        @SuppressWarnings("unchecked")
        List<Double> got = (List<Double>) decompress.invoke(decompressor);
        long t3 = System.nanoTime();
        if (got == null || got.size() != chunkLen) {
            return false;
        }
        for (int i = 0; i < chunkLen; i++) {
            if (Double.doubleToRawLongBits(got.get(i))
                    != Double.doubleToRawLongBits(col.get(start + i))) {
                return false;
            }
        }
        if (lossAcc != null) {
            List<Double> origSlice = col.subList(start, end);
            BenchmarkLoss.accumulateDoubleListLoss(origSlice, got, lossAcc);
        }
        metrics.orig += BenchmarkWireFormat.columnOriginalSize(chunkLen, true);
        metrics.comp += compBytes;
        metrics.ct += t1 - t0;
        metrics.dt += t3 - t2;
        return true;
    }

    private static void elfDoubleCompressAdaptive(
            List<Double> col,
            int start,
            int end,
            int maxChunk,
            Constructor<?> encCtor,
            Constructor<?> decCtor,
            Method addValue,
            Method close,
            Method getBytes,
            Method decompress,
            Field xorField,
            ElfChunkMetrics metrics,
            double[] lossAcc) {
        if (start >= end) {
            return;
        }
        int tryEnd = Math.min(end, start + maxChunk);
        try {
            if (elfDoubleChunkRoundTrip(
                    col,
                    start,
                    tryEnd,
                    encCtor,
                    decCtor,
                    addValue,
                    close,
                    getBytes,
                    decompress,
                    xorField,
                    metrics,
                    lossAcc)) {
                elfDoubleCompressAdaptive(
                        col,
                        tryEnd,
                        end,
                        maxChunk,
                        encCtor,
                        decCtor,
                        addValue,
                        close,
                        getBytes,
                        decompress,
                        xorField,
                        metrics,
                        lossAcc);
                return;
            }
        } catch (ReflectiveOperationException ex) {
            if ("1".equals(System.getenv("COMPRESSION_BENCHMARK_ELF_DEBUG"))) {
                ex.printStackTrace(System.err);
            }
        } catch (Throwable ex) {
            if ("1".equals(System.getenv("COMPRESSION_BENCHMARK_ELF_DEBUG"))) {
                ex.printStackTrace(System.err);
            }
        }
        if (tryEnd - start <= 1) {
            return;
        }
        int mid = start + (tryEnd - start) / 2;
        elfDoubleCompressAdaptive(
                col,
                start,
                mid,
                maxChunk,
                encCtor,
                decCtor,
                addValue,
                close,
                getBytes,
                decompress,
                xorField,
                metrics,
                lossAcc);
        elfDoubleCompressAdaptive(
                col,
                mid,
                end,
                maxChunk,
                encCtor,
                decCtor,
                addValue,
                close,
                getBytes,
                decompress,
                xorField,
                metrics,
                lossAcc);
    }

    private static void elfDoubleCompressAdaptive(
            List<Double> col,
            int start,
            int end,
            int maxChunk,
            Constructor<?> encCtor,
            Constructor<?> decCtor,
            Method addValue,
            Method close,
            Method getBytes,
            Method decompress,
            Field xorField,
            ElfChunkMetrics metrics) {
        elfDoubleCompressAdaptive(
                col,
                start,
                end,
                maxChunk,
                encCtor,
                decCtor,
                addValue,
                close,
                getBytes,
                decompress,
                xorField,
                metrics,
                null);
    }

    static boolean benchElfVariant32Float(
            List<ArrayList<Float>> columns,
            String encoderClassName,
            String decoderClassName,
            String xorBackendField,
            long[] out) {
        File elfJar = resolveElfJarForIsolatedBench();
        if (elfJar == null) {
            return false;
        }
        final URL url;
        try {
            url = elfJar.toURI().toURL();
        } catch (MalformedURLException e) {
            return false;
        }
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        URLClassLoader cl = new ElfIsolatingLoader(new URL[]{url}, ClassLoader.getSystemClassLoader());
        try {
            Class<?> encCl = Class.forName(encoderClassName, true, cl);
            Class<?> decCl = Class.forName(decoderClassName, true, cl);
            Constructor<?> encCtor = encCl.getConstructor();
            Constructor<?> decCtor = decCl.getConstructor(byte[].class);
            Method addValue = encCl.getMethod("addValue", float.class);
            Method close = encCl.getMethod("close");
            Method getBytes = encCl.getMethod("getBytes");
            Method decompress = decCl.getMethod("decompress");
            Field xorField = encCl.getDeclaredField(xorBackendField);
            xorField.setAccessible(true);

            // ElfXORCompressor32 wraps a fixed ~5KiB OutputBitStream buffer; long columns exceed it.
            final int elfChunkPoints = 1600;
            for (List<Float> col : columns) {
                try {
                    int n = col.size();
                    if (n == 0) {
                        continue;
                    }
                    long colOrig = 0;
                    long colComp = 0;
                    long colCt = 0;
                    long colDt = 0;
                    for (int start = 0; start < n; start += elfChunkPoints) {
                        int end = Math.min(n, start + elfChunkPoints);
                        Object compressor = encCtor.newInstance();
                        long t0 = System.nanoTime();
                        for (int i = start; i < end; i++) {
                            addValue.invoke(compressor, col.get(i));
                        }
                        close.invoke(compressor);
                        long t1 = System.nanoTime();
                        Object xor32 = xorField.get(compressor);
                        Method getOutputStream = xor32.getClass().getMethod("getOutputStream");
                        Object bitStream = getOutputStream.invoke(xor32);
                        Method writtenBitsM = bitStream.getClass().getMethod("writtenBits");
                        long wb = ((Number) writtenBitsM.invoke(bitStream)).longValue();
                        int compBytes = (int) ((wb + 7L) / 8L);
                        byte[] rawBuf = (byte[]) getBytes.invoke(compressor);
                        if (compBytes > rawBuf.length) {
                            throw new IllegalStateException("Elf compressed payload larger than bitstream buffer");
                        }
                        byte[] cbytes = Arrays.copyOf(rawBuf, compBytes);

                        long t2 = System.nanoTime();
                        Object decompressor = decCtor.newInstance((Object) cbytes);
                        @SuppressWarnings("unchecked")
                        List<Float> got = (List<Float>) decompress.invoke(decompressor);
                        long t3 = System.nanoTime();
                        int chunkLen = end - start;
                        if (got == null || got.size() != chunkLen) {
                            throw new IllegalStateException("Elf decompress length mismatch");
                        }
                        for (int i = 0; i < chunkLen; i++) {
                            if (Float.floatToIntBits(got.get(i)) != Float.floatToIntBits(col.get(start + i))) {
                                throw new IllegalStateException("Elf round-trip mismatch");
                            }
                        }
                        colOrig += BenchmarkWireFormat.columnOriginalSize(chunkLen, true);
                        colComp += compBytes;
                        colCt += t1 - t0;
                        colDt += t3 - t2;
                    }
                    orig += colOrig;
                    comp += colComp;
                    ct += colCt;
                    dt += colDt;
                } catch (Throwable ignoredPerColumn) {
                    continue;
                }
            }
        } catch (Throwable setupFailed) {
            out[0] = 0;
            return false;
        } finally {
            try {
                cl.close();
            } catch (IOException ignoredClose) {
                // ignore
            }
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
        return orig > 0;
    }

    /**
     * IEEE-754 {@code double} Elf (Elf64Utils + 64-bit XOR / Chimp). Matches {@code AbstractElfCompressor#addValue(double)}.
     */
    static boolean benchElfVariant64Double(
            List<ArrayList<Double>> columns,
            String encoderClassName,
            String decoderClassName,
            String xorBackendField,
            long[] out) {
        return benchElfVariant64Double(
                columns, encoderClassName, decoderClassName, xorBackendField, out, null);
    }

    static boolean benchElfVariant64Double(
            List<ArrayList<Double>> columns,
            String encoderClassName,
            String decoderClassName,
            String xorBackendField,
            long[] out,
            double[] lossAcc) {
        File elfJar = resolveElfJarForIsolatedBench();
        if (elfJar == null) {
            return false;
        }
        final URL[] urls;
        try {
            urls = elfClasspathUrls(elfJar);
        } catch (IOException e) {
            return false;
        }
        long orig = 0;
        long comp = 0;
        long ct = 0;
        long dt = 0;
        URLClassLoader cl = new ElfIsolatingLoader(urls, ClassLoader.getSystemClassLoader());
        try {
            Class<?> encCl = Class.forName(encoderClassName, true, cl);
            Class<?> decCl = Class.forName(decoderClassName, true, cl);
            Constructor<?> encCtor = encCl.getConstructor();
            Constructor<?> decCtor = decCl.getConstructor(byte[].class);
            Method addValue = encCl.getMethod("addValue", double.class);
            Method close = encCl.getMethod("close");
            Method getBytes = encCl.getMethod("getBytes");
            Method decompress = decCl.getMethod("decompress");
            Field xorField = encCl.getDeclaredField(xorBackendField);
            xorField.setAccessible(true);

            for (List<Double> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                ElfChunkMetrics metrics = new ElfChunkMetrics();
                elfDoubleCompressAdaptive(
                        col,
                        0,
                        n,
                        ELF_XOR_CHUNK_POINTS,
                        encCtor,
                        decCtor,
                        addValue,
                        close,
                        getBytes,
                        decompress,
                        xorField,
                        metrics,
                        lossAcc);
                orig += metrics.orig;
                comp += metrics.comp;
                ct += metrics.ct;
                dt += metrics.dt;
            }
        } catch (Throwable setupFailed) {
            if ("1".equals(System.getenv("COMPRESSION_BENCHMARK_ELF_DEBUG"))) {
                setupFailed.printStackTrace(System.err);
            }
            out[0] = 0;
            return false;
        } finally {
            try {
                cl.close();
            } catch (IOException ignoredClose) {
                // ignore
            }
        }
        out[0] = orig;
        out[1] = comp;
        out[2] = ct;
        out[3] = dt;
        return orig > 0;
    }

    /** Camel on full {@code double} columns (no float narrowing). */
    static boolean benchCamelDouble(List<ArrayList<Double>> columns, long[] out) {
        final List<URL> urls;
        try {
            urls = camelClasspathUrls();
        } catch (IOException e) {
            return false;
        }
        if (urls.isEmpty()) {
            out[0] = 0;
            return false;
        }
        URLClassLoader cl =
                new CamelIsolatingLoader(urls.toArray(new URL[0]), ClassLoader.getSystemClassLoader());
        try {
            Class<?> compCl =
                    Class.forName("org.urbcomp.startdb.compress.camel.compressor.CamelCompressor", true, cl);
            Class<?> camelDecCl = Class.forName("yyy.ts.compress.camel.CamelDecompressor", true, cl);
            Constructor<?> compCtor = compCl.getConstructor();
            Constructor<?> camelDecCtor = camelDecCl.getConstructor(byte[].class);
            Method readPair = camelDecCl.getMethod("readPair");
            Method addValue = compCl.getMethod("addValue", double.class);
            Method close = compCl.getMethod("close");
            Method getBytes = compCl.getMethod("getBytes");
            Field camelField = compCl.getDeclaredField("camel");
            camelField.setAccessible(true);

            long orig = 0;
            long comp = 0;
            long ct = 0;
            long dt = 0;
            final int camelChunkPoints = 1000;
            for (List<Double> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                long colOrig = 0;
                long colComp = 0;
                long colCt = 0;
                long colDt = 0;
                for (int start = 0; start < n; start += camelChunkPoints) {
                    try {
                        int end = Math.min(n, start + camelChunkPoints);
                        Object compressor = compCtor.newInstance();
                        long t0 = System.nanoTime();
                        for (int i = start; i < end; i++) {
                            addValue.invoke(compressor, col.get(i));
                        }
                        close.invoke(compressor);
                        long t1 = System.nanoTime();
                        Object camelInner = camelField.get(compressor);
                        Method camelGetOs = camelInner.getClass().getMethod("getOutputStream");
                        Object bitStream = camelGetOs.invoke(camelInner);
                        Method writtenBitsM = bitStream.getClass().getMethod("writtenBits");
                        long wb = ((Number) writtenBitsM.invoke(bitStream)).longValue();
                        byte[] rawBuf = (byte[]) getBytes.invoke(compressor);
                        if (rawBuf == null) {
                            continue;
                        }
                        int compBytes = (int) ((wb + 7L) / 8L);
                        if (compBytes <= 0 || compBytes > rawBuf.length) {
                            throw new IllegalStateException("Camel compressed length out of range");
                        }
                        byte[] payload = Arrays.copyOf(rawBuf, compBytes);
                        int chunkLen = end - start;
                        long t2 = System.nanoTime();
                        Object camelDec = camelDecCtor.newInstance((Object) payload);
                        for (int i = 0; i < chunkLen; i++) {
                            Double v = (Double) readPair.invoke(camelDec);
                            if (v == null) {
                                throw new IllegalStateException("Camel decompress early end");
                            }
                            if (Double.doubleToRawLongBits(v)
                                    != Double.doubleToRawLongBits(col.get(start + i))) {
                                throw new IllegalStateException("Camel round-trip mismatch");
                            }
                        }
                        long t3 = System.nanoTime();
                        colOrig += BenchmarkWireFormat.columnOriginalSize(chunkLen, true);
                        colComp += compBytes;
                        colCt += t1 - t0;
                        colDt += t3 - t2;
                    } catch (Throwable ignoredPerChunk) {
                        if (Boolean.parseBoolean(System.getProperty("web.compression.camel.debug", "false"))) {
                            ignoredPerChunk.printStackTrace();
                        }
                    }
                }
                orig += colOrig;
                comp += colComp;
                ct += colCt;
                dt += colDt;
            }
            out[0] = orig;
            out[1] = comp;
            out[2] = ct;
            out[3] = dt;
            return orig > 0;
        } catch (Throwable setupFailed) {
            out[0] = 0;
            return false;
        } finally {
            try {
                cl.close();
            } catch (IOException ignoredClose) {
                // ignore
            }
        }
    }

    /**
     * Camel (lossless float; upstream uses {@code double} API). Loads Camel and Maven dependencies in a
     * child {@link URLClassLoader} to avoid shading into the benchmark JAR.
     */
    static boolean benchCamelFloat(List<ArrayList<Float>> columns, long[] out) {
        final List<URL> urls;
        try {
            urls = camelClasspathUrls();
        } catch (IOException e) {
            return false;
        }
        if (urls.isEmpty()) {
            out[0] = 0;
            return false;
        }
        URLClassLoader cl =
                new CamelIsolatingLoader(urls.toArray(new URL[0]), ClassLoader.getSystemClassLoader());
        try {
            Class<?> compCl =
                    Class.forName("org.urbcomp.startdb.compress.camel.compressor.CamelCompressor", true, cl);
            Class<?> camelDecCl = Class.forName("yyy.ts.compress.camel.CamelDecompressor", true, cl);
            Constructor<?> compCtor = compCl.getConstructor();
            Constructor<?> camelDecCtor = camelDecCl.getConstructor(byte[].class);
            Method readPair = camelDecCl.getMethod("readPair");
            Method addValue = compCl.getMethod("addValue", double.class);
            Method close = compCl.getMethod("close");
            Method getBytes = compCl.getMethod("getBytes");
            Field camelField = compCl.getDeclaredField("camel");
            camelField.setAccessible(true);

            long orig = 0;
            long comp = 0;
            long ct = 0;
            long dt = 0;
            // CamelDecompressor stops after DEFAULT_BLOCK_SIZE (1000) points per bitstream.
            final int camelChunkPoints = 1000;
            for (List<Float> col : columns) {
                int n = col.size();
                if (n == 0) {
                    continue;
                }
                try {
                    long colOrig = 0;
                    long colComp = 0;
                    long colCt = 0;
                    long colDt = 0;
                    for (int start = 0; start < n; start += camelChunkPoints) {
                        int end = Math.min(n, start + camelChunkPoints);
                        Object compressor = compCtor.newInstance();
                        long t0 = System.nanoTime();
                        for (int i = start; i < end; i++) {
                            addValue.invoke(compressor, col.get(i).doubleValue());
                        }
                        close.invoke(compressor);
                        long t1 = System.nanoTime();
                        Object camelInner = camelField.get(compressor);
                        Method camelGetOs = camelInner.getClass().getMethod("getOutputStream");
                        Object bitStream = camelGetOs.invoke(camelInner);
                        Method writtenBitsM = bitStream.getClass().getMethod("writtenBits");
                        long wb = ((Number) writtenBitsM.invoke(bitStream)).longValue();
                        byte[] rawBuf = (byte[]) getBytes.invoke(compressor);
                        if (rawBuf == null) {
                            continue;
                        }
                        int compBytes = (int) ((wb + 7L) / 8L);
                        if (compBytes <= 0 || compBytes > rawBuf.length) {
                            throw new IllegalStateException("Camel compressed length out of range");
                        }
                        // Full getOut() buffer is fixed-size; trailing zeros make getValues() read up to
                        // DEFAULT_BLOCK_SIZE (1000) points. Trim to the exact bitstream byte length.
                        byte[] payload = Arrays.copyOf(rawBuf, compBytes);
                        int chunkLen = end - start;
                        long t2 = System.nanoTime();
                        Object camelDec = camelDecCtor.newInstance((Object) payload);
                        for (int i = 0; i < chunkLen; i++) {
                            Double v = (Double) readPair.invoke(camelDec);
                            if (v == null) {
                                throw new IllegalStateException("Camel decompress early end");
                            }
                            if (Double.doubleToRawLongBits(v)
                                    != Double.doubleToRawLongBits(col.get(start + i).doubleValue())) {
                                throw new IllegalStateException("Camel round-trip mismatch");
                            }
                        }
                        long t3 = System.nanoTime();
                        colOrig += BenchmarkWireFormat.columnOriginalSize(chunkLen, true);
                        colComp += compBytes;
                        colCt += t1 - t0;
                        colDt += t3 - t2;
                    }
                    orig += colOrig;
                    comp += colComp;
                    ct += colCt;
                    dt += colDt;
                } catch (Throwable ignoredPerColumn) {
                    if (Boolean.parseBoolean(System.getProperty("web.compression.camel.debug", "false"))) {
                        ignoredPerColumn.printStackTrace();
                    }
                    continue;
                }
            }
            out[0] = orig;
            out[1] = comp;
            out[2] = ct;
            out[3] = dt;
            return orig > 0;
        } catch (Throwable setupFailed) {
            if (Boolean.parseBoolean(System.getProperty("web.compression.camel.debug", "false"))) {
                setupFailed.printStackTrace();
            }
            out[0] = 0;
            return false;
        } finally {
            try {
                cl.close();
            } catch (IOException ignoredClose) {
                // ignore
            }
        }
    }

    /**
     * Sim-Piece (lossy) via {@link SimPieceFloatNative} (aligned with Python/C++ ports).
     * {@code originalSize} uses 8 bytes/value on double-wire runs
     * ({@link BenchmarkWireFormat#columnOriginalSize} with {@code doubleWire=true}).
     */
    static boolean benchSimPieceFloat(List<ArrayList<Float>> columns, double relRangeFrac, long[] out) {
        return benchSimPieceFloat(columns, relRangeFrac, out, null);
    }

    static boolean benchSimPieceFloat(
            List<ArrayList<Float>> columns, double relRangeFrac, long[] out, double[] lossAcc) {
        return SimPieceFloatNative.benchFloatColumnsRelRange(columns, relRangeFrac, out, lossAcc);
    }

    static boolean benchSimPieceDouble(
            List<? extends List<Double>> columns, double relRangeFrac, long[] out) {
        return benchSimPieceDouble(columns, relRangeFrac, out, null);
    }

    static boolean benchSimPieceDouble(
            List<? extends List<Double>> columns, double relRangeFrac, long[] out, double[] lossAcc) {
        return SimPieceFloatNative.benchDoubleColumnsRelRange(columns, relRangeFrac, out, lossAcc);
    }

    static boolean benchSimPieceLong(
            List<? extends List<Long>> columns, double relRangeFrac, long[] out) {
        return benchSimPieceLong(columns, relRangeFrac, out, null);
    }

    static boolean benchSimPieceLong(
            List<? extends List<Long>> columns, double relRangeFrac, long[] out, double[] lossAcc) {
        return SimPieceFloatNative.benchLongColumnsRelRange(columns, relRangeFrac, out, lossAcc);
    }

    // -------------------------------------------------------------------------
    // Reference compressed payloads for Python ↔ JAR hex parity (scripts/parity_hex_research_codecs.py).
    // Chunking matches benchElfVariant32Float / benchCamelFloat / benchChimp32Paper / benchSimPieceFloat.
    // -------------------------------------------------------------------------

    public static ArrayList<byte[]> parityCompressedChunksElfOnChimp(ArrayList<Float> col) throws Exception {
        return parityCompressedChunksElfVariant(
                col,
                "org.urbcomp.startdb.compress.elf.compressor32.ElfOnChimpCompressor32",
                "org.urbcomp.startdb.compress.elf.decompressor32.ElfOnChimpDecompressor32",
                "chimp32");
    }

    public static ArrayList<byte[]> parityCompressedChunksElfPlus(ArrayList<Float> col) throws Exception {
        return parityCompressedChunksElfVariant(
                col,
                "org.urbcomp.startdb.compress.elf.compressor32.ElfCompressor32",
                "org.urbcomp.startdb.compress.elf.decompressor32.ElfDecompressor32",
                "xorCompressor32");
    }

    private static ArrayList<byte[]> parityCompressedChunksElfVariant(
            ArrayList<Float> col,
            String encoderClassName,
            String decoderClassName,
            String xorBackendField)
            throws Exception {
        ArrayList<byte[]> chunks = new ArrayList<>();
        if (col == null || col.isEmpty()) {
            return chunks;
        }
        File elfJar = resolveElfJarForIsolatedBench();
        if (elfJar == null) {
            throw new IllegalStateException("Elf JAR not found (build external/compression/elf)");
        }
        final URL url = elfJar.toURI().toURL();
        URLClassLoader cl = new ElfIsolatingLoader(new URL[]{url}, ClassLoader.getSystemClassLoader());
        try {
            Class<?> encCl = Class.forName(encoderClassName, true, cl);
            Class<?> decCl = Class.forName(decoderClassName, true, cl);
            Constructor<?> encCtor = encCl.getConstructor();
            Constructor<?> decCtor = decCl.getConstructor(byte[].class);
            Method addValue = encCl.getMethod("addValue", float.class);
            Method close = encCl.getMethod("close");
            Method getBytes = encCl.getMethod("getBytes");
            Method decompress = decCl.getMethod("decompress");
            Field xorField = encCl.getDeclaredField(xorBackendField);
            xorField.setAccessible(true);
            final int elfChunkPoints = 1600;
            int n = col.size();
            for (int start = 0; start < n; start += elfChunkPoints) {
                int end = Math.min(n, start + elfChunkPoints);
                int chunkLen = end - start;
                Object compressor = encCtor.newInstance();
                for (int i = start; i < end; i++) {
                    addValue.invoke(compressor, col.get(i));
                }
                close.invoke(compressor);
                Object xor32 = xorField.get(compressor);
                Method getOutputStream = xor32.getClass().getMethod("getOutputStream");
                Object bitStream = getOutputStream.invoke(xor32);
                Method writtenBitsM = bitStream.getClass().getMethod("writtenBits");
                long wb = ((Number) writtenBitsM.invoke(bitStream)).longValue();
                int compBytes = (int) ((wb + 7L) / 8L);
                byte[] rawBuf = (byte[]) getBytes.invoke(compressor);
                if (compBytes > rawBuf.length) {
                    throw new IllegalStateException("Elf payload larger than buffer");
                }
                byte[] cbytes = Arrays.copyOf(rawBuf, compBytes);
                Object decompressor = decCtor.newInstance((Object) cbytes);
                @SuppressWarnings("unchecked")
                List<Float> got = (List<Float>) decompress.invoke(decompressor);
                if (got == null || got.size() != chunkLen) {
                    throw new IllegalStateException("Elf decompress length");
                }
                for (int i = 0; i < chunkLen; i++) {
                    if (Float.floatToIntBits(got.get(i)) != Float.floatToIntBits(col.get(start + i))) {
                        throw new IllegalStateException("Elf round-trip");
                    }
                }
                chunks.add(cbytes);
            }
        } finally {
            cl.close();
        }
        return chunks;
    }

    public static ArrayList<byte[]> parityCompressedChunksCamel(ArrayList<Float> col) throws Exception {
        ArrayList<byte[]> chunks = new ArrayList<>();
        if (col == null || col.isEmpty()) {
            return chunks;
        }
        List<URL> urls = camelClasspathUrls();
        if (urls.isEmpty()) {
            throw new IllegalStateException("Camel JAR/deps not found");
        }
        URLClassLoader cl =
                new CamelIsolatingLoader(urls.toArray(new URL[0]), ClassLoader.getSystemClassLoader());
        try {
            Class<?> compCl =
                    Class.forName("org.urbcomp.startdb.compress.camel.compressor.CamelCompressor", true, cl);
            Class<?> camelDecCl = Class.forName("yyy.ts.compress.camel.CamelDecompressor", true, cl);
            Constructor<?> compCtor = compCl.getConstructor();
            Constructor<?> camelDecCtor = camelDecCl.getConstructor(byte[].class);
            Method readPair = camelDecCl.getMethod("readPair");
            Method addValue = compCl.getMethod("addValue", double.class);
            Method close = compCl.getMethod("close");
            Method getBytes = compCl.getMethod("getBytes");
            Field camelField = compCl.getDeclaredField("camel");
            camelField.setAccessible(true);
            final int camelChunkPoints = 1000;
            int n = col.size();
            for (int start = 0; start < n; start += camelChunkPoints) {
                int end = Math.min(n, start + camelChunkPoints);
                int chunkLen = end - start;
                Object compressor = compCtor.newInstance();
                for (int i = start; i < end; i++) {
                    addValue.invoke(compressor, col.get(i).doubleValue());
                }
                close.invoke(compressor);
                Object camelInner = camelField.get(compressor);
                Method camelGetOs = camelInner.getClass().getMethod("getOutputStream");
                Object bitStream = camelGetOs.invoke(camelInner);
                Method writtenBitsM = bitStream.getClass().getMethod("writtenBits");
                long wb = ((Number) writtenBitsM.invoke(bitStream)).longValue();
                byte[] rawBuf = (byte[]) getBytes.invoke(compressor);
                if (rawBuf == null) {
                    throw new IllegalStateException("Camel null buffer");
                }
                int compBytes = (int) ((wb + 7L) / 8L);
                if (compBytes <= 0 || compBytes > rawBuf.length) {
                    throw new IllegalStateException("Camel length");
                }
                byte[] payload = Arrays.copyOf(rawBuf, compBytes);
                Object camelDec = camelDecCtor.newInstance((Object) payload);
                for (int i = 0; i < chunkLen; i++) {
                    Double v = (Double) readPair.invoke(camelDec);
                    if (v == null) {
                        throw new IllegalStateException("Camel early EOS");
                    }
                    if (Double.doubleToRawLongBits(v)
                            != Double.doubleToRawLongBits(col.get(start + i).doubleValue())) {
                        throw new IllegalStateException("Camel round-trip");
                    }
                }
                chunks.add(payload);
            }
        } finally {
            cl.close();
        }
        return chunks;
    }

    public static byte[] parityCompressedChimp32FloatWholeColumn(ArrayList<Float> col) throws Exception {
        if (col == null || col.isEmpty()) {
            throw new IllegalArgumentException("empty column");
        }
        Class<?> chimpCl = Class.forName("gr.aueb.delorean.chimp.Chimp32");
        Class<?> decCl = Class.forName("gr.aueb.delorean.chimp.ChimpDecompressor32");
        Method addValue = chimpCl.getMethod("addValue", float.class);
        Method close = chimpCl.getMethod("close");
        Method getOut = chimpCl.getMethod("getOut");
        Constructor<?> chimpCtor = chimpCl.getConstructor();
        Constructor<?> decCtor = decCl.getConstructor(byte[].class);
        Method readPair = decCl.getMethod("readPair");
        Method getSize = chimpCl.getMethod("getSize");
        int n = col.size();
        Object compressor = chimpCtor.newInstance();
        for (int i = 0; i < n; i++) {
            addValue.invoke(compressor, col.get(i));
        }
        close.invoke(compressor);
        int bits = (Integer) getSize.invoke(compressor);
        int compBytes = (int) (((long) bits + 7L) / 8L);
        byte[] rawBuf = (byte[]) getOut.invoke(compressor);
        byte[] compressed = Arrays.copyOf(rawBuf, compBytes);
        Object decompressor = decCtor.newInstance((Object) compressed);
        int count = 0;
        while (true) {
            Object val = readPair.invoke(decompressor);
            if (val == null) {
                break;
            }
            Method getFloat = val.getClass().getMethod("getFloatValue");
            float got = (Float) getFloat.invoke(val);
            if (count >= n || Float.floatToIntBits(got) != Float.floatToIntBits(col.get(count))) {
                throw new IllegalStateException("Chimp round-trip");
            }
            count++;
        }
        if (count != n) {
            throw new IllegalStateException("Chimp count");
        }
        return compressed;
    }

    public static byte[] parityCompressedSimPiece(ArrayList<Float> col, double relRangeFrac) throws Exception {
        byte[] packed = SimPieceFloatNative.compressColumn(col, relRangeFrac);
        if (!SimPieceFloatNative.verifyTolerance(packed, col, relRangeFrac)) {
            throw new IllegalStateException("SimPiece tolerance");
        }
        return packed;
    }
}
