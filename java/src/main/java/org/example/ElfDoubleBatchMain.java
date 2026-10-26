package org.example;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Subprocess entry for Python to run IEEE-754 {@code double} Elf benches (64-bit wire / Elf64Utils).
 * Stdin: {@code int numCols}, then per column {@code int len} + {@code len} little-endian doubles.
 * Stdout: {@code long orig}, {@code long comp}, {@code long encNs}, {@code long decNs}.
 */
public final class ElfDoubleBatchMain {

    private ElfDoubleBatchMain() {}

    public static void main(String[] args) throws Exception {
        String mode = "elf_plus";
        byte[] payload;
        if (args.length > 0
                && ("elf_on_chimp".equals(args[0])
                        || "elf_plus".equals(args[0])
                        || "chimp".equals(args[0]))) {
            mode = args[0];
            payload = System.in.readAllBytes();
        } else if (args.length > 0) {
            payload = Files.readAllBytes(Path.of(args[0]));
            mode = args.length > 1 ? args[1] : "elf_plus";
        } else {
            payload = System.in.readAllBytes();
        }
        ByteBuffer buf = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        if (buf.remaining() < 4) {
            System.exit(2);
            return;
        }
        int numCols = buf.getInt();
        if (numCols < 0) {
            System.exit(2);
            return;
        }
        List<ArrayList<Double>> columns = new ArrayList<>();
        for (int c = 0; c < numCols; c++) {
            if (buf.remaining() < 4) {
                System.exit(3);
                return;
            }
            int len = buf.getInt();
            if (len < 0 || buf.remaining() < len * 8) {
                System.exit(3);
                return;
            }
            ArrayList<Double> col = new ArrayList<>(len);
            for (int i = 0; i < len; i++) {
                col.add(buf.getDouble());
            }
            columns.add(col);
        }

        long[] t = new long[4];
        boolean ok;
        if ("chimp".equals(mode)) {
            ok = ExternalCompressionBench.benchChimpDouble(columns, t);
        } else if ("elf_on_chimp".equals(mode)) {
            ok =
                    ExternalCompressionBench.benchElfVariant64Double(
                            columns,
                            "org.urbcomp.startdb.compress.elf.compressor.ElfOnChimpCompressor",
                            "org.urbcomp.startdb.compress.elf.decompressor.ElfOnChimpDecompressor",
                            "chimp",
                            t);
        } else {
            ok =
                    ExternalCompressionBench.benchElfVariant64Double(
                            columns,
                            "org.urbcomp.startdb.compress.elf.compressor.ElfCompressor",
                            "org.urbcomp.startdb.compress.elf.decompressor.ElfDecompressor",
                            "xorCompressor",
                            t);
        }
        ByteBuffer bb = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN);
        bb.putLong(t[0]);
        bb.putLong(t[1]);
        bb.putLong(t[2]);
        bb.putLong(t[3]);
        System.out.write(bb.array());
        System.out.flush();
        if (!ok || t[0] <= 0) {
            if ("1".equals(System.getenv("COMPRESSION_BENCHMARK_ELF_DEBUG"))) {
                System.err.println(
                        "ElfDoubleBatchMain failed ok=" + ok + " orig=" + t[0] + " mode=" + mode);
            }
            System.exit(4);
        }
    }
}
