package org.example;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * Canonical byte Huffman for benchmark row {@code huffman}.
 *
 * <p>Layout: {@code u32 orig_len LE} + {@code 256 * u32 freq LE} + {@code u32 num_bits LE} +
 * ceil(num_bits/8) payload bytes. Bits are MSB-first within each payload byte (first bit of stream
 * is bit 7 of the first payload byte). Tree merge tie-break: lower frequency first; equal
 * frequency then lower {@code minSym}. Left child = bit 0, right = 1. Single-symbol payload uses
 * {@code num_bits = 0} and repeats the sole byte {@code orig_len} times.
 */
public final class HuffmanBytes {

    private static final int HEADER = 4 + 1024 + 4;

    private HuffmanBytes() {}

    private static final class Node {
        final long freq;
        final int minSym;
        final int sym; // -1 internal
        final Node left;
        final Node right;

        Node(long freq, int minSym, int sym, Node left, Node right) {
            this.freq = freq;
            this.minSym = minSym;
            this.sym = sym;
            this.left = left;
            this.right = right;
        }

        boolean isLeaf() {
            return left == null && right == null;
        }
    }

    private static final Comparator<Node> NODE_CMP =
            Comparator.comparingLong((Node n) -> n.freq)
                    .thenComparingInt(n -> n.minSym);

    static long[] countFreq(byte[] data) {
        long[] f = new long[256];
        if (data != null) {
            for (byte b : data) {
                f[b & 0xff]++;
            }
        }
        return f;
    }

    static Node buildTree(long[] freq) {
        PriorityQueue<Node> pq = new PriorityQueue<>(NODE_CMP);
        for (int s = 0; s < 256; s++) {
            if (freq[s] > 0) {
                pq.add(new Node(freq[s], s, s, null, null));
            }
        }
        if (pq.isEmpty()) {
            return null;
        }
        if (pq.size() == 1) {
            return pq.poll();
        }
        while (pq.size() > 1) {
            Node a = pq.poll();
            Node b = pq.poll();
            int ms = Math.min(a.minSym, b.minSym);
            long fr = a.freq + b.freq;
            pq.add(new Node(fr, ms, -1, a, b));
        }
        return pq.poll();
    }

    static void assignCodes(Node root, int[] code, int[] len) {
        Arrays.fill(len, 0);
        Arrays.fill(code, 0);
        if (root == null) {
            return;
        }
        if (root.isLeaf()) {
            len[root.sym] = 1;
            code[root.sym] = 0;
            return;
        }
        dfsAssign(root, 0, 0, code, len);
    }

    private static void dfsAssign(Node n, int c, int d, int[] code, int[] len) {
        if (n.isLeaf()) {
            len[n.sym] = d;
            code[n.sym] = d == 0 ? 0 : c;
            return;
        }
        dfsAssign(n.left, (c << 1), d + 1, code, len);
        dfsAssign(n.right, (c << 1) | 1, d + 1, code, len);
    }

    public static byte[] encode(byte[] data) {
        int origLen = data == null ? 0 : data.length;
        long[] freq = countFreq(data);
        ByteBuffer hdr = ByteBuffer.allocate(HEADER).order(ByteOrder.LITTLE_ENDIAN);
        hdr.putInt(origLen);
        for (int i = 0; i < 256; i++) {
            hdr.putInt((int) Math.min(freq[i], 0xffffffffL));
        }

        if (origLen == 0) {
            hdr.putInt(0);
            return hdr.array();
        }

        int onlySym = -1;
        int nz = 0;
        for (int s = 0; s < 256; s++) {
            if (freq[s] > 0) {
                nz++;
                onlySym = s;
            }
        }
        if (nz == 1) {
            hdr.putInt(0);
            return hdr.array();
        }

        Node root = buildTree(freq);
        int[] symCode = new int[256];
        int[] symLen = new int[256];
        assignCodes(root, symCode, symLen);

        BitWriter bw = new BitWriter();
        for (int i = 0; i < origLen; i++) {
            int s = data[i] & 0xff;
            int L = symLen[s];
            int v = symCode[s];
            for (int b = L - 1; b >= 0; b--) {
                bw.append((v >>> b) & 1);
            }
        }
        int numBits = bw.bitLength();
        hdr.putInt(numBits);
        byte[] body = bw.toByteArray();
        byte[] out = new byte[HEADER + body.length];
        System.arraycopy(hdr.array(), 0, out, 0, HEADER);
        System.arraycopy(body, 0, out, HEADER, body.length);
        return out;
    }

    public static byte[] decode(byte[] comp) {
        if (comp.length < HEADER) {
            throw new IllegalArgumentException("huffman: truncated header");
        }
        ByteBuffer bb = ByteBuffer.wrap(comp, 0, HEADER).order(ByteOrder.LITTLE_ENDIAN);
        int origLen = bb.getInt();
        long[] freq = new long[256];
        for (int i = 0; i < 256; i++) {
            freq[i] = bb.getInt() & 0xffffffffL;
        }
        int numBits = bb.getInt();
        if (origLen == 0) {
            return new byte[0];
        }
        int onlySym = -1;
        int nz = 0;
        for (int s = 0; s < 256; s++) {
            if (freq[s] > 0) {
                nz++;
                onlySym = s;
            }
        }
        if (nz == 1) {
            byte[] out = new byte[origLen];
            Arrays.fill(out, (byte) onlySym);
            return out;
        }
        if (numBits < 0 || comp.length < HEADER + (numBits + 7) / 8) {
            throw new IllegalArgumentException("huffman: truncated payload");
        }
        Node root = buildTree(freq);

        BitReader br = new BitReader(comp, HEADER, numBits);
        byte[] out = new byte[origLen];
        for (int i = 0; i < origLen; i++) {
            Node n = root;
            while (!n.isLeaf()) {
                int bit = br.readBit();
                n = bit == 0 ? n.left : n.right;
            }
            out[i] = (byte) n.sym;
        }
        return out;
    }

    public static void bench(byte[] data, long[] result) {
        bench(data, result, null);
    }

    public static void bench(byte[] data, long[] result, double[] lossAcc) {
        long t0 = System.nanoTime();
        byte[] c = encode(data);
        long t1 = System.nanoTime();
        byte[] d = decode(c);
        long t2 = System.nanoTime();
        if (data == null) {
            data = new byte[0];
        }
        if (!Arrays.equals(data, d)) {
            BenchmarkLoss.accumulateByteLoss(data, d, lossAcc);
            throw new IllegalStateException("huffman roundtrip failed len=" + data.length);
        }
        BenchmarkLoss.accumulateByteLoss(data, d, lossAcc);
        result[0] = data.length;
        result[1] = c.length;
        result[2] = t1 - t0;
        result[3] = t2 - t1;
    }

    private static final class BitWriter {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();
        private int cur;
        private int nbits;

        void append(int bit) {
            cur = (cur << 1) | (bit & 1);
            nbits++;
            if (nbits == 8) {
                out.write(cur);
                cur = 0;
                nbits = 0;
            }
        }

        int bitLength() {
            return (out.size() * 8) + nbits;
        }

        byte[] toByteArray() {
            if (nbits > 0) {
                cur <<= (8 - nbits);
                out.write(cur);
            }
            return out.toByteArray();
        }
    }

    private static final class BitReader {
        private final byte[] data;
        private final int start;
        private final int numBits;
        private int pos;

        BitReader(byte[] data, int start, int numBits) {
            this.data = data;
            this.start = start;
            this.numBits = numBits;
            this.pos = 0;
        }

        int readBit() {
            if (pos >= numBits) {
                throw new IllegalStateException("huffman: read past end of bitstream");
            }
            int byteIdx = start + (pos >>> 3);
            int sub = 7 - (pos & 7);
            pos++;
            return (data[byteIdx] >>> sub) & 1;
        }
    }
}
