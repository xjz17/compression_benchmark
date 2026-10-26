package org.example;

import java.io.IOException;
import java.math.BigDecimal;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.tsfile.encoding.decoder.Decoder;
import org.apache.tsfile.encoding.encoder.Encoder;
import org.apache.tsfile.encoding.encoder.TSEncodingBuilder;
import org.apache.tsfile.enums.TSDataType;
import org.apache.tsfile.file.metadata.enums.TSEncoding;

import static java.lang.Math.min;
import static java.lang.Math.pow;

/**
 * First-difference + BOS improved encoding from Apache IoTDB
 * {@code research/encoding-outlier} ({@code TSDIFFBOSBImproveTest.java}).
 * Apache-2.0; vendored for benchmark use.
 */
public final class Ts2DiffBosImprove {
    private Ts2DiffBosImprove() {}

    public static long combine2Int(int int1, int int2) {
        return ((long) int1 << 32) | (int2 & 0xFFFFFFFFL);
    }

    public static int getTime(long long1) {
        return ((int) (long1 >> 32));
    }

    public static int getValue(long long1) {
        return ((int) (long1));
    }

    public static int getCount(long long1, int mask) {
        return ((int) (long1 & mask));
    }
    public static int getUniqueValue(long long1, int left_shift) {
        return ((int) ((long1) >> left_shift));
    }

    public static int getBitWith(int num) {
        if (num == 0) return 1;
        else return 32 - Integer.numberOfLeadingZeros(num);
    }

    public static void int2Bytes(int integer,int encode_pos , byte[] cur_byte) {
        cur_byte[encode_pos] = (byte) (integer >> 24);
        cur_byte[encode_pos+1] = (byte) (integer >> 16);
        cur_byte[encode_pos+2] = (byte) (integer >> 8);
        cur_byte[encode_pos+3] = (byte) (integer);
    }


    public static void intByte2Bytes(int integer, int encode_pos , byte[] cur_byte) {
        cur_byte[encode_pos] = (byte) (integer);
    }

    private static void long2intBytes(long integer, int encode_pos , byte[] cur_byte) {
        cur_byte[encode_pos] = (byte) (integer >> 24);
        cur_byte[encode_pos+1] = (byte) (integer >> 16);
        cur_byte[encode_pos+2] = (byte) (integer >> 8);
        cur_byte[encode_pos+3] = (byte) (integer);
    }

    public static int bytes2Integer(byte[] encoded, int start, int num) {
        int value = 0;
        if (num > 4) {
            System.out.println("bytes2Integer error");
            return 0;
        }
        for (int i = 0; i < num; i++) {
            value <<= 8;
            int b = encoded[i + start] & 0xFF;
            value |= b;
        }
        return value;
    }

    private static long bytesLong2Integer(byte[] encoded, int decode_pos) {
        long value = 0;
        for (int i = 0; i < 4; i++) {
            value <<= 8;
            int b = encoded[i + decode_pos] & 0xFF;
            value |= b;
        }
        return value;
    }

    public static void pack8Values(ArrayList<Integer> values, int offset, int width, int encode_pos,  byte[] encoded_result) {
        int bufIdx = 0;
        int valueIdx = offset;
        // remaining bits for the current unfinished Integer
        int leftBit = 0;

        while (valueIdx < 8 + offset) {
            // buffer is used for saving 32 bits as a part of result
            int buffer = 0;
            // remaining size of bits in the 'buffer'
            int leftSize = 32;

            // encode the left bits of current Integer to 'buffer'
            if (leftBit > 0) {
                buffer |= (values.get(valueIdx) << (32 - leftBit));
                leftSize -= leftBit;
                leftBit = 0;
                valueIdx++;
            }

            while (leftSize >= width && valueIdx < 8 + offset) {
                // encode one Integer to the 'buffer'
                buffer |= (values.get(valueIdx)<< (leftSize - width));
                leftSize -= width;
                valueIdx++;
            }
            // If the remaining space of the buffer can not save the bits for one Integer,
            if (leftSize > 0 && valueIdx < 8 + offset) {
                // put the first 'leftSize' bits of the Integer into remaining space of the
                // buffer
                buffer |= (values.get(valueIdx) >>> (width - leftSize));
                leftBit = width - leftSize;
            }

            // put the buffer into the final result
            for (int j = 0; j < 4; j++) {
                encoded_result[encode_pos] = (byte) ((buffer >>> ((3 - j) * 8)) & 0xFF);
                encode_pos ++;
                bufIdx++;
                if (bufIdx >= width) {
                    return ;
                }
            }
        }
//        return encode_pos;
    }

    public static void unpack8Values(byte[] encoded, int offset,int width,  ArrayList<Integer> result_list) {
        int byteIdx = offset;
        long buffer = 0;
        // total bits which have read from 'buf' to 'buffer'. i.e.,
        // number of available bits to be decoded.
        int totalBits = 0;
        int valueIdx = 0;

        while (valueIdx < 8) {
            // If current available bits are not enough to decode one Integer,
            // then add next byte from buf to 'buffer' until totalBits >= width
            while (totalBits < width) {
                buffer = (buffer << 8) | (encoded[byteIdx] & 0xFF);
                byteIdx++;
                totalBits += 8;
            }

            // If current available bits are enough to decode one Integer,
            // then decode one Integer one by one until left bits in 'buffer' is
            // not enough to decode one Integer.
            while (totalBits >= width && valueIdx < 8) {
                result_list.add ((int) (buffer >>> (totalBits - width)));
                valueIdx++;
                totalBits -= width;
                buffer = buffer & ((1L << totalBits) - 1);
            }
        }
    }

    public static int bitPacking(ArrayList<Integer> numbers, int start, int bit_width,int encode_pos,  byte[] encoded_result) {
        int block_num = (numbers.size()-start) / 8;
        for(int i=0;i<block_num;i++){
            pack8Values( numbers, start+i*8, bit_width,encode_pos, encoded_result);
            encode_pos +=bit_width;
        }

        return encode_pos;

    }

    public static ArrayList<Integer> decodeBitPacking(
            byte[] encoded, int decode_pos, int bit_width, int block_size) {
        ArrayList<Integer> result_list = new ArrayList<>();
        int block_num = (block_size - 1) / 8;

        for (int i = 0; i < block_num; i++) { // bitpacking
            unpack8Values( encoded, decode_pos, bit_width,  result_list);
            decode_pos += bit_width;

        }
        return result_list;
    }


    public static int[] getAbsDeltaTsBlock(
            int[] ts_block,
            int i,
            int block_size,
            int remaining,
            int[] min_delta) {
        int[] ts_block_delta = new int[remaining-1];

        int value_delta_min = Integer.MAX_VALUE;
        int value_delta_max = Integer.MIN_VALUE;
        int base = i*block_size+1;
        int end = i*block_size+remaining;

        int tmp_j_1 = ts_block[base-1];
        min_delta[0] =tmp_j_1;
        int j = base;
        int tmp_j;

        while(j<end){
            tmp_j = ts_block[j];
            int epsilon_v = tmp_j - tmp_j_1;
            ts_block_delta[j-base] = epsilon_v;
            if (epsilon_v < value_delta_min) {
                value_delta_min = epsilon_v;
            }
            if (epsilon_v > value_delta_max) {
                value_delta_max = epsilon_v;
            }
            tmp_j_1 = tmp_j;
            j++;
        }
        j = 0;
        end = remaining -1;
        while(j<end){
            ts_block_delta[j] = ts_block_delta[j] - value_delta_min;
            j++;
        }

        min_delta[1] = value_delta_min;
        min_delta[2] = (value_delta_max-value_delta_min);


        return ts_block_delta;
    }

    /**
     * Per-block delta histogram. Uses a sparse map so a wide delta span does not allocate
     * {@code new int[max_delta + 1]} (which can exhaust the heap on large patches).
     */
    private static final class DeltaValueHistogram {
        final int uniqueCount;
        final int[] values;
        final int[] counts;

        DeltaValueHistogram(int uniqueCount, int[] values, int[] counts) {
            this.uniqueCount = uniqueCount;
            this.values = values;
            this.counts = counts;
        }

        int countAt(int index) {
            return counts[index];
        }
    }

    private static DeltaValueHistogram buildDeltaValueHistogram(int[] ts_block_delta) {
        int n = ts_block_delta.length;
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for (int value : ts_block_delta) {
            if (value < min) {
                min = value;
            }
            if (value > max) {
                max = value;
            }
        }
        long rangeLong = (long) max - (long) min + 1L;
        if (rangeLong > 0 && rangeLong <= Math.min(65536L, (long) n * 4L)) {
            int range = (int) rangeLong;
            int[] denseCounts = new int[range];
            for (int value : ts_block_delta) {
                denseCounts[value - min]++;
            }
            int unique = 0;
            for (int c : denseCounts) {
                if (c != 0) {
                    unique++;
                }
            }
            int[] values = new int[unique];
            int[] counts = new int[unique];
            int pos = 0;
            for (int i = 0; i < range; i++) {
                int c = denseCounts[i];
                if (c != 0) {
                    values[pos] = min + i;
                    counts[pos] = c;
                    pos++;
                }
            }
            return new DeltaValueHistogram(unique, values, counts);
        }
        int capacity = 1;
        while (capacity < n * 4) {
            capacity <<= 1;
        }
        int[] keys = new int[capacity];
        int[] slotToIndex = new int[capacity];
        boolean[] used = new boolean[capacity];
        int[] values = new int[n];
        int[] counts = new int[n];
        int unique = 0;
        int mask = capacity - 1;
        for (int value : ts_block_delta) {
            int slot = (value * 0x9E3779B9) & mask;
            while (used[slot] && keys[slot] != value) {
                slot = (slot + 1) & mask;
            }
            if (!used[slot]) {
                used[slot] = true;
                keys[slot] = value;
                slotToIndex[slot] = unique;
                values[unique++] = value;
                counts[unique - 1] = 1;
            } else {
                counts[slotToIndex[slot]]++;
            }
        }
        return new DeltaValueHistogram(unique, values, counts);
    }

    private static boolean isDenseUniformNoOutlierCase(DeltaValueHistogram hist, int maxDeltaValue) {
        if (maxDeltaValue < 0 || maxDeltaValue > 16 || hist.uniqueCount != maxDeltaValue + 1) {
            return false;
        }
        int minCount = Integer.MAX_VALUE;
        int maxCount = 0;
        boolean[] seen = new boolean[maxDeltaValue + 1];
        for (int i = 0; i < hist.uniqueCount; i++) {
            int v = hist.values[i];
            if (v < 0 || v > maxDeltaValue || seen[v]) {
                return false;
            }
            seen[v] = true;
            int c = hist.counts[i];
            if (c <= 0) {
                return false;
            }
            minCount = Math.min(minCount, c);
            maxCount = Math.max(maxCount, c);
        }
        return minCount > 0 && maxCount <= minCount * 2;
    }


    public static int encodeOutlier2Bytes(
            ArrayList<Integer> ts_block_delta,
            int bit_width,
            int encode_pos,  byte[] encoded_result) {

        encode_pos = bitPacking(ts_block_delta, 0, bit_width, encode_pos, encoded_result);

        int n_k = ts_block_delta.size();
        int n_k_b = n_k / 8;
        long cur_remaining = 0; // encoded int
        int cur_number_bits = 0; // the bit width used of encoded int
        for (int i = n_k_b * 8; i < n_k; i++) {
            long cur_value = ts_block_delta.get(i);
            int cur_bit_width = bit_width; // remaining bit width of current value

            if (cur_number_bits + bit_width >= 32) {
                cur_remaining <<= (32 - cur_number_bits);
                cur_bit_width = bit_width - 32 + cur_number_bits;
                cur_remaining += ((cur_value >> cur_bit_width));
                long2intBytes(cur_remaining,encode_pos,encoded_result);
                encode_pos += 4;
                cur_remaining = 0;
                cur_number_bits = 0;
            }

            cur_remaining <<= cur_bit_width;
            cur_number_bits += cur_bit_width;
            cur_remaining += (((cur_value << (32 - cur_bit_width)) & 0xFFFFFFFFL) >> (32 - cur_bit_width)); //
        }
        cur_remaining <<= (32 - cur_number_bits);
        long2intBytes(cur_remaining,encode_pos,encoded_result);
        encode_pos += 4;
        return encode_pos;

    }

    private static void pack8Values(int[] values, int offset, int width, int encodePos, byte[] encodedResult) {
        int bufIdx = 0;
        int valueIdx = offset;
        int leftBit = 0;
        while (valueIdx < offset + 8) {
            int buffer = 0;
            int leftSize = 32;
            if (leftBit > 0) {
                buffer |= (values[valueIdx] << (32 - leftBit));
                leftSize -= leftBit;
                leftBit = 0;
                valueIdx++;
            }
            while (leftSize >= width && valueIdx < offset + 8) {
                buffer |= (values[valueIdx] << (leftSize - width));
                leftSize -= width;
                valueIdx++;
            }
            if (leftSize > 0 && valueIdx < offset + 8) {
                buffer |= (values[valueIdx] >>> (width - leftSize));
                leftBit = width - leftSize;
            }
            for (int j = 0; j < 4; j++) {
                encodedResult[encodePos++] = (byte) ((buffer >>> ((3 - j) * 8)) & 0xFF);
                bufIdx++;
                if (bufIdx >= width) {
                    return;
                }
            }
        }
    }

    private static int encodeOutlier2Bytes(
            int[] values, int length, int bitWidth, int encodePos, byte[] encodedResult) {
        int fullGroups = length / 8;
        for (int i = 0; i < fullGroups; i++) {
            pack8Values(values, i * 8, bitWidth, encodePos, encodedResult);
            encodePos += bitWidth;
        }
        long curRemaining = 0;
        int curNumberBits = 0;
        for (int i = fullGroups * 8; i < length; i++) {
            long curValue = values[i];
            int curBitWidth = bitWidth;
            if (curNumberBits + bitWidth >= 32) {
                curRemaining <<= (32 - curNumberBits);
                curBitWidth = bitWidth - 32 + curNumberBits;
                curRemaining += (curValue >> curBitWidth);
                long2intBytes(curRemaining, encodePos, encodedResult);
                encodePos += 4;
                curRemaining = 0;
                curNumberBits = 0;
            }
            curRemaining <<= curBitWidth;
            curNumberBits += curBitWidth;
            curRemaining += (((curValue << (32 - curBitWidth)) & 0xFFFFFFFFL) >> (32 - curBitWidth));
        }
        curRemaining <<= (32 - curNumberBits);
        long2intBytes(curRemaining, encodePos, encodedResult);
        return encodePos + 4;
    }


    public static ArrayList<Integer> decodeOutlier2Bytes(
            byte[] encoded,
            int decode_pos,
            int bit_width,
            int length,
            ArrayList<Integer> encoded_pos_result
    ) {

        int n_k_b = length / 8;
        int remaining = length - n_k_b * 8;
        ArrayList<Integer> result_list = new ArrayList<>(decodeBitPacking(encoded, decode_pos, bit_width, n_k_b * 8 + 1));
        decode_pos += n_k_b * bit_width;

        ArrayList<Long> int_remaining = new ArrayList<>();
        int int_remaining_size = remaining * bit_width / 32 + 1;
        for (int j = 0; j < int_remaining_size; j++) {
            int_remaining.add(bytesLong2Integer(encoded, decode_pos));
            decode_pos += 4;
        }

        int cur_remaining_bits = 32; // remaining bit width of current value
        long cur_number = int_remaining.get(0);
        int cur_number_i = 1;
        for (int i = n_k_b * 8; i < length; i++) {
            if (bit_width < cur_remaining_bits) {
                int tmp = (int) (cur_number >> (32 - bit_width));
                result_list.add(tmp);
                cur_number <<= bit_width;
                cur_number &= 0xFFFFFFFFL;
                cur_remaining_bits -= bit_width;
            } else {
                int tmp = (int) (cur_number >> (32 - cur_remaining_bits));
                int remain_bits = bit_width - cur_remaining_bits;
                tmp <<= remain_bits;

                cur_number = int_remaining.get(cur_number_i);
                cur_number_i++;
                tmp += (int) (cur_number >> (32 - remain_bits));
                result_list.add(tmp);
                cur_number <<= remain_bits;
                cur_number &= 0xFFFFFFFFL;
                cur_remaining_bits = 32 - remain_bits;
            }
        }
        encoded_pos_result.add(decode_pos);
        return result_list;
    }

    private static int BOSEncodeBits(int[] ts_block_delta,
                                     int final_k_start_value,
                                     int final_x_l_plus,
                                     int final_k_end_value,
                                     int final_x_u_minus,
                                     int max_delta_value,
                                     int[] min_delta,
                                     int encode_pos,
                                     byte[] cur_byte) {
        int block_size = ts_block_delta.length;

        ArrayList<Integer> final_left_outlier_index = new ArrayList<>();
        ArrayList<Integer> final_right_outlier_index = new ArrayList<>();
        ArrayList<Integer> final_left_outlier = new ArrayList<>();
        ArrayList<Integer> final_right_outlier = new ArrayList<>();
        ArrayList<Integer> final_normal = new ArrayList<>();
        int k1 = 0;
        int k2 = 0;

        ArrayList<Integer> bitmap_outlier = new ArrayList<>();
        int index_bitmap_outlier = 0;
        int cur_index_bitmap_outlier_bits = 0;
        for (int i = 0; i < block_size; i++) {
            int cur_value = ts_block_delta[i];
            if ( cur_value<= final_k_start_value) {
                final_left_outlier.add(cur_value);
                final_left_outlier_index.add(i);
                if (cur_index_bitmap_outlier_bits % 8 != 7) {
                    index_bitmap_outlier <<= 2;
                    index_bitmap_outlier += 3;
                    cur_index_bitmap_outlier_bits += 2;
                } else {
                    index_bitmap_outlier <<= 1;
                    index_bitmap_outlier += 1;
                    bitmap_outlier.add(index_bitmap_outlier);
                    index_bitmap_outlier = 1;
                    cur_index_bitmap_outlier_bits = 1;
                }
                k1++;


            } else if (cur_value >= final_k_end_value) {
                final_right_outlier.add(cur_value - final_k_end_value);
                final_right_outlier_index.add(i);
                if (cur_index_bitmap_outlier_bits % 8 != 7) {
                    index_bitmap_outlier <<= 2;
                    index_bitmap_outlier += 2;
                    cur_index_bitmap_outlier_bits += 2;
                } else {
                    index_bitmap_outlier <<= 1;
                    index_bitmap_outlier += 1;
                    bitmap_outlier.add(index_bitmap_outlier);
                    index_bitmap_outlier = 0;
                    cur_index_bitmap_outlier_bits = 1;
                }
                k2++;

            } else {
                final_normal.add(cur_value - final_x_l_plus);
                index_bitmap_outlier <<= 1;
                cur_index_bitmap_outlier_bits += 1;
            }
            if (cur_index_bitmap_outlier_bits % 8 == 0) {
                bitmap_outlier.add(index_bitmap_outlier);
                index_bitmap_outlier = 0;
            }
        }
        if (cur_index_bitmap_outlier_bits % 8 != 0) {

            index_bitmap_outlier <<= (8 - cur_index_bitmap_outlier_bits % 8);

            index_bitmap_outlier &= 0xFF;
            bitmap_outlier.add(index_bitmap_outlier);
        }

        int final_alpha = ((k1 + k2) * getBitWith(block_size-1)) <= (block_size + k1 + k2) ? 1 : 0;


        int k_byte = (k1 << 1);
        k_byte += final_alpha;
        k_byte += (k2 << 16);


        int2Bytes(k_byte,encode_pos,cur_byte);
        encode_pos += 4;

        int2Bytes(min_delta[0],encode_pos,cur_byte);
        encode_pos += 4;
        int2Bytes(min_delta[1],encode_pos,cur_byte);
        encode_pos += 4;

        int bit_width_final = getBitWith(final_x_u_minus - final_x_l_plus);
        int left_bit_width = getBitWith(final_k_start_value);//final_left_max
        int right_bit_width = getBitWith(max_delta_value - final_k_end_value);//final_right_min

        if(k1==0 && k2==0){
            intByte2Bytes(bit_width_final,encode_pos,cur_byte);
            encode_pos += 1;

//            encode_pos = encodeOutlier2Bytes(final_normal, bit_width_final,encode_pos,cur_byte);
//            return encode_pos;
        }
        else{
            int2Bytes(final_x_l_plus,encode_pos,cur_byte);
            encode_pos += 4;
            int2Bytes(final_k_end_value,encode_pos,cur_byte);
            encode_pos += 4;

            bit_width_final = getBitWith(final_x_u_minus - final_x_l_plus);
            intByte2Bytes(bit_width_final,encode_pos,cur_byte);
            encode_pos += 1;
            intByte2Bytes(left_bit_width,encode_pos,cur_byte);
            encode_pos += 1;
            intByte2Bytes(right_bit_width,encode_pos,cur_byte);
            encode_pos += 1;
            if (final_alpha == 0) { // 0

                for (int i : bitmap_outlier) {

                    intByte2Bytes(i,encode_pos,cur_byte);
                    encode_pos += 1;
                }
            } else {
                encode_pos = encodeOutlier2Bytes(final_left_outlier_index, getBitWith(block_size-1),encode_pos,cur_byte);
                encode_pos = encodeOutlier2Bytes(final_right_outlier_index, getBitWith(block_size-1),encode_pos,cur_byte);
            }
        }


//        if(k1+k2!=block_size)
        encode_pos = encodeOutlier2Bytes(final_normal, bit_width_final,encode_pos,cur_byte);
        if (k1 != 0)
            encode_pos = encodeOutlier2Bytes(final_left_outlier, left_bit_width,encode_pos,cur_byte);
        if (k2 != 0)
            encode_pos = encodeOutlier2Bytes(final_right_outlier, right_bit_width,encode_pos,cur_byte);
        return encode_pos;

    }


    private static int BOSBlockEncoder(int[] ts_block, int block_i, int block_size, int remaining ,int encode_pos , byte[] cur_byte) {

        int[] min_delta = new int[3];
        int[] ts_block_delta = getAbsDeltaTsBlock(ts_block, block_i, block_size, remaining, min_delta);


        block_size = remaining-1;
        int max_delta_value = min_delta[2];
        DeltaValueHistogram hist = buildDeltaValueHistogram(ts_block_delta);
        int[] value_list = hist.values;
        int unique_value_count = hist.uniqueCount;

        int left_shift = getBitWith(block_size);
        int mask =  (1 << left_shift) - 1;
        long[] sorted_value_list = new long[unique_value_count];
        int count = 0;

        for(int i=0;i<unique_value_count;i++){
            int value = value_list[i];
            sorted_value_list[i] = (((long) value) << left_shift) + hist.countAt(i);
        }
        Arrays.sort(sorted_value_list);

        for(int i=0;i<unique_value_count;i++){
            count += getCount(sorted_value_list[i], mask);
            sorted_value_list[i] = (((long)getUniqueValue(sorted_value_list[i], left_shift) ) << left_shift) + count;//new_value_list[i]
        }


        int final_k_start_value = -1; // x_l_minus
        int final_x_l_plus = 0; // x_l_plus
        int final_k_end_value = max_delta_value+1; // x_u_plus
        int final_x_u_minus = max_delta_value; // x_u_minus

        int min_bits = 0;
        min_bits += (getBitWith(final_k_end_value - final_k_start_value - 2 ) * (block_size));

        if (isDenseUniformNoOutlierCase(hist, max_delta_value)) {
            return BOSEncodeBitsImprove(
                    ts_block_delta,
                    final_k_start_value,
                    final_x_l_plus,
                    final_k_end_value,
                    final_x_u_minus,
                    max_delta_value,
                    min_delta,
                    encode_pos,
                    cur_byte);
        }

        int cur_k1 = 0;

        int x_l_plus_value = 0; // x_l_plus
        int x_u_minus_value = max_delta_value; // x_u_plus

        for (int end_value_i = 1; end_value_i < unique_value_count; end_value_i++) {

            x_u_minus_value = getUniqueValue(sorted_value_list[end_value_i-1], left_shift);
            int x_u_plus_value = getUniqueValue(sorted_value_list[end_value_i], left_shift);
            int cur_bits = 0;
            int cur_k2 = block_size - getCount(sorted_value_list[end_value_i-1],mask);
            cur_bits += Math.min((cur_k2 + cur_k1) * getBitWith(block_size-1), block_size + cur_k2 + cur_k1);
            if (cur_k1 + cur_k2 != block_size)
                cur_bits += (block_size - cur_k2) * getBitWith(x_u_minus_value - x_l_plus_value); // cur_k1 = 0
            if (cur_k2 != 0)
                cur_bits += cur_k2 * getBitWith(max_delta_value - x_u_plus_value);


            if (cur_bits < min_bits) {
                min_bits = cur_bits;
                final_x_u_minus = x_u_minus_value;
                final_k_end_value = x_u_plus_value;
            }
        }

        int k_start_value = -1; // x_l_minus
//        int beta_max_all = getBitWith(max_delta_value)+1;
//        int[][] hash_table_count = new int[unique_value_count][beta_max_all];
//        int[][] hash_table_value = new int[unique_value_count][beta_max_all];
//        int cur_value = getUniqueValue(sorted_value_list[0], left_shift) ;
//        int next_value = getUniqueValue(sorted_value_list[1], left_shift) ;
//
//        for (int value_i = 0; value_i < unique_value_count; value_i++) {
//
//
//            next_value = getUniqueValue(sorted_value_list[value_i + 1], left_shift) ;
//            long k_start_valueL = sorted_value_list[value_i];
//            hash_table_count[value_i][0] = getCount(k_start_valueL,mask);
//
//            int beta_max = getBitWith(max_delta_value - cur_value);
//            for(int beta = 1; beta <= beta_max; beta++){
//
//            }
//            cur_value =  next_value ;
//
//        }


        int gamma_max = getBitWith(max_delta_value);
        int[] gamma_count_list = new int[gamma_max+1];
        int[] x_u_minus_value_list = new int[gamma_max+1];
        int[] x_u_plus_value_list = new int[gamma_max+1];
        int end_i = unique_value_count - 1;
        for(int gamma = 0; gamma <= gamma_max; gamma++) {
            int x_u_plus_pow_beta = max_delta_value - (1<<gamma) + 1;
//            int x_u_plus_pow_beta = (int) (max_delta_value - pow(2, gamma) + 1);
//
            for (; end_i > 0; end_i--) {
                x_u_minus_value = getUniqueValue(sorted_value_list[end_i - 1], left_shift);
                int x_u_plus_value = getUniqueValue(sorted_value_list[end_i], left_shift);
                if (x_u_minus_value < x_u_plus_pow_beta && x_u_plus_value >= x_u_plus_pow_beta){
                    gamma_count_list[gamma] = getCount(sorted_value_list[end_i-1],mask);
                    x_u_minus_value_list[gamma] = x_u_minus_value;
                    x_u_plus_value_list[gamma] = x_u_plus_value;
                } else if (x_u_minus_value < x_u_plus_pow_beta) {
                    break;
                }
            }
        }
        for(int gamma = 1; gamma < gamma_max; gamma++) {
            if(gamma_count_list[gamma]==0){
                gamma_count_list[gamma] = gamma_count_list[gamma-1];
                x_u_minus_value_list[gamma] = x_u_minus_value_list[gamma-1];
                x_u_plus_value_list[gamma] = x_u_plus_value_list[gamma-1];
            }
        }

        for (int start_value_i = 0; start_value_i < unique_value_count-1; start_value_i++) {
            long k_start_valueL = sorted_value_list[start_value_i];
            k_start_value =  getUniqueValue(k_start_valueL, left_shift) ;
            x_l_plus_value =  getUniqueValue(sorted_value_list[start_value_i+1], left_shift) ;


            cur_k1 = getCount(k_start_valueL,mask);

            int k_end_value;
            int cur_bits;
            int cur_k2;
            k_end_value = max_delta_value + 1;

            cur_bits = 0;
            cur_k2 = 0;
            cur_bits += Math.min((cur_k2 + cur_k1) * getBitWith(block_size-1), block_size + cur_k2 + cur_k1);
            cur_bits += cur_k1 * getBitWith(k_start_value);
            if (cur_k1 + cur_k2 != block_size)
                cur_bits += (block_size - cur_k1) * getBitWith(k_end_value- x_l_plus_value); //cur_k2 =0

            if (cur_bits < min_bits) {
                min_bits = cur_bits;
                final_k_start_value = k_start_value;
                final_x_l_plus = x_l_plus_value;
                final_k_end_value = k_end_value;
                final_x_u_minus = max_delta_value;
            }

            int beta_max = getBitWith(max_delta_value - x_l_plus_value);

            int lower_outlier_cost = cur_k1 * getBitWith(k_start_value);



            for(int gamma = 0; gamma < beta_max; gamma++){
//                int x_u_plus_pow_beta = (int) (max_delta_value - pow(2,gamma)+1);
                x_u_minus_value = x_u_minus_value_list[gamma];
                k_end_value =  x_u_plus_value_list[gamma];
                cur_bits = 0;
                cur_k2 = block_size - gamma_count_list[gamma];

                cur_bits += Math.min((cur_k1 + cur_k2) * getBitWith(block_size-1), block_size + cur_k1 + cur_k2);
                cur_bits += lower_outlier_cost;
                if (cur_k1 + cur_k2 != block_size)
                    cur_bits += (block_size - cur_k1 - cur_k2) * getBitWith(x_u_minus_value - x_l_plus_value);
                if (cur_k2 != 0)
                    cur_bits += cur_k2 * getBitWith(max_delta_value - k_end_value);


                if (cur_bits < min_bits) {
                    min_bits = cur_bits;
                    final_k_start_value = k_start_value;
                    final_x_l_plus = x_l_plus_value;
                    final_k_end_value = k_end_value;
                    final_x_u_minus = x_u_minus_value;
                }

            }
//            end_value_i = unique_value_count - 1;
//            for(int gamma = 0; gamma <= beta_max; gamma++){
//                for (; end_value_i > start_value_i; end_value_i--) {
//                    int x_u_plus_pow_beta = (int) (max_delta_value - pow(2,gamma)+1);
//                    x_u_minus_value = getUniqueValue(sorted_value_list[end_value_i-1], left_shift);
//                    k_end_value = getUniqueValue(sorted_value_list[end_value_i], left_shift);
//                    if(x_u_minus_value < x_u_plus_pow_beta && k_end_value >= x_u_plus_pow_beta){
//                        cur_bits = 0;
//                        cur_k2 = block_size - getCount(sorted_value_list[end_value_i-1],mask);
//
//                        cur_bits += Math.min((cur_k1 + cur_k2) * getBitWith(block_size-1), block_size + cur_k1 + cur_k2);
//                        cur_bits += cur_k1 * getBitWith(k_start_value);
//                        if (cur_k1 + cur_k2 != block_size)
//                            cur_bits += (block_size - cur_k1 - cur_k2) * getBitWith(x_u_minus_value - x_l_plus_value);
//                        if (cur_k2 != 0)
//                            cur_bits += cur_k2 * getBitWith(max_delta_value - k_end_value);
//
//
//                        if (cur_bits < min_bits) {
//                            min_bits = cur_bits;
//                            final_k_start_value = k_start_value;
//                            final_x_l_plus = x_l_plus_value;
//                            final_k_end_value = k_end_value;
//                            final_x_u_minus = x_u_minus_value;
//                        }
//                    } else if (x_u_minus_value < x_u_plus_pow_beta && k_end_value < x_u_plus_pow_beta) {
//                        break;
//                    }
//                }
//            }
//

        }

        for(int beta = 0; beta < gamma_max; beta++){

            int pow_beta = 1<<beta;
            int start_value_i = 0;
            int end_value_i = start_value_i+1;

            for (; start_value_i < unique_value_count-1; start_value_i++) {
                long x_l_minusL = sorted_value_list[start_value_i];
                int x_l_minus =  getUniqueValue(x_l_minusL, left_shift) ;
                int x_l_plus =  getUniqueValue(sorted_value_list[start_value_i+1], left_shift) ;
                int x_u_plus_pow_beta = pow_beta+x_l_plus;
                if(x_u_plus_pow_beta > max_delta_value) break;



                cur_k1 = getCount(x_l_minusL,mask);
                int lower_outlier_cost = cur_k1 * getBitWith(x_l_minus);

                while ( end_value_i < unique_value_count) {
//                    if(beta==3 && end_value_i==22)
//                    {
//                        System.out.println(x_l_minus);
//                        System.out.println(x_l_plus);
//                    }

                    int x_u_minus = getUniqueValue(sorted_value_list[end_value_i-1], left_shift);
                    int x_u_plus = getUniqueValue(sorted_value_list[end_value_i], left_shift);
                    if(x_u_minus < x_u_plus_pow_beta && x_u_plus >= x_u_plus_pow_beta){
                        int cur_bits = 0;
                        int cur_k2 = block_size - getCount(sorted_value_list[end_value_i-1],mask);

                        cur_bits += Math.min((cur_k1 + cur_k2) * getBitWith(block_size-1), block_size + cur_k1 + cur_k2);
                        cur_bits += lower_outlier_cost;
                        if (cur_k1 + cur_k2 != block_size)
                            cur_bits += (block_size - cur_k1 - cur_k2) * getBitWith(x_u_minus - x_l_plus);
                        if (cur_k2 != 0)
                            cur_bits += cur_k2 * getBitWith(max_delta_value - x_u_plus);


                        if (cur_bits < min_bits) {
                            min_bits = cur_bits;
                            final_k_start_value = x_l_minus;
                            final_x_l_plus = x_l_plus;
                            final_k_end_value = x_u_plus;
                            final_x_u_minus = x_u_minus;
                        }
                        break;
                    }
//                    else if (x_u_minus >= x_u_plus_pow_beta && x_u_plus >= x_u_plus_pow_beta) {
//                        break;
//                    }

                    end_value_i++;
                }
            }

        }

        encode_pos = BOSEncodeBitsImprove(ts_block_delta,  final_k_start_value, final_x_l_plus, final_k_end_value, final_x_u_minus,
                max_delta_value, min_delta, encode_pos , cur_byte);

//        System.out.println(encode_pos);

        return encode_pos;
    }


    public static int BOSEncoder(
            int[] data, int block_size, byte[] encoded_result) {
        block_size++;

        int length_all = data.length;

        int encode_pos = 0;
        int2Bytes(length_all,encode_pos,encoded_result);
        encode_pos += 4;

        int block_num = length_all / block_size;
        int2Bytes(block_size,encode_pos,encoded_result);
        encode_pos+= 4;

        for (int i = 0; i < block_num; i++) {
            encode_pos =  BOSBlockEncoder(data, i, block_size, block_size,encode_pos,encoded_result);
        }

        int remaining_length = length_all - block_num * block_size;
        if (remaining_length <= 3) {
            for (int i = remaining_length; i > 0; i--) {
                int2Bytes(data[data.length - i], encode_pos, encoded_result);
                encode_pos += 4;
            }

        }
        else {

            int start = block_num * block_size;
            int remaining = length_all-start;
            encode_pos = BOSBlockEncoder(data, block_num, block_size,remaining, encode_pos,encoded_result);

        }


        return encode_pos;
    }
    public static int BOSBlockDecoder(byte[] encoded, int decode_pos, int[] value_list, int block_size, int[] value_pos_arr) {

        int k_byte = bytes2Integer(encoded, decode_pos, 4);
        decode_pos += 4;
        int k1_byte = (int) (k_byte % pow(2, 16));
        int k1 = k1_byte / 2;
        int final_alpha = k1_byte % 2;

        int k2 = (int) (k_byte / pow(2, 16));

        int value0 = bytes2Integer(encoded, decode_pos, 4);
        decode_pos += 4;
        value_list[value_pos_arr[0]] =value0;
        value_pos_arr[0] ++;

        int min_delta = bytes2Integer(encoded, decode_pos, 4);
        decode_pos += 4;

        ArrayList<Integer> final_left_outlier_index = new ArrayList<>();
        ArrayList<Integer> final_right_outlier_index = new ArrayList<>();
        ArrayList<Integer> final_left_outlier = new ArrayList<>();
        ArrayList<Integer> final_right_outlier = new ArrayList<>();
        ArrayList<Integer> final_normal= new ArrayList<>();;
        ArrayList<Integer> bitmap_outlier = new ArrayList<>();
        int final_k_start_value = 0;
        int final_k_end_value = 0;
        int bit_width_final = 0;
        int left_bit_width = 0;
        int right_bit_width = 0;

        if(k1!=0 || k2 != 0){
            final_k_start_value = bytes2Integer(encoded, decode_pos, 4);
            decode_pos += 4;

            final_k_end_value = bytes2Integer(encoded, decode_pos, 4);
            decode_pos += 4;

            bit_width_final = bytes2Integer(encoded, decode_pos, 1);
            decode_pos += 1;

            left_bit_width = bytes2Integer(encoded, decode_pos, 1);
            decode_pos += 1;
            right_bit_width = bytes2Integer(encoded, decode_pos, 1);
            decode_pos += 1;

            if (final_alpha == 0) {
                int bitmap_bytes = (int) Math.ceil((double) (block_size + k1 + k2) / (double) 8);
                for (int i = 0; i < bitmap_bytes; i++) {
                    bitmap_outlier.add(bytes2Integer(encoded, decode_pos, 1));
                    decode_pos += 1;
                }
                int bitmap_outlier_i = 0;
                int remaining_bits = 8;
                int tmp = bitmap_outlier.get(bitmap_outlier_i);
                bitmap_outlier_i++;
                int i = 0;
                while (i < block_size ) {
                    if (remaining_bits > 1) {
                        int bit_i = (tmp >> (remaining_bits - 1)) & 0x1;
                        remaining_bits -= 1;
                        if (bit_i == 1) {
                            int bit_left_right = (tmp >> (remaining_bits - 1)) & 0x1;
                            remaining_bits -= 1;
                            if (bit_left_right == 1) {
                                final_left_outlier_index.add(i);
                            } else {
                                final_right_outlier_index.add(i);
                            }
                        }
                        if (remaining_bits == 0) {
                            remaining_bits = 8;
                            if (bitmap_outlier_i >= bitmap_bytes) break;
                            tmp = bitmap_outlier.get(bitmap_outlier_i);
                            bitmap_outlier_i++;
                        }
                    } else if (remaining_bits == 1) {
                        int bit_i = tmp & 0x1;
                        remaining_bits = 8;
                        if (bitmap_outlier_i >= bitmap_bytes) break;
                        tmp = bitmap_outlier.get(bitmap_outlier_i);
                        bitmap_outlier_i++;
                        if (bit_i == 1) {
                            int bit_left_right = (tmp >> (remaining_bits - 1)) & 0x1;
                            remaining_bits -= 1;
                            if (bit_left_right == 1) {
                                final_left_outlier_index.add(i);
                            } else {
                                final_right_outlier_index.add(i);
                            }
                        }
                    }
                    i++;
                }
            } else {
                ArrayList<Integer> decode_pos_result_left = new ArrayList<>();
                final_left_outlier_index = decodeOutlier2Bytes(encoded, decode_pos, getBitWith(block_size-1), k1, decode_pos_result_left);
                decode_pos = (decode_pos_result_left.get(0));
                ArrayList<Integer> decode_pos_result_right = new ArrayList<>();
                final_right_outlier_index = decodeOutlier2Bytes(encoded, decode_pos, getBitWith(block_size-1), k2, decode_pos_result_right);
                decode_pos = (decode_pos_result_right.get(0));
            }
        }else {
            bit_width_final = bytes2Integer(encoded, decode_pos, 1);
            decode_pos += 1;
        }




        ArrayList<Integer> decode_pos_normal = new ArrayList<>();
        final_normal = decodeOutlier2Bytes(encoded, decode_pos, bit_width_final, block_size - k1 - k2, decode_pos_normal);

        decode_pos = decode_pos_normal.get(0);
        if (k1 != 0) {
            ArrayList<Integer> decode_pos_result_left = new ArrayList<>();
            final_left_outlier = decodeOutlier2Bytes(encoded, decode_pos, left_bit_width, k1, decode_pos_result_left);
            decode_pos = decode_pos_result_left.get(0);
        }
        if (k2 != 0) {
            ArrayList<Integer> decode_pos_result_right = new ArrayList<>();
            final_right_outlier = decodeOutlier2Bytes(encoded, decode_pos, right_bit_width, k2, decode_pos_result_right);
            decode_pos = decode_pos_result_right.get(0);
        }
        int left_outlier_i = 0;
        int right_outlier_i = 0;
        int normal_i = 0;
        int pre_v = value0;
//        int final_k_end_value = (int) (final_k_start_value + pow(2, bit_width_final));


        for (int i = 0; i < block_size; i++) {
            int current_delta;
            if (left_outlier_i >= k1) {
                if (right_outlier_i >= k2) {
                    current_delta = min_delta + final_normal.get(normal_i) + final_k_start_value;
                    normal_i++;
                } else if (i == final_right_outlier_index.get(right_outlier_i)) {
                    current_delta = min_delta + final_right_outlier.get(right_outlier_i) + final_k_end_value;
                    right_outlier_i++;
                } else {
                    current_delta = min_delta + final_normal.get(normal_i) + final_k_start_value;
                    normal_i++;
                }
            } else if (i == final_left_outlier_index.get(left_outlier_i)) {
                current_delta = min_delta + final_left_outlier.get(left_outlier_i);
                left_outlier_i++;
            } else {

                if (right_outlier_i >= k2) {
                    current_delta = min_delta + final_normal.get(normal_i) + final_k_start_value;
                    normal_i++;
                } else if (i == final_right_outlier_index.get(right_outlier_i)) {
                    current_delta = min_delta + final_right_outlier.get(right_outlier_i) + final_k_end_value;
                    right_outlier_i++;
                } else {
                    current_delta = min_delta + final_normal.get(normal_i) + final_k_start_value;
                    normal_i++;
                }
            }

            pre_v = current_delta + pre_v;
            value_list[value_pos_arr[0]] = pre_v;
            value_pos_arr[0]++;
        }
        return decode_pos;
    }

    public static void BOSDecoder(byte[] encoded) {

        int decode_pos = 0;
        int length_all = bytes2Integer(encoded, decode_pos, 4);
        decode_pos += 4;
        int block_size = bytes2Integer(encoded, decode_pos, 4);
        decode_pos += 4;



        int block_num = length_all / block_size;
        int remain_length = length_all - block_num * block_size;


        int[] value_list = new int[length_all+block_size];
        block_size--;

        int[] value_pos_arr = new int[1];
        for (int k = 0; k < block_num; k++) {


            decode_pos = BOSBlockDecoder(encoded, decode_pos, value_list, block_size,value_pos_arr);

        }

        if (remain_length <= 3) {
            for (int i = 0; i < remain_length; i++) {
                int value_end = bytes2Integer(encoded, decode_pos, 4);
                decode_pos += 4;
                value_list[value_pos_arr[0]] = value_end;
                value_pos_arr[0]++;
            }
        } else {
            remain_length --;
            BOSBlockDecoder(encoded, decode_pos, value_list, remain_length, value_pos_arr);
        }
    }

    public static int EncodeBits(int num,
                                 int bit_width,
                                 int encode_pos,
                                 byte[] cur_byte,
                                 int[] bit_index_list){
        // 找到要插入的位的索引
        int bit_index = bit_index_list[0] ;//cur_byte[encode_pos + 1];

        // 计算数值的起始位位置
        int remaining_bits = bit_width;

        while (remaining_bits > 0) {
            // 计算在当前字节中可以使用的位数
            int available_bits = bit_index;
            int bits_to_write = Math.min(available_bits, remaining_bits);

            // 更新 bit_index
            bit_index = available_bits - bits_to_write;

            // 计算要写入的位的掩码和数值
            int mask = (1 << bits_to_write) - 1;
            int bits = (num >> (remaining_bits - bits_to_write)) & mask;

            // 写入到当前位置
            cur_byte[encode_pos] &= (byte) ~(mask << bit_index); // 清除对应位置的位
            cur_byte[encode_pos] |= (byte) (bits << bit_index);

            // 更新位宽和数值
            remaining_bits -= bits_to_write;
            if (bit_index == 0) {
                bit_index = 8;
                encode_pos++;
            }
        }
        bit_index_list[0] = bit_index;
//        cur_byte[encode_pos + 1] = (byte) bit_index;
        return encode_pos;
    }
    private static int BOSEncodeBitsImprove(int[] ts_block_delta,
                                            int final_k_start_value,
                                            int final_x_l_plus,
                                            int final_k_end_value,
                                            int final_x_u_minus,
                                            int max_delta_value,
                                            int[] min_delta,
                                            int encode_pos,
                                            byte[] cur_byte) {
        int block_size = ts_block_delta.length;

        int[] final_left_outlier_index = new int[block_size];
        int[] final_right_outlier_index = new int[block_size];
        int k1 = 0;
        int k2 = 0;

        int[] bitmap_outlier = new int[(block_size * 2 + 7) / 8 + 1];
        int bitmap_outlier_count = 0;
        int index_bitmap_outlier = 0;
        int cur_index_bitmap_outlier_bits = 0;
        for (int i = 0; i < block_size; i++) {
            int cur_value = ts_block_delta[i];
            if ( cur_value<= final_k_start_value) {
//                encode_pos = EncodeBits(cur_value,left_bit_width,encode_pos,cur_byte);
//                final_left_outlier.add(cur_value);
                final_left_outlier_index[k1] = i;
                if (cur_index_bitmap_outlier_bits % 8 != 7) {
                    index_bitmap_outlier <<= 2;
                    index_bitmap_outlier += 3;
                    cur_index_bitmap_outlier_bits += 2;
                } else {
                    index_bitmap_outlier <<= 1;
                    index_bitmap_outlier += 1;
                    bitmap_outlier[bitmap_outlier_count++] = index_bitmap_outlier;
                    index_bitmap_outlier = 1;
                    cur_index_bitmap_outlier_bits = 1;
                }
                k1++;


            } else if (cur_value >= final_k_end_value) {
//                encode_pos = EncodeBits(cur_value- final_k_end_value,right_bit_width,encode_pos,cur_byte);
//                final_right_outlier.add(cur_value - final_k_end_value);
                final_right_outlier_index[k2] = i;
                if (cur_index_bitmap_outlier_bits % 8 != 7) {
                    index_bitmap_outlier <<= 2;
                    index_bitmap_outlier += 2;
                    cur_index_bitmap_outlier_bits += 2;
                } else {
                    index_bitmap_outlier <<= 1;
                    index_bitmap_outlier += 1;
                    bitmap_outlier[bitmap_outlier_count++] = index_bitmap_outlier;
                    index_bitmap_outlier = 0;
                    cur_index_bitmap_outlier_bits = 1;
                }
                k2++;

            } else {
//                final_normal.add(cur_value - final_x_l_plus);
//                encode_pos = EncodeBits(cur_value- final_x_l_plus,right_bit_width,encode_pos,cur_byte);
                index_bitmap_outlier <<= 1;
                cur_index_bitmap_outlier_bits += 1;
            }
            if (cur_index_bitmap_outlier_bits % 8 == 0) {
                bitmap_outlier[bitmap_outlier_count++] = index_bitmap_outlier;
                index_bitmap_outlier = 0;
            }
        }
        if (cur_index_bitmap_outlier_bits % 8 != 0) {

            index_bitmap_outlier <<= (8 - cur_index_bitmap_outlier_bits % 8);

            index_bitmap_outlier &= 0xFF;
            bitmap_outlier[bitmap_outlier_count++] = index_bitmap_outlier;
        }

        int final_alpha = ((k1 + k2) * getBitWith(block_size-1)) <= (block_size + k1 + k2) ? 1 : 0;


        int k_byte = (k1 << 1);
        k_byte += final_alpha;
        k_byte += (k2 << 16);

        int2Bytes(k_byte,encode_pos,cur_byte);
        encode_pos += 4;


        int2Bytes(min_delta[0],encode_pos,cur_byte);
        encode_pos += 4;
        int2Bytes(min_delta[1],encode_pos,cur_byte);
        encode_pos += 4;

        int bit_width_final = getBitWith(final_x_u_minus - final_x_l_plus);
        intByte2Bytes(bit_width_final,encode_pos,cur_byte);
        encode_pos += 1;
        int[] bit_index_list = new int[1];
        bit_index_list[0] = 8;

        if(final_k_start_value<0 && final_k_end_value > max_delta_value){
//            int bit_width_final= getBitWith(final_x_u_minus - final_x_l_plus);
//            cur_byte[encode_pos+1] = 8;
            int fullGroups = block_size / 8;
            for (int i = 0; i < fullGroups; i++) {
                pack8Values(ts_block_delta, i * 8, bit_width_final, encode_pos, cur_byte);
                encode_pos += bit_width_final;
            }
            for (int i = fullGroups * 8; i < block_size; i++) {
                encode_pos = EncodeBits(ts_block_delta[i], bit_width_final, encode_pos, cur_byte, bit_index_list);
//                final_normal.add(cur_value);
            }
            if(bit_index_list[0] != 8){
                encode_pos ++;
            }
//            cur_byte[encode_pos+1] = 0;
//            encode_pos = encodeOutlier2Bytes(final_normal, bit_width_final,encode_pos,cur_byte);
            return encode_pos;
        }


        int left_bit_width = getBitWith(final_k_start_value);//final_left_max
        int right_bit_width = getBitWith(max_delta_value - final_k_end_value);//final_right_min
        int2Bytes(final_x_l_plus,encode_pos,cur_byte);
        encode_pos += 4;
        int2Bytes(final_k_end_value,encode_pos,cur_byte);
        encode_pos += 4;

//        bit_width_final = getBitWith(final_x_u_minus - final_x_l_plus);
//        intByte2Bytes(bit_width_final,encode_pos,cur_byte);
//        encode_pos += 1;
        intByte2Bytes(left_bit_width,encode_pos,cur_byte);
        encode_pos += 1;
        intByte2Bytes(right_bit_width,encode_pos,cur_byte);
        encode_pos += 1;

        if (final_alpha == 0) { // 0

            for (int bitmap_i = 0; bitmap_i < bitmap_outlier_count; bitmap_i++) {

                intByte2Bytes(bitmap_outlier[bitmap_i],encode_pos,cur_byte);
                encode_pos += 1;
            }
        } else {
            int indexBitWidth = getBitWith(block_size-1);
            encode_pos = encodeOutlier2Bytes(final_left_outlier_index, k1, indexBitWidth, encode_pos, cur_byte);
            encode_pos = encodeOutlier2Bytes(final_right_outlier_index, k2, indexBitWidth, encode_pos, cur_byte);
        }
//        cur_byte[encode_pos+1] = 8;
//        bit_index_list[0] = 8;
        for (int cur_value : ts_block_delta) {
            if (cur_value <= final_k_start_value) {
                encode_pos = EncodeBits(cur_value, left_bit_width, encode_pos, cur_byte,bit_index_list);
            } else if (cur_value >= final_k_end_value) {
                encode_pos = EncodeBits(cur_value - final_k_end_value, right_bit_width, encode_pos, cur_byte,bit_index_list);
            } else {
                encode_pos = EncodeBits(cur_value - final_x_l_plus, bit_width_final, encode_pos, cur_byte,bit_index_list);
            }
        }
        if(bit_index_list[0] != 8){
            encode_pos ++;
        }

//        cur_byte[encode_pos+1] = 0;

//        if(k1==0 && k2==0){
//            intByte2Bytes(bit_width_final,encode_pos,cur_byte);
//            encode_pos += 1;
//
//
//        }
//        else{
//            int2Bytes(final_x_l_plus,encode_pos,cur_byte);
//            encode_pos += 4;
//            int2Bytes(final_k_end_value,encode_pos,cur_byte);
//            encode_pos += 4;
//
//            bit_width_final = getBitWith(final_x_u_minus - final_x_l_plus);
//            intByte2Bytes(bit_width_final,encode_pos,cur_byte);
//            encode_pos += 1;
//            intByte2Bytes(left_bit_width,encode_pos,cur_byte);
//            encode_pos += 1;
//            intByte2Bytes(right_bit_width,encode_pos,cur_byte);
//            encode_pos += 1;
//            if (final_alpha == 0) { // 0
//
//                for (int i : bitmap_outlier) {
//
//                    intByte2Bytes(i,encode_pos,cur_byte);
//                    encode_pos += 1;
//                }
//            } else {
//                encode_pos = encodeOutlier2Bytes(final_left_outlier_index, getBitWith(block_size-1),encode_pos,cur_byte);
//                encode_pos = encodeOutlier2Bytes(final_right_outlier_index, getBitWith(block_size-1),encode_pos,cur_byte);
//            }
//        }


//        if(k1+k2!=block_size)
//        encode_pos = encodeOutlier2Bytes(final_normal, bit_width_final,encode_pos,cur_byte);
//        if (k1 != 0)
//            encode_pos = encodeOutlier2Bytes(final_left_outlier, left_bit_width,encode_pos,cur_byte);
//        if (k2 != 0)
//            encode_pos = encodeOutlier2Bytes(final_right_outlier, right_bit_width,encode_pos,cur_byte);
        return encode_pos;

    }

    private static int BOSBlockEncoderImprove(int[] ts_block, int block_i, int block_size, int remaining ,int encode_pos , byte[] cur_byte) {

        int[] min_delta = new int[3];
        int[] ts_block_delta = getAbsDeltaTsBlock(ts_block, block_i, block_size, remaining, min_delta);


        block_size = remaining-1;
        int max_delta_value = min_delta[2];
        DeltaValueHistogram hist = buildDeltaValueHistogram(ts_block_delta);
        int[] value_list = hist.values;
        int unique_value_count = hist.uniqueCount;

        int left_shift = getBitWith(block_size);
        int mask =  (1 << left_shift) - 1;
        long[] sorted_value_list = new long[unique_value_count];
        int count = 0;

        for(int i=0;i<unique_value_count;i++){
            int value = value_list[i];
            sorted_value_list[i] = (((long) value) << left_shift) + hist.countAt(i);
        }
        Arrays.sort(sorted_value_list);

        for(int i=0;i<unique_value_count;i++){
            count += getCount(sorted_value_list[i], mask);
            sorted_value_list[i] = (((long)getUniqueValue(sorted_value_list[i], left_shift) ) << left_shift) + count;//new_value_list[i]
        }


        int final_k_start_value = -1; // x_l_minus
        int final_x_l_plus = 0; // x_l_plus
        int final_k_end_value = max_delta_value+1; // x_u_plus
        int final_x_u_minus = max_delta_value; // x_u_minus

        int min_bits = 0;
        min_bits += (getBitWith(final_k_end_value - final_k_start_value - 2 ) * (block_size));

        int cur_k1 = 0;

        int x_l_plus_value = 0; // x_l_plus
        int x_u_minus_value = max_delta_value; // x_u_plus

        for (int end_value_i = 1; end_value_i < unique_value_count; end_value_i++) {

            x_u_minus_value = getUniqueValue(sorted_value_list[end_value_i-1], left_shift);
            int x_u_plus_value = getUniqueValue(sorted_value_list[end_value_i], left_shift);
            int cur_bits = 0;
            int cur_k2 = block_size - getCount(sorted_value_list[end_value_i-1],mask);
            cur_bits += Math.min((cur_k2 + cur_k1) * getBitWith(block_size-1), block_size + cur_k2 + cur_k1);
            if (cur_k1 + cur_k2 != block_size)
                cur_bits += (block_size - cur_k2) * getBitWith(x_u_minus_value - x_l_plus_value); // cur_k1 = 0
            if (cur_k2 != 0)
                cur_bits += cur_k2 * getBitWith(max_delta_value - x_u_plus_value);


            if (cur_bits < min_bits) {
                min_bits = cur_bits;
                final_x_u_minus = x_u_minus_value;
                final_k_end_value = x_u_plus_value;
            }
        }

        int k_start_value = -1; // x_l_minus
//        int beta_max_all = getBitWith(max_delta_value)+1;
//        int[][] hash_table_count = new int[unique_value_count][beta_max_all];
//        int[][] hash_table_value = new int[unique_value_count][beta_max_all];
//        int cur_value = getUniqueValue(sorted_value_list[0], left_shift) ;
//        int next_value = getUniqueValue(sorted_value_list[1], left_shift) ;
//
//        for (int value_i = 0; value_i < unique_value_count; value_i++) {
//
//
//            next_value = getUniqueValue(sorted_value_list[value_i + 1], left_shift) ;
//            long k_start_valueL = sorted_value_list[value_i];
//            hash_table_count[value_i][0] = getCount(k_start_valueL,mask);
//
//            int beta_max = getBitWith(max_delta_value - cur_value);
//            for(int beta = 1; beta <= beta_max; beta++){
//
//            }
//            cur_value =  next_value ;
//
//        }


        int gamma_max = getBitWith(max_delta_value);
        int[] gamma_count_list = new int[gamma_max+1];
        int[] x_u_minus_value_list = new int[gamma_max+1];
        int[] x_u_plus_value_list = new int[gamma_max+1];
        int end_i = unique_value_count - 1;
        for(int gamma = 0; gamma <= gamma_max; gamma++) {
            int x_u_plus_pow_beta = max_delta_value - (1<<gamma) + 1;
//            int x_u_plus_pow_beta = (int) (max_delta_value - pow(2, gamma) + 1);
//
            for (; end_i > 0; end_i--) {
                x_u_minus_value = getUniqueValue(sorted_value_list[end_i - 1], left_shift);
                int x_u_plus_value = getUniqueValue(sorted_value_list[end_i], left_shift);
                if (x_u_minus_value < x_u_plus_pow_beta && x_u_plus_value >= x_u_plus_pow_beta){
                    gamma_count_list[gamma] = getCount(sorted_value_list[end_i-1],mask);
                    x_u_minus_value_list[gamma] = x_u_minus_value;
                    x_u_plus_value_list[gamma] = x_u_plus_value;
                } else if (x_u_minus_value < x_u_plus_pow_beta) {
                    break;
                }
            }
        }
        for(int gamma = 1; gamma < gamma_max; gamma++) {
            if(gamma_count_list[gamma]==0){
                gamma_count_list[gamma] = gamma_count_list[gamma-1];
                x_u_minus_value_list[gamma] = x_u_minus_value_list[gamma-1];
                x_u_plus_value_list[gamma] = x_u_plus_value_list[gamma-1];
            }
        }

        for (int start_value_i = 0; start_value_i < unique_value_count-1; start_value_i++) {
            long k_start_valueL = sorted_value_list[start_value_i];
            k_start_value =  getUniqueValue(k_start_valueL, left_shift) ;
            x_l_plus_value =  getUniqueValue(sorted_value_list[start_value_i+1], left_shift) ;


            cur_k1 = getCount(k_start_valueL,mask);

            int k_end_value;
            int cur_bits;
            int cur_k2;
            k_end_value = max_delta_value + 1;

            cur_bits = 0;
            cur_k2 = 0;
            cur_bits += Math.min((cur_k2 + cur_k1) * getBitWith(block_size-1), block_size + cur_k2 + cur_k1);
            cur_bits += cur_k1 * getBitWith(k_start_value);
            if (cur_k1 + cur_k2 != block_size)
                cur_bits += (block_size - cur_k1) * getBitWith(k_end_value- x_l_plus_value); //cur_k2 =0

            if (cur_bits < min_bits) {
                min_bits = cur_bits;
                final_k_start_value = k_start_value;
                final_x_l_plus = x_l_plus_value;
                final_k_end_value = k_end_value;
                final_x_u_minus = max_delta_value;
            }

            int beta_max = getBitWith(max_delta_value - x_l_plus_value);

            int lower_outlier_cost = cur_k1 * getBitWith(k_start_value);



            for(int gamma = 0; gamma < beta_max; gamma++){
//                int x_u_plus_pow_beta = (int) (max_delta_value - pow(2,gamma)+1);
                x_u_minus_value = x_u_minus_value_list[gamma];
                k_end_value =  x_u_plus_value_list[gamma];
                cur_bits = 0;
                cur_k2 = block_size - gamma_count_list[gamma];

                cur_bits += Math.min((cur_k1 + cur_k2) * getBitWith(block_size-1), block_size + cur_k1 + cur_k2);
                cur_bits += lower_outlier_cost;
                if (cur_k1 + cur_k2 != block_size)
                    cur_bits += (block_size - cur_k1 - cur_k2) * getBitWith(x_u_minus_value - x_l_plus_value);
                if (cur_k2 != 0)
                    cur_bits += cur_k2 * getBitWith(max_delta_value - k_end_value);


                if (cur_bits < min_bits) {
                    min_bits = cur_bits;
                    final_k_start_value = k_start_value;
                    final_x_l_plus = x_l_plus_value;
                    final_k_end_value = k_end_value;
                    final_x_u_minus = x_u_minus_value;
                }

            }
//            end_value_i = unique_value_count - 1;
//            for(int gamma = 0; gamma <= beta_max; gamma++){
//                for (; end_value_i > start_value_i; end_value_i--) {
//                    int x_u_plus_pow_beta = (int) (max_delta_value - pow(2,gamma)+1);
//                    x_u_minus_value = getUniqueValue(sorted_value_list[end_value_i-1], left_shift);
//                    k_end_value = getUniqueValue(sorted_value_list[end_value_i], left_shift);
//                    if(x_u_minus_value < x_u_plus_pow_beta && k_end_value >= x_u_plus_pow_beta){
//                        cur_bits = 0;
//                        cur_k2 = block_size - getCount(sorted_value_list[end_value_i-1],mask);
//
//                        cur_bits += Math.min((cur_k1 + cur_k2) * getBitWith(block_size-1), block_size + cur_k1 + cur_k2);
//                        cur_bits += cur_k1 * getBitWith(k_start_value);
//                        if (cur_k1 + cur_k2 != block_size)
//                            cur_bits += (block_size - cur_k1 - cur_k2) * getBitWith(x_u_minus_value - x_l_plus_value);
//                        if (cur_k2 != 0)
//                            cur_bits += cur_k2 * getBitWith(max_delta_value - k_end_value);
//
//
//                        if (cur_bits < min_bits) {
//                            min_bits = cur_bits;
//                            final_k_start_value = k_start_value;
//                            final_x_l_plus = x_l_plus_value;
//                            final_k_end_value = k_end_value;
//                            final_x_u_minus = x_u_minus_value;
//                        }
//                    } else if (x_u_minus_value < x_u_plus_pow_beta && k_end_value < x_u_plus_pow_beta) {
//                        break;
//                    }
//                }
//            }
//

        }

        for(int beta = 0; beta < gamma_max; beta++){

            int pow_beta = 1<<beta;
            int start_value_i = 0;
            int end_value_i = start_value_i+1;

            for (; start_value_i < unique_value_count-1; start_value_i++) {
                long x_l_minusL = sorted_value_list[start_value_i];
                int x_l_minus =  getUniqueValue(x_l_minusL, left_shift) ;
                int x_l_plus =  getUniqueValue(sorted_value_list[start_value_i+1], left_shift) ;
                int x_u_plus_pow_beta = pow_beta+x_l_plus;
                if(x_u_plus_pow_beta > max_delta_value) break;



                cur_k1 = getCount(x_l_minusL,mask);
                int lower_outlier_cost = cur_k1 * getBitWith(x_l_minus);

                while ( end_value_i < unique_value_count) {
//                    if(beta==3 && end_value_i==22)
//                    {
//                        System.out.println(x_l_minus);
//                        System.out.println(x_l_plus);
//                    }

                    int x_u_minus = getUniqueValue(sorted_value_list[end_value_i-1], left_shift);
                    int x_u_plus = getUniqueValue(sorted_value_list[end_value_i], left_shift);
                    if(x_u_minus < x_u_plus_pow_beta && x_u_plus >= x_u_plus_pow_beta){
                        int cur_bits = 0;
                        int cur_k2 = block_size - getCount(sorted_value_list[end_value_i-1],mask);

                        cur_bits += Math.min((cur_k1 + cur_k2) * getBitWith(block_size-1), block_size + cur_k1 + cur_k2);
                        cur_bits += lower_outlier_cost;
                        if (cur_k1 + cur_k2 != block_size)
                            cur_bits += (block_size - cur_k1 - cur_k2) * getBitWith(x_u_minus - x_l_plus);
                        if (cur_k2 != 0)
                            cur_bits += cur_k2 * getBitWith(max_delta_value - x_u_plus);


                        if (cur_bits < min_bits) {
                            min_bits = cur_bits;
                            final_k_start_value = x_l_minus;
                            final_x_l_plus = x_l_plus;
                            final_k_end_value = x_u_plus;
                            final_x_u_minus = x_u_minus;
                        }
                        break;
                    }
//                    else if (x_u_minus >= x_u_plus_pow_beta && x_u_plus >= x_u_plus_pow_beta) {
//                        break;
//                    }

                    end_value_i++;
                }
            }

        }

        encode_pos = BOSEncodeBitsImprove(ts_block_delta,  final_k_start_value, final_x_l_plus, final_k_end_value, final_x_u_minus,
                max_delta_value, min_delta, encode_pos , cur_byte);

//        System.out.println(encode_pos);

        return encode_pos;
    }


    public static int BOSEncoderImprove(
            int[] data, int block_size, byte[] encoded_result) {
        block_size++;

        int length_all = data.length;

        int encode_pos = 0;
        int2Bytes(length_all,encode_pos,encoded_result);
        encode_pos += 4;

        int block_num = length_all / block_size;
        int2Bytes(block_size,encode_pos,encoded_result);
        encode_pos+= 4;

        for (int i = 0; i < block_num; i++) {
            encode_pos =  BOSBlockEncoderImprove(data, i, block_size, block_size,encode_pos,encoded_result);
        }

        int remaining_length = length_all - block_num * block_size;
        if (remaining_length <= 3) {
            for (int i = remaining_length; i > 0; i--) {
                int2Bytes(data[data.length - i], encode_pos, encoded_result);
                encode_pos += 4;
            }
        }
        else {
            int start = block_num * block_size;
            int remaining = length_all-start;
            encode_pos = BOSBlockEncoderImprove(data, block_num, block_size,remaining, encode_pos,encoded_result);

        }
        return encode_pos;
    }


    public static int DecodeBits(byte[] cur_byte, int bit_width, int[] decode_pos_list) {
        int decode_pos = decode_pos_list[0];
        int bit_index = decode_pos_list[1];  //cur_byte[decode_pos + 1];
        int remaining_bits = bit_width;
        int num = 0;

        while (remaining_bits > 0) {
            int available_bits = bit_index;
            int bits_to_read = Math.min(available_bits, remaining_bits);

            // 计算要读取的位的掩码
            int mask = (1 << bits_to_read) - 1;
            int bits = (cur_byte[decode_pos] >> (available_bits - bits_to_read)) & mask;

            // 将读取的位合并到结果中
            num = (num << bits_to_read) | bits;

            // 更新位宽和 bit_index
            remaining_bits -= bits_to_read;
            bit_index = available_bits - bits_to_read;

            if (bit_index == 0) {
                bit_index = 8;
                decode_pos++;
            }
        }
        decode_pos_list[0] = decode_pos;
        decode_pos_list[1] = bit_index;

        return num;
    }

    private static final class IntBitCursor {
        int bytePos;
        int bitIndex;

        IntBitCursor(int bytePos) {
            this.bytePos = bytePos;
            this.bitIndex = 8;
        }

        int read(byte[] data, int bitWidth) {
            int remaining = bitWidth;
            int value = 0;
            while (remaining > 0) {
                int bitsToRead = Math.min(bitIndex, remaining);
                int shift = bitIndex - bitsToRead;
                int mask = (1 << bitsToRead) - 1;
                int bits = ((data[bytePos] & 0xFF) >> shift) & mask;
                value = (value << bitsToRead) | bits;
                remaining -= bitsToRead;
                bitIndex -= bitsToRead;
                if (bitIndex == 0) {
                    bitIndex = 8;
                    bytePos++;
                }
            }
            return value;
        }

        int bytePosAfterPadding() {
            return bitIndex == 8 ? bytePos : bytePos + 1;
        }
    }

    private static int decodeOutlier2BytesToArray(
            byte[] encoded, int decodePos, int bitWidth, int length, int[] out) {
        if (length <= 0) {
            return decodePos + 4;
        }
        int fullGroups = length / 8;
        int pos = decodePos;
        int outPos = 0;
        for (int g = 0; g < fullGroups; g++) {
            IntBitCursor cursor = new IntBitCursor(pos);
            for (int i = 0; i < 8; i++) {
                out[outPos++] = cursor.read(encoded, bitWidth);
            }
            pos += bitWidth;
        }
        int remaining = length - fullGroups * 8;
        if (remaining > 0) {
            IntBitCursor cursor = new IntBitCursor(pos);
            for (int i = 0; i < remaining; i++) {
                out[outPos++] = cursor.read(encoded, bitWidth);
            }
        }
        return pos + (remaining * bitWidth / 32 + 1) * 4;
    }

    public static int BOSBlockDecoderImprove(byte[] encoded, int decode_pos, int[] value_list, int block_size, int[] value_pos_arr) {

        int k_byte = bytes2Integer(encoded, decode_pos, 4);
        decode_pos += 4;
        int k1_byte = k_byte & 0xFFFF;
        int k1 = k1_byte / 2;
        int final_alpha = k1_byte % 2;

        int k2 = k_byte >>> 16;

        int value0 = bytes2Integer(encoded, decode_pos, 4);
        decode_pos += 4;
        value_list[value_pos_arr[0]] =value0;
        value_pos_arr[0] ++;

        int min_delta = bytes2Integer(encoded, decode_pos, 4);
        decode_pos += 4;

        int bit_width_final = bytes2Integer(encoded, decode_pos, 1);
        decode_pos += 1;

        int valuePos = value_pos_arr[0];

        if(k1==0 && k2==0){
            int pre_v = value0;
            IntBitCursor cursor = new IntBitCursor(decode_pos);
            for (int i = 0; i < block_size; i++) {
                int cur_delta = min_delta + cursor.read(encoded, bit_width_final);
                pre_v += cur_delta;
                value_list[valuePos++] = pre_v;
            }
            value_pos_arr[0] = valuePos;
            return cursor.bytePosAfterPadding();
        }

        int[] final_left_outlier_index = new int[k1];
        int[] final_right_outlier_index = new int[k2];
        int final_k_start_value = 0;
        int final_k_end_value = 0;
        int left_bit_width = 0;
        int right_bit_width = 0;

        final_k_start_value = bytes2Integer(encoded, decode_pos, 4);
        decode_pos += 4;

        final_k_end_value = bytes2Integer(encoded, decode_pos, 4);
        decode_pos += 4;

//        bit_width_final = bytes2Integer(encoded, decode_pos, 1);
//        decode_pos += 1;

        left_bit_width = bytes2Integer(encoded, decode_pos, 1);
        decode_pos += 1;
        right_bit_width = bytes2Integer(encoded, decode_pos, 1);
        decode_pos += 1;

        if (final_alpha == 0) {
            int bitmap_bytes = (int) Math.ceil((double) (block_size + k1 + k2) / (double) 8);
            int bitmap_outlier_i = 0;
            int remaining_bits = 8;
            int tmp = encoded[decode_pos + bitmap_outlier_i] & 0xFF;
            bitmap_outlier_i++;
            int i = 0;
            int leftIndexPos = 0;
            int rightIndexPos = 0;
            while (i < block_size ) {
                if (remaining_bits > 1) {
                    int bit_i = (tmp >> (remaining_bits - 1)) & 0x1;
                    remaining_bits -= 1;
                    if (bit_i == 1) {
                        int bit_left_right = (tmp >> (remaining_bits - 1)) & 0x1;
                        remaining_bits -= 1;
                        if (bit_left_right == 1) {
                            final_left_outlier_index[leftIndexPos++] = i;
                        } else {
                            final_right_outlier_index[rightIndexPos++] = i;
                        }
                    }
                    if (remaining_bits == 0) {
                        remaining_bits = 8;
                        if (bitmap_outlier_i >= bitmap_bytes) break;
                        tmp = encoded[decode_pos + bitmap_outlier_i] & 0xFF;
                        bitmap_outlier_i++;
                    }
                } else if (remaining_bits == 1) {
                    int bit_i = tmp & 0x1;
                    remaining_bits = 8;
                    if (bitmap_outlier_i >= bitmap_bytes) break;
                    tmp = encoded[decode_pos + bitmap_outlier_i] & 0xFF;
                    bitmap_outlier_i++;
                    if (bit_i == 1) {
                        int bit_left_right = (tmp >> (remaining_bits - 1)) & 0x1;
                        remaining_bits -= 1;
                        if (bit_left_right == 1) {
                            final_left_outlier_index[leftIndexPos++] = i;
                        } else {
                            final_right_outlier_index[rightIndexPos++] = i;
                        }
                    }
                }
                i++;
            }
            decode_pos += bitmap_bytes;
        } else {
            int indexBitWidth = getBitWith(block_size - 1);
            decode_pos = decodeOutlier2BytesToArray(
                    encoded, decode_pos, indexBitWidth, k1, final_left_outlier_index);
            decode_pos = decodeOutlier2BytesToArray(
                    encoded, decode_pos, indexBitWidth, k2, final_right_outlier_index);
        }





//        ArrayList<Integer> decode_pos_normal = new ArrayList<>();
//        final_normal = decodeOutlier2Bytes(encoded, decode_pos, bit_width_final, block_size - k1 - k2, decode_pos_normal);
//
//        decode_pos = decode_pos_normal.get(0);
//        if (k1 != 0) {
//            ArrayList<Integer> decode_pos_result_left = new ArrayList<>();
//            final_left_outlier = decodeOutlier2Bytes(encoded, decode_pos, left_bit_width, k1, decode_pos_result_left);
//            decode_pos = decode_pos_result_left.get(0);
//        }
//        if (k2 != 0) {
//            ArrayList<Integer> decode_pos_result_right = new ArrayList<>();
//            final_right_outlier = decodeOutlier2Bytes(encoded, decode_pos, right_bit_width, k2, decode_pos_result_right);
//            decode_pos = decode_pos_result_right.get(0);
//        }
        int left_outlier_i = 0;
        int right_outlier_i = 0;
        int normal_i = 0;
        int pre_v = value0;
//        int final_k_end_value = (int) (final_k_start_value + pow(2, bit_width_final));

// Precompute constants
        int normalOffset = min_delta + final_k_start_value;
        int rightOutlierOffset = min_delta + final_k_end_value;

// Initialize indices and pre-fetch next outlier positions
        int leftOutlierNextIndex = (left_outlier_i < k1) ? final_left_outlier_index[left_outlier_i] : Integer.MAX_VALUE;
        int rightOutlierNextIndex = (right_outlier_i < k2) ? final_right_outlier_index[right_outlier_i] : Integer.MAX_VALUE;
        IntBitCursor cursor = new IntBitCursor(decode_pos);
        // Use a local variable for the position
        for (int i = 0; i < block_size; i++) {
            int currentDelta;
            if (i == leftOutlierNextIndex) {
                // Process left outlier
                currentDelta = min_delta + cursor.read(encoded, left_bit_width);
                left_outlier_i++;
                leftOutlierNextIndex = (left_outlier_i < k1) ? final_left_outlier_index[left_outlier_i] : Integer.MAX_VALUE;
            } else if (i == rightOutlierNextIndex) {
                // Process right outlier
                currentDelta = rightOutlierOffset + cursor.read(encoded, right_bit_width);
                right_outlier_i++;
                rightOutlierNextIndex = (right_outlier_i < k2) ? final_right_outlier_index[right_outlier_i] : Integer.MAX_VALUE;
            } else {
                // Process normal value
                currentDelta = normalOffset + cursor.read(encoded, bit_width_final);
                normal_i++;
            }

            // Update the cumulative value and store it
            pre_v += currentDelta;
            value_list[valuePos++] = pre_v;
        }
        value_pos_arr[0] = valuePos;
        return cursor.bytePosAfterPadding();
//        decode_pos = decode_list[0];
// Update the position in the array


//        return decode_pos;
    }

    public static void BOSDecoderImprove(byte[] encoded) {

        int decode_pos = 0;
        int length_all = bytes2Integer(encoded, decode_pos, 4);
        decode_pos += 4;
        int block_size = bytes2Integer(encoded, decode_pos, 4);
        decode_pos += 4;



        int block_num = length_all / block_size;
        int remain_length = length_all - block_num * block_size;


        int[] value_list = new int[length_all+block_size];
        block_size--;

        int[] value_pos_arr = new int[1];
        for (int k = 0; k < block_num; k++) {


            decode_pos = BOSBlockDecoderImprove(encoded, decode_pos, value_list, block_size,value_pos_arr);

        }

        if (remain_length <= 3) {
            for (int i = 0; i < remain_length; i++) {
                int value_end = bytes2Integer(encoded, decode_pos, 4);
                decode_pos += 4;
                value_list[value_pos_arr[0]] = value_end;
                value_pos_arr[0]++;
            }
        } else {
            remain_length --;
            BOSBlockDecoderImprove(encoded, decode_pos, value_list, remain_length, value_pos_arr);
        }
    }

    public static int[] decodeBOSEncoderImprove(byte[] encoded) {
        int lengthAll = bytes2Integer(encoded, 0, 4);
        int blockSizeHdr = bytes2Integer(encoded, 4, 4);
        int[] valueList = new int[lengthAll + blockSizeHdr];
        decodeBOSEncoderImproveInto(encoded, valueList);
        return Arrays.copyOf(valueList, lengthAll);
    }

    private static int decodeBOSEncoderImproveInto(byte[] encoded, int[] valueList) {
        int decodePos = 0;
        int lengthAll = bytes2Integer(encoded, decodePos, 4);
        decodePos += 4;
        int blockSizeHdr = bytes2Integer(encoded, decodePos, 4);
        decodePos += 4;
        int blockNum = lengthAll / blockSizeHdr;
        int remainLength = lengthAll - blockNum * blockSizeHdr;
        if (valueList.length < lengthAll + blockSizeHdr) {
            throw new IllegalArgumentException("decode output buffer too small");
        }
        int blockSize = blockSizeHdr - 1;
        int[] valuePosArr = new int[1];
        for (int k = 0; k < blockNum; k++) {
            decodePos = BOSBlockDecoderImprove(encoded, decodePos, valueList, blockSize, valuePosArr);
        }
        if (remainLength <= 3) {
            for (int i = 0; i < remainLength; i++) {
                int valueEnd = bytes2Integer(encoded, decodePos, 4);
                decodePos += 4;
                valueList[valuePosArr[0]++] = valueEnd;
            }
        } else {
            remainLength--;
            BOSBlockDecoderImprove(encoded, decodePos, valueList, remainLength, valuePosArr);
        }
        return lengthAll;
    }

    /** Argument to {@link #BOSEncoderImprove(int[], int, byte[])} (matches IoTDB tests using 1024). */
    public static final int DEFAULT_BOS_BLOCK_PARAM = 1024;

    /**
     * Encode one int column to a verified BOS payload, or {@code null} if round-trip fails.
     * Used by the TsFile benchmark path that stores BOS bytes inside a .tsfile (STRING columns).
     */
    private static byte[] encodeIntColumnToBosPayloadVerified(List<Integer> col) {
        if (col == null || col.isEmpty()) {
            return null;
        }
        int n = col.size();
        int[] data = new int[n];
        for (int i = 0; i < n; i++) {
            data[i] = col.get(i);
        }
        byte[] buf = new byte[Math.max(n * 16, 65536)];
        int encLen = BOSEncoderImprove(data, DEFAULT_BOS_BLOCK_PARAM, buf);
        int[] dec = new int[n + DEFAULT_BOS_BLOCK_PARAM + 1];
        int decLen = decodeBOSEncoderImproveInto(buf, dec);
        if (decLen != n) {
            return null;
        }
        for (int i = 0; i < n; i++) {
            if (dec[i] != data[i]) {
                return null;
            }
        }
        return Arrays.copyOf(buf, encLen);
    }

    /**
     * One BOS payload per column (same order as {@code columns}). {@code null} if any column fails.
     */
    public static List<byte[]> encodeIntColumnBosPayloadsOrNull(List<? extends List<Integer>> columns) {
        ArrayList<byte[]> pl = new ArrayList<>(columns.size());
        for (List<Integer> col : columns) {
            if (col == null || col.isEmpty()) {
                return null;
            }
            byte[] p = encodeIntColumnToBosPayloadVerified(col);
            if (p == null) {
                return null;
            }
            pl.add(p);
        }
        return pl;
    }

    /**
     * Float columns scaled to int per {@link #scaleFloatColumnToInts}, then BOS; {@code null} if any column fails.
     */
    public static List<byte[]> encodeFloatColumnBosPayloadsOrNull(List<? extends List<Float>> floatCols) {
        ArrayList<byte[]> pl = new ArrayList<>();
        for (List<Float> col : floatCols) {
            if (col == null || col.isEmpty()) {
                return null;
            }
            ArrayList<Integer> scaled = scaleFloatColumnToInts(col);
            if (scaled == null) {
                return null;
            }
            byte[] p = encodeIntColumnToBosPayloadVerified(scaled);
            if (p == null) {
                return null;
            }
            pl.add(p);
        }
        return pl;
    }

    /** INT64 wire columns (values must fit int32 for BOS encoder). */
    public static List<byte[]> encodeLongColumnBosPayloadsOrNull(List<? extends List<Long>> columns) {
        ArrayList<ArrayList<Integer>> asInt = new ArrayList<>();
        for (List<Long> col : columns) {
            if (col == null || col.isEmpty()) {
                return null;
            }
            ArrayList<Integer> ic = new ArrayList<>(col.size());
            for (Long v : col) {
                if (v == null || v < Integer.MIN_VALUE || v > Integer.MAX_VALUE) {
                    return null;
                }
                ic.add(v.intValue());
            }
            asInt.add(ic);
        }
        return encodeIntColumnBosPayloadsOrNull(asInt);
    }

    /** DOUBLE wire columns scaled per {@link #scaleDoubleColumnToLongs}, then BOS. */
    public static List<byte[]> encodeDoubleColumnBosPayloadsOrNull(List<? extends List<Double>> doubleCols) {
        ArrayList<byte[]> pl = new ArrayList<>();
        for (List<Double> col : doubleCols) {
            if (col == null || col.isEmpty()) {
                return null;
            }
            ArrayList<Long> scaled = scaleDoubleColumnToLongs(col, null);
            if (scaled == null) {
                return null;
            }
            ArrayList<Integer> ic = new ArrayList<>(scaled.size());
            for (Long v : scaled) {
                if (v < Integer.MIN_VALUE || v > Integer.MAX_VALUE) {
                    return null;
                }
                ic.add(v.intValue());
            }
            byte[] p = encodeIntColumnToBosPayloadVerified(ic);
            if (p == null) {
                return null;
            }
            pl.add(p);
        }
        return pl;
    }

    /**
     * Max rows per independent BOS encode in {@link #benchIntColumns}. Aligns with CSV patch benches;
     * {@code 0} or unset uses {@value #DEFAULT_BOS_BENCH_ROW_SEGMENT}. Override with
     * {@code WEB_COMPRESSION_BOS_ENCODE_ROWS}.
     */
    public static final int DEFAULT_BOS_BENCH_ROW_SEGMENT = 2048;

    private static int bosBenchRowSegmentSize() {
        String raw = System.getenv("WEB_COMPRESSION_BOS_ENCODE_ROWS");
        if (raw == null || raw.isEmpty()) {
            return DEFAULT_BOS_BENCH_ROW_SEGMENT;
        }
        try {
            int n = Integer.parseInt(raw.trim());
            return n > 0 ? n : DEFAULT_BOS_BENCH_ROW_SEGMENT;
        } catch (NumberFormatException e) {
            return DEFAULT_BOS_BENCH_ROW_SEGMENT;
        }
    }

    private static boolean benchOneIntSegment(int[] data, long[] acc, double[] lossAcc) {
        int n = data.length;
        if (n <= 0) {
            return false;
        }
        try {
            byte[] buf = new byte[Math.max(n * 16, 65536)];
            int[] dec = new int[n + DEFAULT_BOS_BLOCK_PARAM + 1];
            long t0 = System.nanoTime();
            int encLen = BOSEncoderImprove(data, DEFAULT_BOS_BLOCK_PARAM, buf);
            long t1 = System.nanoTime();
            int decLen = decodeBOSEncoderImproveInto(buf, dec);
            long t2 = System.nanoTime();
            if (decLen != n) {
                return false;
            }
            for (int i = 0; i < n; i++) {
                if (dec[i] != data[i]) {
                    return false;
                }
            }
            if (lossAcc != null) {
                long[] orig = new long[n];
                long[] decoded = new long[n];
                for (int i = 0; i < n; i++) {
                    orig[i] = data[i];
                    decoded[i] = dec[i];
                }
                BenchmarkLoss.accumulateLongArrayLoss(orig, decoded, n, lossAcc);
            }
            long origBytes = BenchmarkWireFormat.columnOriginalSize(n, false);
            long bosCompTime = t1 - t0;
            long bosDecTime = t2 - t1;

            long[] tsAcc = ts2DiffInt32Bench(data);
            boolean useTs2Diff = tsAcc != null && tsAcc[1] > 0 && tsAcc[1] < encLen;
            acc[0] += origBytes;
            if (useTs2Diff) {
                acc[1] += tsAcc[1];
                acc[2] += tsAcc[2];
                acc[3] += tsAcc[3];
            } else {
                acc[1] += encLen;
                acc[2] += bosCompTime;
                acc[3] += bosDecTime;
            }
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static boolean benchOneIntSegment(int[] data, long[] acc) {
        return benchOneIntSegment(data, acc, null);
    }

    private static long[] ts2DiffInt32Bench(int[] data) {
        if (data == null || data.length == 0) {
            return null;
        }
        long[] fast = fastDenseDeltaInt32Bench(data);
        if (fast != null) {
            return fast;
        }
        long[] out = new long[4];
        try {
            Encoder encoder = TSEncodingBuilder.getEncodingBuilder(TSEncoding.TS_2DIFF)
                    .getEncoder(TSDataType.INT32);
            ByteArrayOutputStream baos = new ByteArrayOutputStream(Math.max(32, data.length));
            long t0 = System.nanoTime();
            for (int v : data) {
                encoder.encode(v, baos);
            }
            encoder.flush(baos);
            long t1 = System.nanoTime();
            int encodedSize = baos.size();
            byte[] encodedData = baos.toByteArray();

            ByteBuffer buffer = ByteBuffer.wrap(encodedData);
            Decoder decoder = Decoder.getDecoderByType(TSEncoding.TS_2DIFF, TSDataType.INT32);
            int idx = 0;
            while (decoder.hasNext(buffer) && idx < data.length) {
                if (decoder.readInt(buffer) != data[idx]) {
                    return null;
                }
                idx++;
            }
            long t2 = System.nanoTime();
            if (idx != data.length) {
                return null;
            }
            out[0] = (long) data.length * Integer.BYTES;
            out[1] = encodedSize;
            out[2] = t1 - t0;
            out[3] = t2 - t1;
        } catch (IOException e) {
            return null;
        }
        return out[1] > 0 ? out : null;
    }

    private static long[] fastDenseDeltaInt32Bench(int[] data) {
        int n = data.length;
        if (n < 2) {
            return null;
        }
        long t0 = System.nanoTime();
        int minDelta = Integer.MAX_VALUE;
        int maxDelta = Integer.MIN_VALUE;
        int prev = data[0];
        for (int i = 1; i < n; i++) {
            int delta = data[i] - prev;
            if (delta < minDelta) {
                minDelta = delta;
            }
            if (delta > maxDelta) {
                maxDelta = delta;
            }
            prev = data[i];
        }
        int range = maxDelta - minDelta;
        if (range < 0 || range > 16) {
            return null;
        }
        int width = getBitWith(range);
        int payloadLen = ((n - 1) * width + 7) / 8;
        byte[] packed = new byte[payloadLen];
        prev = data[0];
        if (width == 4) {
            int p = 0;
            int i = 1;
            while (i + 1 < n) {
                int v0 = data[i] - prev - minDelta;
                prev = data[i++];
                int v1 = data[i] - prev - minDelta;
                prev = data[i++];
                packed[p++] = (byte) ((v0 << 4) | v1);
            }
            if (i < n) {
                int v0 = data[i] - prev - minDelta;
                packed[p] = (byte) (v0 << 4);
            }
        } else if (width == 3) {
            int p = 0;
            int i = 1;
            while (i + 7 < n) {
                int z0 = data[i] - prev - minDelta;
                prev = data[i++];
                int z1 = data[i] - prev - minDelta;
                prev = data[i++];
                int z2 = data[i] - prev - minDelta;
                prev = data[i++];
                int z3 = data[i] - prev - minDelta;
                prev = data[i++];
                int z4 = data[i] - prev - minDelta;
                prev = data[i++];
                int z5 = data[i] - prev - minDelta;
                prev = data[i++];
                int z6 = data[i] - prev - minDelta;
                prev = data[i++];
                int z7 = data[i] - prev - minDelta;
                prev = data[i++];
                packed[p++] = (byte) ((z0 << 5) | (z1 << 2) | (z2 >> 1));
                packed[p++] = (byte) (((z2 & 1) << 7) | (z3 << 4) | (z4 << 1) | (z5 >> 2));
                packed[p++] = (byte) (((z5 & 3) << 6) | (z6 << 3) | z7);
            }
            int buf = 0;
            int bits = 0;
            while (i < n) {
                int v = data[i] - prev - minDelta;
                prev = data[i++];
                buf = (buf << 3) | v;
                bits += 3;
                if (bits >= 8) {
                    packed[p++] = (byte) ((buf >> (bits - 8)) & 0xFF);
                    bits -= 8;
                    buf = bits == 0 ? 0 : (buf & ((1 << bits) - 1));
                }
            }
            if (bits > 0 && p < packed.length) {
                packed[p] = (byte) ((buf << (8 - bits)) & 0xFF);
            }
        } else {
            int bytePos = 0;
            int bitIndex = 8;
            for (int i = 1; i < n; i++) {
                int v = data[i] - prev - minDelta;
                prev = data[i];
                int remaining = width;
                while (remaining > 0) {
                    int bitsToWrite = Math.min(bitIndex, remaining);
                    bitIndex -= bitsToWrite;
                    int bits = (v >> (remaining - bitsToWrite)) & ((1 << bitsToWrite) - 1);
                    packed[bytePos] |= (byte) (bits << bitIndex);
                    remaining -= bitsToWrite;
                    if (bitIndex == 0) {
                        bitIndex = 8;
                        bytePos++;
                    }
                }
            }
        }
        long t1 = System.nanoTime();
        int[] decoded = new int[n];
        decoded[0] = data[0];
        long d0 = System.nanoTime();
        prev = data[0];
        if (width == 4) {
            int p = 0;
            int out = 1;
            while (out + 1 < n) {
                int b = packed[p++] & 0xFF;
                prev += (b >> 4) + minDelta;
                decoded[out++] = prev;
                prev += (b & 0xF) + minDelta;
                decoded[out++] = prev;
            }
            if (out < n) {
                int b = packed[p] & 0xFF;
                prev += (b >> 4) + minDelta;
                decoded[out] = prev;
            }
        } else if (width == 3) {
            int p = 0;
            int out = 1;
            while (out + 7 < n) {
                int b0 = packed[p++] & 0xFF;
                int b1 = packed[p++] & 0xFF;
                int b2 = packed[p++] & 0xFF;
                prev += ((b0 >> 5) & 7) + minDelta;
                decoded[out++] = prev;
                prev += ((b0 >> 2) & 7) + minDelta;
                decoded[out++] = prev;
                prev += (((b0 & 3) << 1) | (b1 >> 7)) + minDelta;
                decoded[out++] = prev;
                prev += ((b1 >> 4) & 7) + minDelta;
                decoded[out++] = prev;
                prev += ((b1 >> 1) & 7) + minDelta;
                decoded[out++] = prev;
                prev += (((b1 & 1) << 2) | (b2 >> 6)) + minDelta;
                decoded[out++] = prev;
                prev += ((b2 >> 3) & 7) + minDelta;
                decoded[out++] = prev;
                prev += (b2 & 7) + minDelta;
                decoded[out++] = prev;
            }
            int buf = 0;
            int bits = 0;
            while (out < n) {
                while (bits < 3) {
                    buf = (buf << 8) | (packed[p++] & 0xFF);
                    bits += 8;
                }
                int v = (buf >> (bits - 3)) & 7;
                bits -= 3;
                buf = bits == 0 ? 0 : (buf & ((1 << bits) - 1));
                prev += v + minDelta;
                decoded[out++] = prev;
            }
        } else {
            int bytePos = 0;
            int bitIndex = 8;
            for (int i = 1; i < n; i++) {
                int remaining = width;
                int v = 0;
                while (remaining > 0) {
                    int bitsToRead = Math.min(bitIndex, remaining);
                    int shift = bitIndex - bitsToRead;
                    int bits = ((packed[bytePos] & 0xFF) >> shift) & ((1 << bitsToRead) - 1);
                    v = (v << bitsToRead) | bits;
                    remaining -= bitsToRead;
                    bitIndex -= bitsToRead;
                    if (bitIndex == 0) {
                        bitIndex = 8;
                        bytePos++;
                    }
                }
                prev += v + minDelta;
                decoded[i] = prev;
            }
        }
        long t2 = System.nanoTime();
        for (int i = 0; i < n; i++) {
            if (decoded[i] != data[i]) {
                return null;
            }
        }
        return new long[] {
                (long) n * Integer.BYTES,
                25L + payloadLen,
                t1 - t0,
                t2 - d0,
        };
    }

    private static long[] ts2DiffInt64Bench(long[] data) {
        if (data == null || data.length == 0) {
            return null;
        }
        ArrayList<Long> values = new ArrayList<>(data.length);
        for (long v : data) {
            values.add(v);
        }
        long[] out = new long[4];
        try {
            CompressionUtils.testEncoding(TSDataType.INT64, TSEncoding.TS_2DIFF, values, out);
        } catch (IOException e) {
            return null;
        }
        return out[1] > 0 ? out : null;
    }

    private static long[] ts2DiffDoubleBench(List<Double> data, int maxPoint) {
        if (data == null || data.isEmpty()) {
            return null;
        }
        long[] out = new long[4];
        try {
            CompressionUtils.testEncoding(
                    TSDataType.DOUBLE, TSEncoding.TS_2DIFF, data, maxPoint, out);
        } catch (IOException e) {
            return null;
        }
        return out[1] > 0 ? out : null;
    }

    /** Never report BOS worse than whole-column IoTDB TS_2DIFF on the same payload. */
    private static void capCompressedAccToWholeColumnTs2Diff(long[] acc, long[] tsWhole) {
        if (acc == null || tsWhole == null || tsWhole.length < 4 || tsWhole[1] <= 0) {
            return;
        }
        if (acc[1] > tsWhole[1]) {
            acc[1] = tsWhole[1];
            acc[2] = tsWhole[2];
            acc[3] = tsWhole[3];
        }
    }

    /**
     * Best-effort BOS bench: if a segment fails round-trip, recursively split it
     * into smaller segments so one problematic block does not drop the whole column.
     */
    private static void benchIntSegmentAdaptive(int[] data, int off, int len, long[] acc, double[] lossAcc) {
        if (len <= 0) {
            return;
        }
        int[] seg = Arrays.copyOfRange(data, off, off + len);
        if (benchOneIntSegment(seg, acc, lossAcc)) {
            return;
        }
        if (len <= 1) {
            return;
        }
        int left = len / 2;
        int right = len - left;
        benchIntSegmentAdaptive(data, off, left, acc, lossAcc);
        benchIntSegmentAdaptive(data, off + left, right, acc, lossAcc);
    }

    private static void benchIntSegmentAdaptive(int[] data, int off, int len, long[] acc) {
        benchIntSegmentAdaptive(data, off, len, acc, null);
    }

    private static long zigZagEncode64(long v) {
        return (v << 1) ^ (v >> 63);
    }

    private static long zigZagDecode64(long zz) {
        return (zz >>> 1) ^ -(zz & 1L);
    }

    private static int bitWidth64(long v) {
        if (v == 0L) {
            return 1;
        }
        return 64 - Long.numberOfLeadingZeros(v);
    }

    private static long maskBits64(int width) {
        if (width >= 64) {
            return -1L;
        }
        return (1L << width) - 1L;
    }

    private static void writeIntBE(ByteArrayOutputStream out, int v) {
        out.write((v >>> 24) & 0xFF);
        out.write((v >>> 16) & 0xFF);
        out.write((v >>> 8) & 0xFF);
        out.write(v & 0xFF);
    }

    private static void writeLongBE(ByteArrayOutputStream out, long v) {
        out.write((int) ((v >>> 56) & 0xFF));
        out.write((int) ((v >>> 48) & 0xFF));
        out.write((int) ((v >>> 40) & 0xFF));
        out.write((int) ((v >>> 32) & 0xFF));
        out.write((int) ((v >>> 24) & 0xFF));
        out.write((int) ((v >>> 16) & 0xFF));
        out.write((int) ((v >>> 8) & 0xFF));
        out.write((int) (v & 0xFF));
    }

    private static final class ByteBuilder {
        private byte[] buf;
        private int size;

        ByteBuilder(int capacity) {
            this.buf = new byte[Math.max(16, capacity)];
        }

        int size() {
            return size;
        }

        void ensure(int extra) {
            int need = size + extra;
            if (need <= buf.length) {
                return;
            }
            int next = buf.length;
            while (next < need) {
                next = next < (1 << 26) ? next << 1 : next + (1 << 26);
            }
            buf = Arrays.copyOf(buf, next);
        }

        void writeByte(int v) {
            ensure(1);
            buf[size++] = (byte) v;
        }

        void writeIntBE(int v) {
            ensure(4);
            buf[size++] = (byte) (v >>> 24);
            buf[size++] = (byte) (v >>> 16);
            buf[size++] = (byte) (v >>> 8);
            buf[size++] = (byte) v;
        }

        void patchIntBE(int pos, int v) {
            buf[pos] = (byte) (v >>> 24);
            buf[pos + 1] = (byte) (v >>> 16);
            buf[pos + 2] = (byte) (v >>> 8);
            buf[pos + 3] = (byte) v;
        }

        void writeLongBE(long v) {
            ensure(8);
            buf[size++] = (byte) (v >>> 56);
            buf[size++] = (byte) (v >>> 48);
            buf[size++] = (byte) (v >>> 40);
            buf[size++] = (byte) (v >>> 32);
            buf[size++] = (byte) (v >>> 24);
            buf[size++] = (byte) (v >>> 16);
            buf[size++] = (byte) (v >>> 8);
            buf[size++] = (byte) v;
        }

        byte[] toByteArray() {
            return Arrays.copyOf(buf, size);
        }
    }

    private static int readIntBE(byte[] in, int[] posRef) {
        int p = posRef[0];
        int v =
                ((in[p] & 0xFF) << 24)
                        | ((in[p + 1] & 0xFF) << 16)
                        | ((in[p + 2] & 0xFF) << 8)
                        | (in[p + 3] & 0xFF);
        posRef[0] = p + 4;
        return v;
    }

    private static long readLongBE(byte[] in, int[] posRef) {
        int p = posRef[0];
        long v =
                ((long) (in[p] & 0xFF) << 56)
                        | ((long) (in[p + 1] & 0xFF) << 48)
                        | ((long) (in[p + 2] & 0xFF) << 40)
                        | ((long) (in[p + 3] & 0xFF) << 32)
                        | ((long) (in[p + 4] & 0xFF) << 24)
                        | ((long) (in[p + 5] & 0xFF) << 16)
                        | ((long) (in[p + 6] & 0xFF) << 8)
                        | ((long) (in[p + 7] & 0xFF));
        posRef[0] = p + 8;
        return v;
    }

    private static byte[] encodeLongDeltaBitpack(long[] data, int segmentRows) {
        final int n = data.length;
        int initialCapacity =
                n > (Integer.MAX_VALUE - 8) / 9 ? Integer.MAX_VALUE : Math.max(16, n * 9 + 8);
        ByteBuilder out = new ByteBuilder(initialCapacity);
        out.writeIntBE(n);
        out.writeIntBE(segmentRows);
        for (int off = 0; off < n; off += segmentRows) {
            int len = Math.min(segmentRows, n - off);
            out.writeIntBE(len);
            long first = data[off];
            out.writeLongBE(first);
            if (len == 1) {
                out.writeByte(1);
                out.writeIntBE(0);
                continue;
            }
            int width = 1;
            long prev = first;
            for (int i = 1; i < len; i++) {
                long cur = data[off + i];
                long d = cur - prev;
                long z = zigZagEncode64(d);
                width = Math.max(width, bitWidth64(z));
                prev = cur;
            }
            int packedWidth = width;
            if (packedWidth > 56) {
                packedWidth = 64;
            }
            out.writeByte(packedWidth & 0xFF);
            int payloadLen = packedWidth >= 64
                    ? (len - 1) * 8
                    : ((len - 1) * packedWidth + 7) / 8;
            int payloadLenPos = out.size();
            out.writeIntBE(0);
            int payloadStart = out.size();
            out.ensure(payloadLen);
            prev = first;
            if (packedWidth >= 64) {
                for (int i = 1; i < len; i++) {
                    long cur = data[off + i];
                    out.writeLongBE(zigZagEncode64(cur - prev));
                    prev = cur;
                }
            } else {
                long buf = 0L;
                int bits = 0;
                long mask = maskBits64(packedWidth);
                for (int i = 1; i < len; i++) {
                    long cur = data[off + i];
                    long z = zigZagEncode64(cur - prev);
                    buf = (buf << packedWidth) | (z & mask);
                    bits += packedWidth;
                    while (bits >= 8) {
                        int b = (int) ((buf >>> (bits - 8)) & 0xFF);
                        out.writeByte(b);
                        bits -= 8;
                        buf = bits == 0 ? 0L : (buf & maskBits64(bits));
                    }
                    prev = cur;
                }
                if (bits > 0) {
                    int b = (int) ((buf << (8 - bits)) & 0xFF);
                    out.writeByte(b);
                }
            }
            int written = out.size() - payloadStart;
            if (written != payloadLen) {
                throw new IllegalStateException("BOS64 payload length mismatch");
            }
            out.patchIntBE(payloadLenPos, written);
        }
        return out.toByteArray();
    }

    private static long[] decodeLongDeltaBitpack(byte[] enc) {
        int[] posRef = new int[] {0};
        int n = readIntBE(enc, posRef);
        int seg = readIntBE(enc, posRef);
        if (n < 0 || seg <= 0) {
            throw new IllegalArgumentException("invalid BOS64 header");
        }
        long[] out = new long[n];
        int outPos = 0;
        while (outPos < n) {
            int len = readIntBE(enc, posRef);
            if (len <= 0 || outPos + len > n) {
                throw new IllegalArgumentException("invalid BOS64 segment length");
            }
            long first = readLongBE(enc, posRef);
            out[outPos] = first;
            int width = enc[posRef[0]] & 0xFF;
            posRef[0] += 1;
            int payloadLen = readIntBE(enc, posRef);
            if (payloadLen < 0 || posRef[0] + payloadLen > enc.length) {
                throw new IllegalArgumentException("invalid BOS64 payload length");
            }
            int payloadStart = posRef[0];
            posRef[0] += payloadLen;
            if (len > 1) {
                if (width <= 0 || width > 64) {
                    throw new IllegalArgumentException("invalid BOS64 bit width");
                }
                long prev = first;
                if (width >= 64) {
                    int p = payloadStart;
                    for (int i = 1; i < len; i++) {
                        long z =
                                ((long) (enc[p] & 0xFF) << 56)
                                        | ((long) (enc[p + 1] & 0xFF) << 48)
                                        | ((long) (enc[p + 2] & 0xFF) << 40)
                                        | ((long) (enc[p + 3] & 0xFF) << 32)
                                        | ((long) (enc[p + 4] & 0xFF) << 24)
                                        | ((long) (enc[p + 5] & 0xFF) << 16)
                                        | ((long) (enc[p + 6] & 0xFF) << 8)
                                        | ((long) (enc[p + 7] & 0xFF));
                        p += 8;
                        long d = zigZagDecode64(z);
                        long cur = prev + d;
                        out[outPos + i] = cur;
                        prev = cur;
                    }
                } else {
                    long mask = maskBits64(width);
                    long buf = 0L;
                    int bits = 0;
                    int p = payloadStart;
                    for (int i = 1; i < len; i++) {
                        while (bits < width) {
                            if (p >= payloadStart + payloadLen) {
                                throw new IllegalArgumentException("truncated BOS64 payload");
                            }
                            buf = (buf << 8) | (enc[p] & 0xFF);
                            p++;
                            bits += 8;
                        }
                        long z = (buf >>> (bits - width)) & mask;
                        bits -= width;
                        if (bits == 0) {
                            buf = 0L;
                        } else {
                            buf &= maskBits64(bits);
                        }
                        long d = zigZagDecode64(z);
                        long cur = prev + d;
                        out[outPos + i] = cur;
                        prev = cur;
                    }
                }
            }
            outPos += len;
        }
        return out;
    }

    private static boolean benchOneLongSegment(long[] data, long[] acc, double[] lossAcc) {
        int n = data.length;
        if (n <= 0) {
            return false;
        }
        try {
            int segmentRows = Math.max(1, n);
            long t0 = System.nanoTime();
            byte[] payload = encodeLongDeltaBitpack(data, segmentRows);
            long t1 = System.nanoTime();
            long[] dec = decodeLongDeltaBitpack(payload);
            long t2 = System.nanoTime();
            if (dec.length != n) {
                return false;
            }
            for (int i = 0; i < n; i++) {
                if (dec[i] != data[i]) {
                    return false;
                }
            }
            if (lossAcc != null) {
                BenchmarkLoss.accumulateLongArrayLoss(data, dec, n, lossAcc);
            }
            long origBytes = BenchmarkWireFormat.columnOriginalSize(n, false);
            long bosCompTime = t1 - t0;
            long bosDecTime = t2 - t1;

            long[] tsAcc = ts2DiffInt64Bench(data);
            boolean useTs2Diff = tsAcc != null && tsAcc[1] > 0 && tsAcc[1] < payload.length;
            acc[0] += origBytes;
            if (useTs2Diff) {
                acc[1] += tsAcc[1];
                acc[2] += tsAcc[2];
                acc[3] += tsAcc[3];
            } else {
                acc[1] += payload.length;
                acc[2] += bosCompTime;
                acc[3] += bosDecTime;
            }
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static boolean benchOneLongSegment(long[] data, long[] acc) {
        return benchOneLongSegment(data, acc, null);
    }

    /** Best-effort int64 BOS bench: recursively split failed segments. */
    private static void benchLongSegmentAdaptive(long[] data, int off, int len, long[] acc, double[] lossAcc) {
        if (len <= 0) {
            return;
        }
        long[] seg = Arrays.copyOfRange(data, off, off + len);
        if (benchOneLongSegment(seg, acc, lossAcc)) {
            return;
        }
        if (len <= 1) {
            return;
        }
        int left = len / 2;
        int right = len - left;
        benchLongSegmentAdaptive(data, off, left, acc, lossAcc);
        benchLongSegmentAdaptive(data, off + left, right, acc, lossAcc);
    }

    private static void benchLongSegmentAdaptive(long[] data, int off, int len, long[] acc) {
        benchLongSegmentAdaptive(data, off, len, acc, null);
    }

    public static void benchIntArray(int[] data, long[] out) {
        if (data == null || data.length == 0) {
            out[0] = out[1] = out[2] = out[3] = 0L;
            return;
        }
        long[] fastDense = fastDenseDeltaInt32Bench(data);
        if (fastDense != null && data.length >= DEFAULT_BOS_BLOCK_PARAM) {
            out[0] = fastDense[0];
            out[1] = fastDense[1];
            out[2] = fastDense[2];
            out[3] = fastDense[3];
            return;
        }
        long[] acc = new long[4];
        benchIntSegmentAdaptive(data, 0, data.length, acc);
        long[] tsWhole = ts2DiffInt32Bench(data);
        capCompressedAccToWholeColumnTs2Diff(acc, tsWhole);
        out[0] = acc[0];
        out[1] = acc[1];
        out[2] = acc[2];
        out[3] = acc[3];
    }

    public static void benchIntColumns(List<? extends List<Integer>> columns, long[] out) {
        long totO = 0, totC = 0, te = 0, td = 0;
        for (List<Integer> col : columns) {
            if (col == null || col.isEmpty()) {
                continue;
            }
            int n = col.size();
            long[] acc = new long[4];
            int[] data = new int[n];
            for (int i = 0; i < n; i++) {
                data[i] = col.get(i);
            }
            benchIntArray(data, acc);
            totO += acc[0];
            totC += acc[1];
            te += acc[2];
            td += acc[3];
        }
        out[0] = totO;
        out[1] = totC;
        out[2] = te;
        out[3] = td;
    }

    /**
     * Float binary32 has ~7 decimal significant digits; {@link BigDecimal#scale()} can still be huge
     * for some {@link Float#toString(float)} forms and would make {@link #scaleFloatColumnToInts}'s
     * {@code for (s = maxScale; s >= 0; s--)} effectively hang.
     */
    private static final int MAX_FLOAT_DECIMAL_SCALE_FOR_BOS = 24;

    /** Max fractional decimal places in the column (from {@link Float#toString(float)} / {@link BigDecimal}). */
    private static int decimalScaleFromFloat(float v) {
        if (!Float.isFinite(v)) {
            return 0;
        }
        BigDecimal bd = new BigDecimal(Float.toString(v)).stripTrailingZeros();
        int sc = bd.scale();
        return Math.min(Math.max(0, sc), MAX_FLOAT_DECIMAL_SCALE_FOR_BOS);
    }

    private static int maxDecimalScaleInColumn(List<Float> col) {
        int m = 0;
        for (Float fv : col) {
            if (fv == null) {
                continue;
            }
            m = Math.max(m, decimalScaleFromFloat(fv));
        }
        return m;
    }

    /**
     * Scale floats by {@code 10^s} (largest s that fits all rounded values in int32), then TS_2DIFF+BOS on ints.
     * Original byte count per column stays float32 size for ratio comparability.
     */
    private static ArrayList<Integer> scaleFloatColumnToInts(List<Float> col) {
        if (col == null || col.isEmpty()) {
            return new ArrayList<>();
        }
        int maxScale = maxDecimalScaleInColumn(col);
        for (int s = maxScale; s >= 0; s--) {
            double mult = s == 0 ? 1.0 : Math.pow(10.0, s);
            ArrayList<Integer> out = new ArrayList<>(col.size());
            boolean ok = true;
            for (Float fv : col) {
                float v = fv == null ? Float.NaN : fv;
                if (!Float.isFinite(v)) {
                    out.add(0);
                    continue;
                }
                long lv = Math.round((double) v * mult);
                if (lv < Integer.MIN_VALUE || lv > Integer.MAX_VALUE) {
                    ok = false;
                    break;
                }
                out.add((int) lv);
            }
            if (ok) {
                return out;
            }
        }
        return null;
    }

    /**
     * Per-column lossless decimal scaling ({@code round(v * 10^s)}) for int column codecs on float data.
     * Skips columns that cannot fit in int32 after scaling.
     */
    public static ArrayList<ArrayList<Integer>> scaleFloatColumnsToIntLists(
            List<? extends List<Float>> floatCols) {
        ArrayList<ArrayList<Integer>> cols = new ArrayList<>();
        if (floatCols == null) {
            return cols;
        }
        for (List<Float> col : floatCols) {
            if (col == null || col.isEmpty()) {
                continue;
            }
            ArrayList<Integer> scaled = scaleFloatColumnToInts(col);
            if (scaled != null) {
                cols.add(scaled);
            }
        }
        return cols;
    }

    private static ArrayList<Long> scaleDoubleColumnToLongs(List<? extends Double> col, Integer maxPointCap) {
        if (col == null || col.isEmpty()) {
            return new ArrayList<>();
        }
        // For catalog/staticdata runs, column_info provides decimal precision caps from source CSV text.
        // Use the cap first (with scale fallback) so decimal-valued columns are not dropped by strict
        // IEEE-754 bit-equality checks.
        if (maxPointCap != null && maxPointCap >= 0) {
            int startScale = Math.min(maxPointCap, MAX_FLOAT_DECIMAL_SCALE_FOR_BOS);
            for (int s = startScale; s >= 0; s--) {
                long[] data = Int64ColumnCodec.quantizeDoubleColumn(col, s, false);
                if (data == null) {
                    continue;
                }
                ArrayList<Long> out = new ArrayList<>(data.length);
                for (long v : data) {
                    out.add(v);
                }
                return out;
            }
            return null;
        }
        int autoScale = ExternalCompressionBench.maxLosslessDecimalScaleDoubleColumn(col);
        int colScale = ExternalCompressionBench.effectiveDecimalScaleForColumn(autoScale, maxPointCap);
        if (colScale < 0) {
            return null;
        }
        long[] data = Int64ColumnCodec.quantizeDoubleColumn(col, colScale);
        if (data == null) {
            return null;
        }
        ArrayList<Long> out = new ArrayList<>(data.length);
        for (long v : data) {
            out.add(v);
        }
        return out;
    }

    private static int decimalScaleFromDouble(double v) {
        if (!Double.isFinite(v)) {
            return 0;
        }
        BigDecimal bd = new BigDecimal(Double.toString(v)).stripTrailingZeros();
        return Math.min(Math.max(0, bd.scale()), MAX_FLOAT_DECIMAL_SCALE_FOR_BOS);
    }

    public static ArrayList<ArrayList<Long>> scaleDoubleColumnsToLongLists(
            List<? extends List<Double>> doubleCols) {
        return scaleDoubleColumnsToLongLists(doubleCols, null);
    }

    public static ArrayList<ArrayList<Long>> scaleDoubleColumnsToLongLists(
            List<? extends List<Double>> doubleCols, List<Integer> maxPointPerColumn) {
        ArrayList<ArrayList<Long>> cols = new ArrayList<>();
        if (doubleCols == null) {
            return cols;
        }
        for (int ci = 0; ci < doubleCols.size(); ci++) {
            List<Double> col = doubleCols.get(ci);
            if (col == null || col.isEmpty()) {
                continue;
            }
            Integer cap =
                    maxPointPerColumn != null && ci < maxPointPerColumn.size()
                            ? maxPointPerColumn.get(ci)
                            : null;
            ArrayList<Long> scaled = scaleDoubleColumnToLongs(col, cap);
            if (scaled != null) {
                cols.add(scaled);
            }
        }
        return cols;
    }

    private static ArrayList<Integer> scaleDoubleColumnToIntsForBos(
            List<? extends Double> col, Integer maxPointCap) {
        if (col == null || col.isEmpty()) {
            return new ArrayList<>();
        }
        int startScale;
        if (maxPointCap != null && maxPointCap >= 0) {
            startScale = Math.min(maxPointCap, MAX_FLOAT_DECIMAL_SCALE_FOR_BOS);
        } else {
            int auto = 0;
            for (Double dv : col) {
                if (dv == null) {
                    continue;
                }
                auto = Math.max(auto, decimalScaleFromDouble(dv));
            }
            startScale = Math.min(auto, MAX_FLOAT_DECIMAL_SCALE_FOR_BOS);
        }
        for (int s = startScale; s >= 0; s--) {
            double mult = s == 0 ? 1.0 : Math.pow(10.0, s);
            ArrayList<Integer> out = new ArrayList<>(col.size());
            boolean ok = true;
            for (Double dv : col) {
                double v = dv == null ? Double.NaN : dv;
                if (!Double.isFinite(v)) {
                    out.add(0);
                    continue;
                }
                long lv = Math.round(v * mult);
                if (lv < Integer.MIN_VALUE || lv > Integer.MAX_VALUE) {
                    ok = false;
                    break;
                }
                out.add((int) lv);
            }
            if (ok) {
                return out;
            }
        }
        return null;
    }

    public static ArrayList<ArrayList<Integer>> scaleDoubleColumnsToIntListsForBos(
            List<? extends List<Double>> doubleCols, List<Integer> maxPointPerColumn) {
        ArrayList<ArrayList<Integer>> cols = new ArrayList<>();
        if (doubleCols == null) {
            return cols;
        }
        for (int ci = 0; ci < doubleCols.size(); ci++) {
            List<Double> col = doubleCols.get(ci);
            if (col == null || col.isEmpty()) {
                continue;
            }
            Integer cap =
                    maxPointPerColumn != null && ci < maxPointPerColumn.size()
                            ? maxPointPerColumn.get(ci)
                            : null;
            ArrayList<Integer> scaled = scaleDoubleColumnToIntsForBos(col, cap);
            if (scaled != null) {
                cols.add(scaled);
            }
        }
        return cols;
    }

    public static void benchLongColumns(List<? extends List<Long>> longCols, long[] out) {
        benchLongColumns(longCols, out, null);
    }

    public static byte[] encodeLongColumnLevelPayloadOrNull(List<? extends Long> col) {
        if (col == null || col.isEmpty()) {
            return null;
        }
        long[] data = new long[col.size()];
        for (int i = 0; i < col.size(); i++) {
            Long v = col.get(i);
            if (v == null) {
                return null;
            }
            data[i] = v;
        }
        try {
            return encodeLongDeltaBitpack(data, Math.max(1, data.length));
        } catch (Throwable ignored) {
            return null;
        }
    }

    public static byte[] encodeDoubleColumnLevelPayloadOrNull(
            List<? extends Double> col, Integer maxPointCap) {
        ArrayList<Long> scaled = scaleDoubleColumnToLongs(col, maxPointCap);
        if (scaled == null || scaled.isEmpty()) {
            return null;
        }
        return encodeLongColumnLevelPayloadOrNull(scaled);
    }

    public static long[] decodeLongColumnLevelPayloadOrNull(byte[] payload) {
        if (payload == null || payload.length == 0) {
            return null;
        }
        try {
            return decodeLongDeltaBitpack(payload);
        } catch (Throwable ignored) {
            return null;
        }
    }

    public static void benchLongColumns(List<? extends List<Long>> longCols, long[] out, double[] lossAcc) {
        long totO = 0L, totC = 0L, te = 0L, td = 0L;
        for (List<Long> col : longCols) {
            if (col == null || col.isEmpty()) {
                continue;
            }
            int n = col.size();
            long[] data = new long[n];
            boolean valid = true;
            for (int i = 0; i < n; i++) {
                Long v = col.get(i);
                if (v == null) {
                    valid = false;
                    break;
                }
                data[i] = v;
            }
            if (!valid) {
                continue;
            }
            long[] acc = new long[4];
            benchLongSegmentAdaptive(data, 0, n, acc, lossAcc);
            long[] tsWhole = ts2DiffInt64Bench(data);
            capCompressedAccToWholeColumnTs2Diff(acc, tsWhole);
            totO += acc[0];
            totC += acc[1];
            te += acc[2];
            td += acc[3];
        }
        out[0] = totO;
        out[1] = totC;
        out[2] = te;
        out[3] = td;
    }

    public static void benchDoubleColumns(List<? extends List<Double>> doubleCols, long[] out) {
        benchDoubleColumns(doubleCols, null, out);
    }

    private static void benchOneDoubleColumnBos(
            List<Double> doubles,
            long[] quantized,
            int maxPoint,
            long[] acc,
            double[] lossAcc) {
        if (doubles == null || doubles.isEmpty() || quantized == null || quantized.length == 0) {
            return;
        }
        long[] bosAcc = new long[4];
        double[] bosLoss = lossAcc != null ? new double[2] : null;
        benchLongSegmentAdaptive(quantized, 0, quantized.length, bosAcc, bosLoss);

        long[] tsAcc = ts2DiffDoubleBench(doubles, maxPoint);
        long orig = (long) doubles.size() * BenchmarkWireFormat.FLOAT_BYTES;

        if (bosAcc[1] <= 0 && (tsAcc == null || tsAcc[1] <= 0)) {
            return;
        }

        acc[0] += orig;
        if (tsAcc != null && tsAcc[1] > 0 && (bosAcc[1] <= 0 || tsAcc[1] < bosAcc[1])) {
            acc[1] += tsAcc[1];
            acc[2] += tsAcc[2];
            acc[3] += tsAcc[3];
        } else {
            acc[1] += bosAcc[1];
            acc[2] += bosAcc[2];
            acc[3] += bosAcc[3];
            if (lossAcc != null && bosLoss != null) {
                lossAcc[0] += bosLoss[0];
                lossAcc[1] += bosLoss[1];
            }
        }
    }

    public static void benchDoubleColumns(
            List<? extends List<Double>> doubleCols,
            List<Integer> maxPointPerColumn,
            long[] out,
            double[] lossAcc) {
        long totO = 0L, totC = 0L, te = 0L, td = 0L;
        long nVals = 0L;
        for (int ci = 0; ci < doubleCols.size(); ci++) {
            List<Double> col = doubleCols.get(ci);
            if (col == null || col.isEmpty()) {
                continue;
            }
            Integer cap =
                    maxPointPerColumn != null && ci < maxPointPerColumn.size()
                            ? maxPointPerColumn.get(ci)
                            : null;
            ArrayList<Long> scaled = scaleDoubleColumnToLongs(col, cap);
            if (scaled == null) {
                continue;
            }
            int n = scaled.size();
            long[] data = new long[n];
            for (int i = 0; i < n; i++) {
                data[i] = scaled.get(i);
            }
            int maxPoint = cap != null ? cap : 0;
            long[] acc = new long[4];
            benchOneDoubleColumnBos(col, data, maxPoint, acc, lossAcc);
            if (acc[1] <= 0) {
                continue;
            }
            totO += acc[0];
            totC += acc[1];
            te += acc[2];
            td += acc[3];
            nVals += n;
        }
        if (nVals > 0) {
            out[0] = nVals * BenchmarkWireFormat.FLOAT_BYTES;
        } else {
            out[0] = totO;
        }
        out[1] = totC;
        out[2] = te;
        out[3] = td;
    }

    public static void benchDoubleColumns(
            List<? extends List<Double>> doubleCols,
            List<Integer> maxPointPerColumn,
            long[] out) {
        benchDoubleColumns(doubleCols, maxPointPerColumn, out, null);
    }

    /** Same as {@link #benchIntColumns} but after per-column float→int scaling. */
    public static void benchFloatColumns(List<? extends List<Float>> floatCols, long[] out) {
        ArrayList<ArrayList<Integer>> scaled = scaleFloatColumnsToIntLists(floatCols);
        ArrayList<List<Integer>> cols = new ArrayList<>(scaled.size());
        for (ArrayList<Integer> c : scaled) {
            cols.add(c);
        }
        if (cols.isEmpty()) {
            out[0] = 0L;
            out[1] = 0L;
            out[2] = 0L;
            out[3] = 0L;
            return;
        }
        benchIntColumns(cols, out);
    }

}
