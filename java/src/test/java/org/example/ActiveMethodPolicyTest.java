package org.example;

import java.util.Arrays;

/** Standalone active-method policy regression fixture. */
public final class ActiveMethodPolicyTest {
    public static void main(String[] args) {
        if (!ActiveMethodPolicy.POLICY_ID.equals("paper164-active85-glza-retired-v1")) {
            throw new AssertionError("policy id");
        }
        if (ActiveMethodPolicy.CATALOG_METHOD_COUNT != 86
                || ActiveMethodPolicy.ACTIVE_METHOD_COUNT != 85) {
            throw new AssertionError("method counts");
        }
        if (ActiveMethodPolicy.isActiveAlgorithmKey("glza")) {
            throw new AssertionError("GLZA must be retired");
        }
        if (!ActiveMethodPolicy.isActiveAlgorithmKey("lz4-framed")) {
            throw new AssertionError("LZ4 must remain active");
        }
        if (NativeLzFamilyBench.algorithms().contains("glza")) {
            throw new AssertionError("active Java catalog exposed GLZA");
        }
        try {
            NativeLzFamilyBench.compress(new byte[] {'a', 'b', 'c'}, "glza");
            throw new AssertionError("active Java encode accepted GLZA");
        } catch (IllegalArgumentException expected) {
            if (!expected.getMessage().contains(ActiveMethodPolicy.POLICY_ID)) {
                throw new AssertionError("retirement error lacks policy ID");
            }
        }
        byte[] decoded = NativeLzFamilyBench.decompress(
                new byte[] {0, 'a', 'b', 'c'}, "glza", 3);
        if (!Arrays.equals(decoded, new byte[] {'a', 'b', 'c'})) {
            throw new AssertionError("historical GLZA decode compatibility");
        }
    }
}
