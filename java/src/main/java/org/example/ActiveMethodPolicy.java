package org.example;

/** Active-method gate generated from active_method_policy_v1.json. */
public final class ActiveMethodPolicy {
    public static final String POLICY_ID = "paper164-active85-glza-retired-v1";
    public static final String POLICY_SHA256 =
            "f48677a529247a165898bce319f3e20ef6ba9c305552676344f3ead39d0bfc5d";
    public static final int CATALOG_METHOD_COUNT = 86;
    public static final int ACTIVE_METHOD_COUNT = 85;

    private ActiveMethodPolicy() {
    }

    public static boolean isActiveAlgorithmKey(String algorithm) {
        return algorithm != null && !algorithm.equalsIgnoreCase("glza");
    }
}
