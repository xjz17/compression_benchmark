import policy from "../../../benchmark/reference/methods/active_method_policy_v1.json";

const EXPECTED_SCHEMA = "tscb-active-method-policy-v1";

if (policy.schema !== EXPECTED_SCHEMA) {
  throw new Error(`Unsupported active-method policy schema: ${String(policy.schema)}`);
}
if (policy.catalog_method_count - policy.retired_methods.length !== policy.active_method_count) {
  throw new Error("Active-method policy count mismatch");
}

const normalize = (value) => String(value ?? "").trim().toLowerCase();

const RETIRED_METHOD_NAMES = new Set(
  policy.retired_methods.flatMap((row) => row.method_aliases || []).map(normalize),
);
const RETIRED_ALGORITHM_KEYS = new Set(
  policy.retired_methods.flatMap((row) => row.algorithm_keys || []).map(normalize),
);

export const ACTIVE_METHOD_POLICY = Object.freeze(policy);

export function isActiveMethodName(value) {
  const normalized = normalize(value);
  return Boolean(normalized) && !RETIRED_METHOD_NAMES.has(normalized);
}

export function isActiveAlgorithmKey(value) {
  const normalized = normalize(value).replace(/^\ufeff/, "");
  return Boolean(normalized) && !RETIRED_ALGORITHM_KEYS.has(normalized);
}

export function filterActiveAlgorithmKeys(values) {
  return (Array.isArray(values) ? values : []).filter(isActiveAlgorithmKey);
}
