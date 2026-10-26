/** Shared helpers for Methods page encoding vs byte-codec operator matrices. */

export function resolvePipelineForDataType(row, dtypeFilter) {
  const pipeline = row?.operator_pipeline;
  if (pipeline == null) {
    return null;
  }
  const dtype = String(dtypeFilter || "").trim().toLowerCase();
  if (dtype === "int" || dtype === "float") {
    if (
      pipeline &&
      typeof pipeline === "object" &&
      !Array.isArray(pipeline) &&
      Array.isArray(pipeline[dtype])
    ) {
      return pipeline[dtype];
    }
  }
  if (Array.isArray(pipeline)) {
    return pipeline;
  }
  if (pipeline && typeof pipeline === "object") {
    const types = (row.data_types || [])
      .map((t) => String(t).trim().toLowerCase())
      .filter(Boolean);
    const primary = types[0] || "int";
    if (Array.isArray(pipeline[primary])) {
      return pipeline[primary];
    }
    const firstKey = Object.keys(pipeline).find((key) =>
      Array.isArray(pipeline[key]),
    );
    if (firstKey) {
      return pipeline[firstKey];
    }
  }
  return null;
}

/** @returns {string[]} */
export function pipelineOperatorIds(row, dtypeFilter) {
  return pipelineOperatorGroups(row, dtypeFilter).flatMap((group) => group.operatorIds);
}

export function pipelineOperatorGroups(row, dtypeFilter) {
  const list = resolvePipelineForDataType(row, dtypeFilter);
  if (!Array.isArray(list)) {
    return [];
  }
  return list
    .map((raw) => {
      if (typeof raw === "string") {
        const operatorId = raw.trim();
        return operatorId
          ? { operatorIds: [operatorId], anyOf: false }
          : null;
      }
      if (raw && typeof raw === "object" && Array.isArray(raw.any_of)) {
        const operatorIds = raw.any_of
          .map((item) => String(item || "").trim())
          .filter(Boolean);
        return operatorIds.length
          ? { operatorIds, anyOf: true }
          : null;
      }
      return null;
    })
    .filter(Boolean);
}

/**
 * Step numbers (1-based) for operators in one family, preserving pipeline order.
 * @param {string[]} pipelineIds
 * @param {Set<string>} familyIds
 * @returns {Record<string, number|string>}
 */
export function stepsForOperatorFamily(pipelineOrGroups, familyIds) {
  const steps = {};
  let stepNum = 0;
  const groups = Array.isArray(pipelineOrGroups)
    ? pipelineOrGroups.map((item) =>
        typeof item === "string"
          ? { operatorIds: [item], anyOf: false }
          : item,
      )
    : [];
  for (const group of groups) {
    const operatorIds = Array.isArray(group?.operatorIds)
      ? group.operatorIds.filter((operatorId) => familyIds.has(operatorId))
      : [];
    if (!operatorIds.length) {
      continue;
    }
    stepNum += 1;
    for (const operatorId of operatorIds) {
      if (!(operatorId in steps)) {
        steps[operatorId] = stepNum;
      } else {
        steps[operatorId] = `${steps[operatorId]},${stepNum}`;
      }
    }
  }
  return steps;
}

/** @param {Set<string>} familyIds */
export function methodHasOperatorsInFamily(row, familyIds, dtypeFilter) {
  return pipelineOperatorIds(row, dtypeFilter).some((id) => familyIds.has(id));
}

/**
 * Catalog operators that appear in at least one method pipeline (stable catalog order).
 * @param {Array<{ id: string }>} catalogOperators
 * @param {Array<object>} methods
 * @param {Set<string>} familyIds
 */
export function operatorsUsedByMethods(
  catalogOperators,
  methods,
  familyIds,
  dtypeFilter,
) {
  const used = new Set();
  for (const row of methods) {
    for (const operatorId of pipelineOperatorIds(row, dtypeFilter)) {
      if (familyIds.has(operatorId)) {
        used.add(operatorId);
      }
    }
  }
  return (catalogOperators || []).filter((op) => used.has(op.id));
}
