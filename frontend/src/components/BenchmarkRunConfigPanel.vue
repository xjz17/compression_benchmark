<template>
  <div class="benchmark-run-config">
    <template v-if="includeDatasetDimension">
      <el-row class="option-row option-single">
        <el-col :span="8">
          <span class="option-label">Data Dimension</span>
        </el-col>
        <el-col :span="16">
          <el-radio-group v-model="data_dimension">
            <el-radio value="single_column" size="large">Single Column</el-radio>
            <el-radio value="single_dataset" size="large">Single Dataset</el-radio>
            <el-radio value="single_field" size="large">Single Category</el-radio>
            <el-radio value="all_datasets" size="large">All Datasets</el-radio>
          </el-radio-group>
        </el-col>
      </el-row>

      <el-row v-if="data_dimension !== 'single_field'" class="option-row option-single">
        <el-col :span="8">
          <span class="option-label">Select a Dataset</span>
        </el-col>
        <el-col :span="12">
          <el-select
            v-model="dataset"
            class="home-dataset-select"
            placeholder="Select a dataset"
            :disabled="data_dimension === 'all_datasets'"
            :loading="datasetsLoading"
            filterable
            size="large"
            popper-class="dataset-bundle-select-dropdown"
          >
            <el-option
              v-for="item in datasets"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-col>
      </el-row>

      <el-row v-if="data_dimension === 'single_column'" class="option-row option-single">
        <el-col :span="8">
          <span class="option-label">Select a Column</span>
        </el-col>
        <el-col :span="12">
          <el-select
            v-model="selected_column"
            class="home-dataset-select"
            placeholder="Select a column"
            :disabled="data_dimension !== 'single_column'"
            :loading="columnOptionsLoading"
            filterable
            size="large"
            popper-class="dataset-bundle-select-dropdown"
          >
            <el-option
              v-for="item in columnOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-col>
      </el-row>

      <el-row v-if="data_dimension === 'single_field'" class="option-row option-single">
        <el-col :span="8">
          <span class="option-label">Select a Category</span>
        </el-col>
        <el-col :span="12">
          <el-select
            v-model="selected_category"
            class="home-dataset-select"
            placeholder="Select a dataset category"
            :disabled="data_dimension !== 'single_field'"
            :loading="datasetsLoading"
            filterable
            size="large"
            popper-class="dataset-bundle-select-dropdown"
          >
            <el-option
              v-for="item in categoryOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-col>
      </el-row>
    </template>

    <el-row class="option-row option-single">
      <el-col :span="8">
        <span class="option-label">Implement Language</span>
      </el-col>
      <el-col :span="16">
        <el-radio-group v-model="implement_language">
          <el-radio value="3" size="large">C++</el-radio>
          <el-radio value="1" size="large">Java</el-radio>
          <el-radio value="2" size="large">Python</el-radio>
        </el-radio-group>
      </el-col>
    </el-row>

    <el-row class="option-row">
      <el-col :span="8">
        <span class="option-label">Data Type</span>
      </el-col>
      <el-col :span="16">
        <el-radio-group v-model="data_type">
          <el-radio value="4" size="large" :disabled="!canSelectOverallDataType">Overall</el-radio>
          <el-radio value="1" size="large" :disabled="!canSelectIntDataType">Int</el-radio>
          <el-radio value="2" size="large" :disabled="!canSelectFloatDataType">Float</el-radio>
          <!-- Byte/string inputs are hidden while the benchmark focuses on int and float data. -->
          <!--
          <el-radio value="3" size="large" :disabled="!canSelectStringDataType">Byte</el-radio>
          -->
        </el-radio-group>
      </el-col>
    </el-row>

    <el-row
      v-if="showFloatPrecisionSelector"
      class="option-row option-single"
    >
      <el-col :span="8">
        <span class="option-label">Float column precision</span>
      </el-col>
      <el-col :span="16">
        <el-radio-group v-model="float_precision">
          <el-radio value="all" size="large">All</el-radio>
          <el-radio value="fixed" size="large">Fixed precision</el-radio>
          <el-radio value="non_fixed" size="large">Non-fixed precision</el-radio>
        </el-radio-group>
      </el-col>
    </el-row>

    <el-row class="option-row">
      <el-col :span="8">
        <span class="option-label">Available Algorithms</span>
      </el-col>
      <el-col :span="16">
        <div class="algorithms-display algorithms-by-mode">
          <div class="algo-line">
            <span class="algo-mode-label success">Lossless</span>
            <span>{{ availableAlgorithmsLossless }}</span>
          </div>
          <div v-if="!isStringDataType" class="algo-line">
            <span class="algo-mode-label warning">Lossy</span>
            <span>{{ availableAlgorithmsLossy }}</span>
          </div>
        </div>
      </el-col>
    </el-row>

    <el-collapse v-model="parameterConfigExpanded" class="parameter-config-collapse">
      <el-collapse-item name="parameters">
        <template #title>
          <div class="parameter-config-header">
            <span class="parameter-config-title">Parameter configuration of several compressions</span>
            <!-- <span class="parameter-config-hint">
              XZ, Zstd, Sim-Piece, SZ, SZ3, Machete, Serf
            </span> -->
          </div>
        </template>

        <div class="parameter-config-body">
          <el-row class="option-row">
            <el-col :span="8">
              <span class="option-label">XZ Level</span>
            </el-col>
            <el-col :span="12">
              <el-slider class="slider" v-model="xz_level" :min="0" :max="9" :step="1" show-input />
            </el-col>
          </el-row>

          <el-row class="option-row">
            <el-col :span="8">
              <span class="option-label">Zstd Level</span>
            </el-col>
            <el-col :span="12">
              <el-slider class="slider" v-model="zstd_level" :min="1" :max="9" :step="1" show-input />
            </el-col>
          </el-row>

          <el-row class="option-row">
            <el-col :span="8">
              <span class="option-label">Chimp history</span>
            </el-col>
            <el-col :span="12">
              <el-radio-group v-model="chimp_history">
                <el-radio-button
                  v-for="value in chimpHistoryValues"
                  :key="value"
                  :value="value"
                >
                  {{ value }}
                </el-radio-button>
              </el-radio-group>
            </el-col>
          </el-row>

          <el-row
            v-for="entry in lzbenchLevelControls"
            :key="entry.codecId"
            class="option-row"
          >
            <el-col :span="8">
              <span class="option-label">{{ entry.displayName }} {{ entry.controlLabel || "level" }}</span>
            </el-col>
            <el-col :span="12">
              <el-slider
                class="slider research-slider-marks"
                v-model="lzbench_slider_idx[entry.codecId]"
                :min="0"
                :max="lzbenchSliderMax(entry)"
                :step="1"
                show-stops
                :marks="lzbenchSliderMarks(entry)"
                :format-tooltip="(val) => lzbenchSliderTooltip(entry, val)"
              />
              <div class="slider-param-hint">Selected: {{ lzbenchSliderHint(entry) }}</div>
            </el-col>
          </el-row>

          <el-row v-if="showSpdpLevelSlider" class="option-row">
            <el-col :span="8">
              <span class="option-label">SPDP level</span>
            </el-col>
            <el-col :span="12">
              <el-slider
                class="slider research-eps-slider research-slider-marks"
                v-model="spdp_level"
                :min="0"
                :max="spdpLevelMax"
                :step="1"
                show-stops
                :marks="spdpSliderMarks"
                :format-tooltip="spdpLevelTooltip"
              />
              <div class="slider-param-hint">Prediction table level: {{ spdpLevelCurrentLabel }}</div>
            </el-col>
          </el-row>

          <template v-if="showResearchSliders">
            <el-row class="option-row">
              <el-col :span="8">
                <span class="option-label">Sim-Piece relRange</span>
              </el-col>
              <el-col :span="12">
                <el-slider
                  class="slider research-slider-marks"
                  v-model="simpiece_param"
                  :min="0"
                  :max="simpieceParamMax"
                  :step="1"
                  show-stops
                  :marks="simpieceSliderMarks"
                  :format-tooltip="simpieceTooltip"
                />
                <div class="slider-param-hint">{{ simpieceCurrentLabel }}</div>
              </el-col>
            </el-row>
            <el-row class="option-row">
              <el-col :span="8">
                <span class="option-label">SZ</span>
              </el-col>
              <el-col :span="12">
                <el-radio-group v-model="sz_absrel" class="research-absrel-radio">
                  <el-radio value="ABS">ABS</el-radio>
                  <el-radio value="REL">REL</el-radio>
                </el-radio-group>
                <el-slider
                  class="slider research-eps-slider research-slider-marks"
                  v-model="sz_eps"
                  :min="0"
                  :max="szEpsMax"
                  :step="1"
                  show-stops
                  :marks="szSliderMarks"
                  :format-tooltip="szEpsTooltip"
                />
                <div class="slider-param-hint">Error bound: {{ szEpsCurrentLabel }}</div>
              </el-col>
            </el-row>
            <el-row class="option-row">
              <el-col :span="8">
                <span class="option-label">SZ3</span>
              </el-col>
              <el-col :span="12">
                <el-radio-group v-model="sz3_absrel" class="research-absrel-radio">
                  <el-radio value="ABS">ABS</el-radio>
                  <el-radio value="REL">REL</el-radio>
                </el-radio-group>
                <el-slider
                  class="slider research-eps-slider research-slider-marks"
                  v-model="sz3_eps"
                  :min="0"
                  :max="sz3EpsMax"
                  :step="1"
                  show-stops
                  :marks="sz3SliderMarks"
                  :format-tooltip="sz3EpsTooltip"
                />
                <div class="slider-param-hint">Error bound: {{ sz3EpsCurrentLabel }}</div>
              </el-col>
            </el-row>
            <el-row class="option-row">
              <el-col :span="8">
                <span class="option-label">Machete</span>
              </el-col>
              <el-col :span="12">
                <el-slider
                  class="slider research-eps-slider research-slider-marks"
                  v-model="machete_eps"
                  :min="0"
                  :max="macheteEpsMax"
                  :step="1"
                  show-stops
                  :marks="macheteSliderMarks"
                  :format-tooltip="macheteEpsTooltip"
                />
                <div class="slider-param-hint">Relative error: {{ macheteEpsCurrentLabel }}</div>
              </el-col>
            </el-row>
            <el-row v-if="showZfpTolSlider" class="option-row">
              <el-col :span="8">
                <span class="option-label">ZFP tolerance</span>
              </el-col>
              <el-col :span="12">
                <el-slider
                  class="slider research-eps-slider research-slider-marks"
                  v-model="zfp_tol_eps"
                  :min="0"
                  :max="zfpTolEpsMax"
                  :step="1"
                  show-stops
                  :marks="zfpTolSliderMarks"
                  :format-tooltip="zfpTolEpsTooltip"
                />
                <div class="slider-param-hint">
                  Fixed accuracy (lossy): {{ zfpTolEpsCurrentLabel }} · lossless row: ext_zfp_zfpy
                </div>
              </el-col>
            </el-row>
            <!-- Serf ABS (hidden until Serf returns to catalog)
            <el-row class="option-row">
              <el-col :span="8">
                <span class="option-label">Serf ABS</span>
              </el-col>
              <el-col :span="12">
                <el-slider
                  class="slider research-slider-marks"
                  v-model="serf_param"
                  :min="0"
                  :max="serfParamMax"
                  :step="1"
                  show-stops
                  :marks="serfSliderMarks"
                  :format-tooltip="serfTooltip"
                />
                <div class="slider-param-hint">{{ serfCurrentLabel }}</div>
              </el-col>
            </el-row>
            -->
          </template>
        </div>
      </el-collapse-item>
    </el-collapse>
  </div>
</template>

<script setup>
import { inject, ref } from "vue";
import { BENCHMARK_CONFIG_KEY } from "@/composables/useBenchmarkRunConfig.js";

const bench = inject(BENCHMARK_CONFIG_KEY);
if (!bench) {
  throw new Error("BenchmarkRunConfigPanel requires BENCHMARK_CONFIG_KEY provider");
}

const {
  includeDatasetDimension,
  data_dimension,
  dataset,
  datasets,
  datasetsLoading,
  selected_category,
  categoryOptions,
  selected_column,
  columnOptions,
  columnOptionsLoading,
  implement_language,
  data_type,
  float_precision,
  showFloatPrecisionSelector,
  canSelectIntDataType,
  canSelectFloatDataType,
  canSelectStringDataType,
  canSelectOverallDataType,
  isStringDataType,
  availableAlgorithmsLossless,
  availableAlgorithmsLossy,
  xz_level,
  zstd_level,
  chimp_history,
  chimpHistoryValues,
  showResearchSliders,
  showSpdpLevelSlider,
  showZfpTolSlider,
  simpiece_param,
  simpieceParamMax,
  simpieceSliderMarks,
  simpieceTooltip,
  simpieceCurrentLabel,
  sz_absrel,
  sz_eps,
  szEpsMax,
  szSliderMarks,
  szEpsTooltip,
  szEpsCurrentLabel,
  sz3_absrel,
  sz3_eps,
  sz3EpsMax,
  sz3SliderMarks,
  sz3EpsTooltip,
  sz3EpsCurrentLabel,
  machete_eps,
  macheteEpsMax,
  macheteSliderMarks,
  macheteEpsTooltip,
  macheteEpsCurrentLabel,
  zfp_tol_eps,
  zfpTolEpsMax,
  zfpTolSliderMarks,
  zfpTolEpsTooltip,
  zfpTolEpsCurrentLabel,
  spdp_level,
  spdpLevelMax,
  spdpSliderMarks,
  spdpLevelTooltip,
  spdpLevelCurrentLabel,
  lzbenchLevelControls,
  lzbench_slider_idx,
  lzbenchSliderMax,
  lzbenchSliderMarks,
  lzbenchSliderHint,
  lzbenchSliderTooltip,
  // serf_param,
  // serfParamMax,
  // serfSliderMarks,
  // serfTooltip,
  // serfCurrentLabel,
} = bench;

/** Collapsed by default; set to `['parameters']` to expand. */
const parameterConfigExpanded = ref([]);
</script>

<style scoped>
.parameter-config-collapse {
  margin: 1.25rem 0 0.5rem;
  border: none;
}

.parameter-config-collapse :deep(.el-collapse-item__header) {
  height: auto;
  min-height: 48px;
  line-height: 1.4;
  padding: 0.65rem 1rem;
  background: var(--bc-control-bg, rgba(12, 16, 28, 0.75));
  border: 1px solid var(--bc-border, rgba(120, 170, 255, 0.14));
  border-radius: var(--bc-radius-md, 12px);
  color: var(--bc-text, #e8ecf4);
  font-weight: 600;
  transition:
    background-color var(--bc-transition, 0.2s ease),
    border-color var(--bc-transition, 0.2s ease);
}

.parameter-config-collapse :deep(.el-collapse-item__header:hover) {
  background: var(--bc-accent-soft, rgba(77, 163, 255, 0.08));
  border-color: rgba(120, 170, 255, 0.28);
}

.parameter-config-collapse :deep(.el-collapse-item__header.is-active) {
  border-bottom-left-radius: 0;
  border-bottom-right-radius: 0;
  border-color: var(--bc-border-strong, rgba(120, 170, 255, 0.22));
}

.parameter-config-collapse :deep(.el-collapse-item__arrow) {
  color: var(--bc-cyan, #5ae9ff);
  font-size: 1rem;
}

.parameter-config-collapse :deep(.el-collapse-item__wrap) {
  border: none;
  background: transparent;
}

.parameter-config-collapse :deep(.el-collapse-item__content) {
  padding: 0;
}

.parameter-config-collapse :deep(.el-collapse-item) {
  border: none;
}

.parameter-config-header {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.5rem 1rem;
  padding-right: 0.5rem;
  text-align: left;
}

.parameter-config-title {
  font-size: 0.98rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--bc-text, #e8ecf4);
}

.parameter-config-hint {
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--bc-text-muted, #9aa3b2);
}

.parameter-config-body {
  padding: 0.35rem 1rem 0.85rem;
  background: var(--bc-soft-panel-bg, rgba(8, 10, 18, 0.55));
  border: 1px solid var(--bc-border, rgba(120, 170, 255, 0.14));
  border-top: none;
  border-radius: 0 0 var(--bc-radius-md, 12px) var(--bc-radius-md, 12px);
}

.parameter-config-body .option-row:first-child {
  margin-top: 0.65rem;
}

.parameter-config-body .option-row:last-child {
  margin-bottom: 0.35rem;
}
.algorithms-by-mode .algo-line {
  margin-bottom: 8px;
  line-height: 1.5;
  color: var(--bc-text, #e0e0e0);
}

.algorithms-by-mode .algo-mode-label {
  display: inline-block;
  min-width: 3em;
  margin-right: 8px;
  font-weight: 600;
  font-size: 14px;
}

.algorithms-by-mode .algo-mode-label.success {
  color: #67c23a;
}

.algorithms-by-mode .algo-mode-label.warning {
  color: #e6a23c;
}

.data-type-hint {
  margin: 0.5rem 0 0;
  font-size: 13px;
  line-height: 1.45;
  color: var(--bc-text-subtle, #a8abb2);
}

.slider-param-hint {
  margin-top: 0.35rem;
  font-size: 13px;
  color: var(--bc-text-subtle, #a8abb2);
}

.research-absrel-radio {
  margin-bottom: 0.5rem;
}
</style>
