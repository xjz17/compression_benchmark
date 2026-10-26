<template>
  <AppLayout
    title="Dataset Catalog"
    subtitle="Classification, provenance mapping, and downloadable data assets."
  >
    <div class="dataset-page">
      <section class="surface-panel">
        <header class="panel-head">
          <h2 class="panel-title">Overview</h2>
          <p class="panel-desc">
            High-level totals across datasets currently registered in this benchmark.
          </p>
        </header>
        <div v-loading="summaryLoading" class="dataset-overview-body">
        <div class="panel-inner stat-grid">
          <div class="stat-card">
            <div class="stat-label">Registered datasets</div>
            <div class="stat-value">{{ summary.dataset_count }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Total data points</div>
            <div class="stat-value">{{ formatCount(summary.total_rows) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Total CSV size</div>
            <div class="stat-value">{{ formatBytes(summary.total_size_bytes) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Total compressed size</div>
            <div class="stat-value">{{ formatBytes(totalCompressedSizeBytes) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Overall compression rate</div>
            <div class="stat-value">
              {{ formatCompressionRate(summary.overall_compression_rate) }}
            </div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Average compression rate</div>
            <div class="stat-value">
              {{ formatCompressionRate(summary.average_compression_rate) }}
            </div>
          </div>
        </div>
        <div class="panel-inner category-distribution-wrap">
          <h3 class="category-dist-title">Dataset distribution overview</h3>
          <div v-if="categoryCharts.some((c) => c.rows.length)" class="category-dist-content">
            <div class="category-dist-grid">
              <div
                v-for="chart in categoryCharts"
                :key="chart.key"
                class="category-pie-block"
              >
                <h4 class="category-pie-title">{{ chart.title }}</h4>
                <div
                  class="category-pie-card"
                  :class="{ 'category-bar-card': isCompressionRateChartKey(chart.key) }"
                >
                  <CatalogCompressionRateBarChart
                    v-if="isCompressionRateChartKey(chart.key)"
                    :rows="chart.rows"
                    :title="chart.title"
                    :chart-key="chart.key"
                    clickable
                    :legend-highlight="activeLegendCategory || ''"
                    :color-for-name="categoryColorByName"
                    :format-value="(row) => formatCategoryChartValue(chart.key, row)"
                    @slice-click="applyCategoryFilterFromPie"
                  />
                  <CatalogDistributionPieChart
                    v-else
                    :rows="chart.rows"
                    :title="chart.title"
                    :chart-key="chart.key"
                    clickable
                    :legend-highlight="activeLegendCategory || ''"
                    :color-for-name="categoryColorByName"
                    :format-value="(row) => formatCategoryChartValue(chart.key, row)"
                    @slice-click="applyCategoryFilterFromPie"
                  />
                </div>
              </div>
            </div>
            <div class="category-legend">
              <div
                v-for="name in legendCategories"
                :key="name"
                class="category-legend-item"
                :class="{
                  'is-active':
                    highlightedLegendCategory === name || filterCategory === name,
                }"
                @mouseenter="activeLegendCategory = name"
                @mouseleave="activeLegendCategory = null"
              >
                <span class="category-dot" :style="{ backgroundColor: categoryColorByName(name) }" />
                <span class="category-name">{{ name }}</span>
              </div>
            </div>
          </div>
          <p v-else class="category-empty">No category distribution available.</p>
          <h3 class="category-dist-title category-dist-title-spaced">By data type</h3>
          <div v-if="dtypeCharts.some((c) => c.rows.length)" class="category-dist-content">
            <div class="category-dist-grid">
              <div
                v-for="chart in dtypeCharts"
                :key="chart.key"
                class="category-pie-block"
              >
                <h4 class="category-pie-title">{{ chart.title }}</h4>
                <div
                  class="category-pie-card"
                  :class="{ 'category-bar-card': isCompressionRateChartKey(chart.key) }"
                >
                  <CatalogCompressionRateBarChart
                    v-if="isCompressionRateChartKey(chart.key)"
                    :rows="chart.rows"
                    :title="chart.title"
                    :chart-key="chart.key"
                    clickable
                    :legend-highlight="activeDtypeLegend || ''"
                    :color-for-name="dtypeColorByName"
                    :format-value="(row) => formatDtypeChartValue(chart.key, row)"
                    @slice-click="applyDtypeFilterFromPie"
                  />
                  <CatalogDistributionPieChart
                    v-else
                    :rows="chart.rows"
                    :title="chart.title"
                    :chart-key="chart.key"
                    clickable
                    :legend-highlight="activeDtypeLegend || ''"
                    :color-for-name="dtypeColorByName"
                    :format-value="(row) => formatDtypeChartValue(chart.key, row)"
                    @slice-click="applyDtypeFilterFromPie"
                  />
                </div>
              </div>
            </div>
            <div class="category-legend">
              <div
                v-for="name in dtypeLegendCategories"
                :key="name"
                class="category-legend-item"
                :class="{
                  'is-active':
                    highlightedDtypeLegend === name || filterBenchmarkDtype === name,
                }"
                @mouseenter="activeDtypeLegend = name"
                @mouseleave="activeDtypeLegend = null"
              >
                <span class="category-dot" :style="{ backgroundColor: dtypeColorByName(name) }" />
                <span class="category-name">{{ displayBenchmarkDataType(name) }}</span>
              </div>
            </div>
          </div>
          <p v-else class="category-empty">No data type distribution available.</p>
          <h3 class="category-dist-title category-dist-title-spaced">By float precision</h3>
          <div v-if="floatPrecisionCharts.some((c) => c.rows.length)" class="category-dist-content">
            <div class="category-dist-grid">
              <div
                v-for="chart in floatPrecisionCharts"
                :key="chart.key"
                class="category-pie-block"
              >
                <h4 class="category-pie-title">{{ chart.title }}</h4>
                <div
                  class="category-pie-card"
                  :class="{ 'category-bar-card': isCompressionRateChartKey(chart.key) }"
                >
                  <CatalogCompressionRateBarChart
                    v-if="isCompressionRateChartKey(chart.key)"
                    :rows="chart.rows"
                    :title="chart.title"
                    :chart-key="chart.key"
                    :legend-highlight="activeFloatPrecisionLegend || ''"
                    :color-for-name="floatPrecisionColorByName"
                    :format-value="(row) => formatFloatPrecisionChartValue(chart.key, row)"
                  />
                  <CatalogDistributionPieChart
                    v-else
                    :rows="chart.rows"
                    :title="chart.title"
                    :chart-key="chart.key"
                    :legend-highlight="activeFloatPrecisionLegend || ''"
                    :color-for-name="floatPrecisionColorByName"
                    :format-value="(row) => formatFloatPrecisionChartValue(chart.key, row)"
                  />
                </div>
              </div>
            </div>
            <div class="category-legend">
              <div
                v-for="name in floatPrecisionLegendCategories"
                :key="name"
                class="category-legend-item"
                :class="{ 'is-active': highlightedFloatPrecisionLegend === name }"
                @mouseenter="activeFloatPrecisionLegend = name"
                @mouseleave="activeFloatPrecisionLegend = null"
              >
                <span
                  class="category-dot"
                  :style="{ backgroundColor: floatPrecisionColorByName(name) }"
                />
                <span class="category-name">{{ name }}</span>
              </div>
            </div>
          </div>
          <p v-else class="category-empty">No float precision distribution available.</p>
        </div>
        </div>
        <div class="panel-inner">
          <div class="dataset-catalog-filters dataset-catalog-filters--below-pies">
            <h3 class="category-dist-title category-dist-title-spaced">Catalog filters</h3>
            <p class="dataset-catalog-filters-desc">
              Filter the dataset tables and detail explorer below. Charts above always show all
              datasets.
            </p>
            <el-row class="option-row option-single">
              <el-col :span="8">
                <span class="option-label">Category</span>
              </el-col>
              <el-col :span="16">
                <el-select
                  v-model="filterCategory"
                  class="dataset-filter-select"
                  placeholder="All categories"
                  filterable
                  size="large"
                >
                  <el-option label="All categories" value="all" />
                  <el-option
                    v-for="name in catalogCategoryOptions"
                    :key="name"
                    :label="name"
                    :value="name"
                  />
                </el-select>
              </el-col>
            </el-row>
            <el-row class="option-row option-single">
              <el-col :span="8">
                <span class="option-label">Data type</span>
              </el-col>
              <el-col :span="16">
                <el-radio-group v-model="filterBenchmarkDtype">
                  <el-radio value="all" size="large">All</el-radio>
                  <el-radio value="int" size="large">Int</el-radio>
                  <el-radio value="float" size="large">Float</el-radio>
                  <el-radio value="timestamp" size="large">Timestamp</el-radio>
                  <!-- Byte/string inputs are hidden while the catalog focuses on int, float, and timestamp data. -->
                  <!--
                  <el-radio value="string" size="large">Byte</el-radio>
                  -->
                </el-radio-group>
              </el-col>
            </el-row>
            <el-row
              v-if="showCatalogFloatPrecisionSelector"
              class="option-row option-single"
            >
              <el-col :span="8">
                <span class="option-label">Float column precision</span>
              </el-col>
              <el-col :span="16">
                <el-radio-group v-model="filterFloatPrecision">
                  <el-radio :value="FLOAT_PRECISION_FILTER_ALL" size="large">All</el-radio>
                  <el-radio :value="FLOAT_PRECISION_FILTER_FIXED" size="large">
                    Fixed precision
                  </el-radio>
                  <el-radio :value="FLOAT_PRECISION_FILTER_NON_FIXED" size="large">
                    Non-fixed precision
                  </el-radio>
                </el-radio-group>
              </el-col>
            </el-row>
          </div>
        </div>
      </section>

      <section ref="classificationSectionRef" class="surface-panel">
        <header class="panel-head">
          <h2 class="panel-title">Dataset Classification</h2>
          <p class="panel-desc">
            Category, data types, and source.
            <template v-if="catalogFiltersActive">
              Showing {{ datasets.length }} of {{ allCatalogItems.length }} datasets.
            </template>
          </p>
        </header>
        <div class="panel-inner dataset-classification-table-wrap">
          <el-table
            ref="classificationTableRef"
            :data="datasets"
            class="dataset-classification-table"
            style="width: 100%"
            :loading="loading"
            :max-height="DATASET_CLASSIFICATION_TABLE_MAX_HEIGHT_PX"
            :row-key="(row) => row.key"
            :row-class-name="datasetClassificationRowClassName"
            :empty-text="classificationTableEmptyText"
          >
            <el-table-column label="Dataset" min-width="140" fixed="left">
              <template #default="scope">
                {{ scope.row.display_name || scope.row.key
                }}<template v-if="scope.row.is_user_upload">
                  ({{ scope.row.is_public ? "public" : "private" }})
                </template>
              </template>
            </el-table-column>
            <el-table-column prop="category" label="Category" min-width="130" />
            <el-table-column label="Data types" min-width="140">
              <template #default="scope">
                <div
                  v-if="visibleLogicalTypes(scope.row).length"
                  class="logical-type-tags"
                >
                  <el-tag
                    v-for="t in visibleLogicalTypes(scope.row)"
                    :key="`${scope.row.key}-${t}`"
                    size="small"
                    class="logical-type-tag"
                    type="info"
                    effect="plain"
                  >
                    {{ displayBenchmarkDataType(t) }}
                  </el-tag>
                </div>
                <span v-else class="cell-empty">—</span>
              </template>
            </el-table-column>
            <el-table-column label="Source Declaration" min-width="200">
              <template #default="scope">
                <a
                  v-if="scope.row.source_url"
                  :href="scope.row.source_url"
                  target="_blank"
                  rel="noopener noreferrer"
                  class="source-link"
                >
                  {{ scope.row.source }}
                </a>
                <span v-else-if="scope.row.source">{{ scope.row.source }}</span>
                <span v-else class="cell-empty">—</span>
              </template>
            </el-table-column>
            <el-table-column label="Data points" width="140">
              <template #default="scope">{{ formatCount(scope.row.row_count) }}</template>
            </el-table-column>
            <el-table-column label="CSV size" width="120">
              <template #default="scope">
                {{ formatBytes(scope.row.size_bytes ?? 0) }}
              </template>
            </el-table-column>
            <el-table-column
              v-if="SHOW_CLASSIFICATION_DOWNLOADS_COLUMN"
              label="Downloads"
              min-width="220"
            >
              <template #default="scope">
                <div class="download-actions">
                  <el-button
                    type="primary"
                    size="small"
                    :disabled="!scope.row.has_csv"
                    @click="downloadCsv(scope.row)"
                  >
                    CSV
                  </el-button>
                  <el-button
                    type="success"
                    size="small"
                    :disabled="!scope.row.can_export_tsfile"
                    :loading="tsfileLoadingKey === scope.row.key"
                    @click="downloadTsFile(scope.row)"
                  >
                    TsFile
                  </el-button>
                </div>
              </template>
            </el-table-column>
            <el-table-column label="Papers Used" min-width="620" class-name="papers-used-col">
              <template #default="scope">
                <div
                  v-if="papersForDatasetKey(scope.row.key).length"
                  class="paper-list-wrap"
                  role="region"
                  :aria-label="`Papers for ${scope.row.key}`"
                >
                  <ul class="paper-list">
                    <li
                      v-for="paper in papersForDatasetKey(scope.row.key)"
                      :key="`${scope.row.key}_${paper.title}`"
                    >
                      <template v-if="paper.url">
                        <a
                          :href="paper.url"
                          target="_blank"
                          rel="noopener noreferrer"
                          class="source-link paper-link"
                        >
                          {{ paper.title }}
                        </a>
                      </template>
                      <template v-else>
                        {{ paper.title }}
                      </template>
                      <span v-if="paper.venue || paper.year" class="paper-meta">
                        (
                        <template v-if="paper.venue">{{ paper.venue }}</template><template v-if="paper.venue && paper.year">, </template><template v-if="paper.year">{{ paper.year }}</template>
                        )
                      </span>
                    </li>
                  </ul>
                </div>
                <span v-else class="paper-empty">No paper mapping declared yet.</span>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </section>

      <section class="surface-panel">
        <header class="panel-head">
          <h2 class="panel-title">Bundled statistics and charts</h2>
          <p class="panel-desc">
            Choose a dataset that has a pre-built catalog bundle.
          </p>
        </header>
        <div class="panel-inner detail-explorer">
          <div class="detail-picker">
            <template v-if="bundleDatasetOptions.length">
              <div class="detail-picker-controls">
                <div class="detail-picker-inner">
                  <div class="detail-picker-label-row">
                    <el-icon class="detail-picker-icon" :size="18">
                      <FolderOpened />
                    </el-icon>
                    <label class="detail-picker-label" for="dataset-bundle-select">
                      Dataset
                    </label>
                  </div>
                  <el-select
                    id="dataset-bundle-select"
                    v-model="selectedBundleKey"
                    placeholder="Search or pick a dataset…"
                    filterable
                    clearable
                    size="large"
                    class="detail-select"
                    popper-class="dataset-bundle-select-dropdown"
                  >
                    <el-option
                      v-for="row in bundleDatasetOptions"
                      :key="row.key"
                      :label="bundleOptionLabel(row)"
                      :value="row.key"
                    />
                  </el-select>
                </div>
                <div
                  v-if="selectedBundleKey && !isOverallBundleSelected"
                  class="detail-picker-inner detail-operator-picker"
                >
                  <div class="detail-picker-label-row">
                    <label class="detail-picker-label" for="dataset-operator-select">
                      Operator (29 in 8 classes)
                    </label>
                  </div>
                  <el-select
                    id="dataset-operator-select"
                    v-model="selectedOperator"
                    size="large"
                    class="detail-select detail-operator-select"
                  >
                    <el-option
                      :key="detailOperatorOptions[0].value || 'original'"
                      :label="detailOperatorOptions[0].label"
                      :value="detailOperatorOptions[0].value"
                    />
                    <el-option-group
                      v-for="group in detailOperatorGroups"
                      :key="group.id"
                      :label="group.label"
                    >
                      <el-option
                        v-for="opt in group.options"
                        :key="opt.value"
                        :label="opt.label"
                        :value="opt.value"
                        :disabled="opt.disabled"
                      />
                    </el-option-group>
                  </el-select>
                </div>
              </div>
              <div v-if="selectedBundleKey && !isOverallBundleSelected" class="detail-jump-links">
                <button type="button" class="detail-jump-link" @click="jumpToClassificationRow">
                  Go to this dataset in Dataset Classification
                </button>
                <button type="button" class="detail-jump-link" @click="jumpToHomeWithDataset">
                  Go to Home and show algorithm results for this dataset
                </button>
              </div>
            </template>
            <p v-else class="detail-picker-empty">
              No datasets with statistics bundles are available yet.
            </p>
          </div>

          <template v-if="selectedBundle && selectedBundle.asset_urls">
            <!-- Dataset feature matrix temporarily disabled -->
            <!--
            <div v-if="!isOverallBundleSelected" class="statistics-section detail-statistics-block">
              <h3 class="statistics-title">Dataset feature matrix</h3>
              <p
                v-if="detailFeatureMatrixScopeHint.show"
                class="detail-statistics-scope-hint"
              >
                <template v-if="detailFeatureMatrixScopeHint.mode === 'float_fixed'">
                  Columns: <strong>fixed-precision floating-point</strong> only (catalog filter).
                </template>
                <template v-else-if="detailFeatureMatrixScopeHint.mode === 'float_non_fixed'">
                  Columns: <strong>non-fixed-precision floating-point</strong> only (catalog filter).
                </template>
                <template v-else-if="detailFeatureMatrixScopeHint.mode === 'float_all'">
                  Columns: <strong>floating-point</strong>, fixed and non-fixed precision (catalog filter).
                </template>
                <template v-else-if="detailFeatureMatrixScopeHint.mode === 'int'">
                  Catalog filter: <strong>integer</strong> datasets.
                </template>
                <template v-else>
                  Columns: <strong>all data types</strong>.
                </template>
              </p>
              <p
                v-if="detailStatisticsColumnNames.length"
                class="detail-statistics-column-hint"
              >
                Showing {{ detailStatisticsVisibleColumnNames.length }} of
                {{ detailStatisticsColumnNames.length }} columns. Scroll horizontally to load
                more.
              </p>
              <el-table
                ref="detailStatisticsTableRef"
                v-loading="detailLoading"
                :data="detailTransposedStatistics"
                class="detail-statistics-table"
                style="width: 100%"
              >
                <template #empty>
                  <span class="detail-stat-empty">No data</span>
                </template>
                <el-table-column prop="metric" label="Metric" width="180" fixed />
                <el-table-column
                  v-for="(column, colIdx) in detailStatisticsVisibleColumnNames"
                  :key="column"
                  :prop="column"
                  :label="column"
                  min-width="168"
                  show-overflow-tooltip
                >
                  <template #header>
                    <button
                      type="button"
                      class="statistics-column-link"
                      @click="jumpToHomeWithColumn(column)"
                    >
                      {{ column }}
                    </button>
                    <span
                      v-if="
                        colIdx === detailStatisticsVisibleColumnNames.length - 1 &&
                        detailStatisticsHasMoreColumns
                      "
                      :ref="setStatisticsColumnSentinel"
                      class="statistics-column-sentinel"
                      aria-hidden="true"
                    />
                  </template>
                  <template #default="scope">
                    <template v-if="isDistributionMetric(scope.row.metricKey)">
                      <DistributionMiniBarChart
                        v-if="Array.isArray(scope.row[column]) && scope.row[column].length"
                        :distribution="scope.row[column]"
                        :x-axis-name="distributionXAxisName(scope.row.metricKey)"
                        :aria-label="`${scope.row.metric} for ${column}`"
                      />
                      <span v-else class="cell-empty">-</span>
                    </template>
                    <template v-else>
                      {{ formatStatisticValue(scope.row[column]) }}
                    </template>
                  </template>
                </el-table-column>
              </el-table>
            </div>
            -->

            <div
              v-if="selectedBundleKey"
              class="detail-operator-entropy-boxplots"
            >
              <h3 class="statistics-title">{{ selectedDetailScopeTitle }} entropy change by operator</h3>
              <p class="detail-correlation-desc">
                Before-to-after entropy medians and per-column entropy change distributions across encoding operators for
                {{ selectedBundleLabel }}<template v-if="correlationColumnCount">
                  ({{ formatCount(correlationColumnCount) }} benchmarked columns).
                </template>
              </p>
              <div
                v-loading="operatorStepEntropyLoading"
                class="operator-step-entropy-panel"
              >
                <div class="operator-step-entropy-toolbar">
                  <div>
                    <h4 class="detail-operator-boxplot-title">
                      Algorithm pipeline entropy automata
                    </h4>
                  </div>
                  <el-select
                    v-model="selectedEntropyAlgorithm"
                    class="operator-step-algorithm-select"
                    placeholder="Select algorithm"
                    filterable
                    size="large"
                    :disabled="!operatorStepAlgorithmOptions.length"
                    @change="onEntropyAlgorithmChange"
                  >
                    <el-option
                      v-for="opt in operatorStepAlgorithmOptions"
                      :key="opt.algorithm"
                      :label="opt.label"
                      :value="opt.algorithm"
                    />
                  </el-select>
                </div>
                <div
                  v-if="operatorStepAutomatonViews.length"
                  class="operator-step-automata-grid"
                >
                  <div
                    v-for="view in operatorStepAutomatonViews"
                    :key="`${selectedBundleKey}-${selectedEntropyAlgorithm}-${view.data_type}`"
                    class="operator-step-automaton"
                  >
                    <h4 class="operator-step-automaton-title">{{ view.title }}</h4>
                    <div
                      v-if="view.segments.length"
                      class="operator-step-automaton-track"
                      :aria-label="`${view.title} byte entropy automaton`"
                    >
                      <div class="entropy-state-circle">
                        <span class="entropy-state-type">
                          {{ displayEntropyStateType(view.segments[0].from?.data_type) }}
                        </span>
                        <span class="entropy-state-value">
                          {{ formatEntropyStateValue(view.segments[0].from?.byte_entropy) }}
                        </span>
                      </div>
                      <template
                        v-for="segment in view.segments"
                        :key="segment.id"
                      >
                        <div
                          class="entropy-transition"
                          :class="{ 'has-warning': segmentHasUnknownEntropy(segment) }"
                        >
                          <div class="entropy-transition-line" aria-hidden="true" />
                          <div class="entropy-transition-labels">
                            <span
                              v-for="edge in segment.edges"
                              :key="edge.id"
                              class="entropy-transition-label"
                            >
                              {{ edge.operator_label }}
                              <strong>{{ formatEntropyDelta(edge.byte_entropy_change) }}</strong>
                              <em v-if="edge.status !== 'ok'">{{ entropyStatusLabel(edge.status) }}</em>
                            </span>
                          </div>
                        </div>
                        <div class="entropy-state-circle">
                          <span class="entropy-state-type">
                            {{ displayEntropyStateType(segment.to?.data_type) }}
                          </span>
                          <span class="entropy-state-value">
                            {{ formatEntropyStateValue(segment.to?.byte_entropy) }}
                          </span>
                        </div>
                      </template>
                    </div>
                    <p v-else class="detail-correlation-cell-empty">
                      No {{ view.data_type }} operator-step entropy rows for this algorithm.
                    </p>
                  </div>
                </div>
                <p
                  v-else-if="!operatorStepEntropyLoading"
                  class="detail-correlation-cell-empty"
                >
                  {{ selectedBundleLabel }} does not have operator-step entropy automata yet.
                </p>
              </div>
              <div
                v-loading="correlationHeatmapsLoading"
                class="detail-operator-boxplot-grid"
              >
                <div
                  class="detail-operator-boxplot-frame paper-operator-figure-frame"
                >
                  <h4 class="detail-operator-boxplot-title">
                    {{ selectedByteEntropyChart?.title || "Byte entropy reduction by operator" }}
                  </h4>
                  <p class="detail-operator-boxplot-desc">
                    {{ selectedByteEntropyChart?.description || "Mean byte entropy reduction per operator." }}
                  </p>
                  <EntropyBeforeAfterArrowChart
                    v-if="selectedByteEntropyChart"
                    :key="`${selectedBundleKey}-byte-entropy-arrow`"
                    :chart="selectedByteEntropyChart"
                    :aria-label="`${selectedByteEntropyChart.title} chart`"
                  />
                  <p v-else class="detail-correlation-cell-empty">
                    No byte entropy chart data available.
                  </p>
                </div>
                <div
                  class="detail-operator-boxplot-frame paper-operator-figure-frame"
                >
                  <h4 class="detail-operator-boxplot-title">
                    {{ selectedMarkov3EntropyChart?.title || "Markov-3 byte entropy-rate reduction by operator" }}
                  </h4>
                  <p class="detail-operator-boxplot-desc">
                    {{ selectedMarkov3EntropyChart?.description || "Mean Markov-3 byte entropy-rate reduction per operator." }}
                  </p>
                  <EntropyBeforeAfterArrowChart
                    v-if="selectedMarkov3EntropyChart"
                    :key="`${selectedBundleKey}-markov3-entropy-arrow`"
                    :chart="selectedMarkov3EntropyChart"
                    :aria-label="`${selectedMarkov3EntropyChart.title} chart`"
                  />
                  <p v-else class="detail-correlation-cell-empty">
                    No Markov-3 entropy-rate chart data available.
                  </p>
                </div>
              </div>
            </div>
          </template>
          <p v-else-if="selectedBundleKey && !selectedBundle" class="detail-empty-hint">
            Selected dataset is not in the current catalog list.
          </p>
        </div>
      </section>
    </div>
  </AppLayout>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import axios from "@/utils/httpClient.js";
import { CATALOG_API_TIMEOUT_MS } from "@/constants/apiTimeouts.js";
import { ElMessage } from "element-plus";
import { useRouter } from "vue-router";
import AppLayout from "@/components/AppLayout.vue";
import CatalogDistributionPieChart from "@/components/CatalogDistributionPieChart.vue";
import CatalogCompressionRateBarChart from "@/components/CatalogCompressionRateBarChart.vue";
import LeaderboardBoxplotChart from "@/components/LeaderboardBoxplotChart.vue";
import EntropyBeforeAfterArrowChart from "@/components/EntropyBeforeAfterArrowChart.vue";
import DistributionMiniBarChart from "@/components/DistributionMiniBarChart.vue";
import { useAuth } from "@/composables/useAuth.js";
import {
  FLOAT_PRECISION_FILTER_ALL,
  FLOAT_PRECISION_FILTER_FIXED,
  FLOAT_PRECISION_FILTER_NON_FIXED,
} from "@/composables/useBenchmarkRunConfig.js";
import {
  datasetSelectLabelFromKey,
  datasetSelectLabelFromRow,
} from "@/utils/datasetSelectLabel.js";
import { displayBenchmarkDataType } from "@/utils/benchmarkDataTypeDisplay.js";
import {
  readDatasetStatisticsCache,
  writeDatasetStatisticsCache,
} from "@/utils/datasetCatalogCache.js";
import {
  buildDatasetOperatorOptions,
  fetchEncodingOperatorCatalog,
} from "@/utils/encodingOperatorCatalog.js";
import { OPERATOR_CLASSES } from "@/utils/operatorTaxonomy.js";

const { isAuthenticated } = useAuth();
const router = useRouter();

/** Dataset feature matrix rows (numeric stats + distribution chart rows). */
const DATASET_FEATURE_METRIC_KEYS_BASE = [
  "mean",
  "standard_deviation",
  "spread",
  "byte_entropy",
  "repeat",
  "sortedness",
  "bit_width_distribution",
  "leading_zeros_distribution",
  "trailing_zeros_distribution",
];

const DATASET_FEATURE_METRIC_KEYS_OPERATOR = [
  "mean",
  "standard_deviation",
  "spread",
  "byte_entropy",
  "byte_entropy_change",
  "repeat",
  "sortedness",
  "bit_width_distribution",
  "leading_zeros_distribution",
  "trailing_zeros_distribution",
];

const DATASET_FEATURE_METRIC_LABELS = {
  mean: "Mean",
  standard_deviation: "Standard Deviation",
  spread: "Spread",
  byte_entropy: "Byte Entropy",
  byte_entropy_change: "Byte Entropy Change",
  value_entropy: "Value Entropy",
  value_entropy_change: "Value Entropy Change",
  repeat: "Repeat",
  sortedness: "Sortedness",
  bit_width_distribution: "Bit Width Distribution",
  leading_zeros_distribution: "Leading Zeros Distribution",
  trailing_zeros_distribution: "Trailing Zeros Distribution",
};

const DISTRIBUTION_FIELD_BY_METRIC = {
  bit_width_distribution: "scaling_bit_width_distribution",
  leading_zeros_distribution: "leading_zeros_distribution",
  trailing_zeros_distribution: "trailing_zeros_distribution",
};

const DISTRIBUTION_AXIS_LABELS = {
  bit_width_distribution: "Bit width",
  leading_zeros_distribution: "Leading zeros",
  trailing_zeros_distribution: "Trailing zeros",
};

/** Columns rendered per batch in the Dataset feature matrix (horizontal lazy load). */
const STATISTICS_COLUMNS_BATCH = 20;

/** Max height (px) for Dataset Classification table; Element Plus pins header and scrolls the body. */
/** Set false to hide CSV / TsFile buttons in the classification table. */
const SHOW_CLASSIFICATION_DOWNLOADS_COLUMN = false;
const DATASET_CLASSIFICATION_TABLE_MAX_HEIGHT_PX = 560;
const CLASSIFICATION_SECTION_SCROLL_TOP_GAP_PX = 92;

const OVERALL_BUNDLE_KEY = "overall";
const VISIBLE_BENCHMARK_DTYPES = new Set(["int", "float", "timestamp"]);
const SHOW_VALUE_ENTROPY_METRICS = false;

const loading = ref(false);
const summaryLoading = ref(false);
const tsfileLoadingKey = ref("");
const filterCategory = ref("all");
const filterBenchmarkDtype = ref("all");
const filterFloatPrecision = ref(FLOAT_PRECISION_FILTER_ALL);
const catalogCategoryOptions = ref([]);
const allCatalogItems = ref([]);
const allPaperUsage = ref([]);
const selectedBundleKey = ref("");
const selectedOperator = ref("");
const encodingOperatorCatalog = ref([]);
const classificationSectionRef = ref(null);
const classificationTableRef = ref(null);
const classificationHighlightKey = ref("");
const detailLoading = ref(false);
const detailStatRows = ref([]);
const correlationHeatmapsLoading = ref(false);
const correlationPayload = ref(null);
const overallCorrelationPayload = ref(null);
const operatorStepEntropyLoading = ref(false);
const operatorStepEntropyPayload = ref(null);
const selectedEntropyAlgorithm = ref("");
const statisticsVisibleColumnCount = ref(STATISTICS_COLUMNS_BATCH);
const detailStatisticsTableRef = ref(null);
const statisticsColumnSentinelEl = ref(null);
let statisticsColumnObserver = null;
const statisticsScrollListeners = [];
let classificationHighlightTimer = null;
const CLASSIFICATION_ROW_FLASH_MS = 1000;
const EMPTY_CATALOG_SUMMARY = {
  dataset_count: 0,
  total_rows: 0,
  total_size_bytes: 0,
  total_tsfile_size_bytes: 0,
  total_compressed_size_bytes: 0,
  overall_compression_rate: null,
  average_compression_rate: null,
  total_compression_rate: null,
  category_distribution: [],
  category_size_distribution: [],
  category_data_point_distribution: [],
  category_tsfile_size_distribution: [],
  category_overall_compression_rate_distribution: [],
  category_average_compression_rate_distribution: [],
  dtype_column_distribution: [],
  dtype_data_point_distribution: [],
  dtype_csv_size_distribution: [],
  dtype_tsfile_size_distribution: [],
  dtype_overall_compression_rate_distribution: [],
  dtype_average_compression_rate_distribution: [],
  float_precision_column_distribution: [],
  float_precision_data_point_distribution: [],
  float_precision_csv_size_distribution: [],
  float_precision_tsfile_size_distribution: [],
  float_precision_overall_compression_rate_distribution: [],
  float_precision_average_compression_rate_distribution: [],
};

const summary = ref({ ...EMPTY_CATALOG_SUMMARY });

const totalCompressedSizeBytes = computed(
  () =>
    summary.value?.total_compressed_size_bytes ??
    summary.value?.total_tsfile_size_bytes ??
    0,
);

const catalogFiltersActive = computed(
  () =>
    filterCategory.value !== "all" ||
    filterBenchmarkDtype.value !== "all" ||
    (showCatalogFloatPrecisionSelector.value &&
      filterFloatPrecision.value !== FLOAT_PRECISION_FILTER_ALL),
);

const showCatalogFloatPrecisionSelector = computed(
  () => filterBenchmarkDtype.value === "float",
);

const activeStatisticsFloatPrecision = computed(() => {
  if (
    filterBenchmarkDtype.value === "float" &&
    filterFloatPrecision.value !== FLOAT_PRECISION_FILTER_ALL
  ) {
    return filterFloatPrecision.value;
  }
  return FLOAT_PRECISION_FILTER_ALL;
});

const detailFeatureMatrixScopeHint = computed(() => {
  const dtype = filterBenchmarkDtype.value;
  if (dtype === "float") {
    const fp = activeStatisticsFloatPrecision.value;
    if (fp === FLOAT_PRECISION_FILTER_FIXED) {
      return { show: true, mode: "float_fixed" };
    }
    if (fp === FLOAT_PRECISION_FILTER_NON_FIXED) {
      return { show: true, mode: "float_non_fixed" };
    }
    return { show: true, mode: "float_all" };
  }
  if (dtype === "int") {
    return { show: true, mode: "int" };
  }
  return { show: true, mode: "all" };
});

function normalizeBenchmarkDtype(dtype) {
  const normalized = String(dtype || "").trim().toLowerCase();
  if (normalized === "integer" || normalized === "int" || normalized === "long") {
    return "int";
  }
  if (normalized === "float" || normalized === "double") {
    return "float";
  }
  if (
    normalized === "timestamp" ||
    normalized === "datetime" ||
    normalized === "date" ||
    normalized === "time"
  ) {
    return "timestamp";
  }
  return normalized;
}

function isVisibleBenchmarkDtype(dtype) {
  return VISIBLE_BENCHMARK_DTYPES.has(normalizeBenchmarkDtype(dtype));
}

function uniqueVisibleDtypes(values) {
  const out = [];
  const seen = new Set();
  for (const value of Array.isArray(values) ? values : []) {
    const dtype = normalizeBenchmarkDtype(value);
    if (!dtype || !isVisibleBenchmarkDtype(dtype) || seen.has(dtype)) {
      continue;
    }
    seen.add(dtype);
    out.push(dtype);
  }
  return out;
}

function rowCatalogDtypes(row) {
  const catalogDtypes = uniqueVisibleDtypes(row?.catalog_dtypes || []);
  if (catalogDtypes.length) {
    return catalogDtypes;
  }
  const benchmarkDtypes = uniqueVisibleDtypes(row?.benchmark_dtypes || []);
  if (benchmarkDtypes.length) {
    return benchmarkDtypes;
  }
  return uniqueVisibleDtypes(row?.logical_types || []);
}

function visibleLogicalTypes(row) {
  return rowCatalogDtypes(row);
}

function rowHasVisibleBenchmarkDtype(row) {
  return rowCatalogDtypes(row).length > 0;
}

function catalogRowMatchesCategory(row) {
  if (filterCategory.value === "all") {
    return true;
  }
  const rowCategory = String(row?.category || "").trim();
  return rowCategory === filterCategory.value;
}

function catalogRowMatchesBenchmarkDtype(row) {
  const dtype = filterBenchmarkDtype.value;
  if (!dtype || dtype === "all") {
    return rowHasVisibleBenchmarkDtype(row);
  }
  if (!isVisibleBenchmarkDtype(dtype)) {
    return false;
  }
  const target = normalizeBenchmarkDtype(dtype);
  return rowCatalogDtypes(row).includes(target);
}

function catalogRowMatchesFloatPrecision(row) {
  if (filterBenchmarkDtype.value !== "float") {
    return true;
  }
  const fp = filterFloatPrecision.value;
  if (!fp || fp === FLOAT_PRECISION_FILTER_ALL) {
    return true;
  }
  const counts = row?.float_precision_column_counts;
  if (!counts || typeof counts !== "object") {
    return true;
  }
  return Number(counts[fp] ?? 0) > 0;
}

function catalogRowMatchesFilters(row) {
  return (
    catalogRowMatchesCategory(row) &&
    catalogRowMatchesBenchmarkDtype(row) &&
    catalogRowMatchesFloatPrecision(row)
  );
}

const datasets = computed(() => (allCatalogItems.value || []).filter(catalogRowMatchesFilters));

const papersByDatasetKey = computed(() => {
  const map = new Map();
  for (const entry of allPaperUsage.value || []) {
    map.set(String(entry.dataset), Array.isArray(entry.papers) ? entry.papers : []);
  }
  return map;
});

function papersForDatasetKey(key) {
  return papersByDatasetKey.value.get(String(key || "")) || [];
}

const classificationTableEmptyText = computed(() => {
  if (loading.value) {
    return "Loading datasets…";
  }
  if (catalogFiltersActive.value) {
    return "No datasets match the current filters. Try All for category or data type.";
  }
  return "No datasets registered.";
});

const CATEGORY_COLORS = [
  "#4da3ff",
  "#6ee7ff",
  "#7adf8f",
  "#f7c75b",
  "#ff9f7a",
  "#c792ea",
  "#73c0de",
  "#91cc75",
  "#fac858",
  "#ee6666",
];

/** Others used to share the 7th slot (#73c0de), too close to the blue/cyan slices. */
const CATEGORY_COLOR_OTHERS = "#b8a4cf";

const DTYPE_SLICE_ORDER = ["int", "float", "timestamp"];

const DTYPE_SLICE_COLORS = {
  int: "#4da3ff",
  float: "#7adf8f",
  timestamp: "#f7c75b",
};

const FLOAT_PRECISION_SLICE_ORDER = ["Fixed precision", "Non-fixed precision"];

const FLOAT_PRECISION_SLICE_COLORS = {
  "Fixed precision": "#c792ea",
  "Non-fixed precision": "#ff9f7a",
};

const categoryDistribution = computed(() => {
  const arr = summary.value?.category_distribution;
  return Array.isArray(arr) ? arr : [];
});

const categorySizeDistribution = computed(() => {
  const arr = summary.value?.category_size_distribution;
  return Array.isArray(arr) ? arr : [];
});

const categoryDataPointDistribution = computed(() => {
  const arr = summary.value?.category_data_point_distribution;
  return Array.isArray(arr) ? arr : [];
});

const categoryTsfileSizeDistribution = computed(() => {
  const arr = summary.value?.category_tsfile_size_distribution;
  return Array.isArray(arr) ? arr : [];
});

const categoryOverallCompressionRateDistribution = computed(() => {
  const arr = summary.value?.category_overall_compression_rate_distribution;
  return Array.isArray(arr) ? arr : [];
});

const categoryAverageCompressionRateDistribution = computed(() => {
  const arr = summary.value?.category_average_compression_rate_distribution;
  return Array.isArray(arr) ? arr : [];
});

const categoryCharts = computed(() => [
  {
    key: "count",
    title: "Dataset count",
    rows: categoryDistribution.value,
  },
  {
    key: "data_points",
    title: "Data points",
    rows: categoryDataPointDistribution.value,
  },
  {
    key: "size",
    title: "CSV size",
    rows: categorySizeDistribution.value,
  },
  {
    key: "tsfile_size",
    title: "Compressed size",
    rows: categoryTsfileSizeDistribution.value,
  },
  {
    key: "overall_compression_rate",
    title: "Overall compression rate",
    rows: categoryOverallCompressionRateDistribution.value,
  },
  {
    key: "average_compression_rate",
    title: "Average compression rate",
    rows: categoryAverageCompressionRateDistribution.value,
  },
]);

function visibleDtypeDistributionRows(rows) {
  return (Array.isArray(rows) ? rows : []).filter((row) =>
    isVisibleBenchmarkDtype(row?.category),
  );
}

const dtypeColumnDistribution = computed(() => {
  const arr = summary.value?.dtype_column_distribution;
  return visibleDtypeDistributionRows(arr);
});

const dtypeDataPointDistribution = computed(() => {
  const arr = summary.value?.dtype_data_point_distribution;
  return visibleDtypeDistributionRows(arr);
});

const dtypeCsvSizeDistribution = computed(() => {
  const arr = summary.value?.dtype_csv_size_distribution;
  return visibleDtypeDistributionRows(arr);
});

const dtypeTsfileSizeDistribution = computed(() => {
  const arr = summary.value?.dtype_tsfile_size_distribution;
  return visibleDtypeDistributionRows(arr);
});

const dtypeOverallCompressionRateDistribution = computed(() => {
  const arr = summary.value?.dtype_overall_compression_rate_distribution;
  return visibleDtypeDistributionRows(arr);
});

const dtypeAverageCompressionRateDistribution = computed(() => {
  const arr = summary.value?.dtype_average_compression_rate_distribution;
  return visibleDtypeDistributionRows(arr);
});

const dtypeCharts = computed(() => [
  {
    key: "columns",
    title: "Columns",
    rows: dtypeColumnDistribution.value,
  },
  {
    key: "data_points",
    title: "Data points",
    rows: dtypeDataPointDistribution.value,
  },
  {
    key: "csv_size",
    title: "CSV size",
    rows: dtypeCsvSizeDistribution.value,
  },
  {
    key: "tsfile_size",
    title: "Compressed size",
    rows: dtypeTsfileSizeDistribution.value,
  },
  {
    key: "overall_compression_rate",
    title: "Overall compression rate",
    rows: dtypeOverallCompressionRateDistribution.value,
  },
  {
    key: "average_compression_rate",
    title: "Average compression rate",
    rows: dtypeAverageCompressionRateDistribution.value,
  },
]);

const floatPrecisionColumnDistribution = computed(() => {
  const arr = summary.value?.float_precision_column_distribution;
  return Array.isArray(arr) ? arr : [];
});

const floatPrecisionDataPointDistribution = computed(() => {
  const arr = summary.value?.float_precision_data_point_distribution;
  return Array.isArray(arr) ? arr : [];
});

const floatPrecisionCsvSizeDistribution = computed(() => {
  const arr = summary.value?.float_precision_csv_size_distribution;
  return Array.isArray(arr) ? arr : [];
});

const floatPrecisionTsfileSizeDistribution = computed(() => {
  const arr = summary.value?.float_precision_tsfile_size_distribution;
  return Array.isArray(arr) ? arr : [];
});

const floatPrecisionOverallCompressionRateDistribution = computed(() => {
  const arr = summary.value?.float_precision_overall_compression_rate_distribution;
  return Array.isArray(arr) ? arr : [];
});

const floatPrecisionAverageCompressionRateDistribution = computed(() => {
  const arr = summary.value?.float_precision_average_compression_rate_distribution;
  return Array.isArray(arr) ? arr : [];
});

const floatPrecisionCharts = computed(() => [
  {
    key: "columns",
    title: "Columns",
    rows: floatPrecisionColumnDistribution.value,
  },
  {
    key: "data_points",
    title: "Data points",
    rows: floatPrecisionDataPointDistribution.value,
  },
  {
    key: "csv_size",
    title: "CSV size",
    rows: floatPrecisionCsvSizeDistribution.value,
  },
  {
    key: "tsfile_size",
    title: "Compressed size",
    rows: floatPrecisionTsfileSizeDistribution.value,
  },
  {
    key: "overall_compression_rate",
    title: "Overall compression rate",
    rows: floatPrecisionOverallCompressionRateDistribution.value,
  },
  {
    key: "average_compression_rate",
    title: "Average compression rate",
    rows: floatPrecisionAverageCompressionRateDistribution.value,
  },
]);

const legendCategories = computed(() => {
  const seen = new Set();
  const names = [];
  for (const chart of categoryCharts.value) {
    for (const row of chart.rows || []) {
      const name = String(row?.category || "").trim();
      if (!name || seen.has(name)) continue;
      seen.add(name);
      names.push(name);
    }
  }
  return names;
});

const activeLegendCategory = ref(null);
const highlightedLegendCategory = computed(() => activeLegendCategory.value);

const activeDtypeLegend = ref(null);
const highlightedDtypeLegend = computed(() => activeDtypeLegend.value);

const activeFloatPrecisionLegend = ref(null);
const highlightedFloatPrecisionLegend = computed(
  () => activeFloatPrecisionLegend.value,
);

function applyCategoryFilterFromPie(categoryName) {
  const name = String(categoryName || "").trim();
  if (!name || !catalogCategoryOptions.value.includes(name)) {
    return;
  }
  filterCategory.value = name;
  nextTick(() => {
    scrollSectionWithTopGap(classificationSectionRef, CLASSIFICATION_SECTION_SCROLL_TOP_GAP_PX);
  });
}

function applyDtypeFilterFromPie(dtypeName) {
  const dtype = String(dtypeName || "")
    .trim()
    .toLowerCase();
  if (!isVisibleBenchmarkDtype(dtype)) {
    return;
  }
  filterBenchmarkDtype.value = normalizeBenchmarkDtype(dtype);
  nextTick(() => {
    scrollSectionWithTopGap(classificationSectionRef, CLASSIFICATION_SECTION_SCROLL_TOP_GAP_PX);
  });
}

const dtypeLegendCategories = computed(() => {
  const seen = new Set();
  for (const chart of dtypeCharts.value) {
    for (const row of chart.rows || []) {
      const name = String(row?.category || "").trim();
      if (name) seen.add(name);
    }
  }
  return DTYPE_SLICE_ORDER.filter((name) => seen.has(name));
});

const floatPrecisionLegendCategories = computed(() => {
  const seen = new Set();
  for (const chart of floatPrecisionCharts.value) {
    for (const row of chart.rows || []) {
      const name = String(row?.category || "").trim();
      if (name) seen.add(name);
    }
  }
  return FLOAT_PRECISION_SLICE_ORDER.filter((name) => seen.has(name));
});

const bundleDatasets = computed(() =>
  (datasets.value || []).filter((row) => row.has_asset_bundle && row.asset_urls),
);

const overallBundleOption = computed(() => ({
  key: OVERALL_BUNDLE_KEY,
  display_name: "Overall",
  source: "All bundled datasets",
  has_asset_bundle: true,
  asset_urls: {
    correlation_heatmaps_json: `/api/datasets/catalog-assets/${OVERALL_BUNDLE_KEY}/correlation-heatmaps/`,
    operator_step_entropy_json: `/api/datasets/catalog-assets/${OVERALL_BUNDLE_KEY}/operator-step-entropy/`,
  },
}));

const bundleDatasetOptions = computed(() => [
  overallBundleOption.value,
  ...bundleDatasets.value,
]);

const isOverallBundleSelected = computed(
  () => String(selectedBundleKey.value || "").trim().toLowerCase() === OVERALL_BUNDLE_KEY,
);

const selectedBundle = computed(() =>
  isOverallBundleSelected.value
    ? overallBundleOption.value
    : (datasets.value || []).find((row) => row.key === selectedBundleKey.value),
);

const selectedBundleLabel = computed(() =>
  isOverallBundleSelected.value
    ? "Overall"
    : datasetSelectLabelFromRow(selectedBundle.value)
      || datasetSelectLabelFromKey(selectedBundleKey.value),
);

const selectedDetailScopeTitle = computed(() =>
  isOverallBundleSelected.value ? "Overall" : "Selected dataset",
);

const detailOperatorOptions = computed(() =>
  buildDatasetOperatorOptions(encodingOperatorCatalog.value),
);

const detailOperatorGroups = computed(() =>
  OPERATOR_CLASSES.map((group) => ({
    ...group,
    options: detailOperatorOptions.value.filter((option) => option.classId === group.id),
  })).filter((group) => group.options.length),
);

function bundleOptionLabel(row) {
  return String(row?.key || "").toLowerCase() === OVERALL_BUNDLE_KEY
    ? "Overall"
    : datasetSelectLabelFromRow(row);
}

function statisticsRowBenchmarkDtype(row) {
  return normalizeBenchmarkDtype(
    row?.benchmark_data_type ||
      row?.benchmark_dtype ||
      row?.logical_type ||
      row?.dtype ||
      "",
  );
}

function statisticRowMatchesActiveDtype(row) {
  const rowDtype = statisticsRowBenchmarkDtype(row);
  if (rowDtype && !isVisibleBenchmarkDtype(rowDtype)) {
    return false;
  }
  const activeDtype = normalizeBenchmarkDtype(filterBenchmarkDtype.value);
  if (activeDtype === "all" || !activeDtype) {
    return true;
  }
  if (!rowDtype) {
    return true;
  }
  return rowDtype === activeDtype;
}

const detailVisibleStatRows = computed(() =>
  (detailStatRows.value || []).filter(statisticRowMatchesActiveDtype),
);

const detailStatisticsColumnNames = computed(() => {
  const rows = detailVisibleStatRows.value || [];
  return rows
    .map((r) => r.column_name)
    .filter((x) => x != null && String(x).trim() !== "");
});

const detailStatisticsVisibleColumnNames = computed(() =>
  detailStatisticsColumnNames.value.slice(0, statisticsVisibleColumnCount.value),
);

const detailStatisticsHasMoreColumns = computed(
  () =>
    statisticsVisibleColumnCount.value < detailStatisticsColumnNames.value.length,
);

const correlationColumnCount = computed(
  () => Number(correlationPayload.value?.column_count || 0),
);

function entropyBoxplotChartFor(key, entropyLabel, payload = correlationPayload.value) {
  const block = payload?.operator_entropy_change_boxplots;
  if (!block || typeof block !== "object") {
    return null;
  }
  const chart = block[key];
  const hasDistributionValues = Array.isArray(chart?.values) && chart.values.length;
  const hasMeanChanges = Array.isArray(chart?.mean_changes) && chart.mean_changes.length;
  const hasMeanReductions = Array.isArray(chart?.mean_reductions) && chart.mean_reductions.length;
  if (!chart?.algorithms?.length || (!hasDistributionValues && !hasMeanChanges && !hasMeanReductions)) {
    return null;
  }
  const datasetLabel = selectedBundleLabel.value || "selected dataset";
  const isScalarBarChart = String(chart?.chart_type || "").toLowerCase() === "bar";
  return {
    key,
    ...chart,
    title: isScalarBarChart
      ? (chart.title || `${datasetLabel} ${entropyLabel} change by operator`)
      : `${datasetLabel} ${entropyLabel} change by operator`,
    description: isScalarBarChart
      ? (
          chart.description ||
          `Mean per-row ${entropyLabel} change (after minus before) for each operator.`
        )
      : (
          `Distribution of per-column ${entropyLabel} change (transformed minus raw) ` +
          "for each encoding operator in the selected dataset."
        ),
  };
}

function entropyChartHasOperatorData(chart) {
  if (String(chart?.chart_type || "").toLowerCase() === "bar") {
    const changes = Array.isArray(chart?.mean_changes)
      ? chart.mean_changes
      : (Array.isArray(chart?.mean_reductions) ? chart.mean_reductions : []);
    return Boolean(
      chart &&
      Array.isArray(chart.algorithms) &&
      chart.algorithms.length &&
      changes.some((value) => Number.isFinite(Number(value))),
    );
  }
  const values = Array.isArray(chart?.values) ? chart.values : [];
  return Boolean(
    chart &&
    Array.isArray(chart.algorithms) &&
    chart.algorithms.length &&
    values.some((series) =>
      Array.isArray(series) && series.some((value) => Number.isFinite(Number(value))),
    )
  );
}

function entropySegmentBoxplotChartsFor(key, entropyLabel) {
  const base = entropyBoxplotChartFor(key, entropyLabel);
  if (!base) {
    return [];
  }
  const segments = Array.isArray(base.segments) && base.segments.length
    ? base.segments
    : [base];
  const datasetLabel = selectedBundleLabel.value || "selected dataset";
  return segments.map((segment, index) => {
    const segmentLabel = segment.segment_label || "All columns";
    const segmentKey = segment.segment_key || `all-${index}`;
    return {
      ...segment,
      key: `${key}-${segmentKey}`,
      title: `${datasetLabel} ${entropyLabel} change by operator (${segmentLabel})`,
      description:
        `Distribution of per-column ${entropyLabel} change (transformed minus raw) ` +
        `for ${segmentLabel.toLowerCase()} in the selected dataset.`,
      hasData: entropyChartHasOperatorData(segment),
    };
  });
}

function entropyArrowChartFor(key, entropyLabel) {
  const chart = entropyBoxplotChartFor(key, entropyLabel);
  if (
    !chart ||
    !Array.isArray(chart.before_medians) ||
    !Array.isArray(chart.after_medians)
  ) {
    return null;
  }
  const datasetLabel = selectedBundleLabel.value || "selected dataset";
  return {
    ...chart,
    title: `${datasetLabel} ${entropyLabel} before → after by operator`,
    description: isOverallBundleSelected.value
      ? `Mean per-row ${entropyLabel} reduction for each operator.`
      : `Median raw ${entropyLabel} to median transformed ${entropyLabel} for each operator.`,
    delta_label: isOverallBundleSelected.value ? "Mean Δ" : "Δ median",
    entropy_label: `${entropyLabel} (bits)`,
  };
}

function entropySegmentArrowChartsFor(key, entropyLabel) {
  return entropySegmentBoxplotChartsFor(key, entropyLabel).map((chart) => ({
    ...chart,
    title: `${selectedBundleLabel.value || "selected dataset"} ${entropyLabel} before → after by operator (${chart.segment_label || "All columns"})`,
    description: `Median raw ${entropyLabel} to median transformed ${entropyLabel} for each operator.`,
    delta_label: "Δ median",
    entropy_label: `${entropyLabel} (bits)`,
    hasData:
      chart.hasData &&
      Array.isArray(chart.before_medians) &&
      Array.isArray(chart.after_medians),
  }));
}

function entropyChangeBarChartFor(key, entropyLabel, payload = correlationPayload.value) {
  const chart = entropyBoxplotChartFor(key, entropyLabel, payload);
  if (
    !chart ||
    String(chart.chart_type || "").toLowerCase() !== "bar" ||
    (!Array.isArray(chart.mean_changes) && !Array.isArray(chart.mean_reductions))
  ) {
    return null;
  }
  return {
    ...chart,
    description:
      chart.description ||
      `Mean ${entropyLabel} reduction per operator.`,
    y_label: chart.y_label || `Mean ${entropyLabel} reduction (bits/byte)`,
    hasData: entropyChartHasOperatorData(chart),
  };
}

const valueEntropyArrowChart = computed(() =>
  entropyArrowChartFor(
    "value_entropy_change",
    "value entropy",
  ),
);

const byteEntropyArrowCharts = computed(() =>
  isOverallBundleSelected.value
    ? [
        entropyChangeBarChartFor(
          "byte_entropy_change",
          "byte entropy",
        ),
      ].filter(Boolean).map((chart) => ({
        ...chart,
        compact_height: true,
        hasData: entropyChartHasOperatorData(chart),
      }))
    : entropySegmentArrowChartsFor(
        "byte_entropy_change",
        "byte entropy",
      ),
);

const overallByteEntropyChart = computed(() =>
  (() => {
    const chart = entropyChangeBarChartFor(
      "byte_entropy_change",
      "byte entropy",
      overallCorrelationPayload.value,
    );
    return chart ? { ...chart, compact_height: true } : null;
  })(),
);

const overallMarkov3EntropyChart = computed(() => {
  const chart = overallCorrelationPayload.value?.operator_step_markov3_entropy_rate_chart;
  if (
    !chart ||
    !Array.isArray(chart.algorithms) ||
    !Array.isArray(chart.mean_changes)
  ) {
    return null;
  }
  return {
    ...chart,
    compact_height: true,
  };
});

const datasetByteEntropyChart = computed(() => {
  const chart = entropyChangeBarChartFor(
    "byte_entropy_change",
    "byte entropy",
    correlationPayload.value,
  );
  return chart ? { ...chart, compact_height: true } : null;
});

const datasetMarkov3EntropyChart = computed(() => {
  const chart = correlationPayload.value?.operator_step_markov3_entropy_rate_chart;
  if (
    !chart ||
    !Array.isArray(chart.algorithms) ||
    !Array.isArray(chart.mean_changes)
  ) {
    return null;
  }
  return {
    ...chart,
    compact_height: true,
  };
});

const selectedByteEntropyChart = computed(() =>
  isOverallBundleSelected.value
    ? overallByteEntropyChart.value
    : datasetByteEntropyChart.value,
);

const selectedMarkov3EntropyChart = computed(() =>
  isOverallBundleSelected.value
    ? overallMarkov3EntropyChart.value
    : datasetMarkov3EntropyChart.value,
);

const valueBasedEntropyBoxplotChart = computed(() =>
  entropyBoxplotChartFor(
    "value_entropy_change",
    "value entropy",
  ),
);

const byteBasedEntropyBoxplotCharts = computed(() =>
  entropySegmentBoxplotChartsFor(
    "byte_entropy_change",
    "byte entropy",
  ),
);

const operatorStepAlgorithmOptions = computed(() =>
  Array.isArray(operatorStepEntropyPayload.value?.algorithms)
    ? operatorStepEntropyPayload.value.algorithms
    : [],
);

function automatonSegmentsFor(raw) {
  const states = Array.isArray(raw?.states) ? raw.states : [];
  const edges = Array.isArray(raw?.edges) ? raw.edges : [];
  if (!states.length || !edges.length) {
    return [];
  }
  const stateById = new Map(states.map((state) => [state.id, state]));
  const byStep = new Map();
  for (const edge of edges) {
    const step = Number(edge?.step_index || 0);
    if (!step) continue;
    if (!byStep.has(step)) {
      byStep.set(step, []);
    }
    byStep.get(step).push(edge);
  }
  return [...byStep.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([step, stepEdges]) => {
      const first = stepEdges[0] || {};
      return {
        id: `${raw.data_type}-${step}`,
        step,
        from: stateById.get(first.from),
        to: stateById.get(first.to),
        edges: stepEdges,
      };
    })
    .filter((segment) => segment.from && segment.to);
}

const operatorStepAutomatonViews = computed(() => {
  const automata = operatorStepEntropyPayload.value?.automata || {};
  return ["int", "float"]
    .map((dtype) => {
      const raw = automata?.[dtype];
      if (!raw) return null;
      return {
        ...raw,
        title: `${displayBenchmarkDataType(dtype)} operator pipeline`,
        segments: automatonSegmentsFor(raw),
      };
    })
    .filter(Boolean);
});

function displayEntropyStateType(type) {
  const normalized = String(type || "").trim().toLowerCase();
  if (normalized === "mixed") return "mixed";
  if (normalized === "unknown") return "unknown";
  if (normalized === "byte") return "byte";
  return displayBenchmarkDataType(normalized || "int");
}

function formatEntropyStateValue(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return "H: n/a";
  }
  return `H: ${num.toFixed(2)}`;
}

function formatEntropyDelta(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return "Δ n/a";
  }
  const sign = num > 0 ? "+" : "";
  return `Δ ${sign}${num.toFixed(2)}`;
}

function entropyStatusLabel(status) {
  const value = String(status || "").trim();
  if (!value || value === "ok") {
    return "";
  }
  if (value === "unsupported_transform") {
    return "unsupported";
  }
  if (value === "blocked_by_unsupported_previous_step") {
    return "blocked";
  }
  return value.replaceAll("_", " ");
}

function segmentHasUnknownEntropy(segment) {
  return (segment?.edges || []).some((edge) => !Number.isFinite(Number(edge?.byte_entropy_change)));
}

const detailFeatureMetricKeys = computed(() =>
  selectedOperator.value
    ? DATASET_FEATURE_METRIC_KEYS_OPERATOR
    : DATASET_FEATURE_METRIC_KEYS_BASE,
);

function statisticFieldValue(row, metricKey) {
  if (!row || typeof row !== "object") {
    return null;
  }
  if (metricKey === "byte_entropy") {
    const direct = row.byte_entropy ?? row.entropy;
    return direct === null || direct === undefined || direct === "" ? null : direct;
  }
  const value = row[metricKey];
  return value === null || value === undefined || value === "" ? null : value;
}

const detailTransposedStatistics = computed(() => {
  const rows = detailVisibleStatRows.value || [];
  const names = detailStatisticsVisibleColumnNames.value;
  if (!rows.length || !names.length) return [];
  const byName = Object.fromEntries(
    rows
      .filter((r) => r.column_name != null && String(r.column_name).trim() !== "")
      .map((r) => [String(r.column_name), r]),
  );
  return detailFeatureMetricKeys.value.map((key) => {
    const row = {
      metricKey: key,
      metric: DATASET_FEATURE_METRIC_LABELS[key] || key,
    };
    const distributionField = DISTRIBUTION_FIELD_BY_METRIC[key];
    for (const cn of names) {
      row[cn] = distributionField
        ? byName[cn]?.[distributionField]
        : statisticFieldValue(byName[cn], key);
    }
    return row;
  });
});

function isDistributionMetric(metricKey) {
  return Object.prototype.hasOwnProperty.call(DISTRIBUTION_FIELD_BY_METRIC, metricKey);
}

function distributionXAxisName(metricKey) {
  return DISTRIBUTION_AXIS_LABELS[metricKey] || "Category";
}

function setStatisticsColumnSentinel(el) {
  statisticsColumnSentinelEl.value = el;
}

function findStatisticsTableScrollRoot(tableEl) {
  if (!tableEl) {
    return null;
  }
  const wraps = tableEl.querySelectorAll(".el-scrollbar__wrap");
  for (const wrap of wraps) {
    if (wrap.scrollWidth > wrap.clientWidth + 2) {
      return wrap;
    }
  }
  const bodyWrapper = tableEl.querySelector(".el-table__body-wrapper");
  if (bodyWrapper && bodyWrapper.scrollWidth > bodyWrapper.clientWidth + 2) {
    return bodyWrapper;
  }
  return wraps[0] || bodyWrapper || null;
}

function disconnectStatisticsColumnLazyLoad() {
  statisticsColumnObserver?.disconnect();
  statisticsColumnObserver = null;
  for (const { el, handler } of statisticsScrollListeners) {
    el.removeEventListener("scroll", handler);
  }
  statisticsScrollListeners.length = 0;
  statisticsColumnSentinelEl.value = null;
}

function loadMoreStatisticsColumnsIfNeeded() {
  if (!detailStatisticsHasMoreColumns.value) {
    return false;
  }
  statisticsVisibleColumnCount.value = Math.min(
    detailStatisticsColumnNames.value.length,
    statisticsVisibleColumnCount.value + STATISTICS_COLUMNS_BATCH,
  );
  return true;
}

function onStatisticsTableScroll(event) {
  const el = event.currentTarget;
  if (!el || !detailStatisticsHasMoreColumns.value) {
    return;
  }
  const nearEnd = el.scrollLeft + el.clientWidth >= el.scrollWidth - 120;
  if (nearEnd && loadMoreStatisticsColumnsIfNeeded()) {
    scheduleStatisticsColumnLazySetup();
  }
}

function bindStatisticsTableScrollListeners(tableEl) {
  const seen = new Set();
  const candidates = [
    ...tableEl.querySelectorAll(".el-scrollbar__wrap"),
    tableEl.querySelector(".el-table__body-wrapper"),
  ].filter(Boolean);
  for (const el of candidates) {
    if (seen.has(el)) {
      continue;
    }
    seen.add(el);
    el.addEventListener("scroll", onStatisticsTableScroll, { passive: true });
    statisticsScrollListeners.push({ el, handler: onStatisticsTableScroll });
  }
}

function setupStatisticsColumnIntersectionObserver(scrollRoot) {
  statisticsColumnObserver?.disconnect();
  statisticsColumnObserver = null;
  const sentinel = statisticsColumnSentinelEl.value;
  if (!sentinel || !detailStatisticsHasMoreColumns.value) {
    return;
  }
  const root =
    scrollRoot && scrollRoot.contains(sentinel) ? scrollRoot : null;
  statisticsColumnObserver = new IntersectionObserver(
    (entries) => {
      if (entries.some((entry) => entry.isIntersecting) && loadMoreStatisticsColumnsIfNeeded()) {
        scheduleStatisticsColumnLazySetup();
      }
    },
    { root, threshold: 0.01 },
  );
  statisticsColumnObserver.observe(sentinel);
}

async function ensureStatisticsColumnsCanScroll() {
  await nextTick();
  const tableEl = detailStatisticsTableRef.value?.$el;
  if (!tableEl) {
    return;
  }
  let guard = 0;
  while (detailStatisticsHasMoreColumns.value && guard < 50) {
    const scrollRoot = findStatisticsTableScrollRoot(tableEl);
    if (!scrollRoot) {
      break;
    }
    const canScrollHorizontally = scrollRoot.scrollWidth > scrollRoot.clientWidth + 2;
    if (canScrollHorizontally) {
      break;
    }
    if (!loadMoreStatisticsColumnsIfNeeded()) {
      break;
    }
    guard += 1;
    await nextTick();
  }
}

async function scheduleStatisticsColumnLazySetup() {
  disconnectStatisticsColumnLazyLoad();
  await nextTick();
  await ensureStatisticsColumnsCanScroll();
  await nextTick();

  const tableEl = detailStatisticsTableRef.value?.$el;
  if (!tableEl) {
    return;
  }

  bindStatisticsTableScrollListeners(tableEl);
  if (!detailStatisticsHasMoreColumns.value) {
    return;
  }
  const scrollRoot = findStatisticsTableScrollRoot(tableEl);
  setupStatisticsColumnIntersectionObserver(scrollRoot);
}

function resetStatisticsColumnWindow() {
  statisticsVisibleColumnCount.value = STATISTICS_COLUMNS_BATCH;
  disconnectStatisticsColumnLazyLoad();
}

/** Same numeric cell rules as Home.vue `formatStatisticValue`. */
function formatStatisticValue(value) {
  if (value === null || value === undefined || value === "") return "-";
  const numValue = parseFloat(value);
  return Number.isNaN(numValue) ? "-" : numValue.toFixed(3);
}

const formatCount = (value) => new Intl.NumberFormat("en-US").format(Number(value || 0));

const formatBytes = (bytes) => {
  const n = Number(bytes || 0);
  if (n >= 1024 * 1024 * 1024) return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(2)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(2)} KB`;
  return `${n} B`;
};

const formatCompressionRate = (value) => {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  const numValue = Number(value);
  return Number.isFinite(numValue) ? numValue.toFixed(3) : "-";
};

function isCompressionRateChartKey(chartKey) {
  return chartKey === "overall_compression_rate" || chartKey === "average_compression_rate";
}

const categoryColorByName = (name) => {
  if (name === "Others") {
    return CATEGORY_COLOR_OTHERS;
  }
  const idx = legendCategories.value.indexOf(name);
  if (idx < 0) {
    return CATEGORY_COLORS[0];
  }
  return CATEGORY_COLORS[idx % CATEGORY_COLORS.length];
};

const dtypeColorByName = (name) => DTYPE_SLICE_COLORS[name] || CATEGORY_COLORS[0];

const floatPrecisionColorByName = (name) =>
  FLOAT_PRECISION_SLICE_COLORS[name] || CATEGORY_COLORS[0];

const datasetClassificationRowClassName = ({ row }) =>
  row?.key && String(row.key) === String(classificationHighlightKey.value)
    ? "dataset-focus-row"
    : "";

function waitMs(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

async function scrollToTableRowByIndex(tableRef, rowIndex) {
  if (!Number.isInteger(rowIndex) || rowIndex < 0) return false;
  const tableEl = tableRef?.value?.$el || tableRef?.value;
  if (!tableEl) return false;
  const bodyWrapper = tableEl.querySelector(".el-table__body-wrapper");
  if (!bodyWrapper) return false;

  const setScrollTop = (top) => {
    if (typeof tableRef?.value?.setScrollTop === "function") {
      tableRef.value.setScrollTop(top);
    } else {
      bodyWrapper.scrollTop = top;
    }
  };

  const estimateTargetTop = () => {
    const rows = Array.from(bodyWrapper.querySelectorAll("tbody tr.el-table__row"));
    const rowHeight = Math.max(1, rows[0]?.getBoundingClientRect().height || 40);
    return Math.max(0, rowIndex * rowHeight - bodyWrapper.clientHeight * 0.35);
  };

  for (let attempt = 0; attempt < 8; attempt += 1) {
    setScrollTop(estimateTargetTop());
    await waitMs(24);
    const rows = Array.from(bodyWrapper.querySelectorAll("tbody tr.el-table__row"));
    const targetRow = rows[rowIndex];
    if (!targetRow) {
      continue;
    }
    const preciseTop = Math.max(0, targetRow.offsetTop - bodyWrapper.clientHeight * 0.35);
    setScrollTop(preciseTop);
    return true;
  }
  return false;
}

function scrollSectionWithTopGap(sectionRef, topGapPx = 92) {
  const el = sectionRef?.value;
  if (!el) return;
  const rectTop = el.getBoundingClientRect().top;
  const delta = rectTop - topGapPx;
  window.scrollBy({ top: delta, behavior: "smooth" });
}

function clearClassificationRowFlash() {
  if (classificationHighlightTimer != null) {
    window.clearTimeout(classificationHighlightTimer);
    classificationHighlightTimer = null;
  }
  classificationHighlightKey.value = "";
}

function flashClassificationRow(key) {
  clearClassificationRowFlash();
  void nextTick(() => {
    classificationHighlightKey.value = String(key || "");
    classificationHighlightTimer = window.setTimeout(() => {
      classificationHighlightKey.value = "";
      classificationHighlightTimer = null;
    }, CLASSIFICATION_ROW_FLASH_MS);
  });
}

async function jumpToClassificationRow() {
  const key = String(selectedBundleKey.value || "").trim();
  if (!key) {
    ElMessage.warning("Select a dataset first.");
    return;
  }
  const rowIndex = datasets.value.findIndex((row) => String(row?.key || "") === key);
  if (rowIndex < 0) {
    ElMessage.warning("Selected dataset is not listed in classification table.");
    return;
  }
  scrollSectionWithTopGap(classificationSectionRef, CLASSIFICATION_SECTION_SCROLL_TOP_GAP_PX);
  await waitMs(220);
  await nextTick();
  if (!(await scrollToTableRowByIndex(classificationTableRef, rowIndex))) {
    ElMessage.warning("Selected dataset row is not visible in classification table.");
    return;
  }
  flashClassificationRow(key);
}

async function jumpToHomeWithDataset() {
  const key = String(selectedBundleKey.value || "").trim();
  if (!key) {
    ElMessage.warning("Select a dataset first.");
    return;
  }
  await router.push({
    name: "Home",
    query: {
      data_dimension: "single_dataset",
      dataset: key,
    },
  });
}

async function jumpToHomeWithColumn(columnName) {
  const key = String(selectedBundleKey.value || "").trim();
  const name = String(columnName || "").trim();
  if (!key) {
    ElMessage.warning("Select a dataset first.");
    return;
  }
  if (!name) {
    return;
  }
  const row = (detailStatRows.value || []).find(
    (r) => r.column_name != null && String(r.column_name) === name,
  );
  const rawIdx = row?.column_index;
  if (rawIdx == null || rawIdx === "") {
    ElMessage.warning("Column is not available for benchmark results.");
    return;
  }
  await router.push({
    name: "Home",
    query: {
      data_dimension: "single_column",
      dataset: key,
      column_index: String(rawIdx),
    },
  });
}

function formatCategoryChartValue(chartKey, row) {
  if (chartKey === "count") {
    return formatCount(row?.count ?? row?.value ?? 0);
  }
  if (chartKey === "data_points") {
    return formatCount(row?.data_points ?? row?.value ?? 0);
  }
  if (chartKey === "size") {
    return formatBytes(row?.value ?? row?.size_bytes ?? 0);
  }
  if (chartKey === "tsfile_size") {
    return formatBytes(row?.value ?? row?.tsfile_size_bytes ?? 0);
  }
  if (chartKey === "compression_rate" || chartKey === "overall_compression_rate" || chartKey === "average_compression_rate") {
    return formatCompressionRate(row?.value ?? row?.compression_rate ?? 0);
  }
  return String(row?.value ?? "");
}

function formatDtypeChartValue(chartKey, row) {
  if (chartKey === "columns") {
    return formatCount(row?.count ?? row?.value ?? 0);
  }
  if (chartKey === "data_points") {
    return formatCount(row?.data_points ?? row?.value ?? 0);
  }
  if (chartKey === "csv_size") {
    return formatBytes(row?.value ?? row?.csv_size_bytes ?? 0);
  }
  if (chartKey === "tsfile_size") {
    return formatBytes(row?.value ?? row?.tsfile_size_bytes ?? 0);
  }
  if (chartKey === "compression_rate" || chartKey === "overall_compression_rate" || chartKey === "average_compression_rate") {
    return formatCompressionRate(row?.value ?? row?.compression_rate ?? 0);
  }
  return String(row?.value ?? "");
}

function formatFloatPrecisionChartValue(chartKey, row) {
  return formatDtypeChartValue(chartKey, row);
}

function defaultBundleDatasetKey() {
  return OVERALL_BUNDLE_KEY;
}

function syncSelectedBundleAfterFilter() {
  if (!selectedBundleKey.value) {
    return;
  }
  if (isOverallBundleSelected.value) {
    return;
  }
  if ((datasets.value || []).some((row) => row.key === selectedBundleKey.value)) {
    return;
  }
  selectedBundleKey.value = defaultBundleDatasetKey(datasets.value);
}

function applyCatalogSummary(data) {
  summary.value = {
    ...EMPTY_CATALOG_SUMMARY,
    ...(data?.summary && typeof data.summary === "object" ? data.summary : {}),
  };
  if (Array.isArray(data?.summary?.available_categories)) {
    catalogCategoryOptions.value = data.summary.available_categories;
  }
}

function applyCatalogItems(data) {
  allCatalogItems.value = data?.items || [];
  allPaperUsage.value = (data?.paper_usage || []).filter(
    (entry) => !String(entry.dataset || "").startsWith("user-upload-"),
  );
  if (!catalogCategoryOptions.value.length) {
    const seen = new Set();
    for (const row of allCatalogItems.value) {
      const name = String(row?.category || "").trim();
      if (name) {
        seen.add(name);
      }
    }
    catalogCategoryOptions.value = [...seen].sort();
  }
  const defaultKey = defaultBundleDatasetKey(datasets.value);
  if (defaultKey && !selectedBundleKey.value) {
    selectedBundleKey.value = defaultKey;
  }
  syncSelectedBundleAfterFilter();
}

function buildCatalogRequestParams() {
  const params = {};
  if (isAuthenticated.value) {
    params.include_user_uploads = "true";
  }
  return params;
}

const fetchCatalog = async () => {
  summaryLoading.value = true;
  loading.value = true;

  const baseParams = buildCatalogRequestParams();
  const requestConfig = { timeout: CATALOG_API_TIMEOUT_MS };

  const summaryPromise = axios
    .get("/api/datasets/", { ...requestConfig, params: { ...baseParams, part: "summary" } })
    .then(({ data }) => {
      applyCatalogSummary(data);
    })
    .catch((error) => {
      console.error(error);
      ElMessage.error("Failed to load dataset overview.");
    })
    .finally(() => {
      summaryLoading.value = false;
    });

  const itemsPromise = axios
    .get("/api/datasets/", { ...requestConfig, params: { ...baseParams, part: "items" } })
    .then(({ data }) => {
      applyCatalogItems(data);
    })
    .catch((error) => {
      console.error(error);
      ElMessage.error("Failed to load dataset list.");
    })
    .finally(() => {
      loading.value = false;
    });

  await Promise.allSettled([summaryPromise, itemsPromise]);
};

const loadBundleStatistics = async (key) => {
  detailStatRows.value = [];
  resetStatisticsColumnWindow();
  if (!key || String(key).toLowerCase() === OVERALL_BUNDLE_KEY) {
    return;
  }
  const row = (datasets.value || []).find((r) => r.key === key);
  if (!row?.asset_urls?.statistics_json) {
    return;
  }
  const floatPrecision = activeStatisticsFloatPrecision.value;
  const operator = selectedOperator.value || "";
  const cachedRows = readDatasetStatisticsCache(key, floatPrecision, operator);
  if (cachedRows?.length) {
    detailStatRows.value = cachedRows;
    detailLoading.value = false;
    scheduleStatisticsColumnLazySetup();
  } else {
    detailLoading.value = true;
  }
  try {
    const params = {};
    if (floatPrecision !== FLOAT_PRECISION_FILTER_ALL) {
      params.float_precision = floatPrecision;
    }
    if (operator) {
      params.operator = operator;
    }
    const { data } = await axios.get(row.asset_urls.statistics_json, { params });
    detailStatRows.value = Array.isArray(data?.rows) ? data.rows : [];
    if (detailStatRows.value.length) {
      writeDatasetStatisticsCache(key, detailStatRows.value, floatPrecision, operator);
    }
  } catch (error) {
    console.error(error);
    const status = error?.response?.status;
    if (status === 404 || status === 403) {
      detailStatRows.value = [];
    } else if (!cachedRows?.length) {
      ElMessage.error("Failed to load numeric statistics for this dataset.");
    }
  } finally {
    detailLoading.value = false;
    if ((detailStatRows.value?.length || 0) > 0) {
      scheduleStatisticsColumnLazySetup();
    }
  }
};

const loadCorrelationHeatmaps = async (key) => {
  correlationPayload.value = null;
  if (!key) {
    return;
  }
  const row = String(key).toLowerCase() === OVERALL_BUNDLE_KEY
    ? overallBundleOption.value
    : (datasets.value || []).find((r) => r.key === key);
  const url = correlationHeatmapsUrl(row);
  if (!url) {
    return;
  }
  correlationHeatmapsLoading.value = true;
  try {
    const params = {};
    const floatPrecision = activeStatisticsFloatPrecision.value;
    if (floatPrecision !== FLOAT_PRECISION_FILTER_ALL) {
      params.float_precision = floatPrecision;
    }
    const { data } = await axios.get(url, { params });
    correlationPayload.value = data;
    if (String(key).toLowerCase() === OVERALL_BUNDLE_KEY) {
      overallCorrelationPayload.value = data;
    }
    await nextTick();
  } catch (error) {
    console.error(error);
    correlationPayload.value = null;
  } finally {
    correlationHeatmapsLoading.value = false;
  }
};

const loadOperatorStepEntropy = async (key, algorithm = "") => {
  operatorStepEntropyPayload.value = null;
  if (!key) {
    selectedEntropyAlgorithm.value = "";
    return;
  }
  const row = String(key).toLowerCase() === OVERALL_BUNDLE_KEY
    ? overallBundleOption.value
    : (datasets.value || []).find((r) => r.key === key);
  const url = operatorStepEntropyUrl(row);
  if (!url) {
    selectedEntropyAlgorithm.value = "";
    return;
  }
  operatorStepEntropyLoading.value = true;
  try {
    const params = {};
    if (algorithm) {
      params.algorithm = algorithm;
    }
    const { data } = await axios.get(url, { params });
    operatorStepEntropyPayload.value = data;
    const selected = String(data?.selected_algorithm || "").trim();
    selectedEntropyAlgorithm.value = selected;
  } catch (error) {
    console.error(error);
    operatorStepEntropyPayload.value = null;
    selectedEntropyAlgorithm.value = "";
  } finally {
    operatorStepEntropyLoading.value = false;
  }
};

function onEntropyAlgorithmChange(algorithm) {
  if (!selectedBundleKey.value) {
    return;
  }
  loadOperatorStepEntropy(selectedBundleKey.value, String(algorithm || "").trim());
}

function correlationHeatmapsUrl(row) {
  if (!row?.key) {
    return "";
  }
  const fromAssets = row?.asset_urls?.correlation_heatmaps_json;
  if (fromAssets) {
    return fromAssets;
  }
  if (row.has_asset_bundle || row?.asset_urls?.statistics_json) {
    return `/api/datasets/catalog-assets/${row.key}/correlation-heatmaps/`;
  }
  return "";
}

function operatorStepEntropyUrl(row) {
  if (!row?.key) {
    return "";
  }
  const fromAssets = row?.asset_urls?.operator_step_entropy_json;
  if (fromAssets) {
    return fromAssets;
  }
  if (row.has_asset_bundle || row?.asset_urls?.statistics_json) {
    return `/api/datasets/catalog-assets/${row.key}/operator-step-entropy/`;
  }
  return "";
}

watch(filterBenchmarkDtype, (dtype) => {
  if (dtype !== "all" && !isVisibleBenchmarkDtype(dtype)) {
    filterBenchmarkDtype.value = "all";
    return;
  }
  if (dtype !== "float") {
    filterFloatPrecision.value = FLOAT_PRECISION_FILTER_ALL;
  }
});

watch([filterBenchmarkDtype, filterFloatPrecision, selectedOperator], () => {
  if (!selectedBundleKey.value) {
    return;
  }
  if (isOverallBundleSelected.value) {
    loadCorrelationHeatmaps(selectedBundleKey.value);
    return;
  }
  if (!datasets.value.some((row) => row.key === selectedBundleKey.value)) {
    selectedBundleKey.value = "";
    return;
  }
  loadBundleStatistics(selectedBundleKey.value);
  loadCorrelationHeatmaps(selectedBundleKey.value);
});

watch(selectedBundleKey, (key) => {
  selectedOperator.value = "";
  selectedEntropyAlgorithm.value = "";
  loadBundleStatistics(key);
  loadCorrelationHeatmaps(key);
  loadOperatorStepEntropy(key);
});

onBeforeUnmount(() => {
  disconnectStatisticsColumnLazyLoad();
  clearClassificationRowFlash();
});

function parseDownloadFilename(contentDisposition, fallback) {
  if (!contentDisposition) {
    return fallback;
  }
  const m = contentDisposition.match(/filename="([^"]+)"/);
  return m?.[1] || fallback;
}

async function parseBlobErrorDetail(error, fallback) {
  const data = error?.response?.data;
  if (data instanceof Blob) {
    try {
      const parsed = JSON.parse(await data.text());
      if (parsed?.error) {
        return String(parsed.error);
      }
    } catch {
      /* ignore */
    }
  } else if (error?.response?.data?.error) {
    return String(error.response.data.error);
  }
  return fallback;
}

function triggerBrowserFileDownload(blob, filename) {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

async function downloadDatasetAsset(url, fallbackFilename, errorFallback) {
  const response = await axios.get(url, { responseType: "blob" });
  const filename = parseDownloadFilename(
    response.headers["content-disposition"],
    fallbackFilename,
  );
  triggerBrowserFileDownload(new Blob([response.data]), filename);
}

const downloadCsv = async (row) => {
  if (!row?.csv_download_url) {
    return;
  }
  const base = row.display_name || row.key;
  try {
    await downloadDatasetAsset(row.csv_download_url, `${base}.csv`, "CSV download failed.");
    ElMessage.success("CSV downloaded.");
  } catch (error) {
    console.error(error);
    ElMessage.error(await parseBlobErrorDetail(error, "CSV download failed."));
  }
};

const downloadTsFile = async (row) => {
  if (!row?.tsfile_download_url) {
    return;
  }
  const base = row.display_name || row.key;
  tsfileLoadingKey.value = row.key;
  try {
    await downloadDatasetAsset(
      row.tsfile_download_url,
      `${base}.tsfile`,
      "TsFile download failed.",
    );
    ElMessage.success("TsFile downloaded.");
  } catch (error) {
    console.error(error);
    ElMessage.error(await parseBlobErrorDetail(error, "TsFile download failed."));
  } finally {
    setTimeout(() => {
      if (tsfileLoadingKey.value === row.key) {
        tsfileLoadingKey.value = "";
      }
    }, 400);
  }
};

async function loadEncodingOperatorCatalog() {
  try {
    encodingOperatorCatalog.value = await fetchEncodingOperatorCatalog();
  } catch (error) {
    console.error(error);
    encodingOperatorCatalog.value = [];
  }
}

onMounted(() => {
  loadEncodingOperatorCatalog();
  fetchCatalog();
});

watch(isAuthenticated, () => {
  fetchCatalog();
});

watch(filterCategory, (value) => {
  if (value !== "all" && !catalogCategoryOptions.value.includes(value)) {
    filterCategory.value = "all";
  }
  syncSelectedBundleAfterFilter();
});

watch(filterBenchmarkDtype, () => {
  syncSelectedBundleAfterFilter();
});
</script>

<style scoped>
.dataset-page {
  display: flex;
  flex-direction: column;
  gap: 1.35rem;
}

.dataset-catalog-filters--below-pies {
  margin-top: 1.35rem;
  padding-top: 1.1rem;
  border-top: 1px solid rgba(120, 170, 255, 0.14);
}

.dataset-catalog-filters-desc {
  margin: 0 0 0.85rem;
  font-size: 0.84rem;
  color: var(--bc-text-muted, #9aa3b2);
  line-height: 1.45;
}

.dataset-filter-select {
  width: 100%;
  max-width: 420px;
}

.surface-panel {
  border-radius: var(--bc-radius-lg, 18px);
  border: 1px solid var(--bc-border, rgba(120, 170, 255, 0.14));
  background: var(--bc-panel-bg, rgba(10, 12, 20, 0.45));
  overflow: hidden;
}

.panel-head {
  padding: 1.15rem 1.35rem 0.85rem;
  border-bottom: 1px solid rgba(120, 170, 255, 0.1);
  background: linear-gradient(180deg, rgba(77, 163, 255, 0.06) 0%, transparent 100%);
}

.panel-title {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--bc-text, #e8ecf4);
}

.panel-desc {
  margin: 0.35rem 0 0;
  font-size: 0.85rem;
  line-height: 1.45;
  color: var(--bc-text-muted, #9aa3b2);
}

.panel-inner {
  padding: 0.8rem 1.25rem 1.35rem;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 0.8rem;
}

@media (max-width: 1200px) {
  .stat-grid {
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  }
}

.category-distribution-wrap {
  padding-top: 0;
}

.category-dist-title {
  margin: 0 0 0.8rem;
  font-size: 0.92rem;
  font-weight: 700;
  color: var(--bc-text, #e8ecf4);
}

.category-dist-title-spaced {
  margin-top: 1.35rem;
}

.category-dist-content {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 0.9rem;
}

.category-dist-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 0.55rem;
  width: 100%;
}

.category-pie-block {
  border: 1px solid rgba(120, 170, 255, 0.14);
  border-radius: 10px;
  background: var(--bc-soft-panel-bg, rgba(20, 26, 40, 0.36));
  padding: 0.55rem 0.4rem 0.35rem;
}

.category-pie-title {
  margin: 0 0 0.45rem;
  text-align: center;
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--bc-text, #e8ecf4);
}

.category-pie-card {
  position: relative;
  width: 100%;
  max-width: 200px;
  aspect-ratio: 1;
  height: auto;
  margin: 0 auto;
}

.category-bar-card {
  max-width: 220px;
  aspect-ratio: auto;
  height: 168px;
}

.category-legend {
  width: 100%;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.5rem;
  max-height: 252px;
  overflow-y: auto;
  padding-right: 0.2rem;
}

.category-legend-item {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.5rem;
  align-items: center;
  border: 1px solid rgba(120, 170, 255, 0.14);
  border-radius: 10px;
  background: var(--bc-soft-panel-bg, rgba(20, 26, 40, 0.4));
  padding: 0.42rem 0.56rem;
}

.category-legend-item.is-active {
  border-color: rgba(120, 170, 255, 0.34);
  background: var(--bc-soft-panel-active-bg, rgba(38, 54, 82, 0.35));
}

.category-dot {
  width: 0.7rem;
  height: 0.7rem;
  border-radius: 999px;
  box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.12) inset;
}

.category-name {
  font-size: 0.82rem;
  color: var(--bc-text, #e8ecf4);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.category-empty {
  margin: 0;
  font-size: 0.84rem;
  color: var(--bc-text-muted, #9aa3b2);
}

.stat-card {
  border: 1px solid rgba(120, 170, 255, 0.16);
  background: var(--bc-soft-panel-bg, rgba(20, 26, 40, 0.45));
  border-radius: 12px;
  padding: 0.8rem 0.9rem;
}

.stat-label {
  font-size: 0.78rem;
  color: var(--bc-text-muted, #9aa3b2);
}

.stat-value {
  margin-top: 0.35rem;
  font-size: 1.2rem;
  font-weight: 700;
  color: var(--bc-accent, #4da3ff);
}

.source-link {
  color: var(--bc-cyan, #5ae9ff);
}

.download-actions {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.logical-type-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}

.logical-type-tag {
  margin: 0 !important;
  border-color: rgba(120, 170, 255, 0.35) !important;
  color: var(--bc-text, #e8ecf4) !important;
  background: var(--bc-soft-panel-bg-strong, rgba(20, 26, 40, 0.55)) !important;
}

.cell-empty {
  color: var(--bc-text-muted, #9aa3b2);
}

.dataset-classification-table-wrap :deep(.el-table__body-wrapper) {
  scrollbar-color: var(--bc-scroll-thumb, rgba(120, 170, 255, 0.45))
    var(--bc-scroll-track, rgba(20, 26, 40, 0.5));
}

.dataset-classification-table-wrap :deep(.el-table__body-wrapper)::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

.dataset-classification-table-wrap :deep(.el-table__body-wrapper)::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: var(--bc-scroll-thumb, rgba(120, 170, 255, 0.35));
}

.dataset-classification-table-wrap :deep(.el-table__body-wrapper)::-webkit-scrollbar-track {
  background: var(--bc-scroll-track, rgba(20, 26, 40, 0.45));
  border-radius: 999px;
}

/* Tall paper lists in one row: inner scroll so the table body height stays manageable. */
.dataset-classification-table :deep(.el-table__body td.el-table__cell.papers-used-col) {
  vertical-align: top !important;
}

.paper-list-wrap {
  min-height: 0;
  max-height: 6rem;
  overflow-y: auto;
  overflow-x: hidden;
  line-height: 1.45;
  padding-right: 0.15rem;
  -webkit-overflow-scrolling: touch;
  scrollbar-color: var(--bc-scroll-thumb, rgba(120, 170, 255, 0.45))
    var(--bc-scroll-track, rgba(20, 26, 40, 0.5));
}

.paper-list-wrap::-webkit-scrollbar {
  width: 6px;
}

.paper-list-wrap::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: var(--bc-scroll-thumb, rgba(120, 170, 255, 0.35));
}

.paper-list-wrap::-webkit-scrollbar-track {
  background: var(--bc-scroll-track, rgba(20, 26, 40, 0.45));
  border-radius: 999px;
}

.paper-list {
  margin: 0;
  padding-left: 1rem;
}

.paper-list li {
  margin-bottom: 0.2rem;
}

.paper-link {
  overflow-wrap: anywhere;
  word-break: break-word;
}

.paper-meta {
  color: var(--bc-text-subtle, #a8abb2);
  font-size: 0.92em;
}

.paper-empty {
  color: var(--bc-text-muted, #9aa3b2);
}

.inline-code {
  font-size: 0.82em;
  padding: 0.1rem 0.35rem;
  border-radius: 6px;
  background: var(--bc-soft-panel-bg-strong, rgba(20, 26, 40, 0.75));
  border: 1px solid rgba(120, 170, 255, 0.2);
}

.detail-explorer {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.detail-picker {
  border-radius: 12px;
  border: 1px solid rgba(120, 170, 255, 0.16);
  background: var(--bc-soft-panel-bg, rgba(20, 26, 40, 0.5));
  padding: 0.9rem 1rem 1rem;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.detail-picker-controls {
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}

.detail-picker-inner {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 0.65rem;
}

.detail-picker-label-row {
  display: flex;
  align-items: center;
  gap: 0.45rem;
}

.detail-picker-icon {
  color: rgba(90, 233, 255, 0.75);
  flex-shrink: 0;
}

.detail-picker-label {
  margin: 0;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--bc-text-muted, #9aa3b2);
  cursor: default;
  user-select: none;
}

/* Full width inside bundle picker card; shared chrome in theme.css */
.detail-select {
  width: 100%;
  max-width: 100%;
}

.detail-picker-empty {
  margin: 0;
  font-size: 0.85rem;
  line-height: 1.5;
  color: var(--bc-text-muted, #9aa3b2);
}

.detail-jump-links {
  margin-top: 0.85rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.detail-jump-link {
  width: fit-content;
  padding: 0;
  border: none;
  background: transparent;
  color: var(--bc-cyan, #5ae9ff);
  cursor: pointer;
  font-size: 0.84rem;
  line-height: 1.4;
  text-align: left;
}

.detail-jump-link:hover {
  color: #7ef1ff;
  text-decoration: underline;
}

.dataset-classification-table-wrap :deep(.el-table__body tr.dataset-focus-row > td.el-table__cell),
.dataset-classification-table-wrap
  :deep(.el-table__fixed-body-wrapper tr.dataset-focus-row > td.el-table__cell) {
  animation: dataset-row-flash 1s ease-out forwards;
  position: relative;
  z-index: 2;
}

@keyframes dataset-row-flash {
  0% {
    box-shadow: inset 0 0 0 9999px rgba(90, 233, 255, 0.42);
  }
  40% {
    box-shadow: inset 0 0 0 9999px rgba(90, 233, 255, 0.26);
  }
  100% {
    box-shadow: inset 0 0 0 9999px rgba(90, 233, 255, 0);
  }
}

@media (min-width: 560px) {
  .detail-picker-controls {
    flex-direction: row;
    align-items: flex-end;
    gap: 1rem;
  }

  .detail-picker-controls > .detail-picker-inner {
    flex: 1 1 0;
    min-width: 0;
  }

  .detail-operator-picker {
    flex: 0 1 480px;
    min-width: 340px;
  }

  .detail-operator-picker .detail-picker-label-row {
    min-width: 4.9rem;
  }

  .detail-operator-select {
    min-width: 260px;
    max-width: none;
  }

  .detail-picker-inner {
    flex-direction: row;
    align-items: center;
    gap: 1.1rem;
  }

  .detail-picker-label-row {
    min-width: 6.75rem;
  }

  .detail-select {
    flex: 1;
    max-width: min(100%, 440px);
    min-width: 0;
  }
}

@media (max-width: 900px) {
  .category-dist-grid {
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0.45rem;
  }

  .category-pie-card {
    max-width: 150px;
  }

  .category-bar-card {
    max-width: 170px;
    height: 150px;
  }

  .category-pie-title {
    font-size: 0.72rem;
  }
}

@media (max-width: 600px) {
  .category-dist-content {
    overflow-x: auto;
  }

  .category-dist-grid {
    grid-template-columns: repeat(6, minmax(100px, 1fr));
    min-width: min(100%, 520px);
  }

  .category-pie-card {
    max-width: 120px;
  }

  .category-bar-card {
    max-width: 140px;
    height: 140px;
  }
}

.detail-empty-hint {
  font-size: 0.85rem;
  color: var(--bc-text-muted, #9aa3b2);
}

/* Match Home.vue Dataset insights: statistics-section + charts-section (charts-grid/chart-item from theme.css) */
.detail-statistics-block {
  margin-top: 0;
}

.detail-correlation-heatmaps {
  margin-top: 1.35rem;
  padding-top: 1.1rem;
  border-top: 1px solid var(--bc-border-subtle, rgba(120, 170, 255, 0.18));
}

.detail-correlation-desc {
  margin: 0 0 0.75rem;
  font-size: 0.8rem;
  line-height: 1.4;
  color: var(--bc-text-muted, rgba(220, 230, 255, 0.72));
}

.detail-correlation-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  grid-template-rows: 1fr 1fr;
  gap: 0.75rem;
  width: 100%;
  align-items: stretch;
}

.detail-correlation-cell {
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.detail-correlation-cell--left-top {
  grid-column: 1;
  grid-row: 1;
}

.detail-correlation-cell--algorithm {
  grid-column: 1;
  grid-row: 1 / span 2;
}

.detail-correlation-cell--left-bottom {
  grid-column: 1;
  grid-row: 2;
}

.detail-correlation-cell--operator {
  grid-column: 2;
  grid-row: 1 / span 2;
}

.detail-correlation-cell-empty {
  margin: 0;
  flex: 1 1 auto;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.75rem;
  border: 1px dashed rgba(120, 170, 255, 0.22);
  border-radius: 10px;
  font-size: 0.8rem;
  line-height: 1.4;
  text-align: center;
  color: var(--bc-text-muted, rgba(220, 230, 255, 0.72));
}

.detail-correlation-cell-empty--compact {
  min-height: 0;
}

.detail-correlation-cell-empty code {
  font-size: 0.78em;
}

.detail-operator-entropy-boxplots {
  margin-top: 1.35rem;
  padding-top: 1.1rem;
  border-top: 1px solid var(--bc-border);
}

.detail-operator-boxplot-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.9rem;
}

.detail-operator-boxplot-frame {
  min-width: 0;
  padding: 0.65rem 0.75rem 0.35rem;
  border: 1px solid var(--bc-border);
  border-radius: 10px;
  background: var(--bc-soft-panel-bg);
}

.detail-operator-boxplot-title {
  margin: 0 0 0.35rem;
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--bc-text);
}

.detail-operator-boxplot-desc {
  margin: 0 0 0.5rem;
  font-size: 0.78rem;
  line-height: 1.4;
  color: var(--bc-text-muted);
}

.operator-step-entropy-panel {
  margin: 0 0 1rem;
  padding: 0.75rem;
  border: 1px solid var(--bc-border);
  border-radius: 8px;
  background: var(--bc-soft-panel-bg);
}

.operator-step-entropy-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}

.operator-step-algorithm-select {
  width: min(360px, 100%);
  flex: 0 0 auto;
}

.operator-step-automata-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 0.75rem;
}

.operator-step-automaton {
  min-width: 0;
}

.operator-step-automaton-title {
  margin: 0 0 0.55rem;
  font-size: 0.82rem;
  font-weight: 700;
  color: var(--bc-text);
}

.operator-step-automaton-track {
  width: 100%;
  min-height: 128px;
  display: flex;
  align-items: center;
  gap: 0.45rem;
  overflow-x: auto;
  padding: 0.55rem 0.2rem 0.7rem;
}

.entropy-state-circle {
  width: 82px;
  height: 82px;
  flex: 0 0 82px;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.18rem;
  border: 1.5px solid var(--bc-automaton-edge, rgba(90, 233, 255, 0.72));
  background: var(--bc-automaton-state-bg, rgba(20, 34, 52, 0.96));
  color: var(--bc-automaton-state-text, var(--bc-text));
  box-shadow:
    0 0 0 1px color-mix(in srgb, var(--bc-automaton-edge, #5ae9ff) 16%, transparent),
    0 8px 18px color-mix(in srgb, var(--bc-automaton-edge, #5ae9ff) 12%, transparent);
}

.entropy-state-type {
  max-width: 68px;
  font-size: 0.72rem;
  font-weight: 800;
  text-transform: uppercase;
  text-align: center;
  overflow-wrap: anywhere;
}

.entropy-state-value {
  max-width: 70px;
  font-size: 0.68rem;
  color: var(--bc-automaton-label-text, rgba(220, 230, 255, 0.78));
  text-align: center;
  overflow-wrap: anywhere;
}

.entropy-transition {
  width: 132px;
  min-width: 132px;
  flex: 0 0 132px;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: center;
  gap: 0.34rem;
}

.entropy-transition-line {
  position: relative;
  height: 2px;
  background: var(--bc-automaton-edge, rgba(90, 233, 255, 0.72));
}

.entropy-transition-line::after {
  content: "";
  position: absolute;
  right: -1px;
  top: -4px;
  width: 0;
  height: 0;
  border-top: 5px solid transparent;
  border-bottom: 5px solid transparent;
  border-left: 8px solid var(--bc-automaton-edge, rgba(90, 233, 255, 0.82));
}

.entropy-transition.has-warning .entropy-transition-line,
.entropy-transition.has-warning .entropy-transition-line::after {
  background: var(--bc-automaton-edge-active, rgba(255, 214, 100, 0.72));
  border-left-color: var(--bc-automaton-edge-active, rgba(255, 214, 100, 0.82));
}

.entropy-transition-labels {
  min-height: 44px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  gap: 0.24rem;
}

.entropy-transition-label {
  max-width: 132px;
  padding: 0.18rem 0.34rem;
  border: 1px solid var(--bc-automaton-label-border, rgba(120, 170, 255, 0.24));
  border-radius: 6px;
  background: var(--bc-automaton-label-bg, rgba(12, 18, 30, 0.88));
  color: var(--bc-automaton-label-text, rgba(220, 230, 255, 0.9));
  font-size: 0.66rem;
  line-height: 1.25;
  text-align: center;
  overflow-wrap: anywhere;
  box-shadow: 0 1px 4px color-mix(in srgb, var(--bc-automaton-edge, #5ae9ff) 10%, transparent);
}

.entropy-transition-label strong {
  margin-left: 0.22rem;
  color: var(--bc-automaton-label-active-text, rgba(255, 239, 190, 0.98));
  font-weight: 800;
}

.entropy-transition-label em {
  display: block;
  margin-top: 0.1rem;
  font-style: normal;
  color: var(--bc-automaton-edge-active, rgba(255, 214, 100, 0.9));
}

@media (max-width: 900px) {
  .detail-correlation-grid {
    grid-template-columns: repeat(2, minmax(300px, 1fr));
    overflow-x: auto;
    padding-bottom: 0.25rem;
  }

  .detail-operator-boxplot-grid {
    grid-template-columns: 1fr;
  }

  .operator-step-entropy-toolbar {
    align-items: stretch;
    flex-direction: column;
  }

  .operator-step-algorithm-select {
    width: 100%;
  }

}

.statistics-section {
  margin-bottom: 0;
}

.statistics-title {
  text-align: left;
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--bc-text, #e8ecf4);
  margin: 0 0 0.75rem;
}

.detail-statistics-scope-hint {
  margin: 0 0 0.45rem;
  font-size: 0.82rem;
  color: var(--bc-text-muted, #9aa3b2);
  line-height: 1.45;
}

.detail-statistics-scope-hint strong {
  color: var(--bc-text, #e8ecf4);
  font-weight: 600;
}

.detail-statistics-column-hint {
  margin: 0 0 0.65rem;
  font-size: 0.82rem;
  color: var(--bc-text-muted, #9aa3b2);
}

.detail-statistics-block :deep(.detail-statistics-table .el-table__body td.el-table__cell) {
  overflow: visible;
}

.detail-statistics-block :deep(.detail-statistics-table .el-table__body td.el-table__cell .cell) {
  overflow: visible;
  padding-top: 6px;
  padding-bottom: 6px;
}

.detail-statistics-block :deep(.distribution-bar-chart) {
  margin: 0;
}

.detail-statistics-block :deep(.detail-statistics-table .el-table__header-wrapper th.el-table__cell) {
  height: auto !important;
  vertical-align: top;
  font-size: 0.82rem !important;
  padding: 8px !important;
}

.detail-statistics-block :deep(.detail-statistics-table .el-table__header-wrapper th .cell) {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: unset !important;
  line-height: 1.35;
  word-break: break-word;
  overflow-wrap: anywhere;
}

.statistics-column-link {
  display: block;
  width: 100%;
  padding: 0;
  border: none;
  background: transparent;
  color: var(--bc-cyan, #5ae9ff);
  cursor: pointer;
  font: inherit;
  font-weight: 600;
  text-align: left;
  white-space: normal;
  word-break: break-word;
  overflow-wrap: anywhere;
  line-height: 1.35;
}

.statistics-column-link:hover {
  color: #7ef1ff;
  text-decoration: underline;
}

.statistics-column-link:focus-visible {
  outline: 2px solid rgba(90, 233, 255, 0.45);
  outline-offset: 2px;
  border-radius: 3px;
}

.statistics-column-header {
  display: inline-flex;
  align-items: center;
  max-width: 100%;
}

.statistics-column-sentinel {
  display: inline-block;
  flex: 0 0 auto;
  width: 1px;
  height: 1px;
  margin-left: 2px;
}

.detail-stat-empty {
  color: var(--bc-text-muted, #9aa3b2);
  font-size: 0.9rem;
}

/* Override global theme.css .charts-section box so Dataset matches Home.vue viz-panel charts */
.charts-section.detail-charts-block {
  margin: 0.5rem 0 1rem;
  padding: 0;
  background: transparent;
  border: none;
}

.charts-title {
  text-align: left;
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--bc-text, #e8ecf4);
  margin-bottom: 0.85rem;
}
</style>
