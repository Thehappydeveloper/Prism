<script lang="ts">
  import type { ChartConfiguration } from 'chart.js';
  import ChartCanvas from '../components/ChartCanvas.svelte';
  import { analyzeSpeedAcceleration, discoverSpeedAccelerationColumns, inspectSpeedAccelerationPly } from '../core/api';
  import { gotoRoute } from '../core/router';
  import type {
    DistributionBin,
    ExtraToolWarning,
    NumericStats,
    PlyInspectionResponse,
    SpeedAccelerationAnalysisResponse,
    SpeedAccelerationDiscoverResponse,
  } from '../core/types';

  let rootFolder = '';
  let discovery: SpeedAccelerationDiscoverResponse | null = null;
  let analysis: SpeedAccelerationAnalysisResponse | null = null;
  let xColumn = '';
  let yColumn = '';
  let zColumn = '';
  let fps = 60;
  let plyPath = '';
  let plyInspection: PlyInspectionResponse | null = null;
  let heightAxis: 'x' | 'y' | 'z' = 'y';
  let realHeightM = 1.8;
  let filterOutliers = true;
  let filterMethod: 'iqr' | 'percentile' = 'iqr';
  let percentileKeep = 99;
  let histogramSource: 'raw' | 'filtered' = 'filtered';
  let excludeSuspiciousJumps = true;
  let showRawStats = false;
  let loading = false;
  let phase = 'Waiting for a folder.';
  let error = '';

  $: speedChart = buildHistogramChart(analysis?.speed.histogram || [], 'Speed distribution (m/s)', '#38bdf8');
  $: accelerationChart = buildHistogramChart(analysis?.signed_acceleration.histogram || [], 'Signed acceleration distribution (m/s²)', '#f472b6');
  $: warnings = combinedWarnings(discovery?.warnings || [], analysis?.warnings || []);
  $: sampleInterval = fps > 0 ? 1 / fps : null;
  $: selectedExtent = plyInspection ? plyInspection.extents[heightAxis] : null;
  $: scaleFactor = selectedExtent && realHeightM > 0 ? realHeightM / selectedExtent : null;

  function combinedWarnings(first: ExtraToolWarning[], second: ExtraToolWarning[]): ExtraToolWarning[] {
    return [...first, ...second];
  }

  function formatNumber(value: number | null | undefined): string {
    if (value === null || value === undefined) return 'No data';
    if (value !== 0 && (Math.abs(value) >= 1000 || Math.abs(value) < 0.01)) return value.toExponential(3);
    return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
  }

  function pickDefaultColumn(columns: string[], matcher: RegExp): string {
    return columns.find((column) => matcher.test(column)) || columns[0] || '';
  }

  function availableColumns(): string[] {
    return discovery?.numeric_columns?.length ? discovery.numeric_columns : discovery?.columns || [];
  }

  async function discoverColumns(): Promise<void> {
    loading = true;
    error = '';
    analysis = null;
    discovery = null;
    plyInspection = null;
    phase = 'Scanning folders for CSV files...';
    try {
      const payload = await discoverSpeedAccelerationColumns(rootFolder);
      discovery = payload;
      const columns = payload.numeric_columns?.length ? payload.numeric_columns : payload.columns;
      xColumn = pickDefaultColumn(columns, /(^|[_\s-])(x|pos_x|position_x|x_position)($|[_\s-])/i);
      yColumn = pickDefaultColumn(columns, /(^|[_\s-])(y|pos_y|position_y|y_position)($|[_\s-])/i);
      zColumn = pickDefaultColumn(columns, /(^|[_\s-])(z|pos_z|position_z|z_position)($|[_\s-])/i);
      phase = payload.file_count
        ? `Loaded ${payload.files_loaded} of ${payload.file_count} CSV file${payload.file_count === 1 ? '' : 's'} and detected ${columns.length} numeric column${columns.length === 1 ? '' : 's'}.`
        : 'No CSV files found.';
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      phase = 'Scan failed.';
    } finally {
      loading = false;
    }
  }

  async function analyzeColumns(): Promise<void> {
    loading = true;
    error = '';
    analysis = null;
    phase = 'Reading positions and deriving motion metrics...';
    try {
      analysis = await analyzeSpeedAcceleration(rootFolder, xColumn, yColumn, zColumn, fps, {
        plyPath,
        heightAxis,
        realHeightM,
      }, {
        filterOutliers,
        filterMethod,
        percentileKeep,
        histogramSource,
        excludeSuspiciousJumps,
      });
      phase = `Derived ${analysis.summary.speed_samples.toLocaleString()} speed samples and ${analysis.summary.signed_acceleration_samples.toLocaleString()} signed acceleration samples.`;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      phase = 'Analysis failed.';
    } finally {
      loading = false;
    }
  }

  async function inspectPly(): Promise<void> {
    loading = true;
    error = '';
    analysis = null;
    plyInspection = null;
    phase = 'Inspecting reference PLY point cloud...';
    try {
      plyInspection = await inspectSpeedAccelerationPly(plyPath);
      phase = `Loaded ${plyInspection.point_count.toLocaleString()} PLY points from ${plyInspection.filename}.`;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      phase = 'PLY inspection failed.';
    } finally {
      loading = false;
    }
  }

  function buildHistogramChart(bins: DistributionBin[], title: string, color: string): ChartConfiguration | null {
    if (!bins.length) return null;
    return {
      type: 'bar',
      data: {
        labels: bins.map((bin) => bin.label),
        datasets: [
          {
            label: 'Count',
            data: bins.map((bin) => bin.count),
            backgroundColor: color,
            yAxisID: 'y',
          },
          {
            label: 'Percentage',
            data: bins.map((bin) => Number(bin.percentage.toFixed(2))),
            backgroundColor: 'rgba(148,163,184,0.7)',
            yAxisID: 'y1',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom' },
          title: { display: true, text: title },
          tooltip: {
            callbacks: {
              afterBody: (items) => {
                const bin = bins[items[0]?.dataIndex ?? 0];
                if (!bin) return '';
                return `Range: ${formatNumber(bin.lower)} to ${formatNumber(bin.upper)}`;
              },
            },
          },
        },
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'Count' } },
          y1: {
            beginAtZero: true,
            position: 'right',
            grid: { drawOnChartArea: false },
            title: { display: true, text: 'Percent' },
            ticks: { callback: (value) => `${value}%` },
          },
        },
      },
    };
  }

  const statRows: Array<{ key: keyof NumericStats; label: string }> = [
    { key: 'min', label: 'Minimum' },
    { key: 'max', label: 'Maximum' },
    { key: 'median', label: 'Median' },
    { key: 'mean', label: 'Mean' },
    { key: 'stddev', label: 'Std. dev.' },
  ];

  function outlierText(summary: { total_samples: number; outlier_samples: number; percentage_removed: number }): string {
    return `${summary.total_samples.toLocaleString()} samples analyzed, ${summary.outlier_samples.toLocaleString()} outliers removed (${summary.percentage_removed.toFixed(1)}%).`;
  }

  function formatVector(value: number[] | null | undefined, suffix = ''): string {
    if (!value?.length) return 'No data';
    return `(${value.map((item) => formatNumber(item)).join(', ')})${suffix}`;
  }
</script>

<header class="hero">
  <div class="hero-actions">
    <a class="button-link ghost" href="/" on:click|preventDefault={() => gotoRoute('home')}>Main menu</a>
    <a class="button-link ghost" href="/static/datasets.html" on:click|preventDefault={() => gotoRoute('datasets')}>Datasets</a>
    <a class="button-link ghost" href="/static/index.html" on:click|preventDefault={() => gotoRoute('control')}>Single Run</a>
    <a class="button-link ghost" href="/static/aggregate.html" on:click|preventDefault={() => gotoRoute('batch')}>Batch Planner</a>
    <a class="button-link ghost" href="/static/compare.html" on:click|preventDefault={() => gotoRoute('compare')}>Compare Runs</a>
    <a class="button-link ghost" href="/static/reports.html" on:click|preventDefault={() => gotoRoute('reports')}>Reports</a>
    <a class="button-link active" aria-current="page" href="/static/extra.html" on:click|preventDefault={() => gotoRoute('extra')}>Extra</a>
    <a class="button-link ghost" href="/static/settings.html" on:click|preventDefault={() => gotoRoute('settings')}>Settings</a>
  </div>
</header>

<main>
  <section class="card form-card">
    <div class="section-header">
      <div>
        <p class="eyebrow micro">Input</p>
        <h2>Folder scan</h2>
      </div>
      <span class="pill" class:running={loading}>{loading ? 'Working' : 'Ready'}</span>
    </div>

    <div class="scan-grid">
      <label>Root folder
        <input bind:value={rootFolder} placeholder="data/fov_traces or /absolute/path/to/csvs" disabled={loading} />
      </label>
      <button on:click={discoverColumns} disabled={loading || !rootFolder.trim()}>Scan CSV files</button>
    </div>

    <div class="callout" class:warn={Boolean(error)}>
      <div class="callout-title">{error ? 'Error' : 'Progress'}</div>
      <div class="callout-body">{error || phase}</div>
    </div>
  </section>

  {#if discovery}
    <section class="card">
      <div class="section-header">
        <div>
          <p class="eyebrow micro">Motion source</p>
          <h2>Select position columns</h2>
        </div>
        <span class="pill">{discovery.files_loaded} loaded</span>
      </div>

      <div class="column-grid">
        <label>X Position column
          <select bind:value={xColumn} disabled={loading || !availableColumns().length}>
            {#each availableColumns() as column}
              <option value={column}>{column}</option>
            {/each}
          </select>
        </label>
        <label>Y Position column
          <select bind:value={yColumn} disabled={loading || !availableColumns().length}>
            {#each availableColumns() as column}
              <option value={column}>{column}</option>
            {/each}
          </select>
        </label>
        <label>Z Position column
          <select bind:value={zColumn} disabled={loading || !availableColumns().length}>
            {#each availableColumns() as column}
              <option value={column}>{column}</option>
            {/each}
          </select>
        </label>
      </div>

      <div class="timing-grid">
        <label>Recording frame rate (FPS)
          <input type="number" min="0.0001" step="0.001" bind:value={fps} disabled={loading} />
        </label>
        <div class="stat-cell">
          <span>Sample interval</span>
          <strong>{sampleInterval ? `${formatNumber(sampleInterval)} s` : 'No data'}</strong>
        </div>
      </div>

      <div class="scaling-grid">
        <label>Reference PLY point cloud
          <input bind:value={plyPath} placeholder="/absolute/path/reference.ply" disabled={loading} />
        </label>
        <button class="ghost" on:click={inspectPly} disabled={loading || !plyPath.trim()}>Inspect PLY</button>
        <label>Known height (m)
          <input type="number" min="0.0001" step="0.001" bind:value={realHeightM} disabled={loading} />
        </label>
        <label>Height axis
          <select bind:value={heightAxis} disabled={loading || !plyInspection}>
            <option value="x">X extent {plyInspection ? `(${formatNumber(plyInspection.extents.x)} units)` : ''}</option>
            <option value="y">Y extent {plyInspection ? `(${formatNumber(plyInspection.extents.y)} units)` : ''}</option>
            <option value="z">Z extent {plyInspection ? `(${formatNumber(plyInspection.extents.z)} units)` : ''}</option>
          </select>
        </label>
        <div class="stat-cell">
          <span>Scale factor</span>
          <strong>{scaleFactor ? `${formatNumber(scaleFactor)} m/unit` : 'No data'}</strong>
        </div>
      </div>

      {#if plyInspection}
        <section class="callout info scaling-summary">
          <div class="callout-title">Scale Conversion</div>
          <div class="summary-grid">
            <div><span>1 raw unit</span><strong>{scaleFactor ? `${formatNumber(scaleFactor)} m` : 'No data'}</strong></div>
            <div><span>1000 raw units</span><strong>{scaleFactor ? `${formatNumber(scaleFactor * 1000)} m` : 'No data'}</strong></div>
            <div><span>Selected axis</span><strong>{heightAxis.toUpperCase()}</strong></div>
            <div><span>Measured extent</span><strong>{selectedExtent ? `${formatNumber(selectedExtent)} units` : 'No data'}</strong></div>
            <div><span>Known height</span><strong>{formatNumber(realHeightM)} m</strong></div>
            <div><span>Scale factor</span><strong>{scaleFactor ? `${formatNumber(scaleFactor)} m/unit` : 'No data'}</strong></div>
          </div>
          {#if analysis?.scaling.coordinate_example.original}
            <div class="summary-grid coordinate-summary">
              <div><span>Original trace position</span><strong>{formatVector(analysis.scaling.coordinate_example.original)}</strong></div>
              <div><span>Scaled trace position</span><strong>{formatVector(analysis.scaling.coordinate_example.scaled_m, ' m')}</strong></div>
            </div>
          {/if}
          <details class="details-panel">
            <summary>Details</summary>
            <div class="summary-grid">
              <div><span>Filename</span><strong>{plyInspection.filename}</strong></div>
              <div><span>Points</span><strong>{plyInspection.point_count.toLocaleString()}</strong></div>
              <div><span>X extent</span><strong>{formatNumber(plyInspection.extents.x)} units</strong></div>
              <div><span>Y extent</span><strong>{formatNumber(plyInspection.extents.y)} units</strong></div>
              <div><span>Z extent</span><strong>{formatNumber(plyInspection.extents.z)} units</strong></div>
            </div>
            {#if analysis}
              <div class="callout-body assumptions">{analysis.scaling.assumptions.join(' ')}</div>
            {/if}
          </details>
        </section>
      {/if}

      <div class="filter-grid">
        <label class="pill-check">
          <input type="checkbox" bind:checked={filterOutliers} disabled={loading} />
          Filter outliers
        </label>
        <label>Filtering method
          <select bind:value={filterMethod} disabled={loading || !filterOutliers}>
            <option value="iqr">IQR</option>
            <option value="percentile">Percentile</option>
          </select>
        </label>
        <label>Percentile threshold
          <select bind:value={percentileKeep} disabled={loading || !filterOutliers || filterMethod !== 'percentile'}>
            <option value={99}>99%</option>
            <option value={99.5}>99.5%</option>
            <option value={99.9}>99.9%</option>
          </select>
        </label>
        <label>Histogram source
          <select bind:value={histogramSource} disabled={loading}>
            <option value="filtered">Filtered data</option>
            <option value="raw">Raw data</option>
          </select>
        </label>
        <label class="pill-check">
          <input type="checkbox" bind:checked={excludeSuspiciousJumps} disabled={loading} />
          Exclude suspicious jumps
        </label>
        <label class="pill-check">
          <input type="checkbox" bind:checked={showRawStats} disabled={loading} />
          Show raw statistics
        </label>
      </div>

      <div class="form-footer">
        <button on:click={analyzeColumns} disabled={loading || !xColumn || !yColumn || !zColumn || fps <= 0 || !plyInspection || !plyPath.trim() || realHeightM <= 0}>Analyze scaled motion</button>
      </div>

      <div class="chip-row">
        {#each availableColumns() as column}
          <span class="pill">{column}</span>
        {/each}
      </div>
    </section>
  {/if}

  {#if analysis}
    <section class="dashboard-grid">
      <article class="card metric-card">
        <p class="eyebrow micro">Speed</p>
        <h2>Filtered data</h2>
        <p class="subtle">{outlierText(analysis.speed.outlier_summary)}</p>
        <div class="stat-grid">
          {#each statRows as row}
            <div class="stat-cell">
              <span>{row.label}</span>
              <strong>{formatNumber(analysis.speed.filtered_stats[row.key])}{row.key === 'samples' ? '' : ' m/s'}</strong>
            </div>
          {/each}
        </div>
        {#if showRawStats}
          <h2>Raw data</h2>
          <div class="stat-grid compact-stats">
            {#each statRows as row}
              <div class="stat-cell">
                <span>{row.label}</span>
                <strong>{formatNumber(analysis.speed.raw_stats[row.key])} m/s</strong>
              </div>
            {/each}
          </div>
        {/if}
      </article>

      <article class="card metric-card">
        <p class="eyebrow micro">Signed acceleration</p>
        <h2>Filtered data</h2>
        <p class="subtle">{outlierText(analysis.signed_acceleration.outlier_summary)}</p>
        <div class="stat-grid">
          {#each statRows as row}
            <div class="stat-cell">
              <span>{row.label}</span>
              <strong>{formatNumber(analysis.signed_acceleration.filtered_stats[row.key])}{row.key === 'samples' ? '' : ' m/s²'}</strong>
            </div>
          {/each}
        </div>
        {#if showRawStats}
          <h2>Raw data</h2>
          <div class="stat-grid compact-stats">
            {#each statRows as row}
              <div class="stat-cell">
                <span>{row.label}</span>
                <strong>{formatNumber(analysis.signed_acceleration.raw_stats[row.key])} m/s²</strong>
              </div>
            {/each}
          </div>
        {/if}
      </article>
    </section>

    <section class="card">
      <div class="section-header">
        <div>
          <p class="eyebrow micro">Data quality</p>
          <h2>Quality summary</h2>
        </div>
      </div>
      <div class="summary-grid">
        <div><span>Total samples</span><strong>{analysis.quality.total_samples.toLocaleString()}</strong></div>
        <div><span>Valid samples</span><strong>{analysis.quality.valid_samples.toLocaleString()}</strong></div>
        <div><span>Invalid samples</span><strong>{analysis.quality.invalid_rows_skipped.toLocaleString()}</strong></div>
        <div><span>Outliers removed</span><strong>{analysis.motion_overview.outliers_removed.toLocaleString()}</strong></div>
        <div><span>Suspicious position jumps</span><strong>{analysis.quality.suspicious_position_jumps.toLocaleString()}</strong></div>
        <div><span>Speed outliers detected</span><strong>{analysis.quality.speed_outliers_detected.toLocaleString()}</strong></div>
        <div><span>Acceleration outliers detected</span><strong>{analysis.quality.acceleration_outliers_detected.toLocaleString()}</strong></div>
      </div>
      <div class="callout info">
        <div class="callout-title">Outlier handling</div>
        <div class="callout-body">Extreme values are often caused by tracking noise, dropped frames, invalid coordinates, or coordinate glitches and may not represent actual object motion.</div>
      </div>
    </section>

    <section class="grid two-col">
      <article class="card chart-card">
        <h2>Speed histogram (m/s)</h2>
        <ChartCanvas config={speedChart} height={340} />
        <div class="table-wrap">
          <table class="compact">
            <thead>
              <tr>
                  <th>Range</th>
                <th>Count</th>
                <th>Percent</th>
              </tr>
            </thead>
            <tbody>
              {#each analysis.speed.histogram as bin}
                <tr>
                  <td>{bin.label}</td>
                  <td>{bin.count.toLocaleString()}</td>
                  <td>{bin.percentage.toFixed(2)}%</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </article>
      <article class="card chart-card">
        <h2>Signed acceleration histogram (m/s²)</h2>
        <ChartCanvas config={accelerationChart} height={340} />
        <div class="table-wrap">
          <table class="compact">
            <thead>
              <tr>
                  <th>Range</th>
                <th>Count</th>
                <th>Percent</th>
              </tr>
            </thead>
            <tbody>
              {#each analysis.signed_acceleration.histogram as bin}
                <tr>
                  <td>{bin.label}</td>
                  <td>{bin.count.toLocaleString()}</td>
                  <td>{bin.percentage.toFixed(2)}%</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </article>
    </section>

    <section class="card">
      <div class="section-header">
        <div>
          <p class="eyebrow micro">Analysis</p>
          <h2>Analysis summary</h2>
        </div>
      </div>
      <div class="summary-grid">
        <div><span>Recording duration</span><strong>{formatNumber(analysis.motion_overview.recording_duration_s)} s</strong></div>
        <div><span>Total valid samples</span><strong>{analysis.quality.valid_samples.toLocaleString()}</strong></div>
        <div><span>Total distance travelled</span><strong>{formatNumber(analysis.motion_overview.total_distance_m)} m</strong></div>
        <div><span>Median speed</span><strong>{formatNumber(analysis.motion_overview.median_speed_mps)} m/s</strong></div>
        <div><span>Peak speed</span><strong>{formatNumber(analysis.motion_overview.peak_speed_mps)} m/s</strong></div>
        <div><span>Median acceleration</span><strong>{formatNumber(analysis.motion_overview.median_acceleration_mps2)} m/s²</strong></div>
        <div><span>Peak acceleration</span><strong>{formatNumber(analysis.motion_overview.peak_acceleration_mps2)} m/s²</strong></div>
        <div><span>Outliers removed</span><strong>{analysis.motion_overview.outliers_removed.toLocaleString()}</strong></div>
      </div>
    </section>

    <section class="grid two-col">
      <article class="card">
        <div class="section-header">
          <div>
            <p class="eyebrow micro">Motion overview</p>
            <h2>Movement profile</h2>
          </div>
        </div>
        <div class="summary-grid">
          <div><span>Recording duration</span><strong>{formatNumber(analysis.motion_overview.recording_duration_s)} s</strong></div>
          <div><span>Total distance travelled</span><strong>{formatNumber(analysis.motion_overview.total_distance_m)} m</strong></div>
          <div><span>Average speed</span><strong>{formatNumber(analysis.motion_overview.average_speed_mps)} m/s</strong></div>
          <div><span>Peak speed</span><strong>{formatNumber(analysis.motion_overview.peak_speed_mps)} m/s</strong></div>
          <div><span>Time stationary</span><strong>{formatNumber(analysis.motion_overview.time_stationary_s)} s</strong></div>
          <div><span>Time accelerating</span><strong>{formatNumber(analysis.motion_overview.time_accelerating_s)} s</strong></div>
          <div><span>Time decelerating</span><strong>{formatNumber(analysis.motion_overview.time_decelerating_s)} s</strong></div>
        </div>
      </article>

      <article class="card">
        <div class="section-header">
          <div>
            <p class="eyebrow micro">Outliers</p>
            <h2>Outlier overview</h2>
          </div>
        </div>
        <div class="summary-grid">
          <div><span>Speed outliers removed</span><strong>{analysis.outlier_overview.speed_outliers_removed.toLocaleString()}</strong></div>
          <div><span>Acceleration outliers removed</span><strong>{analysis.outlier_overview.acceleration_outliers_removed.toLocaleString()}</strong></div>
          <div><span>Largest removed speed</span><strong>{formatNumber(analysis.outlier_overview.largest_removed_speed_mps)} m/s</strong></div>
          <div><span>Largest removed acceleration</span><strong>{formatNumber(analysis.outlier_overview.largest_removed_acceleration_mps2)} m/s²</strong></div>
          <div><span>Largest retained speed</span><strong>{formatNumber(analysis.outlier_overview.largest_retained_speed_mps)} m/s</strong></div>
          <div><span>Largest retained acceleration</span><strong>{formatNumber(analysis.outlier_overview.largest_retained_acceleration_mps2)} m/s²</strong></div>
        </div>
      </article>
    </section>
  {/if}

  {#if warnings.length}
    <section class="card">
      <div class="section-header">
        <div>
          <p class="eyebrow micro">Warnings</p>
          <h2>Skipped files and rows</h2>
        </div>
        <span class="pill queued">{warnings.length}</span>
      </div>
      <div class="status-list">
        {#each warnings as warning}
          <div class="status-row failed">
            <div class="status-title">{warning.kind.replace('_', ' ')}</div>
            <div class="status-meta">{warning.path ? `${warning.path}: ` : ''}{warning.message}</div>
          </div>
        {/each}
      </div>
    </section>
  {/if}
</main>

<style>
  .scan-grid,
  .column-grid {
    display: grid;
    grid-template-columns: minmax(260px, 1fr) auto;
    gap: 0.75rem;
    align-items: end;
  }

  .column-grid {
    grid-template-columns: repeat(3, minmax(180px, 1fr));
    margin-top: 0.85rem;
  }

  .timing-grid {
    display: grid;
    grid-template-columns: minmax(220px, 0.7fr) minmax(180px, 0.5fr) auto;
    gap: 0.75rem;
    align-items: end;
    margin-top: 0.85rem;
  }

  .filter-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.75rem;
    align-items: end;
    margin-top: 0.85rem;
  }

  .scaling-grid {
    display: grid;
    grid-template-columns: minmax(260px, 1.2fr) auto minmax(150px, 0.6fr) minmax(210px, 1fr) minmax(170px, 0.7fr);
    gap: 0.75rem;
    align-items: end;
    margin-top: 0.85rem;
  }

  .scaling-summary {
    margin-top: 0.85rem;
  }

  .coordinate-summary,
  .details-panel {
    margin-top: 0.75rem;
  }

  .details-panel summary {
    cursor: pointer;
    color: var(--text-main);
    font-weight: 700;
  }

  .assumptions {
    margin-top: 0.6rem;
  }

  .chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.9rem;
  }

  .dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 1rem;
  }

  .metric-card {
    display: grid;
    gap: 0.7rem;
  }

  .metric-card h2,
  .chart-card h2 {
    margin: 0;
    font-size: 1.05rem;
  }

  .stat-grid,
  .summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.65rem;
  }

  .compact-stats {
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  }

  .stat-cell,
  .summary-grid > div {
    border: 1px solid var(--card-border);
    border-radius: 10px;
    background: var(--soft-bg-alt);
    padding: 0.7rem 0.75rem;
    display: grid;
    gap: 0.25rem;
  }

  .stat-cell span,
  .summary-grid span {
    color: var(--text-subtle);
    font-size: 0.82rem;
  }

  .stat-cell strong,
  .summary-grid strong {
    font-size: 1.05rem;
  }

  .chart-card {
    display: grid;
    gap: 0.75rem;
  }

  button:disabled,
  input:disabled,
  select:disabled {
    opacity: 0.58;
    cursor: not-allowed;
  }

  @media (max-width: 780px) {
    .scan-grid,
    .column-grid,
    .timing-grid,
    .filter-grid,
    .scaling-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
