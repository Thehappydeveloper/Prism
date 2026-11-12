<script lang="ts">
  import type { ChartConfiguration } from 'chart.js';
  import ChartCanvas from '../components/ChartCanvas.svelte';
  import { analyzePointCloudSequence, discoverPointCloudSequence } from '../core/api';
  import { gotoRoute } from '../core/router';
  import type {
    ExtraToolWarning,
    PointCloudSequenceAggregation,
    PointCloudSequenceAnalysisResponse,
    PointCloudSequenceDiscoverResponse,
    PointCloudSequenceFrame,
    PointCloudSequenceSegment,
  } from '../core/types';

  type Axis = 'x' | 'y' | 'z';
  type ViewMode = 'frames' | 'segments';
  type Row = PointCloudSequenceFrame | PointCloudSequenceSegment;
  const radiusColors = ['#38bdf8', '#22c55e', '#eab308', '#f97316', '#ef4444'];

  let rootFolder = '';
  let recursive = true;
  let segmentSize = 30;
  let aggregation: PointCloudSequenceAggregation = 'mean';
  let voxelBits = 10;
  let useNativeVoxels = false;
  let maxPointsForQp = 120000;
  let axis: Axis = 'x';
  let viewMode: ViewMode = 'frames';
  let discovery: PointCloudSequenceDiscoverResponse | null = null;
  let analysis: PointCloudSequenceAnalysisResponse | null = null;
  let loading = false;
  let phase = 'Waiting for a PLY sequence folder.';
  let error = '';

  $: warnings = combinedWarnings(discovery?.warnings || [], analysis?.warnings || []);
  $: rows = viewMode === 'segments' ? analysis?.segments || [] : analysis?.frames || [];
  $: xLabels = rows.map((row) => rowLabel(row));
  $: xyzMinMaxChart = buildLineChart(
    xLabels,
    [
      { label: `Min ${axis.toUpperCase()}`, values: rows.map((row) => numberValue(row, `min_${axis}`)), color: '#38bdf8' },
      { label: `Max ${axis.toUpperCase()}`, values: rows.map((row) => numberValue(row, `max_${axis}`)), color: '#f97316' },
    ],
    `${axis.toUpperCase()} coordinate bounds`,
    'Coordinate value',
  );
  $: rangeChart = buildLineChart(
    xLabels,
    [
      { label: 'Width X', values: rows.map((row) => numberValue(row, 'range_x')), color: '#38bdf8' },
      { label: 'Height Y', values: rows.map((row) => numberValue(row, 'range_y')), color: '#22c55e' },
      { label: 'Depth Z', values: rows.map((row) => numberValue(row, 'range_z')), color: '#f97316' },
    ],
    'Bounding box range',
    'Range',
  );
  $: pointCountChart = buildLineChart(
    xLabels,
    [{ label: 'Point count', values: rows.map((row) => numberValue(row, 'point_count')), color: '#38bdf8' }],
    'Point count',
    'Points',
  );
  $: densityChart = buildLineChart(
    xLabels,
    [{ label: 'Point density', values: rows.map((row) => nullableValue(row, 'density')), color: '#a78bfa' }],
    'Point density',
    'Points / volume',
  );
  $: safetyChart = buildLineChart(
    xLabels,
    [
      { label: 'Geometry QP Safety', values: rows.map((row) => nullableValue(row, 'geometry_qp_safety')), color: '#22c55e' },
      { label: 'Attribute QP Safety', values: rows.map((row) => nullableValue(row, 'attribute_qp_safety')), color: '#f472b6' },
    ],
    'QP safety over time',
    'Score',
    true,
  );
  $: geometryRadiusChart = buildLineChart(
    xLabels,
    [1, 2, 3, 4, 5].map((radius, index) => ({
      label: `G${radius}`,
      values: rows.map((row) => nullableValue(row, `g${radius}`)),
      color: radiusColors[index] || '#38bdf8',
    })),
    'Geometry redundancy radius curve',
    'Score',
    true,
  );
  $: attributeRadiusChart = buildLineChart(
    xLabels,
    [1, 2, 3, 4, 5].map((radius, index) => ({
      label: `A${radius}`,
      values: rows.map((row) => nullableValue(row, `a${radius}`)),
      color: radiusColors[index] || '#38bdf8',
    })),
    'Attribute smoothness radius curve',
    'Score',
    true,
  );
  $: stabilityChart = buildLineChart(
    xLabels,
    [
      { label: 'Geometry stability', values: rows.map((row) => nullableValue(row, 'geometry_stability')), color: '#22c55e' },
      { label: 'Attribute stability', values: rows.map((row) => nullableValue(row, 'attribute_stability')), color: '#f472b6' },
      { label: 'Inverse color entropy', values: rows.map((row) => nullableValue(row, 'inverse_color_entropy')), color: '#38bdf8' },
    ],
    'Stability and color entropy',
    'Score',
    true,
  );

  function combinedWarnings(first: ExtraToolWarning[], second: ExtraToolWarning[]): ExtraToolWarning[] {
    return [...first, ...second];
  }

  function formatNumber(value: number | null | undefined): string {
    if (value === null || value === undefined || Number.isNaN(value)) return 'No data';
    if (value !== 0 && (Math.abs(value) >= 1000 || Math.abs(value) < 0.01)) return value.toExponential(3);
    return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
  }

  function formatScore(value: number | null | undefined): string {
    if (value === null || value === undefined) return 'No data';
    return value.toFixed(3);
  }

  function rowLabel(row: Row): string {
    if ('segment_id' in row) return `S${row.segment_id} (${row.start_frame_index}-${row.end_frame_index})`;
    return String(row.frame_index);
  }

  function numberValue(row: Row, key: string): number {
    const value = (row as unknown as Record<string, unknown>)[key];
    return typeof value === 'number' && Number.isFinite(value) ? value : 0;
  }

  function nullableValue(row: Row, key: string): number | null {
    const value = (row as unknown as Record<string, unknown>)[key];
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  async function scanSequence(): Promise<void> {
    loading = true;
    error = '';
    discovery = null;
    analysis = null;
    phase = 'Scanning for indexed PLY files...';
    try {
      discovery = await discoverPointCloudSequence(rootFolder, recursive);
      phase = discovery.file_count
        ? `Detected ${discovery.file_count.toLocaleString()} indexed PLY file${discovery.file_count === 1 ? '' : 's'} from frame ${discovery.first_frame_index} to ${discovery.last_frame_index}.`
        : 'No indexed PLY files found.';
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      phase = 'Scan failed.';
    } finally {
      loading = false;
    }
  }

  async function runAnalysis(): Promise<void> {
    loading = true;
    error = '';
    analysis = null;
    phase = 'Reading PLY frames and computing sequence metrics...';
    try {
      analysis = await analyzePointCloudSequence({
        rootFolder,
        recursive,
        segmentSize,
        aggregation,
        voxelBits,
        useNativeVoxels,
        maxPointsForQp,
      });
      phase = `Processed ${analysis.summary.processed_frames.toLocaleString()} frame${analysis.summary.processed_frames === 1 ? '' : 's'} and built ${analysis.segments.length.toLocaleString()} segment${analysis.segments.length === 1 ? '' : 's'}.`;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      phase = 'Analysis failed.';
    } finally {
      loading = false;
    }
  }

  function buildLineChart(
    labels: string[],
    series: Array<{ label: string; values: Array<number | null>; color: string }>,
    title: string,
    yTitle: string,
    scoreScale = false,
  ): ChartConfiguration | null {
    if (!labels.length) return null;
    return {
      type: 'line',
      data: {
        labels,
        datasets: series.map((item) => ({
          label: item.label,
          data: item.values,
          borderColor: item.color,
          backgroundColor: item.color,
          borderWidth: 2,
          pointRadius: labels.length > 90 ? 0 : 2,
          spanGaps: true,
          tension: 0.18,
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom' },
          title: { display: true, text: title },
        },
        scales: {
          x: { title: { display: true, text: viewMode === 'segments' ? 'Segment' : 'Frame index' } },
          y: {
            min: scoreScale ? 0 : undefined,
            max: scoreScale ? 1 : undefined,
            title: { display: true, text: yTitle },
          },
        },
      },
    };
  }

  function exportJson(): void {
    if (!analysis) return;
    downloadFile('point-cloud-sequence-analysis.json', JSON.stringify(analysis, null, 2), 'application/json');
  }

  function exportCsv(): void {
    if (!analysis) return;
    const currentAnalysis = analysis;
    const headers = [
      'segment_id',
      'frame_index',
      'file_name',
      'min_x',
      'max_x',
      'min_y',
      'max_y',
      'min_z',
      'max_z',
      'range_x',
      'range_y',
      'range_z',
      'point_count',
      'density',
      'geometry_qp_safety',
      'attribute_qp_safety',
      'g1',
      'g2',
      'g3',
      'g4',
      'g5',
      'a1',
      'a2',
      'a3',
      'a4',
      'a5',
      'geometry_stability',
      'attribute_stability',
      'inverse_color_entropy',
    ];
    const segmentForFrame = new Map<number, number>();
    currentAnalysis.segments.forEach((segment) => {
      currentAnalysis.frames
        .filter((frame) => frame.frame_index >= segment.start_frame_index && frame.frame_index <= segment.end_frame_index)
        .forEach((frame) => segmentForFrame.set(frame.frame_index, segment.segment_id));
    });
    const lines = [headers.join(',')];
    currentAnalysis.frames.forEach((frame) => {
      const record: Record<string, unknown> = { ...frame, segment_id: segmentForFrame.get(frame.frame_index) || '' };
      lines.push(headers.map((header) => csvCell(record[header])).join(','));
    });
    downloadFile('point-cloud-sequence-analysis.csv', `${lines.join('\n')}\n`, 'text/csv');
  }

  function csvCell(value: unknown): string {
    if (value === null || value === undefined) return '';
    const text = String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }

  function downloadFile(filename: string, content: string, type: string): void {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
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
      <label>PLY sequence folder
        <input bind:value={rootFolder} placeholder="/mnt/storage/encoded_content/longdress" disabled={loading} />
      </label>
      <label class="check-row">
        <input type="checkbox" bind:checked={recursive} disabled={loading} />
        Recursive scan
      </label>
      <button on:click={scanSequence} disabled={loading || !rootFolder.trim()}>Scan PLY files</button>
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
          <p class="eyebrow micro">Processing Options</p>
          <h2>Sequence analysis</h2>
        </div>
        <span class="pill">{discovery.file_count.toLocaleString()} detected</span>
      </div>

      <div class="stats-grid">
        <div><span>First frame</span><strong>{discovery.first_frame_index ?? 'No data'}</strong></div>
        <div><span>Last frame</span><strong>{discovery.last_frame_index ?? 'No data'}</strong></div>
        <div><span>PLY files</span><strong>{discovery.file_count.toLocaleString()}</strong></div>
      </div>

      <div class="column-grid">
        <label>Segment size
          <input type="number" min="1" step="1" bind:value={segmentSize} disabled={loading} />
        </label>
        <label>Aggregation
          <select bind:value={aggregation} disabled={loading}>
            <option value="mean">Mean</option>
            <option value="median">Median</option>
            <option value="minimum">Minimum</option>
            <option value="maximum">Maximum</option>
            <option value="standard_deviation">Standard deviation</option>
          </select>
        </label>
        <label>Voxel bit depth
          <select bind:value={voxelBits} disabled={loading}>
            <option value={9}>vox9</option>
            <option value={10}>vox10</option>
            <option value={11}>vox11</option>
            <option value={12}>vox12</option>
          </select>
        </label>
        <label>Max points for QP scoring
          <input type="number" min="0" step="1000" bind:value={maxPointsForQp} disabled={loading} />
        </label>
        <label class="check-row">
          <input type="checkbox" bind:checked={useNativeVoxels} disabled={loading} />
          Use native integer coordinates
        </label>
      </div>

      <div class="actions-row">
        <button on:click={runAnalysis} disabled={loading || !discovery.file_count}>Analyze sequence</button>
      </div>
    </section>
  {/if}

  {#if analysis}
    <section class="card">
      <div class="section-header">
        <div>
          <p class="eyebrow micro">Summary</p>
          <h2>Processed sequence</h2>
        </div>
        <span class="pill">{analysis.summary.processed_frames.toLocaleString()} processed</span>
      </div>

      <div class="stats-grid">
        <div><span>Detected PLY files</span><strong>{analysis.summary.detected_ply_files.toLocaleString()}</strong></div>
        <div><span>Skipped frames</span><strong>{analysis.summary.skipped_frames.toLocaleString()}</strong></div>
        <div><span>Color frames</span><strong>{analysis.summary.color_frames.toLocaleString()}</strong></div>
        <div><span>Sampled frames</span><strong>{analysis.summary.sampled_frames.toLocaleString()}</strong></div>
      </div>
    </section>

    <section class="card">
      <div class="section-header">
        <div>
          <p class="eyebrow micro">QP Safety</p>
          <h2>Score interpretation</h2>
        </div>
      </div>
      <div class="score-grid">
        <div><span>0.80 to 1.00</span><strong>Aggressive QP likely safe</strong></div>
        <div><span>0.60 to 0.80</span><strong>Moderate/aggressive may work</strong></div>
        <div><span>0.40 to 0.60</span><strong>Default QP recommended</strong></div>
        <div><span>0.20 to 0.40</span><strong>Risky, use weaker QP</strong></div>
        <div><span>0.00 to 0.20</span><strong>Very fragile</strong></div>
      </div>
    </section>

    <section class="card">
      <div class="section-header">
        <div>
          <p class="eyebrow micro">Sequence Graphs</p>
          <h2>Frame and segment views</h2>
        </div>
        <div class="toggle-group">
          <button class:active={viewMode === 'frames'} on:click={() => (viewMode = 'frames')}>Frames</button>
          <button class:active={viewMode === 'segments'} on:click={() => (viewMode = 'segments')}>Segments</button>
        </div>
      </div>

      <div class="axis-row">
        <span>XYZ axis</span>
        <button class:active={axis === 'x'} on:click={() => (axis = 'x')}>X</button>
        <button class:active={axis === 'y'} on:click={() => (axis = 'y')}>Y</button>
        <button class:active={axis === 'z'} on:click={() => (axis = 'z')}>Z</button>
      </div>

      <div class="chart-grid">
        <ChartCanvas config={xyzMinMaxChart} />
        <ChartCanvas config={rangeChart} />
        <ChartCanvas config={pointCountChart} />
        <ChartCanvas config={densityChart} />
        <ChartCanvas config={safetyChart} />
        <ChartCanvas config={geometryRadiusChart} />
        <ChartCanvas config={attributeRadiusChart} />
        <ChartCanvas config={stabilityChart} />
      </div>
    </section>

    <section class="card">
      <div class="section-header">
        <div>
          <p class="eyebrow micro">Export</p>
          <h2>Computed results</h2>
        </div>
        <div class="actions-row compact">
          <button on:click={exportCsv}>Export CSV</button>
          <button on:click={exportJson}>Export JSON</button>
        </div>
      </div>

      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>{viewMode === 'segments' ? 'Segment' : 'Frame'}</th>
              <th>Points</th>
              <th>Density</th>
              <th>Geometry QP</th>
              <th>Attribute QP</th>
              <th>Geometry stability</th>
              <th>Color entropy inv.</th>
            </tr>
          </thead>
          <tbody>
            {#each rows.slice(0, 120) as row}
              <tr>
                <td>{rowLabel(row)}</td>
                <td>{formatNumber(nullableValue(row, 'point_count'))}</td>
                <td>{formatNumber(nullableValue(row, 'density'))}</td>
                <td>{formatScore(nullableValue(row, 'geometry_qp_safety'))}</td>
                <td>{formatScore(nullableValue(row, 'attribute_qp_safety'))}</td>
                <td>{formatScore(nullableValue(row, 'geometry_stability'))}</td>
                <td>{formatScore(nullableValue(row, 'inverse_color_entropy'))}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>
  {/if}

  {#if warnings.length}
    <section class="card">
      <div class="section-header">
        <div>
          <p class="eyebrow micro">Warnings</p>
          <h2>Skipped or partial data</h2>
        </div>
        <span class="pill">{warnings.length}</span>
      </div>
      <div class="warning-list">
        {#each warnings as warning}
          <div>
            <strong>{warning.kind}</strong>
            <span>{warning.message}</span>
            {#if warning.path}<code>{warning.path}</code>{/if}
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
    grid-template-columns: minmax(240px, 1fr) repeat(auto-fit, minmax(170px, auto));
    gap: 0.9rem;
    align-items: end;
  }

  .column-grid {
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    margin-top: 1rem;
  }

  .check-row {
    min-height: 2.55rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .stats-grid,
  .score-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.75rem;
    margin-top: 0.85rem;
  }

  .stats-grid > div,
  .score-grid > div {
    border: 1px solid var(--card-border);
    border-radius: 0.5rem;
    padding: 0.85rem;
    background: var(--soft-bg-alt);
  }

  .stats-grid span,
  .score-grid span {
    display: block;
    color: var(--text-muted);
    font-size: 0.78rem;
    margin-bottom: 0.25rem;
  }

  .stats-grid strong,
  .score-grid strong {
    color: var(--text-main);
    font-size: 0.98rem;
  }

  .actions-row,
  .axis-row,
  .toggle-group {
    display: flex;
    gap: 0.55rem;
    flex-wrap: wrap;
    align-items: center;
  }

  .actions-row {
    margin-top: 1rem;
  }

  .actions-row.compact {
    margin-top: 0;
  }

  .axis-row {
    margin: 0.7rem 0 0.9rem;
  }

  .axis-row span {
    color: var(--text-muted);
    font-size: 0.86rem;
  }

  .axis-row button,
  .toggle-group button {
    min-width: 3.2rem;
    padding-inline: 0.75rem;
  }

  .axis-row button.active,
  .toggle-group button.active {
    background: var(--accent-main);
    border-color: var(--accent-main);
    color: white;
  }

  .chart-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(390px, 1fr));
    gap: 1rem;
  }

  .table-wrap {
    overflow: auto;
    margin-top: 0.5rem;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    min-width: 780px;
  }

  th,
  td {
    text-align: left;
    border-bottom: 1px solid var(--card-border);
    padding: 0.6rem 0.5rem;
    white-space: nowrap;
  }

  th {
    color: var(--text-muted);
    font-size: 0.78rem;
    font-weight: 700;
  }

  .warning-list {
    display: grid;
    gap: 0.65rem;
  }

  .warning-list > div {
    display: grid;
    gap: 0.2rem;
    border: 1px solid rgba(251, 191, 36, 0.28);
    border-radius: 0.5rem;
    padding: 0.75rem;
    background: rgba(251, 191, 36, 0.08);
  }

  .warning-list span {
    color: var(--text-muted);
  }

  @media (max-width: 760px) {
    .scan-grid,
    .column-grid,
    .chart-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
