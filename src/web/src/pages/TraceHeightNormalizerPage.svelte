<script lang="ts">
  import { discoverTraceHeightColumns, inspectTraceHeightPly, transformDataset } from '../core/api';
  import { gotoRoute } from '../core/router';
  import type {
    DatasetExportMode,
    DatasetTransformationResponse,
    ExtraToolWarning,
    PlyInspectionResponse,
    SpeedAccelerationDiscoverResponse,
  } from '../core/types';

  let traceRootFolder = '';
  let plyInputPath = '';
  let outputFolder = '';
  let discovery: SpeedAccelerationDiscoverResponse | null = null;
  let result: DatasetTransformationResponse | null = null;
  let xColumn = '';
  let yColumn = '';
  let zColumn = '';
  let referencePlyPath = '';
  let plyInspection: PlyInspectionResponse | null = null;
  let heightAxis: 'x' | 'y' | 'z' = 'y';
  let outputHeightAxis: 'x' | 'y' | 'z' = 'y';
  let realHeightM = 2;
  let transformTraces = true;
  let transformPointClouds = false;
  let heightAxisStartsAtZero = true;
  let positiveCoordinates = false;
  let exportMode: DatasetExportMode = 'scaled';
  let voxelBits = 10;
  let workerCount = 4;
  let fovResampleEnabled = true;
  let fovResampleTarget: number | null = null;
  let fovResampleManual = false;
  let loading = false;
  let phase = 'Waiting for inputs.';
  let error = '';

  $: warnings = combinedWarnings(discovery?.warnings || [], result?.warnings || []);
  $: if (!transformPointClouds && exportMode !== 'scaled') exportMode = 'scaled';
  $: axisSwapEnabled = heightAxis !== outputHeightAxis;
  $: swappedInspectionExtents = plyInspection ? extentsAfterSwap(plyInspection.extents, heightAxis, outputHeightAxis) : null;
  $: selectedExtent = plyInspection ? plyInspection.extents[heightAxis] : null;
  $: scaleFactor = selectedExtent && realHeightM > 0 ? realHeightM / selectedExtent : null;
  $: effectiveReferencePath = showPlyControls ? (referencePlyPath.trim() || plyInputPath.trim()) : plyInputPath.trim();
  $: needsVoxelBits = exportMode === 'voxelized' || exportMode === 'both';
  $: voxelizationEnabled = transformPointClouds && needsVoxelBits;
  $: showTraceControls = transformTraces;
  $: showPlyControls = transformPointClouds;
  $: inferredFovResampleTarget = Number(plyInspection?.reference?.frame_count || 0) > 1
    ? Number(plyInspection?.reference?.frame_count)
    : null;
  $: if (fovResampleEnabled && !fovResampleManual && inferredFovResampleTarget && fovResampleTarget !== inferredFovResampleTarget) {
    fovResampleTarget = inferredFovResampleTarget;
  }
  $: showShiftSummary = positiveCoordinates || Boolean(result?.scaling.positive_coordinates);
  $: exportLabel = describeExportMode(exportMode, axisSwapEnabled, positiveCoordinates, transformTraces, transformPointClouds);
  $: canRun = !loading
    && Boolean(outputFolder.trim())
    && Boolean(effectiveReferencePath)
    && realHeightM > 0
    && (transformTraces || transformPointClouds)
    && (!transformTraces || Boolean(discovery && xColumn && yColumn && zColumn && traceRootFolder.trim()))
    && (!transformPointClouds || Boolean(plyInputPath.trim()))
    && (!voxelizationEnabled || voxelBits > 0)
    && workerCount > 0;

  function combinedWarnings(first: ExtraToolWarning[], second: ExtraToolWarning[]): ExtraToolWarning[] {
    return [...first, ...second];
  }

  function formatNumber(value: number | null | undefined): string {
    if (value === null || value === undefined) return 'No data';
    if (value !== 0 && (Math.abs(value) >= 1000 || Math.abs(value) < 0.01)) return value.toExponential(3);
    return value.toLocaleString(undefined, { maximumFractionDigits: 5 });
  }

  function pickDefaultColumn(columns: string[], matcher: RegExp): string {
    return columns.find((column) => matcher.test(column)) || columns[0] || '';
  }

  function availableColumns(): string[] {
    return discovery?.numeric_columns?.length ? discovery.numeric_columns : discovery?.columns || [];
  }

  function extentsAfterSwap(
    extents: Record<'x' | 'y' | 'z', number>,
    sourceAxis: 'x' | 'y' | 'z',
    outputAxis: 'x' | 'y' | 'z',
  ): Record<'x' | 'y' | 'z', number> {
    const next = { ...extents };
    if (sourceAxis !== outputAxis) {
      const previous = next[outputAxis];
      next[outputAxis] = next[sourceAxis];
      next[sourceAxis] = previous;
    }
    return next;
  }

  function describeExportMode(
    mode: DatasetExportMode | string,
    swapped: boolean,
    shifted: boolean,
    tracesEnabled: boolean,
    plyEnabled: boolean,
  ): string {
    const targets = plyEnabled && tracesEnabled ? 'PLY + traces' : plyEnabled ? 'PLY' : 'traces';
    if (mode === 'voxelized') return `voxelized ${targets}`;
    if (mode === 'both') return `${swapped ? 'transformed' : 'scaled'} + voxelized ${targets}`;
    const base = swapped ? 'transformed' : 'scaled';
    return `${base}${shifted ? ' + shifted' : ''} ${targets}`;
  }

  function resultExportLabel(value: DatasetTransformationResponse): string {
    return value.export_label || describeExportMode(
      value.export_mode,
      Boolean(value.scaling.axis_swap.enabled),
      Boolean(value.scaling.positive_coordinates),
      Boolean(value.traces || value.traces_voxelized),
      Boolean(value.point_clouds || value.point_clouds_voxelized),
    );
  }

  function axisSwapLabel(sourceAxis: 'x' | 'y' | 'z', outputAxis: 'x' | 'y' | 'z'): string {
    return sourceAxis === outputAxis ? 'None' : `${sourceAxis.toUpperCase()} <-> ${outputAxis.toUpperCase()}`;
  }

  async function discoverColumns(): Promise<void> {
    loading = true;
    error = '';
    result = null;
    discovery = null;
    phase = 'Scanning folders for CSV files...';
    try {
      const payload = await discoverTraceHeightColumns(traceRootFolder);
      discovery = payload;
      const columns = payload.numeric_columns?.length ? payload.numeric_columns : payload.columns;
      xColumn = pickDefaultColumn(columns, /(^|[_\s-])(x|pos_x|position_x|x_position|hmdpx)($|[_\s-])/i);
      yColumn = pickDefaultColumn(columns, /(^|[_\s-])(y|pos_y|position_y|y_position|hmdpy)($|[_\s-])/i);
      zColumn = pickDefaultColumn(columns, /(^|[_\s-])(z|pos_z|position_z|z_position|hmdpz)($|[_\s-])/i);
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

  async function inspectPly(): Promise<void> {
    loading = true;
    error = '';
    result = null;
    plyInspection = null;
    phase = 'Inspecting reference PLY point cloud...';
    try {
      plyInspection = await inspectTraceHeightPly(effectiveReferencePath);
      const reference = plyInspection.reference?.mode === 'sequence'
        ? `first sequence frame ${plyInspection.reference.reference_frame_number}`
        : plyInspection.filename;
      phase = `Loaded ${plyInspection.point_count.toLocaleString()} PLY points from ${reference}.`;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      phase = 'PLY inspection failed.';
    } finally {
      loading = false;
    }
  }

  async function runTransformation(): Promise<void> {
    loading = true;
    error = '';
    result = null;
      phase = 'Scaling dataset coordinates and writing transformed outputs...';
    try {
      result = await transformDataset({
        traceRootFolder,
        plyInputPath,
        outputFolder,
        xColumn,
        yColumn,
        zColumn,
        referencePlyPath: effectiveReferencePath,
        heightAxis,
        outputHeightAxis,
        realHeightM,
        transformTraces,
        transformPointClouds,
        heightAxisStartsAtZero,
        positiveCoordinates,
        voxelBits: voxelizationEnabled ? Number(voxelBits) : null,
        exportMode,
        workerCount: Number(workerCount || 1),
        fovResampleEnabled,
        fovResampleTarget: fovResampleEnabled && fovResampleTarget ? Number(fovResampleTarget) : null,
      });
      phase = `Exported ${result.summary.trace_outputs.toLocaleString()} trace file${result.summary.trace_outputs === 1 ? '' : 's'} and ${result.summary.point_cloud_outputs.toLocaleString()} PLY frame${result.summary.point_cloud_outputs === 1 ? '' : 's'}.`;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      phase = 'Dataset transformation failed.';
    } finally {
      loading = false;
    }
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

    <div class="filter-grid">
      <label class="pill-check">
        <input type="checkbox" bind:checked={transformTraces} disabled={loading} />
        Transform FoV traces
      </label>
      <label class="pill-check">
        <input type="checkbox" bind:checked={transformPointClouds} disabled={loading} />
        Transform PLY sequence
      </label>
      <label class="pill-check">
        <input type="checkbox" bind:checked={positiveCoordinates} disabled={loading} />
        Shift to positive coordinates
      </label>
    </div>

    {#if showTraceControls}
      <div class="scan-grid">
        <label>Trace root folder
          <input bind:value={traceRootFolder} placeholder="data/fov_traces or /absolute/path/to/csvs" disabled={loading} />
        </label>
        <button on:click={discoverColumns} disabled={loading || !traceRootFolder.trim()}>Scan CSV files</button>
      </div>
    {/if}

    <div class="callout" class:warn={Boolean(error)}>
      <div class="callout-title">{error ? 'Error' : 'Progress'}</div>
      <div class="callout-body">{error || phase}</div>
    </div>
  </section>

  {#if transformTraces || transformPointClouds}
    <section class="card">
      <div class="section-header">
        <div>
          <p class="eyebrow micro">Transform setup</p>
          <h2>Scale and export</h2>
        </div>
        {#if discovery}
          <span class="pill">{discovery.files_loaded} loaded</span>
        {:else if transformPointClouds && !transformTraces}
          <span class="pill">PLY only</span>
        {:else}
          <span class="pill">Ready</span>
        {/if}
      </div>

      {#if discovery}
        <div class="summary-grid scan-summary">
          <div><span>Total files found</span><strong>{discovery.file_count.toLocaleString()}</strong></div>
          <div><span>Successfully loaded</span><strong>{discovery.files_loaded.toLocaleString()}</strong></div>
          <div><span>Skipped or invalid</span><strong>{discovery.files_skipped.toLocaleString()}</strong></div>
        </div>
      {/if}

      {#if showTraceControls && discovery}
        <div class="subsection">
          <h3>Trace columns</h3>
          <div class="column-grid">
            <label>X position column
              <select bind:value={xColumn} disabled={loading || !availableColumns().length}>
                {#each availableColumns() as column}<option value={column}>{column}</option>{/each}
              </select>
            </label>
            <label>Y position column
              <select bind:value={yColumn} disabled={loading || !availableColumns().length}>
                {#each availableColumns() as column}<option value={column}>{column}</option>{/each}
              </select>
            </label>
            <label>Z position column
              <select bind:value={zColumn} disabled={loading || !availableColumns().length}>
                {#each availableColumns() as column}<option value={column}>{column}</option>{/each}
              </select>
            </label>
          </div>
          <div class="filter-grid">
            <label class="pill-check helper-check">
              <input type="checkbox" bind:checked={fovResampleEnabled} disabled={loading} />
              <span>
                <strong>Resample FoV CSV files</strong>
                <small>Write one temporally aligned FoV row per target sample.</small>
              </span>
            </label>
            <label>FoV CSV FPS / resampling target
              <input
                type="number"
                min="1"
                step="1"
                bind:value={fovResampleTarget}
                disabled={loading || !fovResampleEnabled}
                on:input={() => (fovResampleManual = true)}
              />
              <span class="field-help">
                {inferredFovResampleTarget
                  ? `Inferred ${inferredFovResampleTarget.toLocaleString()} from the PLY sequence; override to write a different number of FoV rows per CSV.`
                  : 'Inspect a PLY sequence folder to infer this from its frame count, or enter a manual target.'}
              </span>
            </label>
          </div>
        </div>
      {/if}

      <div class="subsection">
        <h3>Reference scale</h3>
        <div class="scaling-grid">
          <label>{showPlyControls ? 'PLY sequence input' : 'Reference PLY point cloud'}
            <input bind:value={plyInputPath} placeholder="/absolute/path/reference.ply or /absolute/path/ply_sequence" disabled={loading} />
          </label>
          <button class="ghost" on:click={inspectPly} disabled={loading || !effectiveReferencePath}>Inspect PLY</button>
          <label class="desired-height-field">Desired height (m)
            <input type="number" min="0.0001" step="0.001" bind:value={realHeightM} disabled={loading} />
          </label>
          <label>Original height axis
            <select bind:value={heightAxis} disabled={loading || !plyInspection}>
              <option value="x">Original X extent {plyInspection ? `(${formatNumber(plyInspection.extents.x)} units)` : ''}</option>
              <option value="y">Original Y extent {plyInspection ? `(${formatNumber(plyInspection.extents.y)} units)` : ''}</option>
              <option value="z">Original Z extent {plyInspection ? `(${formatNumber(plyInspection.extents.z)} units)` : ''}</option>
            </select>
          </label>
          <div class="stat-cell">
            <span>Scale factor</span>
            <strong>{scaleFactor ? `${formatNumber(scaleFactor)} m/unit` : 'No data'}</strong>
          </div>
        </div>
        <label class="pill-check helper-check">
          <input type="checkbox" bind:checked={heightAxisStartsAtZero} disabled={loading} />
          <span>
            <strong>Height axis starts at 0</strong>
            <small>After scaling, shift only the selected height axis so its minimum value is 0.</small>
          </span>
        </label>
      </div>

      <div class="subsection">
        <h3>Coordinate transform</h3>
        <div class="filter-grid">
          <label>New height axis
            <select bind:value={outputHeightAxis} disabled={loading}>
              <option value="x">X</option>
              <option value="y">Y</option>
              <option value="z">Z</option>
            </select>
          </label>
          <div class="stat-cell">
            <span>Export mode</span>
            <strong>{exportLabel}</strong>
          </div>
          {#if showShiftSummary}
            <div class="stat-cell">
              <span>Shift to positive</span>
              <strong>{positiveCoordinates ? 'Enabled' : 'Disabled'}</strong>
            </div>
          {/if}
        </div>
      </div>

      {#if showPlyControls}
        <div class="subsection">
          <h3>PLY export</h3>
          <div class="filter-grid">
            <label>Geometry export
              <select bind:value={exportMode} disabled={loading}>
                <option value="scaled">Scaled data</option>
                <option value="voxelized">Voxelized data</option>
                <option value="both">Scaled and voxelized</option>
              </select>
            </label>
            <label>PLY workers
              <input type="number" min="1" step="1" bind:value={workerCount} disabled={loading} />
            </label>
            <label>Reference override
              <input bind:value={referencePlyPath} placeholder="Optional reference PLY or sequence folder" disabled={loading} />
            </label>
            {#if voxelizationEnabled}
              <label>Geometry bit depth
                <select bind:value={voxelBits} disabled={loading}>
                  <option value={9}>vox9</option>
                  <option value={10}>vox10</option>
                  <option value={11}>vox11</option>
                  <option value={12}>vox12</option>
                  <option value={13}>vox13</option>
                  <option value={14}>vox14</option>
                </select>
              </label>
              <div class="stat-cell">
                <span>Voxel range</span>
                <strong>[0, {(2 ** Number(voxelBits || 0)) - 1}]</strong>
              </div>
            {/if}
          </div>
        </div>
      {/if}

      {#if plyInspection}
        <section class="callout info scaling-summary">
          <div class="callout-title">Scale Conversion</div>
          <div class="summary-grid">
            <div><span>1 raw unit</span><strong>{scaleFactor ? `${formatNumber(scaleFactor)} m` : 'No data'}</strong></div>
            <div><span>1000 raw units</span><strong>{scaleFactor ? `${formatNumber(scaleFactor * 1000)} m` : 'No data'}</strong></div>
            <div><span>Axis swap</span><strong>{axisSwapLabel(heightAxis, outputHeightAxis)}</strong></div>
            <div><span>Original height axis</span><strong>{heightAxis.toUpperCase()}</strong></div>
            <div><span>New height axis</span><strong>{outputHeightAxis.toUpperCase()}</strong></div>
            <div><span>Measured original extent</span><strong>{selectedExtent ? `${formatNumber(selectedExtent)} units` : 'No data'}</strong></div>
            <div><span>Desired height</span><strong>{formatNumber(realHeightM)} m</strong></div>
            <div><span>Scale factor</span><strong>{scaleFactor ? `${formatNumber(scaleFactor)} m/unit` : 'No data'}</strong></div>
            <div><span>Height axis starts at 0</span><strong>{heightAxisStartsAtZero ? 'Enabled' : 'Disabled'}</strong></div>
          </div>
          <details class="details-panel">
            <summary>Details</summary>
            <div class="summary-grid">
              <div><span>Reference mode</span><strong>{plyInspection.reference?.mode || 'file'}</strong></div>
              <div><span>Filename</span><strong>{plyInspection.filename}</strong></div>
              <div><span>Points</span><strong>{plyInspection.point_count.toLocaleString()}</strong></div>
              <div><span>Original X extent</span><strong>{formatNumber(plyInspection.extents.x)} units</strong></div>
              <div><span>Original Y extent</span><strong>{formatNumber(plyInspection.extents.y)} units</strong></div>
              <div><span>Original Z extent</span><strong>{formatNumber(plyInspection.extents.z)} units</strong></div>
              {#if swappedInspectionExtents}
                <div><span>Post-swap X extent</span><strong>{formatNumber(swappedInspectionExtents.x)} units</strong></div>
                <div><span>Post-swap Y extent</span><strong>{formatNumber(swappedInspectionExtents.y)} units</strong></div>
                <div><span>Post-swap Z extent</span><strong>{formatNumber(swappedInspectionExtents.z)} units</strong></div>
              {/if}
            </div>
          </details>
        </section>
      {/if}

      <div class="output-grid">
        <label>Output directory
          <input bind:value={outputFolder} placeholder="/absolute/path/output_dataset" disabled={loading} />
        </label>
      </div>

      <div class="form-footer">
        <button on:click={runTransformation} disabled={!canRun}>Transform dataset</button>
      </div>

      {#if discovery}
        <div class="chip-row">
          {#each availableColumns() as column}
            <span class="pill">{column}</span>
          {/each}
        </div>
      {/if}
    </section>
  {/if}

  {#if result}
    <section class="dashboard-grid">
      <article class="card metric-card">
        <p class="eyebrow micro">Transform</p>
        <h2>Global transform</h2>
        <div class="summary-grid">
          <div><span>Original height axis</span><strong>{result.scaling.axis_swap.source_height_axis.toUpperCase()}</strong></div>
          <div><span>New height axis</span><strong>{result.scaling.axis_swap.output_height_axis.toUpperCase()}</strong></div>
          <div><span>Axis swap</span><strong>{result.scaling.axis_swap.enabled ? `${result.scaling.axis_swap.source_height_axis.toUpperCase()} <-> ${result.scaling.axis_swap.output_height_axis.toUpperCase()}` : 'None'}</strong></div>
          <div><span>Scale factor</span><strong>{formatNumber(result.scaling.scale_factor_m_per_unit)} m/unit</strong></div>
          <div><span>Height axis starts at 0</span><strong>{result.scaling.height_axis_starts_at_zero ? 'Enabled' : 'Disabled'}</strong></div>
          {#if result.scaling.positive_coordinates}
            <div><span>Positive shift</span><strong>Enabled</strong></div>
          {/if}
          {#if result.scaling.voxelization.enabled}
            <div><span>Voxelization</span><strong>vox{result.scaling.voxelization.geometry_bit_depth}</strong></div>
          {/if}
          <div><span>Output folder</span><strong>{result.output_folder}</strong></div>
        </div>
        {#if result.scaling.debug_details}
          <details class="details-panel">
            <summary>Debug details</summary>
            <div class="summary-grid">
              <div><span>Original X extent</span><strong>{formatNumber(result.scaling.debug_details.original_reference_extents_units.x)} units</strong></div>
              <div><span>Original Y extent</span><strong>{formatNumber(result.scaling.debug_details.original_reference_extents_units.y)} units</strong></div>
              <div><span>Original Z extent</span><strong>{formatNumber(result.scaling.debug_details.original_reference_extents_units.z)} units</strong></div>
              <div><span>Post-swap X extent</span><strong>{formatNumber(result.scaling.debug_details.swapped_reference_extents_units.x)} units</strong></div>
              <div><span>Post-swap Y extent</span><strong>{formatNumber(result.scaling.debug_details.swapped_reference_extents_units.y)} units</strong></div>
              <div><span>Post-swap Z extent</span><strong>{formatNumber(result.scaling.debug_details.swapped_reference_extents_units.z)} units</strong></div>
              <div><span>Measured height extent</span><strong>{formatNumber(result.scaling.debug_details.measured_height_extent_units)} units</strong></div>
              <div><span>Desired height</span><strong>{formatNumber(result.scaling.debug_details.desired_real_world_height_m ?? result.scaling.debug_details.known_real_world_height_m)} m</strong></div>
              <div><span>Computed scale factor</span><strong>{formatNumber(result.scaling.debug_details.computed_scale_factor_m_per_unit)} m/unit</strong></div>
              <div><span>Height axis minimum before zero-shift</span><strong>{formatNumber(result.scaling.debug_details.height_axis_minimum_before_zero_shift_m)} m</strong></div>
              <div><span>Height axis zero-shift offset</span><strong>{formatNumber(result.scaling.debug_details.height_axis_zero_shift_offset_m)} m</strong></div>
              <div><span>Height axis minimum after zero-shift</span><strong>{formatNumber(result.scaling.debug_details.height_axis_minimum_after_zero_shift_m)} m</strong></div>
              <div><span>Final X extent</span><strong>{formatNumber(result.scaling.debug_details.final_exported_extents_m.x)} m</strong></div>
              <div><span>Final Y extent</span><strong>{formatNumber(result.scaling.debug_details.final_exported_extents_m.y)} m</strong></div>
              <div><span>Final Z extent</span><strong>{formatNumber(result.scaling.debug_details.final_exported_extents_m.z)} m</strong></div>
            </div>
          </details>
        {/if}
      </article>

      <article class="card metric-card">
        <p class="eyebrow micro">Outputs</p>
        <h2>Export summary</h2>
        <div class="summary-grid">
          {#if result.traces || result.traces_voxelized}
            <div><span>Trace files</span><strong>{result.summary.trace_outputs.toLocaleString()}</strong></div>
            {#if result.trace_resampling?.enabled}
              <div><span>FoV CSV FPS / resampling target</span><strong>{Number(result.trace_resampling.target_samples || 0).toLocaleString()}</strong></div>
            {/if}
          {/if}
          {#if result.point_clouds || result.point_clouds_voxelized}
            <div><span>PLY frames</span><strong>{result.summary.point_cloud_outputs.toLocaleString()}</strong></div>
            <div><span>Workers</span><strong>{result.summary.worker_count.toLocaleString()}</strong></div>
          {/if}
          <div><span>Export mode</span><strong>{resultExportLabel(result)}</strong></div>
          <div><span>Axis swap</span><strong>{result.scaling.axis_swap.enabled ? `${result.scaling.axis_swap.source_height_axis.toUpperCase()} <-> ${result.scaling.axis_swap.output_height_axis.toUpperCase()}` : 'None'}</strong></div>
          <div><span>Scale factor</span><strong>{formatNumber(result.scaling.scale_factor_m_per_unit)} m/unit</strong></div>
          <div><span>Height axis starts at 0</span><strong>{result.scaling.height_axis_starts_at_zero ? 'Enabled' : 'Disabled'}</strong></div>
          {#if result.scaling.positive_coordinates}
            <div><span>Shift to positive</span><strong>Enabled</strong></div>
          {/if}
          {#if result.scaling.voxelization.enabled}
            <div><span>Voxelization</span><strong>Enabled</strong></div>
            <div><span>Geometry bit depth</span><strong>vox{result.scaling.voxelization.geometry_bit_depth}</strong></div>
          {/if}
        </div>
      </article>
    </section>

    <section class="card">
      <div class="section-header">
        <div>
          <p class="eyebrow micro">Details</p>
          <h2>Written outputs</h2>
        </div>
      </div>
      <div class="summary-grid">
        {#if result.traces}
          <div><span>{result.scaling.axis_swap.enabled ? 'Transformed traces' : 'Scaled traces'}</span><strong>{result.traces.summary.output_files.toLocaleString()} files</strong></div>
        {/if}
        {#if result.traces_voxelized}
          <div><span>Voxelized traces</span><strong>{result.traces_voxelized.summary.output_files.toLocaleString()} files</strong></div>
        {/if}
        {#if result.point_clouds}
          <div><span>{result.scaling.axis_swap.enabled ? 'Transformed PLY' : 'Scaled PLY'}</span><strong>{result.point_clouds.summary.frames_processed.toLocaleString()} frames</strong></div>
        {/if}
        {#if result.point_clouds_voxelized}
          <div><span>Voxelized PLY</span><strong>{result.point_clouds_voxelized.summary.frames_processed.toLocaleString()} frames</strong></div>
        {/if}
      </div>
    </section>
  {/if}

  {#if warnings.length}
    <section class="card">
      <div class="section-header">
        <div>
          <p class="eyebrow micro">Warnings</p>
          <h2>Skipped files, rows, and points</h2>
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
  main {
    display: grid;
    gap: 1rem;
  }

  .form-card,
  .card {
    display: grid;
    gap: 1rem;
  }

  .section-header {
    gap: 1rem;
  }

  .scan-grid,
  .column-grid,
  .scaling-grid,
  .filter-grid,
  .output-grid {
    display: grid;
    gap: 0.9rem;
    align-items: end;
  }

  .scan-grid {
    grid-template-columns: minmax(280px, 1fr) auto;
  }

  .column-grid {
    grid-template-columns: repeat(3, minmax(180px, 1fr));
  }

  .scaling-grid {
    grid-template-columns: minmax(280px, 1.2fr) auto minmax(118px, 0.35fr) minmax(210px, 0.85fr) minmax(170px, 0.65fr);
  }

  .filter-grid {
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  }

  .output-grid {
    grid-template-columns: minmax(280px, 1fr);
  }

  .subsection {
    display: grid;
    gap: 0.75rem;
    padding-top: 0.95rem;
    border-top: 1px solid var(--card-border);
  }

  .subsection h3 {
    margin: 0;
    color: var(--text-main);
    font-size: 0.98rem;
  }

  label {
    display: grid;
    gap: 0.42rem;
    color: var(--text-muted);
    font-weight: 700;
  }

  label.pill-check {
    display: flex;
    gap: 0.55rem;
    align-items: center;
    min-height: 42px;
  }

  .helper-check {
    justify-content: flex-start;
    align-items: flex-start;
    padding: 0.8rem 0.9rem;
  }

  .helper-check span {
    display: grid;
    gap: 0.2rem;
  }

  .helper-check small {
    color: var(--text-subtle);
    font-weight: 600;
    line-height: 1.35;
  }

  .field-help {
    color: var(--text-subtle);
    font-size: 0.83rem;
    font-weight: 600;
    line-height: 1.35;
  }

  .desired-height-field {
    max-width: 9rem;
  }

  .desired-height-field input {
    width: 100%;
    min-width: 0;
  }

  .pill {
    margin-left: 0.35rem;
  }

  .callout {
    padding: 0.95rem 1rem;
  }

  .scaling-summary,
  .details-panel {
    margin-top: 0.25rem;
  }

  .details-panel summary {
    cursor: pointer;
    color: var(--text-main);
    font-weight: 800;
    padding: 0.35rem 0;
  }

  .details-panel .summary-grid {
    margin-top: 0.7rem;
  }

  .dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 1rem;
  }

  .metric-card {
    align-content: start;
  }

  .metric-card h2 {
    margin: 0;
    font-size: 1.05rem;
  }

  .summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
    gap: 0.75rem;
  }

  .stat-cell,
  .summary-grid > div {
    display: grid;
    gap: 0.35rem;
    min-width: 0;
    padding: 0.8rem 0.85rem;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    background: var(--soft-bg-alt);
  }

  .stat-cell span,
  .summary-grid span {
    color: var(--text-subtle);
    font-size: 0.82rem;
  }

  .stat-cell strong,
  .summary-grid strong {
    overflow-wrap: anywhere;
    color: var(--text-main);
    font-size: 1rem;
  }

  .chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.25rem;
  }

  .status-list {
    display: grid;
    gap: 0.7rem;
    max-height: 360px;
    overflow: auto;
    padding: 0.75rem;
  }

  .status-row {
    padding: 0.85rem 0.9rem;
  }

  button:disabled,
  input:disabled,
  select:disabled {
    opacity: 0.58;
    cursor: not-allowed;
  }

  @media (max-width: 860px) {
    .scan-grid,
    .column-grid,
    .scaling-grid,
    .filter-grid,
    .dashboard-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
