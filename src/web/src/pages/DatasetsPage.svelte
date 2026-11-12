<script lang="ts">
  import { onMount } from 'svelte';
  import { deleteDataset, getDatasets, getOptions, saveDataset } from '../core/api';
  import { gotoRoute } from '../core/router';
  import type { DatasetProfile, DatasetScaleMode } from '../core/types';

  let datasets: DatasetProfile[] = [];
  let angleConventionChoices: Array<{ id: string; label: string }> = [];
  let loading = true;
  let saving = false;
  let statusMessage = '';
  let error = '';

  let id = '';
  let name = '';
  let mpdFolder = '';
  let contentDir = '';
  let scaleMode: DatasetScaleMode = 'known_extent_mpd';
  let dimension: 'height' | 'width' | 'depth' = 'height';
  let axis: 'x' | 'y' | 'z' = 'y';
  let realExtentM = 2;
  let sourceExtentUnits = 1;
  let manualScale = 1;
  let angleConvention = 'unity_zxy';
  let maxSpeed = 1.5;
  let maxAcceleration = 3.0;
  let applyToFovTrace = true;
  let translationEnabled = false;
  let translationX = 0;
  let translationY = 0;
  let translationZ = 0;
  let notes = '';

  $: alreadyNormalized = scaleMode === 'already_normalized';

  function resetForm(): void {
    id = '';
    name = '';
    mpdFolder = '';
    contentDir = '';
    scaleMode = 'known_extent_mpd';
    dimension = 'height';
    axis = 'y';
    realExtentM = 2;
    sourceExtentUnits = 1;
    manualScale = 1;
    angleConvention = angleConventionChoices[0]?.id || 'unity_zxy';
    maxSpeed = 1.5;
    maxAcceleration = 3.0;
    applyToFovTrace = true;
    translationEnabled = false;
    translationX = 0;
    translationY = 0;
    translationZ = 0;
    notes = '';
  }

  async function refreshDatasets(): Promise<void> {
    try {
      const payload = await getDatasets();
      datasets = payload.datasets || [];
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  function editDataset(dataset: DatasetProfile): void {
    id = dataset.id;
    name = dataset.name;
    mpdFolder = dataset.mpd_folder || '';
    contentDir = dataset.content_dir || '';
    scaleMode = dataset.already_normalized === true ? 'already_normalized' : ((dataset.scale_mode as DatasetScaleMode) || 'manual');
    dimension = ((dataset.calibration?.dimension as typeof dimension) || 'height');
    axis = ((dataset.calibration?.axis as typeof axis) || 'y');
    realExtentM = Number(dataset.calibration?.real_extent_m ?? 2);
    sourceExtentUnits = Number(dataset.calibration?.source_extent_units ?? 1);
    manualScale = Number(dataset.transform?.scale ?? 1);
    angleConvention = dataset.angle_convention || angleConventionChoices[0]?.id || 'unity_zxy';
    maxSpeed = Number(dataset.motion_limits?.max_speed ?? 1.5);
    maxAcceleration = Number(dataset.motion_limits?.max_acceleration ?? 3.0);
    applyToFovTrace = dataset.apply_to_fov_trace !== false;
    const t = dataset.transform?.translation || [0, 0, 0];
    translationEnabled = Boolean(t.some((value) => Number(value) !== 0));
    translationX = Number(t[0] ?? 0);
    translationY = Number(t[1] ?? 0);
    translationZ = Number(t[2] ?? 0);
    notes = dataset.notes || '';
  }

  async function saveCurrentDataset(): Promise<void> {
    saving = true;
    error = '';
    statusMessage = '';
    try {
      const payload = {
        id,
        name,
        mpd_folder: mpdFolder,
        content_dir: contentDir,
        scale_mode: scaleMode,
        scale: manualScale,
        angle_convention: angleConvention,
        calibration: {
          dimension,
          axis,
          real_extent_m: realExtentM,
          source_extent_units: sourceExtentUnits,
        },
        motion_limits: {
          max_speed: maxSpeed,
          max_acceleration: maxAcceleration,
        },
        translation: translationEnabled ? [translationX, translationY, translationZ] : [0, 0, 0],
        apply_to_fov_trace: applyToFovTrace,
        already_normalized: scaleMode === 'already_normalized',
        update_existing: Boolean(id),
        notes,
      };
      const response = await saveDataset(payload);
      datasets = response.datasets || [];
      statusMessage = 'Dataset saved.';
      resetForm();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      saving = false;
    }
  }

  async function removeDataset(dataset: DatasetProfile): Promise<void> {
    if (!window.confirm(`Delete dataset "${dataset.name}"?`)) return;
    try {
      const response = await deleteDataset(dataset.id);
      datasets = response.datasets || [];
      if (id === dataset.id) resetForm();
      statusMessage = 'Dataset deleted.';
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  function computedScaleText(): string {
    if (alreadyNormalized) return '1';
    if (scaleMode === 'already_meters') return '1';
    if (scaleMode === 'manual') return Number(manualScale || 0).toPrecision(5);
    const source = scaleMode === 'known_extent_manual' ? Number(sourceExtentUnits || 0) : NaN;
    if (scaleMode === 'known_extent_manual' && source > 0) {
      return (Number(realExtentM || 0) / source).toPrecision(5);
    }
    return 'computed from MPD tile bounds on save';
  }

  function scaleModeLabel(mode: string | undefined): string {
    if (mode === 'already_meters') return 'Already in meters';
    if (mode === 'already_normalized') return 'Already normalized';
    if (mode === 'manual') return 'Enter scale manually';
    if (mode === 'known_extent_manual') return 'Real size + source range';
    return 'Real size + MPD tile extents';
  }

  function scaleModeHelp(): string {
    if (scaleMode === 'already_normalized') return 'Use this when the Dataset Transformation Tool has already scaled, shifted, and aligned the dataset.';
    if (scaleMode === 'already_meters') return 'Use this when PLY points, tile bounds, and FoV traces are already expressed in meters.';
    if (scaleMode === 'manual') return 'Use this when you already know the multiplier from dataset units to meters.';
    if (scaleMode === 'known_extent_manual') return 'Enter the real-world size and the matching native coordinate range yourself. Prism computes scale = real size / source range.';
    return 'Recommended when MPD tile bounds tightly cover the PLY. Prism reads tile extents from the MPD and avoids scanning PLY frames.';
  }

  function translationText(dataset: DatasetProfile): string {
    const t = dataset.transform?.translation || [0, 0, 0];
    return `[${t.map((value) => Number(value || 0).toFixed(3)).join(', ')}]`;
  }

  async function loadPageData(): Promise<void> {
    try {
      const [optionsPayload, datasetPayload] = await Promise.all([getOptions(), getDatasets()]);
      angleConventionChoices = optionsPayload.angle_conventions || [];
      angleConvention = angleConventionChoices[0]?.id || angleConvention;
      datasets = datasetPayload.datasets || [];
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  function onDimensionChanged(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement;
    dimension = target.value as typeof dimension;
    if (dimension === 'height') axis = 'y';
    if (dimension === 'width') axis = 'x';
    if (dimension === 'depth') axis = 'z';
  }

  onMount(loadPageData);
</script>

<header class="hero">
  <div class="hero-actions">
    <a class="button-link ghost" href="/" on:click|preventDefault={() => gotoRoute('home')}>Main menu</a>
    <a class="button-link active" aria-current="page" href="/static/datasets.html" on:click|preventDefault={() => gotoRoute('datasets')}>Datasets</a>
    <a class="button-link ghost" href="/static/index.html" on:click|preventDefault={() => gotoRoute('control')}>Single Run</a>
    <a class="button-link ghost" href="/static/aggregate.html" on:click|preventDefault={() => gotoRoute('batch')}>Batch Planner</a>
    <a class="button-link ghost" href="/static/compare.html" on:click|preventDefault={() => gotoRoute('compare')}>Compare Runs</a>
    <a class="button-link ghost" href="/static/reports.html" on:click|preventDefault={() => gotoRoute('reports')}>Reports</a>
    <a class="button-link ghost" href="/static/extra.html" on:click|preventDefault={() => gotoRoute('extra')}>Extra</a>
    <a class="button-link ghost" href="/static/settings.html" on:click|preventDefault={() => gotoRoute('settings')}>Settings</a>
  </div>
</header>

<main>
  <section class="card form-card">
    <div class="section-header">
      <div>
        <h2>{id ? 'Edit Dataset' : 'Add Dataset'}</h2>
      </div>
      <button class="ghost small" on:click={resetForm}>New dataset</button>
    </div>

    <div class="dataset-grid">
      <label>Dataset name
        <input bind:value={name} placeholder="Longdress calibrated" />
      </label>
      <label>MPD folder
        <input bind:value={mpdFolder} placeholder="/data/longdress/v3c" />
      </label>
      <label>Content dir (PLY)
        <input bind:value={contentDir} placeholder="/data/longdress/ply" />
      </label>
      <label>Calibration method
        <select bind:value={scaleMode}>
          <option value="already_normalized">Already normalized</option>
          <option value="known_extent_mpd">Real size + MPD tile extents</option>
          <option value="known_extent_manual">Real size + source range</option>
          <option value="manual">Enter scale manually</option>
          <option value="already_meters">Already in meters</option>
        </select>
        <span class="field-help">{scaleModeHelp()}</span>
      </label>

      {#if !alreadyNormalized && scaleMode === 'manual'}
        <label>Scale to meters
          <input type="number" step="any" min="0" bind:value={manualScale} />
        </label>
      {:else if !alreadyNormalized && (scaleMode === 'known_extent_mpd' || scaleMode === 'known_extent_manual')}
        <label>Dimension
          <select value={dimension} on:change={onDimensionChanged}>
            <option value="height">Height</option>
            <option value="width">Width</option>
            <option value="depth">Depth</option>
          </select>
        </label>
        <label>Axis
          <select bind:value={axis}>
            <option value="x">X</option>
            <option value="y">Y</option>
            <option value="z">Z</option>
          </select>
        </label>
        <label>Real extent (m)
          <input type="number" step="any" min="0" bind:value={realExtentM} />
        </label>
        {#if scaleMode === 'known_extent_manual'}
          <label>Source extent
            <input type="number" step="any" min="0" bind:value={sourceExtentUnits} />
          </label>
        {/if}
      {/if}

      <label class="span-2">Angle convention
        <select bind:value={angleConvention}>
          {#each angleConventionChoices as choice}
            <option value={choice.id}>{choice.label}</option>
          {/each}
        </select>
        <span class="field-help">Stored with the dataset because FoV trace rotations must match the dataset coordinate system.</span>
      </label>

      <label>Max speed (m/s)
        <input type="number" step="any" min="0" bind:value={maxSpeed} />
        <span class="field-help">Dataset-level limit used to normalize pose-trace speed for motion confidence.</span>
      </label>
      <label>Max acceleration (m/s^2)
        <input type="number" step="any" min="0" bind:value={maxAcceleration} />
        <span class="field-help">Dataset-level limit used to normalize pose-trace acceleration for motion confidence.</span>
      </label>

      {#if !alreadyNormalized}
        <label class="toggle-field dataset-toggle">
          Apply transform to FoV traces
          <span class="check-row">
            <input type="checkbox" bind:checked={applyToFovTrace} />
          </span>
        </label>
        <label class="toggle-field dataset-toggle">
          Translation override
          <span class="check-row">
            <input type="checkbox" bind:checked={translationEnabled} />
          </span>
        </label>
      {/if}

      {#if !alreadyNormalized && translationEnabled}
        <label>Translation X (m)
          <input type="number" step="any" bind:value={translationX} />
        </label>
        <label>Translation Y (m)
          <input type="number" step="any" bind:value={translationY} />
        </label>
        <label>Translation Z (m)
          <input type="number" step="any" bind:value={translationZ} />
        </label>
      {/if}

      <label class="span-2">Notes
        <input bind:value={notes} placeholder="Optional calibration notes" />
      </label>
    </div>

    <div class="form-footer">
      <button disabled={saving || !name || !mpdFolder || !contentDir} on:click={saveCurrentDataset}>
        {saving ? 'Saving…' : 'Save dataset'}
      </button>
      <span class="status">Effective scale: {computedScaleText()}</span>
    </div>
    {#if error}<p class="navis-export-error">{error}</p>{/if}
    {#if statusMessage}<p class="subtle">{statusMessage}</p>{/if}
  </section>

  <section class="card">
    <div class="section-header">
      <h2>Registered datasets</h2>
      <button class="ghost small" on:click={refreshDatasets}>Refresh</button>
    </div>
    {#if loading}
      <p class="subtle">Loading datasets…</p>
    {:else if !datasets.length}
      <p class="subtle">No datasets registered yet.</p>
    {:else}
      <div class="dataset-list">
        {#each datasets as dataset}
          <article class="dataset-card">
            <div class="dataset-card-head">
              <div>
                <p class="eyebrow micro">Dataset</p>
                <h3>{dataset.name}</h3>
              </div>
              <div class="section-actions">
                <button class="ghost small" on:click={() => editDataset(dataset)}>Edit</button>
                <button class="ghost small" on:click={() => removeDataset(dataset)}>Delete</button>
              </div>
            </div>
            <div class="dataset-paths">
              <div>
                <span>MPD folder</span>
                <code>{dataset.mpd_folder}</code>
              </div>
              <div>
                <span>PLY directory</span>
                <code>{dataset.content_dir}</code>
              </div>
            </div>
            <div class="dataset-summary-grid">
              <div>
                <span>Calibration</span>
                <strong>{dataset.already_normalized ? 'Already normalized' : scaleModeLabel(dataset.scale_mode)}</strong>
              </div>
              <div>
                <span>Scale</span>
                <strong>{Number(dataset.transform?.scale ?? 1).toPrecision(5)}</strong>
              </div>
              <div>
                <span>Angle convention</span>
                <strong>{dataset.angle_convention || 'unity_zxy'}</strong>
              </div>
              <div>
                <span>Motion limits</span>
                <strong>{Number(dataset.motion_limits?.max_speed ?? 1.5).toPrecision(3)} m/s, {Number(dataset.motion_limits?.max_acceleration ?? 3.0).toPrecision(3)} m/s^2</strong>
              </div>
              <div>
                <span>FoV trace positions</span>
                <strong>{dataset.apply_to_fov_trace === false ? 'Unchanged' : 'Transformed'}</strong>
              </div>
              <div>
                <span>Translation</span>
                <strong>{translationText(dataset)}</strong>
              </div>
            </div>
            {#if dataset.extent_stats?.axis}
              <div class="extent-row">
                {#each Object.entries(dataset.extent_stats.axis) as [axisName, stats]}
                  <span>{axisName.toUpperCase()} {stats.extent.toPrecision(5)}</span>
                {/each}
              </div>
            {/if}
          </article>
        {/each}
      </div>
    {/if}
  </section>
</main>

<style>
  .dataset-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
  }

  .span-2 {
    grid-column: span 2;
  }

  .field-help {
    display: block;
    margin-top: 0.32rem;
    color: var(--text-muted);
    font-size: 0.84rem;
    line-height: 1.35;
  }

  .dataset-toggle {
    min-height: 74px;
  }

  .dataset-list {
    display: grid;
    gap: 12px;
  }

  .dataset-card {
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 16px;
    display: grid;
    grid-template-columns: 1fr;
    gap: 14px;
  }

  .dataset-card h3 {
    margin: 0 0 4px;
  }

  .dataset-card-head {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: flex-start;
  }

  .dataset-paths,
  .dataset-summary-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
  }

  .dataset-paths > div,
  .dataset-summary-grid > div {
    min-width: 0;
    border: 1px solid var(--field-border);
    border-radius: 8px;
    padding: 10px;
    background: rgba(148, 163, 184, 0.06);
  }

  .dataset-paths span,
  .dataset-summary-grid span {
    display: block;
    color: var(--text-muted);
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
  }

  .dataset-paths code,
  .dataset-summary-grid strong {
    display: block;
    min-width: 0;
    overflow-wrap: anywhere;
  }

  .extent-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
  }

  .extent-row {
    color: var(--text-muted);
    font-size: 0.86rem;
  }

  @media (max-width: 760px) {
    .dataset-grid,
    .dataset-paths,
    .dataset-summary-grid {
      grid-template-columns: 1fr;
    }

    .dataset-card-head {
      flex-direction: column;
    }

    .span-2,
    .extent-row {
      grid-column: auto;
    }
  }
</style>
