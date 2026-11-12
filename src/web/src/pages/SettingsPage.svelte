  <script lang="ts">
  import { onMount } from 'svelte';
  import { gotoRoute } from '../core/router';
  import ThemeSwitcher from '../components/ThemeSwitcher.svelte';
  import { loadLaunchPreflightMode, saveLaunchPreflightMode } from '../core/launchPreflight';
  import { getHeadsetConfig, getNavisConfig, getNavisStatus, getPrismConfig, getTraceColumnConfig, getTraceRootConfig, saveHeadsetConfig, saveNavisConfig, savePrismConfig, saveTraceColumnConfig, saveTraceRootConfig } from '../core/api';
  import type { HeadsetConfig, HeadsetPreset, LaunchPreflightMode, NavisConfig, NavisStatus, PrismConfig, TraceColumnAliasSettings, TraceColumnField, TraceRootConfig } from '../core/types';

  let loading = true;
  let saving = false;
  let checking = false;
  let error = '';
  let statusMessage = '';
  let navisStatus: NavisStatus | null = null;
  let settingsPath = '';
  let headsetSettingsPath = '';
  let traceSettingsPath = '';
  let traceColumnSettingsPath = '';
  let prismSettingsPath = '';
  let headsetStatusMessage = '';
  let traceStatusMessage = '';
  let traceColumnStatusMessage = '';
  let prismStatusMessage = '';
  let launchPreflightMode: LaunchPreflightMode = loadLaunchPreflightMode();
  let savingHeadsets = false;
  let savingTraceRoots = false;
  let savingTraceColumns = false;
  let savingPrism = false;
  let headsetConfig: HeadsetConfig = { default_preset: 'prism_default', presets: [] };
  let traceConfig: TraceRootConfig = { network_root: 'data/network_traces', fov_root: 'data/fov_traces' };
  let traceColumnAliases: TraceColumnAliasSettings = { fov: {}, network: {} };
  let traceColumnFields: Record<string, Record<string, TraceColumnField>> = {};
  let prismConfig: PrismConfig = {
    record_fov_stage_latency: false,
    batch_queue_coalesce_ms: 250,
    batch_queue_max_coalesce_ms: 2500,
    batch_ply_io_workers: 1,
    batch_segment_workers: 0,
    batch_segment_workers_auto_cap: 4,
    batch_group_max_runs: 0,
    batch_prefetch_segments: 1,
    training_max_parallel_runs: 1,
    training_cpu_core_limit: navigator.hardwareConcurrency || 1,
    training_require_single_worker_for_parallel: true,
  };

  let navisHost = 'http://127.0.0.1';
  let navisPort = '9000';
  let healthPath = '/health';
  let exportPath = '/api/prism/runs';
  let timeoutS = '0.35';
  let autoExport = false;
  let authToken = '';

  function splitBaseUrl(baseUrl: string): { host: string; port: string } {
    const trimmed = (baseUrl || '').trim();
    if (!trimmed) return { host: 'http://127.0.0.1', port: '9000' };
    try {
      const parsed = new URL(trimmed);
      const host = `${parsed.protocol}//${parsed.hostname}`;
      const port = parsed.port || '';
      return { host, port };
    } catch {
      return { host: trimmed, port: '' };
    }
  }

  function composeBaseUrl(host: string, port: string): string {
    let normalizedHost = (host || '').trim().replace(/\/+$/, '');
    if (!normalizedHost) return '';
    if (!/^https?:\/\//i.test(normalizedHost)) normalizedHost = `http://${normalizedHost}`;
    const normalizedPort = (port || '').trim();
    try {
      const parsed = new URL(normalizedHost);
      if (normalizedPort) parsed.port = normalizedPort;
      return parsed.origin;
    } catch {
      return normalizedPort ? `${normalizedHost}:${normalizedPort}` : normalizedHost;
    }
  }

  function normalizePath(path: string, fallback: string): string {
    const trimmed = (path || '').trim();
    if (!trimmed) return fallback;
    return trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
  }

  function applySettings(settings: NavisConfig | null | undefined): void {
    if (!settings) return;
    const split = splitBaseUrl(settings.base_url || '');
    navisHost = split.host;
    navisPort = split.port;
    healthPath = settings.health_path || '/health';
    exportPath = settings.export_path || '/api/prism/runs';
    timeoutS = String(settings.timeout_s ?? 0.35);
    autoExport = Boolean(settings.auto_export);
    authToken = settings.auth_token || '';
  }

  function statusLabel(): string {
    if (!navisStatus?.configured) return 'Not configured';
    if (navisStatus.reachable) return 'Online';
    return 'Offline';
  }

  function statusClass(): 'queued' | 'success' | 'failed' {
    if (!navisStatus?.configured) return 'queued';
    return navisStatus.reachable ? 'success' : 'failed';
  }

  function applyHeadsetSettings(settings: HeadsetConfig | null | undefined): void {
    if (!settings) return;
    headsetConfig = {
      default_preset: settings.default_preset || settings.presets?.[0]?.id || '',
      angle_convention: settings.angle_convention || 'unity_zxy',
      presets: (settings.presets || []).map((preset) => ({ ...preset })),
    };
  }

  function applyTraceSettings(settings: TraceRootConfig | null | undefined): void {
    if (!settings) return;
    traceConfig = {
      network_root: settings.network_root || 'data/network_traces',
      fov_root: settings.fov_root || 'data/fov_traces',
    };
  }

  function applyTraceColumnSettings(
    settings: TraceColumnAliasSettings | null | undefined,
    fields: Record<string, Record<string, TraceColumnField>> | null | undefined,
  ): void {
    traceColumnFields = fields || {};
    const next: TraceColumnAliasSettings = {};
    for (const traceType of Object.keys(traceColumnFields)) {
      next[traceType] = {};
      for (const fieldId of Object.keys(traceColumnFields[traceType] || {})) {
        const field = traceColumnFields[traceType]?.[fieldId];
        next[traceType][fieldId] = [...(settings?.[traceType]?.[fieldId] || field?.aliases || [])];
      }
    }
    traceColumnAliases = next;
  }

  function applyPrismSettings(settings: PrismConfig | null | undefined): void {
    if (!settings) return;
    prismConfig = {
      record_fov_stage_latency: Boolean(settings.record_fov_stage_latency),
      batch_queue_coalesce_ms: Number(settings.batch_queue_coalesce_ms ?? 250),
      batch_queue_max_coalesce_ms: Number(settings.batch_queue_max_coalesce_ms ?? 2500),
      batch_ply_io_workers: Number(settings.batch_ply_io_workers ?? 1),
      batch_segment_workers: Number(settings.batch_segment_workers ?? 1),
      batch_segment_workers_auto_cap: Number(settings.batch_segment_workers_auto_cap ?? 4),
      batch_group_max_runs: Number(settings.batch_group_max_runs ?? 0),
      batch_prefetch_segments: Number(settings.batch_prefetch_segments ?? 1),
      training_max_parallel_runs: Number(settings.training_max_parallel_runs ?? 1),
      training_cpu_core_limit: Number(settings.training_cpu_core_limit ?? navigator.hardwareConcurrency ?? 1),
      training_require_single_worker_for_parallel: settings.training_require_single_worker_for_parallel !== false,
    };
  }

  async function loadConfig(): Promise<void> {
    error = '';
    try {
      const [navisPayload, headsetPayload, tracePayload, traceColumnPayload, prismPayload] = await Promise.all([
        getNavisConfig(),
        getHeadsetConfig(),
        getTraceRootConfig(),
        getTraceColumnConfig(),
        getPrismConfig(),
      ]);
      settingsPath = navisPayload.settings_path || '';
      navisStatus = navisPayload.status || null;
      applySettings(navisPayload.settings);
      headsetSettingsPath = headsetPayload.settings_path || '';
      applyHeadsetSettings(headsetPayload.settings);
      traceSettingsPath = tracePayload.settings_path || '';
      applyTraceSettings(tracePayload.settings);
      traceColumnSettingsPath = traceColumnPayload.settings_path || '';
      applyTraceColumnSettings(traceColumnPayload.settings, traceColumnPayload.fields);
      prismSettingsPath = prismPayload.settings_path || '';
      applyPrismSettings(prismPayload.settings);
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  async function saveSettings(): Promise<void> {
    if (saving) return;
    saving = true;
    error = '';
    statusMessage = '';
    try {
      const payload: Partial<NavisConfig> = {
        base_url: composeBaseUrl(navisHost, navisPort),
        health_path: normalizePath(healthPath, '/health'),
        export_path: normalizePath(exportPath, '/api/prism/runs'),
        timeout_s: Number(timeoutS) > 0 ? Number(timeoutS) : 0.35,
        auto_export: autoExport,
        auth_token: authToken.trim(),
      };
      const result = await saveNavisConfig(payload);
      settingsPath = result.settings_path || settingsPath;
      navisStatus = result.status || null;
      applySettings(result.settings);
      statusMessage = 'Navis settings saved.';
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      saving = false;
    }
  }

  async function refreshAvailability(): Promise<void> {
    if (checking) return;
    checking = true;
    error = '';
    statusMessage = '';
    try {
      navisStatus = await getNavisStatus();
      statusMessage = navisStatus.reachable ? 'Navis is reachable.' : (navisStatus.error || 'Navis is unreachable.');
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      checking = false;
    }
  }

  function addHeadsetPreset(): void {
    const idx = headsetConfig.presets.length + 1;
    const preset: HeadsetPreset = {
      id: `custom_headset_${idx}`,
      label: `Custom Headset ${idx}`,
      fov_h_deg: 97,
      fov_v_deg: 93,
    };
    headsetConfig = {
      ...headsetConfig,
      presets: [...headsetConfig.presets, preset],
      default_preset: headsetConfig.default_preset || preset.id,
    };
  }

  function updateHeadsetPreset(id: string, next: Partial<HeadsetPreset>): void {
    headsetConfig = {
      ...headsetConfig,
      presets: headsetConfig.presets.map((preset) => (preset.id === id ? { ...preset, ...next } : preset)),
    };
  }

  function handleHeadsetIdChange(id: string, event: Event): void {
    const target = event.currentTarget as HTMLInputElement | null;
    const nextId = target?.value || '';
    updateHeadsetPreset(id, { id: nextId });
    if (headsetConfig.default_preset === id) {
      headsetConfig = { ...headsetConfig, default_preset: nextId };
    }
  }

  function handleHeadsetLabelChange(id: string, event: Event): void {
    const target = event.currentTarget as HTMLInputElement | null;
    updateHeadsetPreset(id, { label: target?.value || '' });
  }

  function handleHeadsetFovHChange(id: string, event: Event): void {
    const target = event.currentTarget as HTMLInputElement | null;
    updateHeadsetPreset(id, { fov_h_deg: Number(target?.value || 0) });
  }

  function handleHeadsetFovVChange(id: string, event: Event): void {
    const target = event.currentTarget as HTMLInputElement | null;
    updateHeadsetPreset(id, { fov_v_deg: Number(target?.value || 0) });
  }

  function removeHeadsetPreset(id: string): void {
    const nextPresets = headsetConfig.presets.filter((preset) => preset.id !== id);
    headsetConfig = {
      default_preset: headsetConfig.default_preset === id ? (nextPresets[0]?.id || '') : headsetConfig.default_preset,
      presets: nextPresets,
    };
  }

  async function saveHeadsets(): Promise<void> {
    if (savingHeadsets) return;
    savingHeadsets = true;
    error = '';
    headsetStatusMessage = '';
    try {
      const payload: HeadsetConfig = {
        default_preset: headsetConfig.default_preset,
        angle_convention: headsetConfig.angle_convention || 'unity_zxy',
        presets: headsetConfig.presets.map((preset) => ({
          id: preset.id.trim(),
          label: preset.label.trim(),
          fov_h_deg: Number(preset.fov_h_deg),
          fov_v_deg: Number(preset.fov_v_deg),
        })),
      };
      const result = await saveHeadsetConfig(payload);
      headsetSettingsPath = result.settings_path || headsetSettingsPath;
      applyHeadsetSettings(result.settings);
      headsetStatusMessage = 'Headset presets saved.';
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      savingHeadsets = false;
    }
  }

  async function saveTraceRoots(): Promise<void> {
    if (savingTraceRoots) return;
    savingTraceRoots = true;
    error = '';
    traceStatusMessage = '';
    try {
      const result = await saveTraceRootConfig({
        network_root: traceConfig.network_root.trim(),
        fov_root: traceConfig.fov_root.trim(),
      });
      traceSettingsPath = result.settings_path || traceSettingsPath;
      applyTraceSettings(result.settings);
      traceStatusMessage = 'Trace roots saved.';
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      savingTraceRoots = false;
    }
  }

  function aliasesText(traceType: string, fieldId: string): string {
    return (traceColumnAliases[traceType]?.[fieldId] || []).join(', ');
  }

  function updateTraceColumnAliases(traceType: string, fieldId: string, event: Event): void {
    const target = event.currentTarget as HTMLInputElement | null;
    const aliases = (target?.value || '')
      .split(',')
      .map((alias) => alias.trim())
      .filter(Boolean);
    traceColumnAliases = {
      ...traceColumnAliases,
      [traceType]: {
        ...(traceColumnAliases[traceType] || {}),
        [fieldId]: aliases,
      },
    };
  }

  async function saveTraceColumns(): Promise<void> {
    if (savingTraceColumns) return;
    savingTraceColumns = true;
    error = '';
    traceColumnStatusMessage = '';
    try {
      const result = await saveTraceColumnConfig(traceColumnAliases);
      traceColumnSettingsPath = result.settings_path || traceColumnSettingsPath;
      applyTraceColumnSettings(result.settings, result.fields);
      traceColumnStatusMessage = 'Trace column mappings saved.';
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      savingTraceColumns = false;
    }
  }

  async function savePrismRuntime(): Promise<void> {
    if (savingPrism) return;
    savingPrism = true;
    error = '';
    prismStatusMessage = '';
    try {
      const result = await savePrismConfig({
        record_fov_stage_latency: prismConfig.record_fov_stage_latency,
        batch_queue_coalesce_ms: Number(prismConfig.batch_queue_coalesce_ms ?? 250),
        batch_queue_max_coalesce_ms: Number(prismConfig.batch_queue_max_coalesce_ms ?? 2500),
        batch_ply_io_workers: Number(prismConfig.batch_ply_io_workers ?? 1),
        batch_segment_workers: Number(prismConfig.batch_segment_workers ?? 1),
        batch_segment_workers_auto_cap: Number(prismConfig.batch_segment_workers_auto_cap ?? 4),
        batch_group_max_runs: Number(prismConfig.batch_group_max_runs ?? 0),
        batch_prefetch_segments: Number(prismConfig.batch_prefetch_segments ?? 1),
        training_max_parallel_runs: Number(prismConfig.training_max_parallel_runs ?? 1),
        training_cpu_core_limit: Number(prismConfig.training_cpu_core_limit ?? navigator.hardwareConcurrency ?? 1),
        training_require_single_worker_for_parallel: Boolean(prismConfig.training_require_single_worker_for_parallel),
      });
      prismSettingsPath = result.settings_path || prismSettingsPath;
      applyPrismSettings(result.settings);
      saveLaunchPreflightMode(launchPreflightMode);
      prismStatusMessage = 'Prism runtime settings saved.';
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      savingPrism = false;
    }
  }

  onMount(async () => {
    await loadConfig();
    loading = false;
  });
</script>

<header class="hero">
  <div class="hero-actions">
    <a class="button-link ghost" href="/" on:click|preventDefault={() => gotoRoute('home')}>Main menu</a>
    <a class="button-link ghost" href="/static/datasets.html" on:click|preventDefault={() => gotoRoute('datasets')}>Datasets</a>
    <a class="button-link ghost" href="/static/index.html" on:click|preventDefault={() => gotoRoute('control')}>Single Run</a>
    <a class="button-link ghost" href="/static/aggregate.html" on:click|preventDefault={() => gotoRoute('batch')}>Batch Planner</a>
    <a class="button-link ghost" href="/static/compare.html" on:click|preventDefault={() => gotoRoute('compare')}>Compare Runs</a>
    <a class="button-link ghost" href="/static/reports.html" on:click|preventDefault={() => gotoRoute('reports')}>Reports</a>
    <a class="button-link ghost" href="/static/extra.html" on:click|preventDefault={() => gotoRoute('extra')}>Extra</a>
    <a class="button-link active" aria-current="page" href="/static/settings.html" on:click|preventDefault={() => gotoRoute('settings')}>Settings</a>
  </div>
</header>

<main>
  <section class="card form-card">
    <div class="section-header">
      <h2>Appearance</h2>
    </div>
    <ThemeSwitcher />
  </section>

  <section class="card form-card">
    <div class="section-header">
      <h2>Navis Connection</h2>
      <span class={`pill ${statusClass()}`}>{statusLabel()}</span>
    </div>

    {#if loading}
      <p class="subtle">Loading settings...</p>
    {:else}
      <div class="grid two-col dense">
        <label>Navis host URL
          <input bind:value={navisHost} placeholder="http://127.0.0.1" />
        </label>
        <label>Navis port
          <input bind:value={navisPort} placeholder="9000" />
        </label>
        <label>Health path
          <input bind:value={healthPath} placeholder="/health" />
        </label>
        <label>Export path
          <input bind:value={exportPath} placeholder="/api/prism/runs" />
        </label>
        <label>Timeout (seconds)
          <input bind:value={timeoutS} type="number" min="0.05" step="0.05" />
        </label>
        <label>Auth token (optional)
          <input bind:value={authToken} placeholder="Bearer token (optional)" />
        </label>
      </div>

      <div class="toggle-row" style="margin-top:0.65rem;">
        <label class="pill-toggle">
          <input type="checkbox" bind:checked={autoExport} />
          Auto-export completed runs
        </label>
      </div>

      <div class="section-actions" style="margin-top:0.8rem;">
        <button type="button" on:click={saveSettings} disabled={saving}>
          {saving ? 'Saving...' : 'Save settings'}
        </button>
        <button type="button" class="ghost" on:click={refreshAvailability} disabled={checking}>
          {checking ? 'Checking...' : 'Refresh availability'}
        </button>
      </div>

      {#if navisStatus}
        <div class="callout" style="margin-top:0.8rem;">
          <p class="micro subtle">
            Checked URL: {navisStatus.checked_url || '-'} · HTTP status: {navisStatus.status_code ?? '-'} · Latency: {navisStatus.latency_ms ?? '-'} ms
          </p>
          {#if navisStatus.error}
            <p class="micro" style="color: var(--error-text); margin: 0.2rem 0 0;">{navisStatus.error}</p>
          {/if}
        </div>
      {/if}

      {#if settingsPath}
        <p class="micro subtle" style="margin-top:0.6rem;">Saved in: <code>{settingsPath}</code></p>
      {/if}
      {#if statusMessage}
        <p class="status" style="margin-top:0.4rem;">{statusMessage}</p>
      {/if}
      {#if error}
        <p class="status-error">{error}</p>
      {/if}
    {/if}
  </section>

  <section class="card form-card" style="margin-top:1rem;">
    <div class="section-header">
      <h2>Predictor Diagnostics</h2>
    </div>

    <p class="subtle">Enable predictor timing and detailed diagnostics for new runs when you need stage-level timing plus internal predictor state.</p>

    <div class="toggle-row" style="margin-top:0.8rem;">
      <label class="pill-toggle">
        <input type="checkbox" bind:checked={prismConfig.record_fov_stage_latency} />
        Record predictor diagnostics JSON
      </label>
    </div>

    <label style="display:block; margin-top:0.8rem;">Launch validation
      <select bind:value={launchPreflightMode}>
        <option value="issues">Show review only for warnings or errors</option>
        <option value="always">Always show review before queueing</option>
        <option value="never">Do not show review; still block invalid runs</option>
      </select>
    </label>
    <p class="subtle">Healthy runs queue immediately by default. This preference is stored in this browser.</p>

    <div class="inline-grid" style="margin-top:0.8rem;">
      <label>Batch quiet wait (ms)
        <input type="number" min="0" max="10000" step="50" bind:value={prismConfig.batch_queue_coalesce_ms} />
      </label>
      <label>Batch max wait (ms)
        <input type="number" min="0" max="60000" step="100" bind:value={prismConfig.batch_queue_max_coalesce_ms} />
      </label>
      <label>Grouped PLY CPU workers
        <input type="number" min="1" max="16" step="1" bind:value={prismConfig.batch_ply_io_workers} />
      </label>
      <label>Grouped segment workers (0 = Auto)
        <input type="number" min="0" max="16" step="1" bind:value={prismConfig.batch_segment_workers} />
      </label>
      <label>Auto segment-worker cap
        <input type="number" min="1" max="16" step="1" bind:value={prismConfig.batch_segment_workers_auto_cap} disabled={Number(prismConfig.batch_segment_workers) !== 0} />
      </label>
      <label>Max grouped runs
        <input type="number" min="0" max="10000" step="1" bind:value={prismConfig.batch_group_max_runs} />
      </label>
      <label>Prefetch segments
        <input type="number" min="0" max="1" step="1" bind:value={prismConfig.batch_prefetch_segments} />
      </label>
    </div>
    <p class="micro subtle" style="margin-top:0.45rem;">
      PLY workers load shared frame windows. Segment workers run per-run predictor/ABR steps inside a grouped segment. Set segment workers to 0 for Auto: Prism recalculates the count before each segment from available RAM and CPU count, up to the cap. Keep the cap low for expensive FoV predictors.
    </p>

    <div class="inline-grid" style="margin-top:0.8rem;">
      <label>Training parallel runs
        <input type="number" min="1" max="256" step="1" bind:value={prismConfig.training_max_parallel_runs} />
      </label>
      <label>Training CPU core limit
        <input type="number" min="1" max="256" step="1" bind:value={prismConfig.training_cpu_core_limit} />
      </label>
      <label class="check-row">
        <input type="checkbox" bind:checked={prismConfig.training_require_single_worker_for_parallel} />
        Parallel only for single-worker runs
      </label>
    </div>
    <p class="micro subtle" style="margin-top:0.45rem;">
      Training runs are queued when the parallel run limit or estimated CPU core usage would be exceeded.
    </p>

    <div class="section-actions" style="margin-top:0.8rem;">
      <button type="button" on:click={savePrismRuntime} disabled={savingPrism}>
        {savingPrism ? 'Saving...' : 'Save runtime settings'}
      </button>
    </div>

    {#if prismSettingsPath}
      <p class="micro subtle" style="margin-top:0.6rem;">Saved in: <code>{prismSettingsPath}</code></p>
    {/if}
    {#if prismStatusMessage}
      <p class="status" style="margin-top:0.4rem;">{prismStatusMessage}</p>
    {/if}
  </section>

  <section class="card form-card" style="margin-top:1rem;">
    <div class="section-header">
      <h2>Trace Roots</h2>
    </div>

    <p class="subtle">These roots define where Prism searches for traces and where folder-based grouping begins in the trace pickers.</p>

    <div class="grid two-col dense" style="margin-top:0.8rem;">
      <label>Network trace root
        <input bind:value={traceConfig.network_root} placeholder="data/network_traces" />
      </label>
      <label>FoV trace root
        <input bind:value={traceConfig.fov_root} placeholder="data/fov_traces" />
      </label>
    </div>

    <div class="section-actions" style="margin-top:0.8rem;">
      <button type="button" on:click={saveTraceRoots} disabled={savingTraceRoots}>
        {savingTraceRoots ? 'Saving...' : 'Save trace roots'}
      </button>
    </div>

    {#if traceSettingsPath}
      <p class="micro subtle" style="margin-top:0.6rem;">Saved in: <code>{traceSettingsPath}</code></p>
    {/if}
    {#if traceStatusMessage}
      <p class="status" style="margin-top:0.4rem;">{traceStatusMessage}</p>
    {/if}
  </section>

  <section class="card form-card" style="margin-top:1rem;">
    <div class="section-header">
      <div>
        <h2>Trace Columns</h2>
        <p class="subtle">Define fallback CSV headers used when the canonical PRISM column is not present.</p>
      </div>
      <div class="section-actions">
        <button type="button" on:click={saveTraceColumns} disabled={savingTraceColumns}>
          {savingTraceColumns ? 'Saving...' : 'Save column mappings'}
        </button>
      </div>
    </div>

    {#if Object.keys(traceColumnFields).length === 0}
      <p class="subtle">Loading trace column mappings...</p>
    {:else}
      <div class="trace-column-editor">
        {#each Object.entries(traceColumnFields) as [traceType, fields]}
          <div class="trace-column-group">
            <h3>{traceType === 'fov' ? 'FoV traces' : 'Network traces'}</h3>
            <div class="trace-column-grid">
              {#each Object.entries(fields) as [fieldId, field]}
                <label>
                  <span>{field.label}{field.required ? '' : ' (optional)'}</span>
                  <small>Canonical: <code>{field.canonical}</code></small>
                  <input
                    value={aliasesText(traceType, fieldId)}
                    on:input={(e) => updateTraceColumnAliases(traceType, fieldId, e)}
                    placeholder="alias_a, alias_b"
                  />
                </label>
              {/each}
            </div>
          </div>
        {/each}
      </div>
    {/if}

    {#if traceColumnSettingsPath}
      <p class="micro subtle" style="margin-top:0.6rem;">Saved in: <code>{traceColumnSettingsPath}</code></p>
    {/if}
    {#if traceColumnStatusMessage}
      <p class="status" style="margin-top:0.4rem;">{traceColumnStatusMessage}</p>
    {/if}
  </section>

  <section class="card form-card" style="margin-top:1rem;">
    <div class="section-header">
      <div>
        <h2>Headsets</h2>
        <p class="subtle">Manage the horizontal and vertical FoV angles used by FoV predictors.</p>
      </div>
      <div class="section-actions">
        <button type="button" class="ghost" on:click={addHeadsetPreset}>Add headset</button>
        <button type="button" on:click={saveHeadsets} disabled={savingHeadsets}>
          {savingHeadsets ? 'Saving...' : 'Save headsets'}
        </button>
      </div>
    </div>

    <div class="headset-editor">
      {#each headsetConfig.presets as preset}
        <article class="headset-row">
          <label class="default-radio">
            <input
              type="radio"
              name="default-headset"
              checked={headsetConfig.default_preset === preset.id}
              on:change={() => (headsetConfig = { ...headsetConfig, default_preset: preset.id })}
            />
            Default
          </label>
          <label>Label
            <input
              value={preset.label}
              on:input={(e) => handleHeadsetLabelChange(preset.id, e)}
            />
          </label>
          <label>Preset id
            <input
              value={preset.id}
              on:change={(e) => handleHeadsetIdChange(preset.id, e)}
            />
          </label>
          <label>Horizontal FoV
            <input
              type="number"
              min="1"
              max="360"
              step="0.1"
              value={preset.fov_h_deg}
              on:input={(e) => handleHeadsetFovHChange(preset.id, e)}
            />
          </label>
          <label>Vertical FoV
            <input
              type="number"
              min="1"
              max="360"
              step="0.1"
              value={preset.fov_v_deg}
              on:input={(e) => handleHeadsetFovVChange(preset.id, e)}
            />
          </label>
          <button type="button" class="ghost small" on:click={() => removeHeadsetPreset(preset.id)} disabled={headsetConfig.presets.length <= 1}>
            Delete
          </button>
        </article>
      {/each}
    </div>

    {#if headsetSettingsPath}
      <p class="micro subtle" style="margin-top:0.6rem;">Saved in: <code>{headsetSettingsPath}</code></p>
    {/if}
    {#if headsetStatusMessage}
      <p class="status" style="margin-top:0.4rem;">{headsetStatusMessage}</p>
    {/if}
  </section>
</main>

<style>
  .headset-editor {
    display: grid;
    gap: 0.75rem;
    margin-top: 1rem;
  }

  .trace-column-editor {
    display: grid;
    gap: 1rem;
    margin-top: 1rem;
  }

  .trace-column-group h3 {
    margin: 0 0 0.65rem;
    font-size: 0.95rem;
  }

  .trace-column-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.75rem;
  }

  .trace-column-grid label {
    min-width: 0;
  }

  .trace-column-grid small {
    display: block;
    margin: 0.25rem 0 0.35rem;
    color: var(--muted);
  }

  .headset-row {
    display: grid;
    grid-template-columns: 110px minmax(160px, 1.2fr) minmax(150px, 1fr) minmax(120px, 0.7fr) minmax(120px, 0.7fr) auto;
    gap: 0.65rem;
    align-items: end;
    padding: 0.8rem;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    background: rgba(148, 163, 184, 0.06);
  }

  .headset-row label,
  .headset-row input {
    min-width: 0;
    max-width: 100%;
    box-sizing: border-box;
  }

  .default-radio {
    min-height: 43px;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  @media (max-width: 760px) {
    .headset-row {
      grid-template-columns: 1fr;
      align-items: stretch;
    }
  }
</style>
