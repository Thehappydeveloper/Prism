<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { getDiagnostics, getOptions, listRuns, startRun, subscribeRuns } from '../core/api';
  import type { AppOptions, RunConfig, RunMetrics, RunSummary } from '../core/types';

  interface MediaPair {
    id: string;
    v3c_folder: string;
    content_dir: string;
    fov_traces: string[];
  }

  interface PlannedRun {
    id: string;
    config: RunConfig;
    status: 'planned' | 'queued' | 'failed';
    runId?: string;
    error?: string;
  }

  let options: AppOptions | null = null;
  let loading = true;
  let error = '';

  let selectedAbrs = new Set<string>();
  let selectedPredictors = new Set<string>(['None']);

  let use4g = true;
  let use5g = true;
  let netSamples = 3;
  let fovSamples = 3;
  let seedText = '';

  let rtt = 0.06;
  let bufferWindowSegments = 10;
  let enableMidCancel = false;

  let mediaPairs: MediaPair[] = [];
  let plannedRuns: PlannedRun[] = [];
  let planMessage = 'No plan yet.';

  let liveRuns: RunSummary[] = [];
  let diagnostics: RunMetrics | null = null;

  let unsubscribeRuns: (() => void) | null = null;

  function addPair(): void {
    mediaPairs = [
      ...mediaPairs,
      { id: crypto.randomUUID(), v3c_folder: '', content_dir: '', fov_traces: [] },
    ];
  }

  function removePair(id: string): void {
    mediaPairs = mediaPairs.filter((p) => p.id !== id);
  }

  function updatePair(id: string, next: Partial<MediaPair>): void {
    mediaPairs = mediaPairs.map((p) => (p.id === id ? { ...p, ...next } : p));
  }

  function handlePairV3CChange(id: string, event: Event): void {
    const target = event.currentTarget as HTMLInputElement | null;
    updatePair(id, { v3c_folder: target?.value || '' });
  }

  function handlePairContentChange(id: string, event: Event): void {
    const target = event.currentTarget as HTMLInputElement | null;
    updatePair(id, { content_dir: target?.value || '' });
  }

  function handlePairFovChange(id: string, event: Event): void {
    const target = event.currentTarget as HTMLInputElement | null;
    const traces = (target?.value || '')
      .split(',')
      .map((x) => x.trim())
      .filter(Boolean);
    updatePair(id, { fov_traces: traces });
  }

  function toggleSelection(setRef: Set<string>, key: string): Set<string> {
    const next = new Set(setRef);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    return next;
  }

  function hashSeed(input: string): number {
    if (!input) return Date.now() >>> 0;
    let h = 2166136261;
    for (let i = 0; i < input.length; i += 1) {
      h ^= input.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function mulberry32(seed: number): () => number {
    return () => {
      let t = (seed += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function sampleArray<T>(arr: T[], count: number, rng: () => number): T[] {
    if (count >= arr.length) return [...arr];
    const pool = [...arr];
    const out: T[] = [];
    for (let i = 0; i < count && pool.length; i += 1) {
      const idx = Math.floor(rng() * pool.length);
      out.push(pool.splice(idx, 1)[0]!);
    }
    return out;
  }

  function planRuns(): void {
    if (!options) return;

    const abrs = Array.from(selectedAbrs);
    const predictors = Array.from(selectedPredictors);
    const net4 = (options.network_traces || []).filter((t) => /(^|\/)4g(\/|$)|lte/i.test(t));
    const net5 = (options.network_traces || []).filter((t) => /(^|\/)5g(\/|$)/i.test(t));
    const netOther = (options.network_traces || []).filter((t) => !net4.includes(t) && !net5.includes(t));

    const rng = mulberry32(hashSeed(seedText));

    let selectedNetwork: string[] = [];
    if (use4g) selectedNetwork = selectedNetwork.concat(sampleArray(net4, netSamples, rng));
    if (use5g) selectedNetwork = selectedNetwork.concat(sampleArray(net5, netSamples, rng));
    if (!use4g && !use5g) selectedNetwork = sampleArray(netOther.length ? netOther : options.network_traces, netSamples, rng);

    const planned: PlannedRun[] = [];

    for (const pair of mediaPairs) {
      if (!pair.v3c_folder || !pair.content_dir) continue;
      const sampledFov = sampleArray(pair.fov_traces, Math.max(1, fovSamples), rng);
      for (const abr of abrs) {
        for (const predictor of predictors) {
          for (const networkTrace of selectedNetwork) {
            for (const fovTrace of sampledFov) {
              planned.push({
                id: crypto.randomUUID(),
                status: 'planned',
                config: {
                  v3c_folder: pair.v3c_folder,
                  content_dir: pair.content_dir,
                  network_trace: networkTrace,
                  fov_trace: fovTrace,
                  abr,
                  predictor,
                  rtt,
                  buffer_window_segments: bufferWindowSegments,
                  enable_mid_download_cancel: enableMidCancel,
                },
              });
            }
          }
        }
      }
    }

    plannedRuns = planned;
    planMessage = planned.length
      ? `${planned.length} runs planned (${abrs.length} ABR x ${predictors.length} predictors x ${selectedNetwork.length} net x sampled FoVs).`
      : 'No runs planned. Select ABRs, predictors, traces, and at least one media pair.';
  }

  async function startBatch(): Promise<void> {
    if (!plannedRuns.length) return;
    const next = [...plannedRuns];

    for (let i = 0; i < next.length; i += 1) {
      const plan = next[i]!;
      try {
        // eslint-disable-next-line no-await-in-loop
        const res = await startRun(plan.config);
        next[i] = { ...plan, status: 'queued', runId: res.run_id };
      } catch (err) {
        next[i] = { ...plan, status: 'failed', error: err instanceof Error ? err.message : String(err) };
      }
      plannedRuns = [...next];
    }
  }

  async function refreshInsights(): Promise<void> {
    try {
      const finished = liveRuns.filter((r) => r.status === 'finished').map((r) => r.run_id);
      if (!finished.length) {
        diagnostics = null;
        return;
      }
      diagnostics = await getDiagnostics({ run_ids: finished });
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  function abrRows() {
    const metrics = diagnostics;
    if (!metrics) return [] as Array<{ abr: string; runs: number; avgQoe: number; avgStall: number; avgBitrate: number }>;
    const buckets: Record<string, { runs: number; qoe: number[]; stall: number[]; bitrate: number[] }> = {};

    Object.keys(metrics.qoe || {}).forEach((runId) => {
      const abr = metrics.meta?.[runId]?.abr || 'Unknown';
      if (!buckets[abr]) buckets[abr] = { runs: 0, qoe: [], stall: [], bitrate: [] };
      buckets[abr].runs += 1;
      buckets[abr].qoe.push(...(metrics.qoe?.[runId] || []));
      buckets[abr].stall.push(Number(metrics.stall?.[runId]?.duration || 0));
      const bq = metrics.bitrate_vs_quality?.[runId] || [];
      bq.forEach(([bitrate]) => buckets[abr]!.bitrate.push(Number(bitrate || 0)));
    });

    const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
    return Object.entries(buckets).map(([abr, b]) => ({
      abr,
      runs: b.runs,
      avgQoe: avg(b.qoe),
      avgStall: avg(b.stall),
      avgBitrate: avg(b.bitrate),
    }));
  }

  onMount(async () => {
    try {
      options = await getOptions();
      selectedAbrs = new Set([options.abr_classes[0] || 'BandwidthAwareABR']);
      mediaPairs = [{ id: crypto.randomUUID(), v3c_folder: '', content_dir: '', fov_traces: options.fov_traces.slice(0, 3) }];
      liveRuns = await listRuns();
      unsubscribeRuns = subscribeRuns((runs) => {
        liveRuns = runs;
      });
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  });

  onDestroy(() => {
    unsubscribeRuns?.();
  });
</script>

<header class="hero">
  <div>
    <p class="eyebrow">Prism VPCC Lab</p>
    <h1>Aggregate Explorer</h1>
    <p class="lede">Plan and launch multi-run experiments across ABRs, FoV predictors, traces, and content.</p>
  </div>
  <div class="hero-actions">
    <a class="button-link ghost" href="/">Main menu</a>
    <a class="button-link ghost" href="/static/index.html">Single runs</a>
  </div>
</header>

<main>
  <section class="card form-card aggregate-form">
    <div class="section-header">
      <h2>Plan experiment set</h2>
      <p class="subtle">Registry-based frontend. Add parameters here, not giant monolithic scripts.</p>
    </div>

    {#if loading}
      <p class="subtle">Loading options…</p>
    {:else}
      <div class="planner-grid">
        <div class="panel-block abr-panel">
          <p class="eyebrow micro">ABRs</p>
          <div class="check-list split">
            {#each options?.abr_classes || [] as abr}
              <label class="pill-toggle">
                <input type="checkbox" checked={selectedAbrs.has(abr)} on:change={() => (selectedAbrs = toggleSelection(selectedAbrs, abr))} />
                {abr}
              </label>
            {/each}
          </div>
        </div>

        <div class="panel-block fov-panel">
          <p class="eyebrow micro">FoV predictors</p>
          <div class="check-list split">
            <label class="pill-toggle"><input type="checkbox" checked={selectedPredictors.has('None')} on:change={() => (selectedPredictors = toggleSelection(selectedPredictors, 'None'))} /> None</label>
            {#each options?.fov_predictors || [] as predictor}
              <label class="pill-toggle">
                <input
                  type="checkbox"
                  checked={selectedPredictors.has(predictor)}
                  on:change={() => (selectedPredictors = toggleSelection(selectedPredictors, predictor))}
                />
                {predictor}
              </label>
            {/each}
          </div>
        </div>

        <div class="panel-block trace-panel span-2">
          <p class="eyebrow micro">Traces</p>
          <div class="trace-box">
            <label class="pill-toggle"><input type="checkbox" bind:checked={use4g} /> 4G</label>
            <label class="pill-toggle"><input type="checkbox" bind:checked={use5g} /> 5G</label>
            <label>Network samples per class
              <input type="number" min="1" bind:value={netSamples} />
            </label>
            <label>FoV samples per media pair
              <input type="number" min="1" bind:value={fovSamples} />
            </label>
            <label>Seed (optional)
              <input bind:value={seedText} placeholder="e.g. 42" />
            </label>
          </div>
        </div>

        <div class="panel-block media-panel span-2">
          <div class="pair-list-head">
            <p class="eyebrow micro">Media pairs</p>
            <button class="ghost small" on:click={addPair}>Add dataset</button>
          </div>
          <div class="pair-list">
            {#each mediaPairs as pair}
              <div class="pair-card">
                <label>V3C folder
                  <input value={pair.v3c_folder} on:change={(e) => handlePairV3CChange(pair.id, e)} />
                </label>
                <label>Content dir (PLY)
                  <input value={pair.content_dir} on:change={(e) => handlePairContentChange(pair.id, e)} />
                </label>
                <label>FoV traces (comma-separated)
                  <input
                    value={pair.fov_traces.join(',')}
                    on:change={(e) => handlePairFovChange(pair.id, e)}
                    placeholder={(options?.fov_traces || []).slice(0, 2).join(', ')}
                  />
                </label>
                <button class="ghost small" on:click={() => removePair(pair.id)}>Remove</button>
              </div>
            {/each}
          </div>
        </div>

        <div class="panel-block controls-panel span-2">
          <div class="inline-grid">
            <label>RTT (s)
              <input type="number" step="0.01" bind:value={rtt} />
            </label>
            <label>Fetch window (segments)
              <input type="number" min="0" bind:value={bufferWindowSegments} />
            </label>
            <label class="pill-toggle">
              <input type="checkbox" bind:checked={enableMidCancel} />
              Adaptive abort
            </label>
          </div>
        </div>
      </div>

      <div class="form-footer">
        <button on:click={planRuns}>Plan runs</button>
        <button class="ghost" disabled={!plannedRuns.length} on:click={startBatch}>Start batch</button>
        <span class="status">{planMessage}</span>
      </div>
    {/if}
  </section>

  <section class="card">
    <div class="section-header">
      <h2>Planned runs</h2>
    </div>
    <div class="status-list">
      {#if !plannedRuns.length}
        <p class="subtle">No planned runs yet.</p>
      {:else}
        {#each plannedRuns as plan}
          <div class="status-item">
            <strong>{plan.config.abr}</strong> | {plan.config.predictor || 'None'} | {plan.config.network_trace} | {plan.config.fov_trace}
            <span class="pill {plan.status === 'failed' ? 'failed' : plan.status === 'queued' ? 'success' : 'queued'}">{plan.status}</span>
          </div>
        {/each}
      {/if}
    </div>
  </section>

  <section class="card">
    <div class="section-header">
      <h2>Batch insights</h2>
      <div class="section-actions">
        <button class="ghost small" on:click={refreshInsights}>Refresh stats</button>
      </div>
    </div>

    {#if error}
      <p class="subtle">{error}</p>
    {/if}

    <div class="table-wrap">
      <table class="compact">
        <thead>
          <tr>
            <th>ABR</th>
            <th>Runs</th>
            <th>Avg QoE</th>
            <th>Avg stall (s)</th>
            <th>Avg bitrate (Mbps)</th>
          </tr>
        </thead>
        <tbody>
          {#each abrRows() as row}
            <tr>
              <td>{row.abr}</td>
              <td>{row.runs}</td>
              <td>{row.avgQoe.toFixed(3)}</td>
              <td>{row.avgStall.toFixed(2)}</td>
              <td>{row.avgBitrate.toFixed(2)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>
</main>
