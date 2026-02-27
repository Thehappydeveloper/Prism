<script lang="ts">
  import type { AppOptions, RunConfig } from '../../../core/types';

  export let options: AppOptions | null = null;
  export let form: RunConfig;
  export let statusMessage = '';
  export let submitting = false;
  export let loading = false;

  export let onSubmit: () => void = () => {};
  export let onLoadCache: () => void = () => {};
  export let onClearCache: () => void = () => {};
</script>

<section class="card form-card">
  <div class="section-header">
    <h2>Start a new run</h2>
    <div class="section-actions">
      <button class="ghost small" on:click={onLoadCache}>Load cached params</button>
      <button class="ghost small" on:click={onClearCache}>Clear cache</button>
    </div>
  </div>

  <form on:submit|preventDefault={onSubmit}>
    <div class="run-grid">
      <label>V3C folder (MPD inside)
        <input bind:value={form.v3c_folder} required placeholder="/data/v3c_content" />
      </label>
      <label>Content dir (PLY)
        <input bind:value={form.content_dir} required placeholder="/data/ply_frames" />
      </label>
      <label class="compact">RTT (s)
        <input type="number" step="0.01" bind:value={form.rtt} />
      </label>
      <label>Network trace
        <input list="network-traces" bind:value={form.network_trace} required />
        <datalist id="network-traces">
          {#each options?.network_traces || [] as trace}
            <option value={trace}></option>
          {/each}
        </datalist>
      </label>
      <label>FoV trace
        <input list="fov-traces" bind:value={form.fov_trace} required />
        <datalist id="fov-traces">
          {#each options?.fov_traces || [] as trace}
            <option value={trace}></option>
          {/each}
        </datalist>
      </label>
      <label>ABR
        <select bind:value={form.abr}>
          {#each options?.abr_classes || [] as abr}
            <option value={abr}>{abr}</option>
          {/each}
        </select>
      </label>
      <label>FoV predictor
        <select bind:value={form.predictor}>
          <option value="None">None</option>
          {#each options?.fov_predictors || [] as predictor}
            <option value={predictor}>{predictor}</option>
          {/each}
        </select>
      </label>
      <label class="compact">Fetch window (segments)
        <input type="number" min="0" step="1" bind:value={form.buffer_window_segments} />
      </label>
      <label class="pill-toggle">
        <input type="checkbox" bind:checked={form.enable_mid_download_cancel} />
        Adaptive abort
      </label>
    </div>
    <div class="form-footer">
      <button type="submit" disabled={submitting || loading}>Start run</button>
      <span class="status">{statusMessage}</span>
    </div>
  </form>
</section>
