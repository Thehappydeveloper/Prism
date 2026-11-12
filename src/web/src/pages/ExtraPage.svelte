<script lang="ts">
  import { gotoRoute } from '../core/router';
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

<main class="extra-layout">
  <section class="card">
    <div class="section-header">
      <div>
        <p class="eyebrow micro">Tools</p>
        <h2>Available tools</h2>
      </div>
    </div>

    <div class="tool-grid">
      <a class="tool-card" href="/static/training.html" on:click|preventDefault={() => gotoRoute('training')}>
        <div class="tool-icon" aria-hidden="true">◎</div>
        <div>
          <h3>Training Setup</h3>
          <p>Register training traces, train FoV or bandwidth predictors, inspect results, and export models for Prism simulations.</p>
        </div>
        <span class="button-link">Open</span>
      </a>
      <a class="tool-card" href="/static/speed-acceleration.html" on:click|preventDefault={() => gotoRoute('speedAcceleration')}>
        <div class="tool-icon" aria-hidden="true">↗</div>
        <div>
          <h3>Speed &amp; Acceleration Analyzer</h3>
          <p>Scan local CSV folders, select motion columns, and review statistics with distribution histograms.</p>
        </div>
        <span class="button-link">Open</span>
      </a>
      <a class="tool-card" href="/static/point-cloud-sequence-analyzer.html" on:click|preventDefault={() => gotoRoute('pointCloudSequence')}>
        <div class="tool-icon" aria-hidden="true">▦</div>
        <div>
          <h3>Point Cloud Sequence Analyzer</h3>
          <p>Analyze PLY frame geometry, density, temporal stability, and V-PCC QP safety scores.</p>
        </div>
        <span class="button-link">Open</span>
      </a>
      <a class="tool-card" href="/static/dataset-transformation.html" on:click|preventDefault={() => gotoRoute('traceHeightNormalizer')}>
        <div class="tool-icon" aria-hidden="true">⇄</div>
        <div>
          <h3>Dataset Transformation Tool</h3>
          <p>Normalize FoV traces and PLY sequences with shared scaling, axis swaps, positive shifting, and optional vox export.</p>
        </div>
        <span class="button-link">Open</span>
      </a>
    </div>
  </section>
</main>

<style>
  .extra-layout {
    max-width: 980px;
  }

  .tool-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 0.9rem;
    margin-top: 0.9rem;
  }

  .tool-card {
    display: grid;
    grid-template-columns: 2.4rem minmax(0, 1fr) auto;
    gap: 0.85rem;
    align-items: center;
    padding: 1rem;
    border: 1px solid var(--card-border);
    border-radius: 12px;
    background: var(--soft-bg-alt);
    color: var(--text-main);
    text-decoration: none;
    transition: transform 0.15s ease, border-color 0.2s ease, background-color 0.2s ease;
  }

  .tool-card:hover {
    transform: translateY(-1px);
    border-color: var(--ghost-border);
    background: var(--soft-bg);
    text-decoration: none;
  }

  .tool-icon {
    width: 2.25rem;
    height: 2.25rem;
    display: grid;
    place-items: center;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.28);
    color: var(--accent-main);
    font-weight: 800;
    font-size: 1.15rem;
  }

  .tool-card h3 {
    margin: 0 0 0.25rem;
    font-size: 1rem;
  }

  .tool-card p {
    margin: 0;
    color: var(--text-muted);
    line-height: 1.38;
  }

  @media (max-width: 760px) {
    .tool-card {
      grid-template-columns: 2.4rem minmax(0, 1fr);
    }

    .tool-card .button-link {
      grid-column: 2;
      width: fit-content;
    }
  }
</style>
