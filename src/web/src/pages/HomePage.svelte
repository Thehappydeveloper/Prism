<script lang="ts">
  import { gotoRoute } from '../core/router';
</script>

<header class="hero home-hero">
  <div class="hero-actions">
    <a class="button-link active" aria-current="page" href="/" on:click|preventDefault={() => gotoRoute('home')}>Main menu</a>
    <a class="button-link ghost" href="/static/datasets.html" on:click|preventDefault={() => gotoRoute('datasets')}>Datasets</a>
    <a class="button-link ghost" href="/static/index.html" on:click|preventDefault={() => gotoRoute('control')}>Single Run</a>
    <a class="button-link ghost" href="/static/aggregate.html" on:click|preventDefault={() => gotoRoute('batch')}>Batch Planner</a>
    <a class="button-link ghost" href="/static/compare.html" on:click|preventDefault={() => gotoRoute('compare')}>Compare Runs</a>
    <a class="button-link ghost" href="/static/reports.html" on:click|preventDefault={() => gotoRoute('reports')}>Reports</a>
    <a class="button-link ghost" href="/static/extra.html" on:click|preventDefault={() => gotoRoute('extra')}>Extra</a>
    <a class="button-link ghost" href="/static/settings.html" on:click|preventDefault={() => gotoRoute('settings')}>Settings</a>
  </div>
</header>

<main class="home-layout">
  <section class="home-panel workflow-panel">
    <div class="workflow-head">
      <div>
        <p class="eyebrow micro">Workflow</p>
        <h2>Start with the dataset, then run experiments</h2>
      </div>
    </div>

    <div class="workflow-list">
      <article class="workflow-step">
        <div class="step-number">1</div>
        <div class="step-copy">
          <h3>Dataset Registry</h3>
          <p>Bind each MPD/PLY source to its coordinate convention and meter-scale calibration.</p>
        </div>
        <a class="button-link" href="/static/datasets.html" on:click|preventDefault={() => gotoRoute('datasets')}>Open</a>
      </article>

      <article class="workflow-step">
        <div class="step-number">6</div>
        <div class="step-copy"><h3>Reports</h3><p>Validate completed experiment cohorts and create frozen table/data snapshots for publication.</p></div>
        <a class="button-link" href="/static/reports.html" on:click|preventDefault={() => gotoRoute('reports')}>Open</a>
      </article>

      <article class="workflow-step">
        <div class="step-number">2</div>
        <div class="step-copy">
          <h3>Training</h3>
          <p>Register training traces, train FoV or bandwidth predictors, inspect results, and export models for simulations.</p>
        </div>
        <a class="button-link" href="/static/training.html" on:click|preventDefault={() => gotoRoute('training')}>Open</a>
      </article>

      <article class="workflow-step">
        <div class="step-number">3</div>
        <div class="step-copy">
          <h3>Single Run</h3>
          <p>Launch one playback to confirm traces, predictors, ABR choices, and dataset calibration behave as expected.</p>
        </div>
        <a class="button-link" href="/static/index.html" on:click|preventDefault={() => gotoRoute('control')}>Open</a>
      </article>

      <article class="workflow-step">
        <div class="step-number">4</div>
        <div class="step-copy">
          <h3>Batch Planner</h3>
          <p>Queue combinations of datasets, traces, ABRs, and predictors after the single-run setup looks right.</p>
        </div>
        <a class="button-link" href="/static/aggregate.html" on:click|preventDefault={() => gotoRoute('batch')}>Open</a>
      </article>

      <article class="workflow-step">
        <div class="step-number">5</div>
        <div class="step-copy">
          <h3>Compare Runs</h3>
          <p>Review quality, stall, buffer, FoV, and QoE views across completed runs.</p>
        </div>
        <a class="button-link" href="/static/compare.html" on:click|preventDefault={() => gotoRoute('compare')}>Open</a>
      </article>

      <article class="workflow-step secondary-step">
        <div class="step-icon" aria-hidden="true">↗</div>
        <div class="step-copy">
          <h3>Extra</h3>
          <p>Open standalone tools such as CSV-based speed and acceleration analysis.</p>
        </div>
        <a class="button-link" href="/static/extra.html" on:click|preventDefault={() => gotoRoute('extra')}>Open</a>
      </article>

      <article class="workflow-step secondary-step">
        <div class="step-icon" aria-hidden="true">⚙</div>
        <div class="step-copy">
          <h3>Settings</h3>
          <p>Adjust headsets, trace roots, Navis export, and diagnostic recording when the workspace defaults need to change.</p>
        </div>
        <a class="button-link" href="/static/settings.html" on:click|preventDefault={() => gotoRoute('settings')}>Open</a>
      </article>
    </div>
  </section>
</main>

<style>
  .home-hero {
    padding-bottom: 0.75rem;
  }

  .home-layout {
    max-width: 980px;
  }

  .home-panel {
    border: 1px solid var(--card-border);
    border-radius: 14px;
    background: var(--card-bg);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.24);
  }

  .workflow-panel {
    padding: 1.15rem;
  }

  .workflow-head {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-start;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid var(--card-border);
  }

  .workflow-head h2 {
    margin: 0.15rem 0 0;
    font-size: 1.2rem;
  }

  .workflow-list {
    display: grid;
  }

  .workflow-step {
    display: grid;
    grid-template-columns: 2.25rem minmax(0, 1fr) auto;
    gap: 0.85rem;
    align-items: center;
    padding: 1rem 0;
    border-bottom: 1px solid var(--card-border);
  }

  .workflow-step:last-child {
    border-bottom: 0;
    padding-bottom: 0.2rem;
  }

  .secondary-step {
    background: rgba(148, 163, 184, 0.04);
    margin: 0 -0.35rem;
    padding-left: 0.35rem;
    padding-right: 0.35rem;
    border-radius: 8px;
  }

  .step-number,
  .step-icon {
    width: 2rem;
    height: 2rem;
    display: grid;
    place-items: center;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.28);
    color: var(--accent-main);
    font-weight: 750;
  }

  .step-icon {
    font-size: 0.95rem;
  }

  .step-copy h3 {
    margin: 0 0 0.22rem;
    font-size: 1rem;
  }

  .step-copy p {
    margin: 0;
    color: var(--text-muted);
    line-height: 1.38;
  }

  @media (max-width: 860px) {
    .workflow-head,
    .workflow-step {
      grid-template-columns: 1fr;
    }

    .workflow-step {
      align-items: start;
    }
  }
</style>
