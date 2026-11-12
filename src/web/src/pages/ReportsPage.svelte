<script lang="ts">
  import { onMount } from 'svelte';
  import { buildReport, listReportBuilds, listReports, updateReportFigureFontSize, validateReport, type ReportBuild, type ReportDefinition, type ReportResult } from '../core/api';
  import { gotoRoute } from '../core/router';

  type BuildWithReport = ReportBuild & { reportId: string };
  let reports: ReportDefinition[] = [];
  let selected = '';
  let result: ReportResult | null = null;
  let builds: ReportBuild[] = [];
  let busy = false;
  let error = '';
  let activity = '';
  let completedSteps = 0;
  let totalSteps = 0;
  let previewTable: string[][] = [];
  let previewTableName = '';
  let previewMarkdown = '';
  let copyMessage = '';
  let previewFigureUrl = '';
  let previewFigureName = '';
  let previewDownloadUrl = '';
  let previewDownloadName = '';
  let pointCloudOverlay = true;
  let motionTraceOverlay = true;
  let figureFontSizePt = 9;

  $: selectedReport = reports.find((report) => report.id === selected);
  $: latestBuild = builds[0];
  $: progress = totalSteps ? Math.round((completedSteps / totalSteps) * 100) : 0;
  $: topLevelReports = reports.filter((report) => !report.parent_report);
  $: childReports = selectedReport?.included_reports?.map((id) => reports.find((report) => report.id === id)).filter(Boolean) as ReportDefinition[] || [];

  function buildUrl(reportId: string, buildId: string, output: string): string {
    return `/reports/${encodeURIComponent(reportId)}/${encodeURIComponent(buildId)}/${output.split('/').map(encodeURIComponent).join('/')}`;
  }
  function parseMarkdownTable(text: string): string[][] {
    const rows = text.split('\n').map((line) => line.trim()).filter((line) => line.startsWith('|') && line.endsWith('|')).map((line) => line.slice(1, -1).split('|').map((cell) => cell.trim()));
    return rows.filter((row) => !row.every((cell) => /^:?-{3,}:?$/.test(cell)));
  }
  async function refresh(): Promise<void> {
    try { reports = (await listReports()).reports || []; if (!selected && reports[0]?.id) await selectReport(reports[0].id); }
    catch (err) { error = err instanceof Error ? err.message : String(err); }
  }
  async function selectReport(id: string): Promise<void> {
    selected = id; result = null; previewTable = []; previewMarkdown = ''; copyMessage = ''; previewFigureUrl = ''; previewDownloadUrl = '';
    const report = reports.find((item) => item.id === id);
    pointCloudOverlay = Boolean(report?.render_options?.point_cloud_overlay?.default ?? true);
    motionTraceOverlay = Boolean(report?.render_options?.motion_trace_overlay?.default ?? true);
    figureFontSizePt = Number(report?.render_options?.figure_font_size_pt?.default ?? 9);
    try { builds = (await listReportBuilds(id)).builds || []; }
    catch (err) { error = err instanceof Error ? err.message : String(err); builds = []; }
  }
  async function validate(): Promise<void> {
    if (!selected) return;
    busy = true; error = ''; activity = 'Checking source runs…'; completedSteps = 0; totalSteps = 1;
    try { result = await validateReport(selected); completedSteps = 1; }
    catch (err) { error = err instanceof Error ? err.message : String(err); }
    finally { busy = false; activity = ''; }
  }
  async function build(): Promise<void> {
    if (!selected) return;
    busy = true; error = ''; completedSteps = 0; totalSteps = 3;
    try {
      activity = 'Step 1 of 3: validating required experiment runs…';
      result = await validateReport(selected);
      if (result.status === 'error') return;
      completedSteps = 1;
      activity = 'Step 2 of 3: regenerating tables and PDF figures…';
      result = await buildReport(selected, selectedReport?.render_options ? {
        ...(selectedReport.render_options.point_cloud_overlay ? { point_cloud_overlay: pointCloudOverlay } : {}),
        ...(selectedReport.render_options.motion_trace_overlay ? { motion_trace_overlay: motionTraceOverlay } : {}),
      } : undefined);
      if (!result.built) throw new Error((result.issues || []).map((issue) => issue.message).join(' ') || 'The report build did not complete.');
      completedSteps = 2;
      activity = 'Step 3 of 3: indexing the new immutable artifact bundle…';
      builds = (await listReportBuilds(selected)).builds || [];
      const newest = builds[0];
      if (newest?.outputs?.includes('paper-table.md')) {
        await previewTableFor({ ...newest, reportId: selected }, 'paper-table.md');
      }
      completedSteps = 3;
    }
    catch (err) { error = err instanceof Error ? err.message : String(err); }
    finally { busy = false; activity = ''; }
  }
  async function buildPaperArtifacts(): Promise<void> {
    await selectReport('argus_mmsys2027');
    await build();
  }
  function openBatchPlan(reportId: string, planId: string): void {
    window.location.assign(`/static/aggregate.html?report=${encodeURIComponent(reportId)}&plan=${encodeURIComponent(planId)}`);
  }
  async function adjustSvgFontSize(): Promise<void> {
    if (!selected) return;
    busy = true; error = ''; activity = 'Updating the figure font only…';
    try {
      await updateReportFigureFontSize(selected, figureFontSizePt);
      builds = (await listReportBuilds(selected)).builds || [];
      const figure = builds[0]?.outputs.find((output) => output.endsWith('.svg') || output.endsWith('.png'));
      if (figure && builds[0]) previewFigureFor({ ...builds[0], reportId: selected }, figure);
    } catch (err) { error = err instanceof Error ? err.message : String(err); }
    finally { busy = false; activity = ''; }
  }
  async function previewTableFor(build: BuildWithReport, output: string): Promise<void> {
    try {
      const response = await fetch(buildUrl(build.reportId, build.build_id, output));
      if (!response.ok) throw new Error(`Could not load ${output}.`);
      previewMarkdown = await response.text(); previewTable = parseMarkdownTable(previewMarkdown); previewTableName = `${build.build_id} · ${output}`; copyMessage = ''; previewFigureUrl = ''; previewDownloadUrl = buildUrl(build.reportId, build.build_id, output); previewDownloadName = output;
    } catch (err) { error = err instanceof Error ? err.message : String(err); }
  }
  function previewFigureFor(build: BuildWithReport, output: string): void {
    previewDownloadUrl = buildUrl(build.reportId, build.build_id, output); previewFigureUrl = `${previewDownloadUrl}?v=${Date.now()}`; previewFigureName = `${build.build_id} · ${output}`; previewTable = []; previewMarkdown = ''; copyMessage = ''; previewDownloadName = output;
  }
  async function copyMarkdown(): Promise<void> {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(previewMarkdown);
      } else {
        const textarea = document.createElement('textarea');
        textarea.value = previewMarkdown;
        textarea.setAttribute('readonly', '');
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        const copied = document.execCommand('copy');
        textarea.remove();
        if (!copied) throw new Error('The browser denied the copy request.');
      }
      copyMessage = 'Markdown copied.';
    }
    catch (err) { copyMessage = `Could not copy: ${err instanceof Error ? err.message : String(err)}`; }
  }
  function outputKind(output: string): 'table' | 'figure' | null { if (output.endsWith('.md')) return 'table'; if (output.endsWith('.pdf') || output.endsWith('.png') || output.endsWith('.svg')) return 'figure'; return null; }
  function artifactLabel(output: string): string {
    const labels: Record<string, string> = {
      'tables.md': 'Paper input snapshot',
      'raycast_process_scaling/paper-table.md': 'Table: process scaling',
      'paper-table.md': 'Report results table',
    };
    if (labels[output]) return labels[output];
    const name = output.split('/').pop() || output;
    const compact = name.match(/^segment_(\d+)_all_predictors\.png$/);
    if (compact) return `Segment ${Number(compact[1])}: ground truth + five predictors`;
    const pair = name.match(/^segment_(\d+)_(.+)_vs_ground_truth\.png$/);
    if (pair) return `Segment ${Number(pair[1])}: ${(pair[2] || 'predictor').replace(/_/g, ' ')} vs ground truth`;
    if (name === 'segments_12_14_all_predictors.png') return 'S12 and S14: 1×2 segment grid of six-panel comparisons';
    return name.replace(/\.(pdf|png)$/, '').replace(/_/g, ' ');
  }
  onMount(refresh);
</script>

<header class="hero">
  <div class="hero-actions">
    <a class="button-link ghost" href="/" on:click|preventDefault={() => gotoRoute('home')}>Main menu</a>
    <a class="button-link ghost" href="/static/datasets.html" on:click|preventDefault={() => gotoRoute('datasets')}>Datasets</a>
    <a class="button-link ghost" href="/static/index.html" on:click|preventDefault={() => gotoRoute('control')}>Single Run</a>
    <a class="button-link ghost" href="/static/aggregate.html" on:click|preventDefault={() => gotoRoute('batch')}>Batch Planner</a>
    <a class="button-link ghost" href="/static/compare.html" on:click|preventDefault={() => gotoRoute('compare')}>Compare Runs</a>
    <a class="button-link active" aria-current="page" href="/static/reports.html" on:click|preventDefault={() => gotoRoute('reports')}>Reports</a>
    <a class="button-link ghost" href="/static/extra.html" on:click|preventDefault={() => gotoRoute('extra')}>Extra</a>
    <a class="button-link ghost" href="/static/settings.html" on:click|preventDefault={() => gotoRoute('settings')}>Settings</a>
  </div>
</header>

<main class="reports-layout">
  <section class="card introduction">
    <div><p class="eyebrow micro">Publication workflow</p><h1>Paper reports</h1><p class="subtle">The Prism paper is the parent report. Its child reports show the exact batch-plan JSON required for their source runs, then preserve versioned report artifacts.</p></div>
    <div class="paper-action"><strong>Argus MMSys 2027 hierarchy</strong><span>Open the Argus reproduction report to see every included report and its plan. Plan buttons pre-load the JSON in Batch Planner for review before launch.</span><button on:click={() => selectReport('argus_mmsys2027')} disabled={busy}>Open Argus reproduction</button></div>
  </section>

  {#if busy}
    <section class="card progress-card" aria-live="polite"><div><strong>{activity}</strong><span>{completedSteps} / {totalSteps} completed</span></div><progress value={completedSteps} max={totalSteps || 1}>{progress}%</progress><small>{progress}% complete. Figure generation may take several minutes because it recomputes outputs from run artifacts.</small></section>
  {/if}
  {#if error}<p class="error-text" role="alert">{error}</p>{/if}

  <section class="card report-grid">
    <aside class="report-library">
      <h2>Report hierarchy</h2><p class="subtle compact">The paper report contains all Prism paper reports. Select a report to inspect its source plan(s) and artifacts.</p>
      {#if !reports.length}<p class="subtle">No report definitions found.</p>{/if}
      {#each topLevelReports as report}
        <button class:selected={selected === report.id} class="report-card" type="button" on:click={() => selectReport(report.id)}><strong>{report.title}</strong><span>{report.description || report.id}</span><small>{report.outputs.some((output) => output.startsWith('figures/')) ? 'Tables + figures' : 'Source snapshot'}</small></button>
        {#each report.included_reports || [] as childId}
          {@const child = reports.find((item) => item.id === childId)}
          {#if child}<button class:selected={selected === child.id} class="report-card child-report" type="button" on:click={() => selectReport(child.id)}><strong>↳ {child.title}</strong><span>{child.description || child.id}</span></button>{/if}
        {/each}
      {/each}
    </aside>
    <div class="report-workspace">
      <div class="workspace-title"><div><p class="eyebrow micro">Selected artifact set</p><h2>{selectedReport?.title || 'Select a report'}</h2></div>{#if latestBuild}<span class="pill ready">Latest build available</span>{/if}</div>
      <p class="subtle">{selectedReport?.description || 'Select an artifact set from the library.'}</p>
      <div class="explain"><strong>Workflow</strong><span>First use a plan button below to load its named JSON in Batch Planner, review inputs, and launch only if needed. Once runs are complete, validate and regenerate report artifacts here.</span></div>
      {#if selectedReport?.batch_plans?.length}<section class="plan-list"><strong>Source batch plan{selectedReport.batch_plans.length === 1 ? '' : 's'}</strong><p class="subtle compact">Each button loads the exact JSON plan into Batch Planner; it does not launch runs automatically.</p>{#each selectedReport.batch_plans as plan}<div class="plan-row"><span><b>{plan.label}</b><code>{plan.path}</code></span><button class="ghost" disabled={busy} on:click={() => openBatchPlan(selected, plan.id)}>Generate from this plan</button></div>{/each}</section>{/if}
      {#if childReports.length}<section class="plan-list"><strong>Included reports</strong><p class="subtle compact">The parent report contains these reports; each exposes its own plan below when selected.</p>{#each childReports as child}<button class="text-button" on:click={() => selectReport(child.id)}>Open {child.title}</button>{/each}</section>{/if}
      <div class="actions"><button class="ghost" disabled={busy || !selected} on:click={validate}>Validate report inputs</button><button disabled={busy || !selected} on:click={build}>{busy ? 'Working…' : 'Regenerate report artifacts'}</button>{#if selectedReport?.render_options?.figure_font_size_pt}<label class="render-option">{selectedReport.render_options.figure_font_size_pt.label || 'Final figure font size (pt)'} <input type="number" min={selectedReport.render_options.figure_font_size_pt.min} max={selectedReport.render_options.figure_font_size_pt.max} step="0.5" bind:value={figureFontSizePt} disabled={busy} /></label><button class="ghost" disabled={busy} on:click={adjustSvgFontSize}>Apply to figure only</button>{/if}{#if selectedReport?.render_options?.point_cloud_overlay}<label class="render-option"><input type="checkbox" bind:checked={pointCloudOverlay} disabled={busy} /> {selectedReport.render_options.point_cloud_overlay.label || 'Overlay point cloud'}</label>{/if}{#if selectedReport?.render_options?.motion_trace_overlay}<label class="render-option"><input type="checkbox" bind:checked={motionTraceOverlay} disabled={busy} /> {selectedReport.render_options.motion_trace_overlay.label || 'Overlay motion trace'}</label>{/if}</div>
      {#if result}<div class={`result ${result.status}`}><strong>{result.status === 'ready' ? 'Inputs ready' : result.status}</strong><p>{result.runs_finished} finished / {result.runs_found} required runs</p>{#each result.issues as issue}<p class={issue.severity}>{issue.severity}: {issue.message}</p>{/each}{#if result.built}<p>New build <code>{result.build_id}</code> is ready below.</p>{/if}</div>{/if}
      <section class="artifact-browser"><div class="artifact-heading"><h3>Generated artifacts</h3><p class="subtle compact">Tables and figures live in timestamped builds. Preview an artifact here; download is available from its preview.</p></div>
        {#if !builds.length}<div class="empty-state">No build exists for this set yet. Select <strong>Generate this set</strong> or regenerate the complete paper set above.</div>
        {:else}{#each builds.slice(0, 1) as build}<article class="build-card"><div class="build-meta"><strong>{build.build_id}</strong><span>{build.runs ?? '—'} source runs</span></div><div class="artifact-links">{#each build.outputs as output}{@const kind = outputKind(output)}{#if kind === 'table'}<button class="text-button" on:click={() => previewTableFor({ ...build, reportId: selected }, output)}>Preview {artifactLabel(output)}</button>{:else if kind === 'figure'}<button class="text-button" on:click={() => previewFigureFor({ ...build, reportId: selected }, output)}>Preview {artifactLabel(output)}</button>{/if}{/each}</div></article>{/each}{/if}
      </section>
    </div>
  </section>

  {#if previewTable.length}<section class="card preview-card"><div class="preview-heading"><div><p class="eyebrow micro">Table preview</p><h2>{previewTableName}</h2></div><div class="preview-actions"><button class="ghost small" on:click={copyMarkdown}>Copy Markdown</button><a class="button-link ghost small" href={previewDownloadUrl} download={previewDownloadName}>Download</a><button class="ghost small" on:click={() => previewTable = []}>Close</button></div></div>{#if copyMessage}<p class="copy-message" aria-live="polite">{copyMessage}</p>{/if}<div class="table-scroll"><table><thead><tr>{#each previewTable[0] || [] as cell}<th>{cell}</th>{/each}</tr></thead><tbody>{#each previewTable.slice(1) as row}<tr>{#each row as cell}<td>{cell}</td>{/each}</tr>{/each}</tbody></table></div></section>{/if}
  {#if previewFigureUrl}<section class="card preview-card"><div class="preview-heading"><div><p class="eyebrow micro">Figure preview</p><h2>{previewFigureName}</h2></div><div class="preview-actions"><a class="button-link ghost small" href={previewDownloadUrl} download={previewDownloadName}>Download figure</a><button class="ghost small" on:click={() => previewFigureUrl = ''}>Close</button></div></div>{#if previewDownloadName.endsWith('.png') || previewDownloadName.endsWith('.svg')}<img class="figure-image-preview" src={previewFigureUrl} alt={previewFigureName} />{:else}<object class="figure-preview" data={previewFigureUrl} type="application/pdf" title={previewFigureName}><a href={previewDownloadUrl} target="_blank" rel="noreferrer">Open the PDF figure</a></object>{/if}</section>{/if}
</main>

<style>
  .reports-layout { max-width: 1240px; margin: 0 auto; padding: 1.2rem; display: grid; gap: 1rem; } h1, h2, h3, p { margin-top: 0; } h1 { margin-bottom: .35rem; } h2 { margin-bottom: .3rem; } h3 { margin-bottom: .2rem; }
  .introduction { display: grid; grid-template-columns: minmax(0, 1fr) minmax(290px, .42fr); gap: 1.25rem; align-items: center; }.paper-action { display: grid; gap: .35rem; padding: 1rem; border-radius: .7rem; background: color-mix(in srgb, var(--accent, #0891b2) 10%, transparent); border: 1px solid color-mix(in srgb, var(--accent, #0891b2) 28%, var(--border, #cbd5e1)); }.paper-action span, .subtle { color: var(--muted, #64748b); }.compact { margin-bottom: .6rem; } button { width: fit-content; }
  .progress-card { display: grid; gap: .55rem; }.progress-card > div, .workspace-title, .preview-heading, .build-meta { display: flex; align-items: center; justify-content: space-between; gap: .8rem; } progress { width: 100%; height: .7rem; accent-color: var(--accent, #0891b2); }.progress-card small { color: var(--muted, #64748b); }
  .report-grid { display: grid; grid-template-columns: minmax(270px, .8fr) minmax(0, 1.7fr); gap: 1.25rem; }.report-library { padding-right: 1.1rem; border-right: 1px solid var(--border, #cbd5e1); }.report-card { width: 100%; text-align: left; display: grid; gap: .3rem; padding: .8rem; margin-bottom: .55rem; border: 1px solid var(--border, #cbd5e1); border-radius: .55rem; background: var(--panel, #fff); color: inherit; }.report-card.child-report { margin-left: 1rem; width: calc(100% - 1rem); }.report-card.selected { border-color: var(--accent, #0891b2); box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent, #0891b2) 18%, transparent); }.report-card span, .report-card small { color: var(--muted, #64748b); }
  .explain { display: grid; gap: .25rem; padding: .8rem; border-radius: .55rem; background: color-mix(in srgb, #0891b2 8%, transparent); color: var(--muted, #475569); }.plan-list { display: grid; gap: .45rem; margin-top: 1rem; padding: .8rem; border: 1px solid var(--border, #cbd5e1); border-radius: .55rem; }.plan-list p { margin: 0; }.plan-row { display:flex; align-items:center; justify-content:space-between; gap:.8rem; padding:.55rem; border-radius:.4rem; background:color-mix(in srgb, var(--accent, #0891b2) 6%, transparent); }.plan-row span { display:grid; gap:.2rem; }.plan-row code { font-size:.75rem; color:var(--muted, #64748b); }.actions, .artifact-links { display: flex; flex-wrap: wrap; gap: .55rem; margin: 1rem 0; }.render-option { display: inline-flex; align-items: center; gap: .3rem; font-size: .9rem; color: var(--muted, #475569); }.render-option input[type=number] { width: 5rem; }.result { border-left: 3px solid #0891b2; padding: .7rem .9rem; background: color-mix(in srgb, #0891b2 7%, transparent); }.result p { margin: .3rem 0 0; }.result.error { border-color: #dc2626; }.warning { color: #a16207; }.error, .error-text { color: #b91c1c; } code { word-break: break-all; }
  .artifact-browser { margin-top: 1.25rem; border-top: 1px solid var(--border, #cbd5e1); padding-top: 1rem; }.empty-state { padding: 1rem; color: var(--muted, #64748b); border: 1px dashed var(--border, #cbd5e1); border-radius: .55rem; }.build-card { padding: .75rem 0; border-bottom: 1px solid var(--border, #cbd5e1); }.build-card:last-child { border-bottom: 0; }.build-meta span { color: var(--muted, #64748b); font-size: .88rem; }.artifact-links { margin: .5rem 0 0; align-items: center; }.text-button { font-size: .9rem; font-weight: 600; border: 0; padding: 0; background: transparent; color: var(--accent, #0891b2); text-decoration: underline; cursor: pointer; }.pill { border-radius: 999px; padding: .26rem .55rem; font-size: .78rem; white-space: nowrap; }.pill.ready { color: #167744; background: color-mix(in srgb, #22c55e 15%, transparent); }
  .preview-card { overflow: hidden; }.preview-heading { margin-bottom: .9rem; }.preview-heading h2 { font-size: 1rem; }.preview-actions { display: flex; gap: .5rem; align-items: center; }.copy-message { margin: -.35rem 0 .65rem; color: #167744; font-size: .88rem; }.small { font-size: .85rem; }.table-scroll { overflow: auto; max-height: 500px; border: 1px solid var(--border, #cbd5e1); border-radius: .5rem; } table { width: 100%; border-collapse: collapse; font-size: .9rem; } th, td { padding: .6rem .7rem; border-bottom: 1px solid var(--border, #cbd5e1); text-align: left; white-space: nowrap; } th { position: sticky; top: 0; background: var(--panel, #fff); }.figure-preview { width: 100%; height: min(72vh, 760px); border: 1px solid var(--border, #cbd5e1); border-radius: .5rem; background: #fff; }.figure-image-preview { display: block; width: 100%; max-height: 78vh; object-fit: contain; border: 1px solid var(--border, #cbd5e1); border-radius: .5rem; background: #fff; }
  @media (max-width: 820px) { .introduction, .report-grid { grid-template-columns: 1fr; }.report-library { padding: 0 0 1rem; border-right: 0; border-bottom: 1px solid var(--border, #cbd5e1); } }
</style>
