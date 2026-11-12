<script lang="ts">
  import { gotoRoute } from '../../../core/router';
  import { fovPredictorLabel } from '../../../core/predictorLabels';
  import type { AlgorithmParamField, AppOptions, DatasetProfile, HeadsetPreset, ParamMap, RunConfig, RunFormSession } from '../../../core/types';
  import {
    analyzeTracePaths,
    groupTraces,
    summarizeSingleTrace,
    traceDisplayName,
    type TraceFilterOption,
    type TraceGroupingOption,
    type TraceSortMode,
  } from '../../../core/traceBrowser';
  import {
    deletePreset,
    getPresetParams,
    listPresetNames,
    loadHyperparamPresets,
    saveHyperparamPresets,
    type HyperparamPresetStore,
    type PresetScope,
    upsertPreset,
  } from '../../../core/hyperparamPresets';

  export let options: AppOptions | null = null;
  export let form: RunConfig = {
    v3c_folder: '',
    content_dir: '',
    network_trace: '',
    network_profile: null,
    fov_trace: '',
    abr: 'BandwidthAwareABR',
    predictor: 'None',
    bandwidth_predictor: 'None',
    rtt: 0.06,
    buffer_window_segments: 2,
    segment_limit: 0,
    enable_mid_download_cancel: false,
  };
  export let submitting = false;
  export let loading = false;

  export let onSubmit: (config: RunConfig) => void = () => {};
  export let onRefreshOptions: () => void = () => {};
  export let onSaveSession: () => void = () => {};
  export let onLoadSession: (name: string) => void = () => {};
  export let onDeleteSession: (name: string) => void = () => {};
  export let savedSessions: RunFormSession[] = [];

  const CUSTOM_PRESET = '__custom__';
  type TracePickerMode = 'network' | 'fov';
  type ParamSpecGroup = { category: string; specs: AlgorithmParamField[] };
  let presets: HyperparamPresetStore = loadHyperparamPresets();
  let showEditor = false;
  let editorScope: PresetScope = 'abr';
  let editorPresetSelection = CUSTOM_PRESET;
  let editorPresetName = '';
  let editorParams: ParamMap = {};
  let editorError = '';
  let abrParamSpecs: AlgorithmParamField[] = [];
  let predictorName = '';
  let fovParamSpecs: AlgorithmParamField[] = [];
  let bandwidthPredictorName = '';
  let bandwidthParamSpecs: AlgorithmParamField[] = [];
  let predictorChoices: string[] = ['None'];
  let bandwidthPredictorChoices: string[] = ['None'];
  let predictorHelp = 'FoV predictors are optional for this ABR.';
  let abrPresetNames: string[] = [];
  let fovPresetNames: string[] = [];
  let bandwidthPresetNames: string[] = [];
  let headsetPresets: HeadsetPreset[] = [];
  let headsetChoices: Array<{ id: string; label: string }> = [];
  let datasetChoices: DatasetProfile[] = [];
  let tracePickerMode: TracePickerMode | null = null;
  let traceGroupingId = '';
  let traceFilterText = '';
  let traceSortMode: TraceSortMode = 'name';
  let tracePickerSelection = '';
  let traceFilterSelections: Record<string, string[]> = {};
  let selectedSessionName = '';
  let networkSource: 'trace' | 'profile' = 'trace';
  $: abrParamSpecs = options?.abr_param_specs?.[form.abr] || [];
  $: predictorChoices = compatiblePredictorChoices(form.abr);
  $: bandwidthPredictorChoices = ['None', ...(options?.bandwidth_predictors || [])];
  $: predictorHelp = describePredictorSupport(predictorChoices);
  $: if (options) normalizePredictorSelection();
  $: if (options) normalizeBandwidthPredictorSelection();
  $: predictorName = form.predictor && form.predictor !== 'None' ? form.predictor : '';
  $: bandwidthPredictorName = form.bandwidth_predictor && form.bandwidth_predictor !== 'None' ? form.bandwidth_predictor : '';
  $: fovParamSpecs = predictorName ? options?.fov_param_specs?.[predictorName] || [] : [];
  $: bandwidthParamSpecs = bandwidthPredictorName ? options?.bandwidth_param_specs?.[bandwidthPredictorName] || [] : [];
  $: abrPresetNames = listPresetNames(presets, 'abr', form.abr || '');
  $: fovPresetNames = predictorName ? listPresetNames(presets, 'fov', predictorName) : [];
  $: bandwidthPresetNames = bandwidthPredictorName ? listPresetNames(presets, 'bandwidth', bandwidthPredictorName) : [];
  $: headsetPresets = options?.headset_presets || [];
  $: headsetChoices = headsetPresets.map((preset) => ({
    id: preset.id,
    label: `${preset.label} · ${preset.fov_h_deg}° horizontal · ${preset.fov_v_deg}° vertical`,
  }));
  $: datasetChoices = options?.datasets || [];
  $: networkSource = form.network_profile ? 'profile' : 'trace';
  $: normalizeDatasetSelection();
  $: if (!form.headset_preset && options?.default_headset_preset) {
    applyHeadsetPreset(options.default_headset_preset);
  }
  $: if (form.abr_preset_name && !abrPresetNames.includes(form.abr_preset_name)) {
    setScopePresetName('abr', undefined);
  }
  $: if (form.predictor_preset_name && !fovPresetNames.includes(form.predictor_preset_name)) {
    setScopePresetName('fov', undefined);
  }
  $: if (form.bandwidth_predictor_preset_name && !bandwidthPresetNames.includes(form.bandwidth_predictor_preset_name)) {
    setScopePresetName('bandwidth', undefined);
  }

  function sameParamMap(a: ParamMap, b: ParamMap): boolean {
    const aKeys = Object.keys(a);
    const bKeys = Object.keys(b);
    if (aKeys.length !== bKeys.length) return false;
    return aKeys.every((k) => a[k] === b[k]);
  }

  function defaultParams(specs: AlgorithmParamField[] | undefined | null): ParamMap {
    const out: ParamMap = {};
    (specs || []).forEach((spec) => {
      out[spec.name] = spec.default;
    });
    return out;
  }

  function coerceScalar(spec: AlgorithmParamField, value: unknown): string | number | boolean | null {
    if (spec.kind === 'boolean') {
      return Boolean(value);
    }
    if (spec.kind === 'integer') {
      if (value === null || value === undefined || value === '') return spec.default;
      const parsed = Number.parseInt(String(value), 10);
      return Number.isFinite(parsed) ? parsed : spec.default;
    }
    if (spec.kind === 'number') {
      if (value === null || value === undefined || value === '') {
        return spec.default === null ? null : spec.default;
      }
      const parsed = Number.parseFloat(String(value));
      return Number.isFinite(parsed) ? parsed : spec.default;
    }
    if (value === null || value === undefined) return '';
    return String(value);
  }

  function normalizeParams(specs: AlgorithmParamField[] | undefined | null, raw: ParamMap | undefined): ParamMap {
    const safeSpecs = specs || [];
    const defaults = defaultParams(specs);
    const out: ParamMap = { ...defaults };
    if (!raw) return out;
    safeSpecs.forEach((spec) => {
      if (!(spec.name in raw)) return;
      out[spec.name] = coerceScalar(spec, raw[spec.name]);
    });
    return out;
  }

  function getScopeSpecs(scope: PresetScope): AlgorithmParamField[] {
    if (scope === 'abr') return abrParamSpecs || [];
    if (scope === 'bandwidth') return bandwidthParamSpecs || [];
    return fovParamSpecs || [];
  }

  function getScopeAlgo(scope: PresetScope): string {
    if (scope === 'abr') return form.abr;
    if (scope === 'bandwidth') return bandwidthPredictorName;
    return predictorName;
  }

  function getScopeParams(scope: PresetScope): ParamMap {
    if (scope === 'abr') return form.abr_params || {};
    if (scope === 'bandwidth') return form.bandwidth_predictor_params || {};
    return form.predictor_params || {};
  }

  function setScopeParams(scope: PresetScope, params: ParamMap): void {
    if (scope === 'abr') {
      form = { ...form, abr_params: { ...params } };
      return;
    }
    if (scope === 'bandwidth') {
      form = { ...form, bandwidth_predictor_params: { ...params } };
      return;
    }
    form = { ...form, predictor_params: { ...params } };
  }

  function setScopePresetName(scope: PresetScope, name: string | undefined): void {
    if (scope === 'abr') {
      form = { ...form, abr_preset_name: name };
      return;
    }
    if (scope === 'bandwidth') {
      form = { ...form, bandwidth_predictor_preset_name: name };
      return;
    }
    form = { ...form, predictor_preset_name: name };
  }

  function groupedParamSpecs(specs: AlgorithmParamField[], advanced: boolean): ParamSpecGroup[] {
    const groups = new Map<string, AlgorithmParamField[]>();
    specs
      .filter((spec) => Boolean(spec.advanced) === advanced)
      .forEach((spec) => {
        const category = spec.category || (advanced ? 'Advanced' : 'General');
        const current = groups.get(category) || [];
        current.push(spec);
        groups.set(category, current);
      });
    return Array.from(groups.entries()).map(([category, groupSpecs]) => ({ category, specs: groupSpecs }));
  }

  function paramHelp(spec: AlgorithmParamField): string {
    const details = spec.help || spec.annotation || 'No description provided.';
    return `${spec.label}: ${details}`;
  }

  function normalizeFormParamsBeforeSubmit(): void {
    const normalizedAbr = normalizeParams(abrParamSpecs, form.abr_params || {});
    const normalizedFov = predictorName ? normalizeParams(fovParamSpecs, form.predictor_params || {}) : {};
    const normalizedBandwidth = bandwidthPredictorName ? normalizeParams(bandwidthParamSpecs, form.bandwidth_predictor_params || {}) : {};
    const cleanNetwork = form.network_profile
      ? { network_profile: form.network_profile, network_trace: undefined }
      : { network_profile: undefined, network_trace: form.network_trace };
    form = {
      ...form,
      ...cleanNetwork,
      abr_params: normalizedAbr,
      predictor_params: normalizedFov,
      bandwidth_predictor_params: normalizedBandwidth,
    };
  }

  function setNetworkSource(source: 'trace' | 'profile'): void {
    if (source === 'trace') {
      form = { ...form, network_profile: null };
    } else {
      form = { ...form, network_trace: '', network_profile: form.network_profile || options?.network_profiles?.[0]?.id || 'ideal' };
    }
  }

  function networkProfileSummary(profileId: string | null | undefined): string {
    const profile = (options?.network_profiles || []).find((item) => item.id === profileId);
    if (!profile) return '';
    const rate = profile.rate_mbit == null ? 'unlimited' : `${profile.rate_mbit} Mbps`;
    return `${profile.description} · ${profile.delay_ms} ms delay · ${profile.jitter_ms} ms jitter · ${profile.loss_pct}% loss · ${rate}`;
  }

  function onNetworkProfileChanged(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement | null;
    form = { ...form, network_trace: '', network_profile: target?.value || 'ideal' };
  }

  function applyHeadsetPreset(presetId: string): void {
    const preset = headsetPresets.find((entry) => entry.id === presetId);
    if (!preset) return;
    form = {
      ...form,
      headset_preset: preset.id,
      fov_h_deg: preset.fov_h_deg,
      fov_v_deg: preset.fov_v_deg,
    };
  }

  function normalizeDatasetSelection(): void {
    if (!form.dataset_id) return;
    if (datasetChoices.some((dataset) => dataset.id === form.dataset_id)) return;
    form = { ...form, dataset_id: '', v3c_folder: '', content_dir: '', mpd: undefined };
  }

  function applyDataset(datasetId: string): void {
    const dataset = datasetChoices.find((entry) => entry.id === datasetId);
    if (!dataset) {
      form = { ...form, dataset_id: '', v3c_folder: '', content_dir: '', mpd: undefined };
      return;
    }
    form = {
      ...form,
      dataset_id: dataset.id,
      v3c_folder: dataset.mpd_folder,
      mpd: dataset.mpd,
      content_dir: dataset.content_dir,
      angle_convention: dataset.angle_convention,
    };
  }

  function onDatasetChanged(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement;
    applyDataset(target.value);
  }

  function onHeadsetPresetChanged(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement;
    applyHeadsetPreset(target.value);
  }

  function onAbrChanged(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement;
    const nextAbr = target.value;
    const currentPredictor = form.predictor || 'None';
    const nextChoices = compatiblePredictorChoices(nextAbr);
    const nextPredictor = nextChoices.includes(currentPredictor) ? currentPredictor : (nextChoices[0] || 'None');
    const predictorChanged = nextPredictor !== currentPredictor;
    form = {
      ...form,
      abr: nextAbr,
      predictor: nextPredictor,
      predictor_params: predictorChanged ? {} : form.predictor_params,
      predictor_preset_name: predictorChanged ? undefined : form.predictor_preset_name,
    };
  }

  function onPredictorChanged(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement;
    const nextPredictor = target.value || 'None';
    const predictorChanged = nextPredictor !== (form.predictor || 'None');
    form = {
      ...form,
      predictor: nextPredictor,
      predictor_params: predictorChanged ? {} : form.predictor_params,
      predictor_preset_name: predictorChanged ? undefined : form.predictor_preset_name,
    };
  }

  function onBandwidthPredictorChanged(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement;
    const nextPredictor = target.value || 'None';
    const predictorChanged = nextPredictor !== (form.bandwidth_predictor || 'None');
    form = {
      ...form,
      bandwidth_predictor: nextPredictor,
      bandwidth_predictor_params: predictorChanged ? {} : form.bandwidth_predictor_params,
      bandwidth_predictor_preset_name: predictorChanged ? undefined : form.bandwidth_predictor_preset_name,
    };
  }

  function activeTraceList(): string[] {
    if (tracePickerMode === 'network') return options?.network_traces || [];
    return options?.fov_traces || [];
  }

  function abrRequiresPredictor(abr: string): boolean {
    return Boolean(options?.abr_requires_predictor?.[abr]);
  }

  function abrSupportsPredictor(abr: string): boolean {
    return Boolean(options?.abr_supports_predictor?.[abr]);
  }

  function compatiblePredictorChoices(abr: string): string[] {
    const predictors = options?.fov_predictors || [];
    if (abrRequiresPredictor(abr)) return predictors;
    if (!abrSupportsPredictor(abr)) return ['None'];
    return ['None', ...predictors];
  }

  function firstCompatiblePredictorChoice(abr: string): string {
    return compatiblePredictorChoices(abr)[0] || 'None';
  }

  function normalizePredictorSelection(): void {
    const currentPredictor = form.predictor || 'None';
    const nextChoices = compatiblePredictorChoices(form.abr);
    if (!nextChoices.length || nextChoices.includes(currentPredictor)) return;
    form = {
      ...form,
      predictor: nextChoices[0] || 'None',
      predictor_params: {},
      predictor_preset_name: undefined,
    };
  }

  function normalizeBandwidthPredictorSelection(): void {
    const currentPredictor = form.bandwidth_predictor || 'None';
    if (bandwidthPredictorChoices.includes(currentPredictor)) return;
    form = {
      ...form,
      bandwidth_predictor: 'None',
      bandwidth_predictor_params: {},
      bandwidth_predictor_preset_name: undefined,
    };
  }

  function describePredictorSupport(choices: string[]): string {
    if (choices.length === 1 && choices[0] === 'None') {
      return 'This ABR does not support FoV predictors.';
    }
    if (!choices.includes('None')) {
      return 'This ABR requires a FoV predictor.';
    }
    return 'FoV predictors are optional for this ABR.';
  }

  function activeTraceAnalysis() {
    const root =
      tracePickerMode === 'network'
        ? options?.network_root || ''
        : options?.fov_root || '';
    return analyzeTracePaths(activeTraceList(), root);
  }

  function traceGroupingOptions(): TraceGroupingOption[] {
    return activeTraceAnalysis().groupingOptions;
  }

  function activeTraceFilters(): TraceFilterOption[] {
    return activeTraceAnalysis().filterOptions;
  }

  function resetTraceFilters(): void {
    traceFilterSelections = Object.fromEntries(activeTraceFilters().map((filter) => [filter.id, []]));
  }

  function activeTraceGroups() {
    return groupTraces(
      activeTraceList(),
      tracePickerMode === 'network'
        ? options?.network_root || ''
        : options?.fov_root || '',
      traceGroupingId || activeTraceAnalysis().defaultGroupingId,
      traceFilterSelections,
      traceFilterText,
      traceSortMode,
    );
  }

  function tracePickerTitle(): string {
    if (tracePickerMode === 'network') return 'Select network trace';
    return 'Select FoV trace';
  }

  function tracePickerHint(): string {
    return 'Choose how to group paths, optionally filter them, then click a trace to use it.';
  }

  function openTracePicker(mode: TracePickerMode): void {
    tracePickerMode = mode;
    traceGroupingId = activeTraceAnalysis().defaultGroupingId;
    traceFilterText = '';
    traceSortMode = 'name';
    tracePickerSelection = mode === 'network' ? form.network_trace || '' : form.fov_trace || '';
    resetTraceFilters();
  }

  function closeTracePicker(): void {
    tracePickerMode = null;
    traceGroupingId = '';
    traceFilterText = '';
    tracePickerSelection = '';
    traceFilterSelections = {};
  }

  function chooseTrace(trace: string): void {
    if (tracePickerMode === 'network') {
      form = { ...form, network_trace: trace };
    } else if (tracePickerMode === 'fov') {
      form = { ...form, fov_trace: trace };
    }
    tracePickerSelection = trace;
    closeTracePicker();
  }

  function selectedTraceSummary(mode: TracePickerMode): string {
    const value = mode === 'network' ? form.network_trace : form.fov_trace;
    return value ? summarizeSingleTrace(value) : '';
  }

  function traceButtonLabel(mode: TracePickerMode): string {
    const value = mode === 'network' ? form.network_trace : form.fov_trace;
    return value ? 'Selected' : 'Browse';
  }

  function traceButtonDescription(mode: TracePickerMode): string {
    if (mode === 'network') return 'Choose network trace';
    return 'Choose FoV trace; embedded tile columns are used automatically when present';
  }

  function toggleTraceFilter(filterId: string, value: string): void {
    const current = new Set(traceFilterSelections[filterId] || []);
    if (current.has(value)) current.delete(value);
    else current.add(value);
    traceFilterSelections = {
      ...traceFilterSelections,
      [filterId]: Array.from(current).sort((a, b) => a.localeCompare(b)),
    };
  }

  function clearTraceFilter(filterId: string): void {
    traceFilterSelections = {
      ...traceFilterSelections,
      [filterId]: [],
    };
  }

  function traceFilterCount(filterId: string): number {
    return traceFilterSelections[filterId]?.length || 0;
  }

  function onFormSubmit(): void {
    normalizeFormParamsBeforeSubmit();
    const cleanForm = { ...form };
    delete cleanForm.oracle_gap_suite;
    delete cleanForm.oracle_gap_modes;
    delete cleanForm.oracle_mode;
    delete cleanForm.oracle_group_id;
    delete cleanForm.baseline_run_id;
    onSubmit(cleanForm);
  }

  function onSavedSessionChanged(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement;
    selectedSessionName = target.value;
  }

  function loadSelectedSession(): void {
    if (!selectedSessionName) return;
    onLoadSession(selectedSessionName);
  }

  function deleteSelectedSession(): void {
    if (!selectedSessionName) return;
    onDeleteSession(selectedSessionName);
    selectedSessionName = '';
  }

  function applyPreset(scope: PresetScope, presetName: string): void {
    if (presetName === CUSTOM_PRESET) {
      setScopePresetName(scope, undefined);
      return;
    }
    const algoName = getScopeAlgo(scope);
    if (!algoName) return;
    const preset = getPresetParams(presets, scope, algoName, presetName);
    if (!preset) {
      setScopePresetName(scope, undefined);
      return;
    }
    const normalized = normalizeParams(getScopeSpecs(scope), preset);
    setScopeParams(scope, normalized);
    setScopePresetName(scope, presetName);
  }

  function onAbrPresetChanged(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement;
    applyPreset('abr', target.value);
  }

  function onFovPresetChanged(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement;
    applyPreset('fov', target.value);
  }

  function onBandwidthPresetChanged(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement;
    applyPreset('bandwidth', target.value);
  }

  function openEditor(scope: PresetScope): void {
    const algoName = getScopeAlgo(scope);
    if (!algoName) return;
    editorScope = scope;
    editorError = '';
    editorParams = { ...normalizeParams(getScopeSpecs(scope), getScopeParams(scope)) };
    const currentPreset =
      scope === 'abr' ? form.abr_preset_name : scope === 'bandwidth' ? form.bandwidth_predictor_preset_name : form.predictor_preset_name;
    editorPresetSelection = currentPreset || CUSTOM_PRESET;
    editorPresetName = currentPreset || '';
    showEditor = true;
  }

  function closeEditor(): void {
    showEditor = false;
    editorError = '';
  }

  function onEditorPresetSelectionChanged(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement;
    const next = target.value;
    editorPresetSelection = next;
    editorError = '';
    if (next === CUSTOM_PRESET) {
      editorPresetName = '';
      editorParams = { ...normalizeParams(getScopeSpecs(editorScope), getScopeParams(editorScope)) };
      return;
    }
    const algoName = getScopeAlgo(editorScope);
    const preset = getPresetParams(presets, editorScope, algoName, next);
    if (!preset) return;
    editorPresetName = next;
    editorParams = { ...normalizeParams(getScopeSpecs(editorScope), preset) };
  }

  function editorValue(spec: AlgorithmParamField): string {
    const val = editorParams[spec.name];
    if (val === null || val === undefined) return '';
    return String(val);
  }

  function onEditorBoolChanged(specName: string, checked: boolean): void {
    editorParams = { ...editorParams, [specName]: checked };
  }

  function onEditorBoolInput(specName: string, event: Event): void {
    const target = event.currentTarget as HTMLInputElement;
    onEditorBoolChanged(specName, target.checked);
  }

  function onEditorScalarChanged(spec: AlgorithmParamField, event: Event): void {
    const target = event.currentTarget as HTMLInputElement | HTMLSelectElement;
    const raw = target.value;
    editorParams = { ...editorParams, [spec.name]: coerceScalar(spec, raw) };
  }

  function applyEditorValues(): void {
    const normalized = normalizeParams(getScopeSpecs(editorScope), editorParams);
    setScopeParams(editorScope, normalized);
    setScopePresetName(editorScope, undefined);
    closeEditor();
  }

  function persistPresets(nextStore: HyperparamPresetStore): void {
    presets = nextStore;
    saveHyperparamPresets(nextStore);
  }

  function saveEditorPreset(updateExisting: boolean): void {
    const algoName = getScopeAlgo(editorScope);
    if (!algoName) return;
    const name = editorPresetName.trim();
    if (!name) {
      editorError = 'Preset name is required.';
      return;
    }
    const names = listPresetNames(presets, editorScope, algoName);
    const exists = names.includes(name);
    if (updateExisting && !exists) {
      editorError = 'Preset does not exist. Save as new first.';
      return;
    }
    if (!updateExisting && exists) {
      editorError = 'Preset already exists. Use update instead.';
      return;
    }

    const normalized = normalizeParams(getScopeSpecs(editorScope), editorParams);
    const next = upsertPreset(presets, editorScope, algoName, name, normalized);
    persistPresets(next);
    setScopeParams(editorScope, normalized);
    setScopePresetName(editorScope, name);
    closeEditor();
  }

  function deleteEditorPreset(): void {
    const algoName = getScopeAlgo(editorScope);
    if (!algoName) return;
    const name = editorPresetSelection !== CUSTOM_PRESET ? editorPresetSelection : editorPresetName.trim();
    if (!name) {
      editorError = 'Select a preset to delete.';
      return;
    }
    const names = listPresetNames(presets, editorScope, algoName);
    if (!names.includes(name)) {
      editorError = 'Preset not found.';
      return;
    }
    if (!window.confirm(`Delete preset "${name}"?`)) return;

    const next = deletePreset(presets, editorScope, algoName, name);
    persistPresets(next);
    const currentPreset =
      editorScope === 'abr'
        ? form.abr_preset_name
        : editorScope === 'bandwidth'
          ? form.bandwidth_predictor_preset_name
          : form.predictor_preset_name;
    if (currentPreset === name) {
      setScopePresetName(editorScope, undefined);
    }
    editorPresetSelection = CUSTOM_PRESET;
    editorPresetName = '';
    editorError = '';
  }

  function onBackdropPointerDown(): void {
    closeEditor();
  }
</script>

<section class="card form-card">
  <div class="run-form-toolbar">
    <h2>New run</h2>
    <details class="form-tools-menu">
      <summary>Form tools</summary>
      <div class="form-tools-popover">
        <label>Saved session
          <select bind:value={selectedSessionName} on:change={onSavedSessionChanged}>
            <option value="">Select saved session</option>
            {#each savedSessions as session}
              <option value={session.name}>{session.name}</option>
            {/each}
          </select>
        </label>
        <button type="button" class="ghost small" on:click={onSaveSession}>Save current fields</button>
        <button type="button" class="ghost small" on:click={loadSelectedSession} disabled={!selectedSessionName}>Load selected session</button>
        <button type="button" class="ghost small" on:click={deleteSelectedSession} disabled={!selectedSessionName}>Delete selected session</button>
        <button type="button" class="ghost small" on:click={onRefreshOptions}>Refresh datasets</button>
      </div>
    </details>
  </div>

  <form on:submit|preventDefault={onFormSubmit}>
    <div class="run-grid run-grid-core">
      <label>Dataset
        <select value={form.dataset_id || ''} required on:change={onDatasetChanged}>
          <option value="">Select registered dataset</option>
          {#each datasetChoices as dataset}
            <option value={dataset.id}>{dataset.name}</option>
          {/each}
        </select>
        {#if form.dataset_id}
          <p class="field-selection-copy">
            {form.v3c_folder || 'MPD folder'} · {form.content_dir || 'PLY directory'}
          </p>
        {/if}
      </label>
      <div class="network-source-field">
        <span class="field-label">Network</span>
        <div class="button-group mode-toggle network-source-toggle">
          <button type="button" class={`ghost small ${networkSource === 'trace' ? 'active' : ''}`} on:click={() => setNetworkSource('trace')}>Trace</button>
          <button type="button" class={`ghost small ${networkSource === 'profile' ? 'active' : ''}`} on:click={() => setNetworkSource('profile')}>Profile</button>
        </div>
        {#if networkSource === 'profile'}
          <select
            value={form.network_profile || options?.network_profiles?.[0]?.id || 'ideal'}
            on:change={onNetworkProfileChanged}
          >
            {#each options?.network_profiles || [] as profile}
              <option value={profile.id}>{profile.id} · {profile.description}</option>
            {/each}
          </select>
          {#if networkProfileSummary(form.network_profile)}
            <p class="field-selection-copy">{networkProfileSummary(form.network_profile)}</p>
          {/if}
        {:else}
          <button
            type="button"
            class="ghost small trace-picker-button"
            aria-label={traceButtonDescription('network')}
            data-selected={form.network_trace ? 'true' : 'false'}
            on:click={() => openTracePicker('network')}
          >
            {traceButtonLabel('network')}
          </button>
          {#if selectedTraceSummary('network')}
            <p class="field-selection-copy">{selectedTraceSummary('network')}</p>
          {/if}
        {/if}
      </div>
      <label>FoV trace
        <button
          type="button"
          class="ghost small trace-picker-button"
          aria-label={traceButtonDescription('fov')}
          data-selected={form.fov_trace ? 'true' : 'false'}
          on:click={() => openTracePicker('fov')}
        >
          {traceButtonLabel('fov')}
        </button>
        {#if selectedTraceSummary('fov')}
          <p class="field-selection-copy">{selectedTraceSummary('fov')}</p>
        {/if}
      </label>
      <label>ABR
        <select bind:value={form.abr} on:change={onAbrChanged}>
          {#each options?.abr_classes || [] as abr}
            <option value={abr}>{abr}</option>
          {/each}
        </select>
      </label>
      <label>FoV predictor
        <select bind:value={form.predictor} title={predictorHelp} on:change={onPredictorChanged}>
          {#each predictorChoices as predictor}
            <option value={predictor}>{fovPredictorLabel(predictor)}</option>
          {/each}
        </select>
      </label>
      <label>Bandwidth predictor
        <select bind:value={form.bandwidth_predictor} on:change={onBandwidthPredictorChanged}>
          {#each bandwidthPredictorChoices as predictor}
            <option value={predictor}>{predictor}</option>
          {/each}
        </select>
      </label>
    </div>

    <details class="advanced-panel">
      <summary>Advanced options and hyperparameter presets</summary>
      <div class="advanced-content">
        <div class="run-grid run-grid-advanced">
          <label class="compact option-card metric-card rtt-card">RTT (s)
            <input type="number" step="0.01" bind:value={form.rtt} />
          </label>
          <label class="compact option-card metric-card fetch-card">Fetch window
            <input type="number" min="0" step="1" bind:value={form.buffer_window_segments} />
          </label>
          <label class="compact option-card metric-card" title="Run only the first N segments; 0 runs the full video.">Stop after segments
            <input type="number" min="0" step="1" bind:value={form.segment_limit} />
          </label>
          <label class="compact option-card toggle-field abort-card">
            Adaptive abort
            <span class="check-row advanced-check simple-check">
              <input type="checkbox" bind:checked={form.enable_mid_download_cancel} aria-label="Enable adaptive abort" />
            </span>
          </label>

          <div class="preset-inline option-card abr-preset-card">
            <label>ABR preset
              <select value={form.abr_preset_name || CUSTOM_PRESET} on:change={onAbrPresetChanged}>
                <option value={CUSTOM_PRESET}>Custom</option>
                {#each abrPresetNames as name}
                  <option value={name}>{name}</option>
                {/each}
              </select>
            </label>
            <button
              type="button"
              class="ghost small tune-icon-button"
              title="Tune ABR hyperparameters"
              aria-label="Tune ABR hyperparameters"
              on:click={() => openEditor('abr')}
              disabled={!abrParamSpecs.length}
            >⚙</button>
          </div>

          <div class="preset-inline option-card fov-preset-card">
            <label>FoV preset
              <select value={form.predictor_preset_name || CUSTOM_PRESET} on:change={onFovPresetChanged} disabled={!predictorName}>
                <option value={CUSTOM_PRESET}>Custom</option>
                {#each fovPresetNames as name}
                  <option value={name}>{name}</option>
                {/each}
              </select>
            </label>
            <button
              type="button"
              class="ghost small tune-icon-button"
              title="Tune FoV hyperparameters"
              aria-label="Tune FoV hyperparameters"
              on:click={() => openEditor('fov')}
              disabled={!predictorName || !fovParamSpecs.length}
            >⚙</button>
          </div>

          <div class="preset-inline option-card bandwidth-preset-card">
            <label>Bandwidth preset
              <select value={form.bandwidth_predictor_preset_name || CUSTOM_PRESET} on:change={onBandwidthPresetChanged} disabled={!bandwidthPredictorName}>
                <option value={CUSTOM_PRESET}>Custom</option>
                {#each bandwidthPresetNames as name}
                  <option value={name}>{name}</option>
                {/each}
              </select>
            </label>
            <button
              type="button"
              class="ghost small tune-icon-button"
              title="Tune bandwidth predictor hyperparameters"
              aria-label="Tune bandwidth predictor hyperparameters"
              on:click={() => openEditor('bandwidth')}
              disabled={!bandwidthPredictorName || !bandwidthParamSpecs.length}
            >⚙</button>
          </div>

          <div class="preset-inline option-card headset-preset-card">
            <label>Headset preset
              <select value={form.headset_preset || options?.default_headset_preset || ''} on:change={onHeadsetPresetChanged}>
                {#each headsetChoices as preset}
                  <option value={preset.id}>{preset.label}</option>
                {/each}
              </select>
            </label>
          </div>
        </div>
      </div>
    </details>

    <div class="form-footer">
      <button type="submit" disabled={submitting || loading}>Start run</button>
    </div>
  </form>
</section>

{#if showEditor}
  <div class="param-modal-backdrop" on:pointerdown|self={onBackdropPointerDown}>
    <section class="card param-modal">
      <div class="section-header">
        <div>
          <h3>{editorScope === 'abr' ? 'ABR hyperparameters' : editorScope === 'bandwidth' ? 'Bandwidth hyperparameters' : 'FoV hyperparameters'}</h3>
          <p class="subtle">{getScopeAlgo(editorScope)}</p>
        </div>
        <div class="section-actions">
          <button type="button" class="ghost small" on:click={closeEditor}>Close</button>
        </div>
      </div>

      <div class="param-modal-grid">
        <label>Load preset
          <select value={editorPresetSelection} on:change={onEditorPresetSelectionChanged}>
            <option value={CUSTOM_PRESET}>Custom / current</option>
            {#each listPresetNames(presets, editorScope, getScopeAlgo(editorScope)) as name}
              <option value={name}>{name}</option>
            {/each}
          </select>
        </label>

        <label>Preset name
          <input bind:value={editorPresetName} placeholder="e.g. conservative" />
        </label>
      </div>

      <div class="param-fields">
        {#if !getScopeSpecs(editorScope).length}
          <p class="subtle">No tunable constructor params exposed for this algorithm.</p>
        {:else}
          {#each groupedParamSpecs(getScopeSpecs(editorScope), false) as group}
            <section class="param-group">
              <h4>{group.category}</h4>
              <div class="param-group-grid">
                {#each group.specs as spec}
                  <label class="param-field">
                    <span class="param-label-row">
                      <span>{spec.label}</span>
                      <span class="param-info" title={paramHelp(spec)} aria-label={paramHelp(spec)}>i</span>
                    </span>
                    {#if spec.kind === 'boolean'}
                      <input
                        type="checkbox"
                        checked={Boolean(editorParams[spec.name])}
                        on:change={(e) => onEditorBoolInput(spec.name, e)}
                      />
                    {:else if spec.choices?.length}
                      <select value={editorValue(spec)} on:change={(e) => onEditorScalarChanged(spec, e)}>
                        <option value="">Select exported model</option>
                        {#each spec.choices as choice}
                          <option value={choice.value}>{choice.label}</option>
                        {/each}
                        {#if editorValue(spec) && !spec.choices.some((choice) => choice.value === editorValue(spec))}
                          <option value={editorValue(spec)}>{editorValue(spec)}</option>
                        {/if}
                      </select>
                    {:else}
                      <input
                        type={spec.kind === 'string' ? 'text' : 'number'}
                        step={spec.kind === 'integer' ? '1' : 'any'}
                        value={editorValue(spec)}
                        on:change={(e) => onEditorScalarChanged(spec, e)}
                      />
                    {/if}
                    {#if spec.annotation}
                      <span class="micro subtle">{spec.annotation}</span>
                    {/if}
                  </label>
                {/each}
              </div>
            </section>
          {/each}

          {#if groupedParamSpecs(getScopeSpecs(editorScope), true).length}
            <details class="param-advanced-groups">
              <summary>Advanced parameters</summary>
              <div class="param-advanced-body">
                {#each groupedParamSpecs(getScopeSpecs(editorScope), true) as group}
                  <section class="param-group">
                    <h4>{group.category}</h4>
                    <div class="param-group-grid">
                      {#each group.specs as spec}
                        <label class="param-field">
                          <span class="param-label-row">
                            <span>{spec.label}</span>
                            <span class="param-info" title={paramHelp(spec)} aria-label={paramHelp(spec)}>i</span>
                          </span>
                          {#if spec.kind === 'boolean'}
                            <input
                              type="checkbox"
                              checked={Boolean(editorParams[spec.name])}
                              on:change={(e) => onEditorBoolInput(spec.name, e)}
                            />
                          {:else if spec.choices?.length}
                            <select value={editorValue(spec)} on:change={(e) => onEditorScalarChanged(spec, e)}>
                              <option value="">Select exported model</option>
                              {#each spec.choices as choice}
                                <option value={choice.value}>{choice.label}</option>
                              {/each}
                              {#if editorValue(spec) && !spec.choices.some((choice) => choice.value === editorValue(spec))}
                                <option value={editorValue(spec)}>{editorValue(spec)}</option>
                              {/if}
                            </select>
                          {:else}
                            <input
                              type={spec.kind === 'string' ? 'text' : 'number'}
                              step={spec.kind === 'integer' ? '1' : 'any'}
                              value={editorValue(spec)}
                              on:change={(e) => onEditorScalarChanged(spec, e)}
                            />
                          {/if}
                          {#if spec.annotation}
                            <span class="micro subtle">{spec.annotation}</span>
                          {/if}
                        </label>
                      {/each}
                    </div>
                  </section>
                {/each}
              </div>
            </details>
          {/if}
        {/if}
      </div>

      {#if editorError}
        <p class="subtle error">{editorError}</p>
      {/if}

      <div class="section-actions">
        <button type="button" class="ghost small" on:click={applyEditorValues}>Apply as custom</button>
        <button type="button" class="ghost small" on:click={() => saveEditorPreset(false)}>Save as preset</button>
        <button type="button" class="ghost small" on:click={() => saveEditorPreset(true)}>Update preset</button>
        <button type="button" class="ghost small" on:click={deleteEditorPreset}>Delete preset</button>
      </div>
    </section>
  </div>
{/if}

{#if tracePickerMode}
  <div class="param-modal-backdrop" on:pointerdown|self={closeTracePicker}>
    <section class="card param-modal trace-picker-modal">
      <div class="section-header">
        <div>
          <h3>{tracePickerTitle()}</h3>
          <p class="subtle">{tracePickerHint()}</p>
        </div>
        <div class="section-actions">
          <button type="button" class="ghost small" on:click={closeTracePicker}>Close</button>
        </div>
      </div>

      <div class="trace-picker-toolbar">
        <label>Group by
          <select bind:value={traceGroupingId}>
            {#each traceGroupingOptions() as option}
              <option value={option.id}>{option.label}</option>
            {/each}
          </select>
        </label>
        <label>Filter
          <input bind:value={traceFilterText} placeholder="Search groups or paths" />
        </label>
        <label>Sort
          <select bind:value={traceSortMode}>
            <option value="name">Name</option>
            <option value="count">Count</option>
          </select>
        </label>
      </div>

      {#if activeTraceFilters().length}
        <div class="trace-filter-strip">
          {#each activeTraceFilters() as filter}
            <details class="trace-filter-menu">
              <summary>
                {filter.label}
                {#if traceFilterCount(filter.id)}
                  <span>{traceFilterCount(filter.id)}</span>
                {/if}
              </summary>
              <div class="trace-filter-menu-body">
                <div class="trace-filter-chip-row">
                  {#each filter.values as value}
                    <button
                      type="button"
                      class:selected={traceFilterSelections[filter.id]?.includes(value)}
                      class="trace-filter-chip"
                      on:click={() => toggleTraceFilter(filter.id, value)}
                    >
                      {value}
                    </button>
                  {/each}
                </div>
                <button type="button" class="ghost small" on:click={() => clearTraceFilter(filter.id)}>Clear</button>
              </div>
            </details>
          {/each}
        </div>
      {/if}

      {#key `${tracePickerMode}:${traceGroupingId}:${traceFilterText}:${traceSortMode}:${JSON.stringify(traceFilterSelections)}`}
        <div class="trace-picker-groups">
          {#each activeTraceGroups() as group}
            <section class="trace-group-card">
              <div class="trace-group-head">
                <div>
                  <h4>{group.name}</h4>
                  <p class="micro subtle">{group.traces.length} traces</p>
                </div>
              </div>

              <div class="trace-option-list">
                {#each group.traces as trace}
                  <button
                    type="button"
                    class:selected={tracePickerSelection === trace}
                    class="trace-option trace-option-button"
                    on:click={() => chooseTrace(trace)}
                    title={trace}
                  >
                    <span class="trace-option-mark">{tracePickerSelection === trace ? 'Selected' : 'Use'}</span>
                    <span>{traceDisplayName(trace)}</span>
                  </button>
                {/each}
              </div>
            </section>
          {/each}
        </div>
      {/key}

      <div class="section-actions">
        <button type="button" class="ghost small" on:click={closeTracePicker}>Cancel</button>
      </div>
    </section>
  </div>
{/if}

<style>
  .form-card {
    padding: 0;
    overflow: visible;
    background: var(--card-bg);
  }

  .form-card form {
    padding: 0.46rem 0.62rem 0.62rem;
  }

  .preset-inline {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: end;
    gap: 0.24rem;
  }

  .form-card :global(input),
  .form-card :global(select),
  .form-card button {
    width: 100%;
    min-width: 0;
    min-height: 1.85rem;
    padding-block: 0.28rem;
  }

  .form-card label {
    gap: 0.14rem;
    font-size: 0.8rem;
    line-height: 1.05;
  }

  .network-source-field {
    display: grid;
    gap: 0.18rem;
    align-content: start;
    font-size: 0.8rem;
    line-height: 1.05;
  }

  .field-label {
    color: var(--text-main);
  }

  .network-source-toggle {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.24rem;
  }

  .run-grid-core {
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.52rem 0.62rem;
    align-items: start;
  }

  .run-form-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.8rem;
    padding: 0.4rem 0.62rem;
    border-bottom: 1px solid var(--card-border);
    background: rgba(148, 163, 184, 0.06);
    border-radius: 14px 14px 0 0;
  }

  .run-form-toolbar h2 {
    margin: 0;
    font-size: 0.98rem;
    white-space: nowrap;
  }

  .form-tools-menu {
    position: relative;
  }

  .form-tools-menu > summary {
    list-style: none;
    cursor: pointer;
    border: 1px solid var(--ghost-border);
    background: var(--soft-bg);
    color: var(--text-muted);
    border-radius: 10px;
    padding: 0.26rem 0.44rem;
    font-size: 0.8rem;
    line-height: 1.2;
    user-select: none;
  }

  .form-tools-menu > summary::-webkit-details-marker {
    display: none;
  }

  .form-tools-menu > summary::after {
    content: ' ▾';
    opacity: 0.8;
  }

  .form-tools-menu[open] > summary::after {
    content: ' ▴';
  }

  .form-tools-popover {
    position: absolute;
    top: calc(100% + 0.45rem);
    right: 0;
    z-index: 20;
    min-width: 220px;
    display: grid;
    gap: 0.45rem;
    padding: 0.55rem;
    border-radius: 12px;
    border: 1px solid var(--modal-border);
    background: var(--modal-surface);
    box-shadow: 0 16px 42px rgba(2, 6, 23, 0.65);
  }

  .form-tools-popover button {
    width: 100%;
    text-align: left;
  }

  .advanced-panel {
    margin-top: 0.34rem;
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 10px;
    background: rgba(148, 163, 184, 0.035);
    box-shadow: none;
    overflow: hidden;
  }

  .advanced-panel > summary {
    cursor: pointer;
    padding: 0.36rem 0.5rem;
    font-weight: 650;
    font-size: 0.84rem;
    color: var(--text-muted);
    list-style: none;
  }

  .advanced-panel > summary::-webkit-details-marker {
    display: none;
  }

  .advanced-panel > summary::after {
    content: '▾';
    float: right;
    opacity: 0.8;
  }

  .advanced-panel:not([open]) > summary::after {
    content: '▸';
  }

  .advanced-content {
    border-top: 1px solid rgba(148, 163, 184, 0.14);
    padding: 0.42rem 0.48rem 0.48rem;
  }

  .run-grid-advanced {
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.52rem 0.62rem;
    align-items: end;
  }

  .run-grid-advanced > label,
  .run-grid-advanced .preset-inline label {
    min-width: 0;
    overflow: hidden;
  }

  .option-card {
    display: grid;
    gap: 0.14rem;
    align-content: start;
    padding: 0;
    border: 0;
    border-radius: 0;
    background: transparent;
  }

  .toggle-field {
    justify-content: start;
  }

  .advanced-check {
    height: 1.85rem;
    min-height: 1.85rem;
    padding: 0;
    border: 1px solid var(--field-border);
    border-radius: 8px;
    background: var(--field-bg);
    font-weight: 500;
    color: var(--text-main);
    width: 100%;
    min-width: 0;
    box-sizing: border-box;
  }

  .simple-check {
    justify-content: center;
    align-items: center;
    height: 1.85rem;
    min-height: 1.85rem;
  }

  .simple-check input {
    width: 1rem;
    height: 1rem;
  }

  .field-group-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-main);
  }

  .metric-card,
  .abort-card {
    align-self: end;
  }

  .rtt-card {
    grid-column: auto;
  }

  .abr-preset-card,
  .fov-preset-card {
    grid-column: span 2;
  }

  .fetch-card,
  .abort-card {
    grid-column: auto;
  }

  .headset-preset-card {
    display: block;
    grid-column: span 2;
    align-self: end;
  }

  .bandwidth-preset-card {
    grid-column: span 2;
  }

  .abort-card {
    max-width: none;
  }

  .trace-picker-button {
    width: 100%;
    justify-content: center;
    background: var(--field-bg);
    border-color: var(--field-border);
    color: var(--text-main);
    font-weight: 500;
    letter-spacing: 0;
    box-shadow: none;
  }

  .trace-picker-button[data-selected='true'] {
    color: var(--accent-main);
  }

  .tune-icon-button {
    width: 1.85rem;
    min-width: 1.85rem;
    padding: 0;
    font-size: 0.95rem;
    line-height: 1;
  }

  .trace-picker-button:hover {
    transform: none;
    background: var(--soft-bg);
    border-color: var(--ghost-border);
    box-shadow: none;
  }

  .field-selection-copy {
    margin: 0.08rem 0 0;
    color: var(--text-muted);
    font-size: 0.7rem;
    line-height: 1.18;
    max-height: 1.75rem;
    overflow: hidden;
    overflow-wrap: anywhere;
  }

  .trace-picker-modal {
    width: min(920px, 96vw);
  }

  .trace-picker-groups {
    display: grid;
    gap: 0.8rem;
  }

  .trace-picker-toolbar {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    flex-wrap: wrap;
  }

  .trace-filter-strip {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
  }

  .trace-filter-menu {
    min-width: 0;
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.025);
  }

  .trace-filter-menu > summary {
    list-style: none;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.52rem 0.7rem;
    font-size: 0.84rem;
    color: var(--text-main);
  }

  .trace-filter-menu > summary::-webkit-details-marker {
    display: none;
  }

  .trace-filter-menu > summary span {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 1.1rem;
    height: 1.1rem;
    padding: 0 0.28rem;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    font-size: 0.72rem;
    font-weight: 700;
  }

  .trace-filter-menu-body {
    display: grid;
    gap: 0.55rem;
    padding: 0 0.7rem 0.7rem;
  }

  .trace-filter-chip-row {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
  }

  .trace-filter-chip {
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(255, 255, 255, 0.025);
    color: var(--text-main);
    border-radius: 999px;
    padding: 0.38rem 0.6rem;
    font-size: 0.78rem;
  }

  .trace-filter-chip.selected {
    border-color: rgba(56, 189, 248, 0.35);
    background: rgba(56, 189, 248, 0.12);
    color: #67e8f9;
  }

  .trace-group-card {
    display: grid;
    gap: 0.75rem;
    padding: 0.9rem;
    border-radius: 14px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    background: rgba(255, 255, 255, 0.03);
  }

  .trace-group-head h4 {
    margin: 0;
  }

  .trace-group-head p {
    margin: 0.18rem 0 0;
  }

  .trace-option-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.55rem 0.7rem;
  }

  .trace-option {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.75rem;
    min-width: 0;
    padding: 0.55rem 0.65rem;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(255, 255, 255, 0.025);
  }

  .trace-option-button {
    width: 100%;
    text-align: left;
    cursor: pointer;
  }

  .trace-option-button.selected {
    border-color: rgba(56, 189, 248, 0.35);
    background: rgba(56, 189, 248, 0.08);
  }

  .trace-option span {
    min-width: 0;
    overflow-wrap: anywhere;
  }

  .trace-option-mark {
    flex: 0 0 auto;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #38bdf8;
  }

  .param-modal-backdrop {
    position: fixed;
    inset: 0;
    z-index: 1400;
    background: var(--overlay-backdrop);
    backdrop-filter: blur(4px);
    display: grid;
    place-items: center;
    padding: 1rem;
  }

  .param-modal {
    width: min(860px, 96vw);
    max-height: 88vh;
    overflow: auto;
    display: grid;
    gap: 0.8rem;
    background: var(--modal-surface);
    border: 1px solid var(--modal-border);
    box-shadow: 0 22px 64px rgba(2, 6, 23, 0.8);
  }

  .param-modal-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 0.65rem;
  }

  .param-fields {
    display: grid;
    gap: 0.75rem;
  }

  .param-group {
    display: grid;
    gap: 0.48rem;
    padding: 0.58rem;
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.025);
  }

  .param-group h4 {
    margin: 0;
    font-size: 0.78rem;
    color: var(--text-muted);
    letter-spacing: 0;
  }

  .param-group-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.65rem;
  }

  .param-field {
    min-width: 0;
  }

  .param-label-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    min-width: 0;
  }

  .param-label-row > span:first-child {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .param-info {
    flex: 0 0 auto;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1rem;
    height: 1rem;
    border-radius: 50%;
    border: 1px solid rgba(148, 163, 184, 0.38);
    color: var(--text-muted);
    font-size: 0.68rem;
    font-weight: 700;
    line-height: 1;
    cursor: help;
  }

  .param-advanced-groups {
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 10px;
    background: rgba(148, 163, 184, 0.035);
  }

  .param-advanced-groups > summary {
    cursor: pointer;
    list-style: none;
    padding: 0.5rem 0.58rem;
    color: var(--text-muted);
    font-size: 0.82rem;
    font-weight: 650;
  }

  .param-advanced-groups > summary::-webkit-details-marker {
    display: none;
  }

  .param-advanced-groups > summary::after {
    content: '▾';
    float: right;
  }

  .param-advanced-groups:not([open]) > summary::after {
    content: '▸';
  }

  .param-advanced-body {
    display: grid;
    gap: 0.75rem;
    padding: 0 0.58rem 0.58rem;
  }

  .error {
    color: var(--error-text);
  }

  .form-footer {
    margin-top: 0.42rem;
  }

  .form-footer button {
    width: auto;
    min-width: 7rem;
    min-height: 2rem;
    padding: 0.38rem 0.78rem;
  }

  @media (max-width: 980px) {
    .run-grid-core {
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    }
  }

  @media (max-width: 760px) {
    .run-grid-core {
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }
  }

  @media (max-width: 840px) {
    .form-card form {
      padding: 0.48rem 0.58rem 0.65rem;
    }

    .run-grid-advanced {
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }

    .preset-inline {
      grid-template-columns: 1fr;
      align-items: stretch;
    }

    .metric-card,
    .abr-preset-card,
    .fov-preset-card,
    .abort-card {
      max-width: none;
    }

    .trace-option-list {
      grid-template-columns: 1fr;
    }

    .trace-picker-toolbar {
      flex-direction: column;
      align-items: stretch;
    }

    .trace-filter-strip {
      flex-direction: column;
    }

    .form-tools-popover {
      right: auto;
      left: 0;
      width: min(100%, 280px);
    }
  }

  @media (max-width: 620px) {
    .run-form-toolbar {
      align-items: stretch;
      flex-direction: column;
      gap: 0.5rem;
      padding: 0.6rem 0.8rem;
    }

    .run-grid-core {
      grid-template-columns: 1fr;
    }

    .form-tools-menu > summary {
      width: 100%;
      text-align: center;
    }

    .form-tools-popover {
      width: 100%;
    }

    .run-grid-advanced {
      grid-template-columns: 1fr;
    }

    .metric-card,
    .abr-preset-card,
    .fov-preset-card,
    .bandwidth-preset-card,
    .headset-preset-card,
    .abort-card {
      grid-column: auto;
    }
  }
</style>
