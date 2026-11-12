import type { QoEModel, QoEParams } from './types';

export interface QoEState {
  selectedId: string;
  paramsById: Record<string, QoEParams>;
}

export const FALLBACK_QOE_MODEL: QoEModel = {
  id: 'WeightedQoEHeuristic',
  label: 'Weighted harmonic QoE',
  equation: 'QoE = harmonic(visible-tile bitrate utility) - smoothness - low_buffer - stall',
  fields: [
    { name: 'attr', label: 'Attr weight', default: 0.35, min: 0, step: 0.01 },
    { name: 'geom', label: 'Geom weight', default: 0.35, min: 0, step: 0.01 },
    { name: 'occp', label: 'Occp weight', default: 0.2, min: 0, step: 0.01 },
    { name: 'atlas', label: 'Atlas weight', default: 0, min: 0, step: 0.01 },
    { name: 'default_component', label: 'Default comp', default: 0.1, min: 0, step: 0.01 },
    {
      name: 'fov_saliency',
      label: 'FoV saliency',
      default: 1.25,
      min: 0,
      step: 0.05,
      help: 'Shapes visible-tile weights; invisible tiles score zero',
    },
    { name: 'temporal_penalty', label: 'Temporal penalty', default: 0.2, min: 0, step: 0.01 },
    { name: 'stall_per_second', label: 'Stall /s penalty', default: 0.15, min: 0, step: 0.01 },
    { name: 'stall_fixed', label: 'Stall fixed', default: 0.05, min: 0, step: 0.01 },
  ],
  defaults: {
    attr: 0.35,
    geom: 0.35,
    occp: 0.2,
    atlas: 0,
    default_component: 0.1,
    fov_saliency: 1.25,
    temporal_penalty: 0.2,
    stall_per_second: 0.15,
    stall_fixed: 0.05,
  },
};

export function normalizeQoeModels(raw: QoEModel[] | undefined): QoEModel[] {
  if (!raw || !raw.length) return [FALLBACK_QOE_MODEL];
  const filtered = raw.filter((m) => {
    const id = (m.id || '').trim();
    const label = (m.label || '').trim().toLowerCase();
    return id !== 'QoEFunction' && label !== 'qoe function';
  });
  const normalized = filtered.map((m) => ({
    id: m.id || m.label || 'QoEModel',
    label: m.label || m.id || 'QoE model',
    equation: m.equation || '',
    description: m.description || '',
    implementation: m.implementation || m.id,
    fields: m.fields || [],
    defaults: m.defaults || {},
  }));
  return normalized.length ? normalized : [FALLBACK_QOE_MODEL];
}

export function defaultsForModel(model: QoEModel): QoEParams {
  const defaults: QoEParams = { ...(model.defaults || {}) };
  (model.fields || []).forEach((f) => {
    if (defaults[f.name] === undefined) defaults[f.name] = Number(f.default ?? 0);
  });
  if (defaults.default_component === undefined && defaults.default !== undefined) defaults.default_component = defaults.default;
  if (defaults.default === undefined && defaults.default_component !== undefined) defaults.default = defaults.default_component;
  return defaults;
}

export function sanitizeQoeParams(model: QoEModel, input: QoEParams): QoEParams {
  const defaults = defaultsForModel(model);
  const merged: QoEParams = { ...defaults, ...(input || {}) };
  Object.keys(merged).forEach((key) => {
    const value = Number(merged[key]);
    merged[key] = Number.isFinite(value) ? value : (defaults[key] ?? 0);
  });
  return merged;
}

export function loadQoeState(storageKey: string, models: QoEModel[]): QoEState {
  const normalized = models.length ? models : [FALLBACK_QOE_MODEL];
  try {
    const parsed = JSON.parse(localStorage.getItem(storageKey) || '{}');
    const selectedId = normalized.some((m) => m.id === parsed.selectedId) ? parsed.selectedId : normalized[0]!.id;
    const paramsById: Record<string, QoEParams> = {};
    normalized.forEach((model) => {
      paramsById[model.id] = sanitizeQoeParams(model, parsed.params?.[model.id] || {});
    });
    return { selectedId, paramsById };
  } catch {
    const paramsById: Record<string, QoEParams> = {};
    normalized.forEach((model) => {
      paramsById[model.id] = defaultsForModel(model);
    });
    return { selectedId: normalized[0]!.id, paramsById };
  }
}

export function saveQoeState(storageKey: string, state: QoEState): void {
  localStorage.setItem(storageKey, JSON.stringify({ selectedId: state.selectedId, params: state.paramsById }));
}
