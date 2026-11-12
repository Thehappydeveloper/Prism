import type { ChartConfiguration } from 'chart.js';
import type { RunMetrics } from '../../core/types';
import { labelForRun } from './metrics';

export type ExportUnit = 'in' | 'cm' | 'mm' | 'px';
export type ExportFormat = 'svg' | 'pdf' | 'png';
export type LegendPosition = 'bottom' | 'top' | 'left' | 'right';

export interface PublicationSettings {
  fontFamily: string;
  baseFontPt: number;
  legendPosition: LegendPosition;
  showTitle: boolean;
  grayscale: boolean;
}

export interface FigureExportSettings {
  preset: string;
  width: number;
  height: number;
  unit: ExportUnit;
  dpi: number;
  format: ExportFormat;
}

export const DEFAULT_PUBLICATION_SETTINGS: PublicationSettings = {
  fontFamily: 'Linux Libertine, Libertine, Times New Roman, serif',
  baseFontPt: 8,
  legendPosition: 'bottom',
  showTitle: false,
  grayscale: false,
};

export const FIGURE_PRESETS: Array<{ id: string; label: string; width: number; height: number; unit: ExportUnit }> = [
  { id: 'acm-single', label: 'ACM single column', width: 3.33, height: 2.35, unit: 'in' },
  { id: 'acm-double', label: 'ACM double column', width: 7.0, height: 3.7, unit: 'in' },
  { id: 'full-page', label: 'Full page width', width: 7.0, height: 7.8, unit: 'in' },
  { id: 'custom', label: 'Custom', width: 7.0, height: 4.0, unit: 'in' },
];

export const DEFAULT_EXPORT_SETTINGS: FigureExportSettings = {
  preset: 'acm-single', width: 3.33, height: 2.35, unit: 'in', dpi: 300, format: 'svg',
};

const COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000', '#F0E442'];
const GRAYS = ['#111111', '#5c5c5c', '#929292', '#c1c1c1', '#383838', '#777777', '#ababab', '#d2d2d2'];
const DASHES: number[][] = [[], [6, 3], [2, 2], [8, 3, 2, 3], [10, 3]];
const MARKERS = ['circle', 'rect', 'triangle', 'rectRot', 'star', 'crossRot'] as const;

function hash(text: string): number {
  let value = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    value ^= text.charCodeAt(index);
    value = Math.imul(value, 16777619);
  }
  return value >>> 0;
}

function baseIdentity(identity: string): string {
  return identity
    .replace(/\s*\(\d+\s+runs?\)\s*$/i, '')
    .replace(/\s*\[[^\]]+\]\s*$/, '')
    .trim();
}

export function compactSeriesLabel(label: string): string {
  return label
    .replace(/\s*\(\d+\s+runs?\)\s*$/i, '')
    .replace('GroundTruthTileFovPredictor', 'GT-Tile')
    .replace('GroundTruthFovPredictor', 'GT-FoV')
    .replace('ReachabilityRaycastPredictor', 'Argus')
    .replace('LinearPoseExtrapolationPredictor', 'Linear pose')
    .replace('LinearExtrapolationPredictor', 'Linear')
    .replace('DistanceBasedTileImportancePredictor', 'Distance')
    .replace('LastObservedFovPredictor', 'Last observed')
    .replace('TrainedFovPredictor', 'Trained')
    .replace(/\s*\[default\]/i, '')
    .replace(/three_second_spread/gi, '3s-spread')
    .replace(/two[_ ]+sec(?:ond)?/gi, '2s')
    .trim();
}

export interface StableSeriesStyle {
  color: string;
  borderDash: number[];
  pointStyle: typeof MARKERS[number];
  colorIndex: number;
  variantIndex: number;
}

export function stableSeriesStyle(identity: string, grayscale = false): StableSeriesStyle {
  const baseHash = hash(baseIdentity(identity));
  const fullHash = hash(identity);
  const colorIndex = baseHash % COLORS.length;
  const variantIndex = fullHash % Math.max(DASHES.length, MARKERS.length);
  return {
    color: grayscale ? GRAYS[colorIndex]! : COLORS[colorIndex]!,
    borderDash: DASHES[variantIndex % DASHES.length]!,
    pointStyle: MARKERS[variantIndex % MARKERS.length]!,
    colorIndex,
    variantIndex,
  };
}

function rgba(hex: string, alpha: number): string {
  const clean = hex.replace('#', '');
  const value = Number.parseInt(clean, 16);
  return `rgba(${(value >> 16) & 255},${(value >> 8) & 255},${value & 255},${alpha})`;
}

export function figurePixels(settings: FigureExportSettings): { width: number; height: number } {
  const perInch = settings.unit === 'cm' ? 2.54 : settings.unit === 'mm' ? 25.4 : 1;
  if (settings.unit === 'px') return { width: Math.round(settings.width), height: Math.round(settings.height) };
  return {
    width: Math.max(1, Math.round((settings.width / perInch) * settings.dpi)),
    height: Math.max(1, Math.round((settings.height / perInch) * settings.dpi)),
  };
}

export function figureInches(settings: FigureExportSettings): { width: number; height: number } {
  if (settings.unit === 'in') return { width: settings.width, height: settings.height };
  if (settings.unit === 'cm') return { width: settings.width / 2.54, height: settings.height / 2.54 };
  if (settings.unit === 'mm') return { width: settings.width / 25.4, height: settings.height / 25.4 };
  return { width: settings.width / settings.dpi, height: settings.height / settings.dpi };
}

function identityLabels(metrics: RunMetrics, aliases: Record<string, string>): Set<string> {
  return new Set(Object.keys(metrics.meta || {}).map((id) => labelForRun(metrics, id, aliases)));
}

export function sortCategoricalChart(config: ChartConfiguration, direction: 'asc' | 'desc' | 'none'): void {
  if (direction === 'none' || config.type !== 'bar' || !Array.isArray(config.data.labels) || !config.data.datasets.length) return;
  const primary = config.data.datasets[0]?.data || [];
  if (!primary.every((value) => typeof value === 'number')) return;
  const order = primary.map((value, index) => ({ value: Number(value), index }))
    .sort((left, right) => direction === 'asc' ? left.value - right.value : right.value - left.value)
    .map((entry) => entry.index);
  config.data.labels = order.map((index) => config.data.labels![index]);
  config.data.datasets.forEach((dataset) => {
    if (Array.isArray(dataset.data) && dataset.data.length === order.length) dataset.data = order.map((index) => dataset.data[index]!);
  });
}

export function applyPublicationStyle(
  config: ChartConfiguration,
  metrics: RunMetrics,
  aliases: Record<string, string>,
  publication: PublicationSettings,
  title: string,
): ChartConfiguration {
  const identities = identityLabels(metrics, aliases);
  const orderedIdentities = Array.from(identities).sort((left, right) => right.length - left.length);
  const labels = (config.data.labels || []).map(String);
  const labelsAreIdentities = labels.length > 0 && labels.every((label) => identities.has(label));
  const fontPx = publication.baseFontPt * (96 / 72);
  const options: any = config.options || {};
  options.responsive = true;
  options.maintainAspectRatio = false;
  options.animation = false;
  options.layout = { ...(options.layout || {}), padding: 4 };
  options.font = { ...(options.font || {}), family: publication.fontFamily, size: fontPx, lineHeight: 1.15 };
  options.color = '#111827';
  options.plugins = options.plugins || {};
  options.plugins.title = {
    ...(options.plugins.title || {}), display: publication.showTitle, text: title,
    font: { family: publication.fontFamily, size: fontPx * 1.12, weight: 'bold' }, padding: { bottom: 4 },
  };
  options.plugins.legend = {
    ...(options.plugins.legend || {}), position: publication.legendPosition,
    labels: {
      ...(options.plugins.legend?.labels || {}), usePointStyle: true, pointStyleWidth: fontPx,
      boxWidth: fontPx, boxHeight: fontPx * 0.75, padding: fontPx * 0.7,
      font: { family: publication.fontFamily, size: fontPx * 0.88 },
    },
  };
  for (const scale of Object.values(options.scales || {}) as any[]) {
    scale.border = { ...(scale.border || {}), color: '#4b5563', width: 0.75 };
    scale.grid = { ...(scale.grid || {}), color: 'rgba(107,114,128,0.18)', lineWidth: 0.5, tickLength: 3 };
    scale.ticks = { ...(scale.ticks || {}), color: '#111827', padding: 2, maxRotation: 35, minRotation: 0, autoSkip: true, font: { family: publication.fontFamily, size: fontPx * 0.88 } };
    scale.title = { ...(scale.title || {}), color: '#111827', padding: 2, font: { family: publication.fontFamily, size: fontPx, weight: 'normal' } };
  }

  config.data.datasets.forEach((dataset: any, datasetIndex) => {
    const datasetLabel = String(dataset.label || '');
    const seriesIdentity = orderedIdentities.find((identity) => datasetLabel === identity || datasetLabel.startsWith(`${identity} `)) || '';
    if (seriesIdentity) {
      const style = stableSeriesStyle(seriesIdentity, publication.grayscale);
      const variantStyle = stableSeriesStyle(datasetLabel, publication.grayscale);
      dataset.borderColor = dataset._band ? rgba(style.color, 0.2) : style.color;
      dataset.backgroundColor = dataset._band ? rgba(style.color, 0.12) : (config.type === 'bar' ? rgba(style.color, 0.72) : style.color);
      dataset.borderDash = datasetLabel === seriesIdentity ? style.borderDash : variantStyle.borderDash;
      dataset.pointStyle = datasetLabel === seriesIdentity ? style.pointStyle : variantStyle.pointStyle;
      dataset.pointRadius = dataset.pointRadius ?? 2.7;
      dataset.pointHoverRadius = dataset.pointHoverRadius ?? 4;
      dataset.borderWidth = 1.5;
    } else if (labelsAreIdentities && config.type === 'bar') {
      if (config.data.datasets.length === 1) {
        dataset.backgroundColor = labels.map((label) => rgba(stableSeriesStyle(label, publication.grayscale).color, 0.76));
        dataset.borderColor = labels.map((label) => stableSeriesStyle(label, publication.grayscale).color);
      } else {
        // In grouped/stacked bars the datasets are metrics or components, not
        // predictors. Allocate colors by dataset order so two metric names can
        // never hash to the same color. Dataset order is stable within a plugin,
        // and the legend reads the exact color and marker assigned here.
        const palette = publication.grayscale ? GRAYS : COLORS;
        const semanticColor = palette[datasetIndex % palette.length]!;
        dataset.backgroundColor = rgba(semanticColor, 0.72);
        dataset.borderColor = semanticColor;
        dataset.pointStyle = MARKERS[datasetIndex % MARKERS.length]!;
      }
      dataset.borderWidth = 1;
      dataset.borderSkipped = false;
    }
  });

  const hasReliabilityPairs = config.data.datasets.some((dataset: any) => / observed$/.test(String(dataset.label || '')))
    && config.data.datasets.some((dataset: any) => /\d+% lower bound$/.test(String(dataset.label || '')));
  if (hasReliabilityPairs) {
    options.plugins.legend.labels.generateLabels = (chart: any) => {
      const seen = new Set<string>();
      const generated: any[] = [];
      chart.data.datasets.forEach((dataset: any, datasetIndex: number) => {
        const raw = String(dataset.label || '');
        const identity = raw.replace(/ observed$/, '').replace(/ \d+% lower bound$/, '');
        if (/Selected threshold/.test(raw)) {
          generated.push({ text: raw, datasetIndex, strokeStyle: dataset.borderColor, lineDash: dataset.borderDash || [], pointStyle: false, fillStyle: 'transparent' });
        } else if (!seen.has(identity)) {
          seen.add(identity);
          generated.push({ text: compactSeriesLabel(identity), datasetIndex, strokeStyle: dataset.borderColor, fillStyle: dataset.borderColor, pointStyle: dataset.pointStyle || 'circle', lineWidth: 1.5 });
        }
      });
      return generated;
    };
  } else {
    options.plugins.legend.labels.generateLabels = (chart: any) => chart.data.datasets
      .map((dataset: any, datasetIndex: number) => ({
        text: compactSeriesLabel(String(dataset.label || '')),
        datasetIndex,
        fillStyle: Array.isArray(dataset.backgroundColor) ? dataset.backgroundColor[0] : dataset.backgroundColor,
        strokeStyle: Array.isArray(dataset.borderColor) ? dataset.borderColor[0] : dataset.borderColor,
        lineDash: dataset.borderDash || [],
        lineWidth: dataset.borderWidth || 1,
        pointStyle: dataset.pointStyle || 'circle',
        hidden: !chart.isDatasetVisible(datasetIndex),
      }))
      .filter((item: any) => !chart.data.datasets[item.datasetIndex]?._band);
  }

  if (config.type === 'bar' && labelsAreIdentities && config.data.datasets.length === 1) {
    options.plugins.legend.display = false;
  }

  const exactIdentitySeries = config.data.datasets.every((dataset: any) => identities.has(String(dataset.label || '')));
  if (exactIdentitySeries && !config.data.datasets.some((dataset: any) => dataset.fill)) {
    config.data.datasets.sort((left: any, right: any) => String(left.label || '').localeCompare(String(right.label || '')));
  }

  const secondaryValueAxis = config.data.datasets.some((dataset: any) => dataset.yAxisID && dataset.yAxisID !== 'y')
    || Object.keys(options.scales || {}).some((scaleId) => /^y.+/.test(scaleId));
  const shouldAutoHorizontal = config.type === 'bar'
    && labelsAreIdentities
    && options.indexAxis !== 'y'
    && !secondaryValueAxis
    && (labels.length > 7 || Math.max(...labels.map((label) => compactSeriesLabel(label).length), 0) > 22);
  if (shouldAutoHorizontal) {
    options.indexAxis = 'y';
    const oldX = options.scales?.x || {};
    const oldY = options.scales?.y || {};
    options.scales = { ...options.scales, x: oldY, y: oldX };
    options.scales.y.ticks = { ...(options.scales.y.ticks || {}), maxRotation: 0, autoSkip: false };
  }
  if (config.type === 'bar' && labelsAreIdentities) {
    options.scales = options.scales || {};
    const categoryScaleId = options.indexAxis === 'y' ? 'y' : 'x';
    const categoryScale = options.scales?.[categoryScaleId] || (options.scales[categoryScaleId] = {});
    const previousCallback = categoryScale.ticks?.callback;
    categoryScale.ticks = {
      ...(categoryScale.ticks || {}),
      autoSkip: false,
      callback: function compactPredictorTick(this: any, value: string | number, index: number, ticks: unknown[]) {
        const raw = labels[Number(value)] ?? labels[index];
        return raw ? compactSeriesLabel(raw) : (typeof previousCallback === 'function' ? previousCallback.call(this, value, index, ticks) : String(value));
      },
    };
  }
  config.options = options;
  return config;
}

export function stableStyleMap(metrics: RunMetrics, aliases: Record<string, string>, grayscale = false): Record<string, StableSeriesStyle> {
  return Object.fromEntries(Array.from(identityLabels(metrics, aliases)).sort().map((label) => [label, stableSeriesStyle(label, grayscale)]));
}
