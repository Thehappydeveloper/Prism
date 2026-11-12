import { Chart, registerables, type ChartConfiguration } from 'chart.js';
import type { FigureExportSettings, PublicationSettings } from './publication';
import { compactSeriesLabel, figureInches, figurePixels } from './publication';

Chart.register(...registerables);

function filenamePart(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'figure';
}

function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function copy<T>(value: T): T {
  if (Array.isArray(value)) return value.map((entry) => copy(entry)) as T;
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    Object.entries(value as Record<string, unknown>).forEach(([key, entry]) => { out[key] = copy(entry); });
    return out as T;
  }
  return value;
}

function scaleNumber(container: any, key: string, ratio: number): void {
  if (typeof container?.[key] === 'number') container[key] *= ratio;
}

function rasterConfig(config: ChartConfiguration, dpi: number): ChartConfiguration {
  const cloned: any = copy(config);
  const ratio = dpi / 96;
  cloned.options = { ...(cloned.options || {}), responsive: false, maintainAspectRatio: false, animation: false, devicePixelRatio: 1 };
  scaleNumber(cloned.options.font, 'size', ratio);
  scaleNumber(cloned.options.layout?.padding, 'top', ratio);
  scaleNumber(cloned.options.layout?.padding, 'right', ratio);
  scaleNumber(cloned.options.layout?.padding, 'bottom', ratio);
  scaleNumber(cloned.options.layout?.padding, 'left', ratio);
  const plugins = cloned.options.plugins || {};
  scaleNumber(plugins.title?.font, 'size', ratio);
  scaleNumber(plugins.legend?.labels?.font, 'size', ratio);
  for (const key of ['boxWidth', 'boxHeight', 'pointStyleWidth', 'padding']) scaleNumber(plugins.legend?.labels, key, ratio);
  Object.values(cloned.options.scales || {}).forEach((scale: any) => {
    scaleNumber(scale.ticks?.font, 'size', ratio);
    scaleNumber(scale.title?.font, 'size', ratio);
    scaleNumber(scale.ticks, 'padding', ratio);
    scaleNumber(scale.grid, 'lineWidth', ratio);
    scaleNumber(scale.border, 'width', ratio);
  });
  cloned.data.datasets.forEach((dataset: any) => {
    for (const key of ['borderWidth', 'pointRadius', 'pointHoverRadius', 'barThickness', 'maxBarThickness']) {
      if (typeof dataset[key] === 'number') dataset[key] *= ratio;
      else if (Array.isArray(dataset[key])) dataset[key] = dataset[key].map((value: unknown) => typeof value === 'number' ? value * ratio : value);
    }
    if (Array.isArray(dataset.borderDash)) dataset.borderDash = dataset.borderDash.map((value: number) => value * ratio);
  });
  cloned.plugins = [
    ...(cloned.plugins || []),
    { id: 'publication-white-background', beforeDraw: (chart: Chart) => {
      chart.ctx.save(); chart.ctx.globalCompositeOperation = 'destination-over'; chart.ctx.fillStyle = '#ffffff';
      chart.ctx.fillRect(0, 0, chart.width, chart.height); chart.ctx.restore();
    } },
  ];
  return cloned;
}

async function exportPng(config: ChartConfiguration, title: string, settings: FigureExportSettings): Promise<void> {
  const pixels = figurePixels(settings);
  const canvas = document.createElement('canvas');
  canvas.width = pixels.width; canvas.height = pixels.height;
  const chart = new Chart(canvas, rasterConfig(config, settings.dpi));
  chart.resize(pixels.width, pixels.height);
  chart.update('none');
  await new Promise((resolve) => requestAnimationFrame(resolve));
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
  chart.destroy();
  if (!blob) throw new Error('The browser could not encode the PNG.');
  download(blob, `${filenamePart(title)}-${settings.dpi}dpi.png`);
}

function esc(value: unknown): string {
  return String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function numeric(value: unknown): number | null {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function color(value: unknown, index = 0, fallback = '#374151'): string {
  if (Array.isArray(value)) return String(value[index] ?? value[0] ?? fallback);
  return typeof value === 'string' ? value : fallback;
}

function niceRange(values: number[], requestedMin: unknown, requestedMax: unknown): [number, number] {
  let min = numeric(requestedMin) ?? Math.min(...values, 0);
  let max = numeric(requestedMax) ?? Math.max(...values, 1);
  if (min === max) { min -= 0.5; max += 0.5; }
  const padding = (max - min) * 0.04;
  if (numeric(requestedMin) === null) min -= padding;
  if (numeric(requestedMax) === null) max += padding;
  return [min, max];
}

function marker(shape: string, x: number, y: number, size: number, fill: string): string {
  if (shape === 'rect' || shape === 'rectRounded') return `<rect x="${x - size}" y="${y - size}" width="${2 * size}" height="${2 * size}" rx="${shape === 'rectRounded' ? size * 0.35 : 0}" fill="${fill}"/>`;
  if (shape === 'triangle') return `<path d="M ${x} ${y - size} L ${x + size} ${y + size} L ${x - size} ${y + size} Z" fill="${fill}"/>`;
  if (shape === 'rectRot') return `<path d="M ${x} ${y - size} L ${x + size} ${y} L ${x} ${y + size} L ${x - size} ${y} Z" fill="${fill}"/>`;
  if (shape === 'crossRot') return `<path d="M ${x-size} ${y-size} L ${x+size} ${y+size} M ${x+size} ${y-size} L ${x-size} ${y+size}" stroke="${fill}" stroke-width="1.2"/>`;
  return `<circle cx="${x}" cy="${y}" r="${size}" fill="${fill}"/>`;
}

function shorten(label: string, limit = 38): string {
  return label.length <= limit ? label : `${label.slice(0, limit - 1)}…`;
}

function tickLabel(scale: any, value: number): string {
  const callback = scale?.ticks?.callback;
  if (typeof callback === 'function') {
    try {
      const result = callback(value, value, []);
      if (result !== undefined && result !== null) return String(result);
    } catch { /* fall through to numeric formatting */ }
  }
  return Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(2).replace(/0+$/, '').replace(/\.$/, '');
}

function tickValues(min: number, max: number, scale: any): number[] {
  if (Number(scale?.ticks?.stepSize) === 1 && max - min <= 20) {
    const start = Math.ceil(min); const end = Math.floor(max);
    return Array.from({ length: Math.max(0, end - start + 1) }, (_, index) => start + index);
  }
  return Array.from({ length: 6 }, (_, index) => min + (max - min) * index / 5);
}

function compactAxisTitle(value: unknown): string {
  return String(value || '')
    .replace('Probability of meeting requirement', 'Requirement success probability')
    .replace('Observed visibility frequency', 'Observed frequency')
    .replace('Predicted tile visibility probability', 'Predicted visibility probability');
}

/** Lightweight vector renderer for the bar, line, and scatter configurations used by Advanced Compare. */
export function chartToSvg(config: ChartConfiguration, title: string, settings: FigureExportSettings, publication: PublicationSettings): string {
  const inches = figureInches(settings);
  const width = inches.width * 72;
  const height = inches.height * 72;
  const font = publication.fontFamily.split(',')[0]!.replace(/["']/g, '').trim();
  const base = publication.baseFontPt;
  const labels = (config.data.labels || []).map(String);
  const datasets: any[] = config.data.datasets as any[];
  const xOptions: any = (config.options as any)?.scales?.x || {};
  const yOptions: any = (config.options as any)?.scales?.y || {};
  const y1Options: any = (config.options as any)?.scales?.y1 || null;
  const hasSecondaryY = Boolean(y1Options && datasets.some((dataset) => dataset.yAxisID === 'y1'));
  const legendEnabled = (config.options as any)?.plugins?.legend?.display !== false;
  const rawLegendDatasets = legendEnabled ? datasets.filter((dataset) => !dataset._band) : [];
  const hasReliabilityPairs = rawLegendDatasets.some((dataset) => / observed$/.test(String(dataset.label || '')))
    && rawLegendDatasets.some((dataset) => /\d+% lower bound$/.test(String(dataset.label || '')));
  const legendDatasets: any[] = [];
  if (hasReliabilityPairs) {
    const seen = new Set<string>();
    rawLegendDatasets.forEach((dataset) => {
      const raw = String(dataset.label || '');
      if (/Selected threshold/.test(raw)) return;
      const identity = raw.replace(/ observed$/, '').replace(/ \d+% lower bound$/, '');
      if (!seen.has(identity)) {
        seen.add(identity);
        legendDatasets.push({ ...dataset, label: compactSeriesLabel(identity), borderDash: [] });
      }
    });
    legendDatasets.push(
      { label: 'Observed', borderColor: '#374151', backgroundColor: '#374151', borderDash: [4, 3], pointStyle: 'circle' },
      { label: 'Confidence lower bound', borderColor: '#374151', backgroundColor: '#374151', borderDash: [], pointStyle: 'rect' },
    );
  } else {
    rawLegendDatasets.forEach((dataset) => legendDatasets.push({ ...dataset, label: compactSeriesLabel(String(dataset.label || '')) }));
  }
  const legendItems = legendDatasets.map((dataset) => String(dataset.label || ''));
  const legendItemWidth = (item: string) => Math.min(105, Math.max(45, item.length * base * 0.42 + 16));
  const countLegendRows = () => {
    let rows = 1; let used = 0; const available = Math.max(40, width - 18);
    legendItems.forEach((item) => {
      const itemWidth = legendItemWidth(item);
      if (used > 0 && used + itemWidth > available) { rows += 1; used = 0; }
      used += itemWidth;
    });
    return rows;
  };
  const legendRows = publication.legendPosition === 'bottom' || publication.legendPosition === 'top'
    ? countLegendRows() : 1;
  const horizontal = config.type === 'bar' && (config.options as any)?.indexAxis === 'y';
  const longLabel = labels.reduce((max, label) => Math.max(max, compactSeriesLabel(label).length), 0);
  const scaleLabelLength = Number(yOptions?.ticks?.stepSize) === 1
    ? Array.from({ length: 20 }, (_, index) => compactSeriesLabel(tickLabel(yOptions, index))).reduce((max, label) => Math.max(max, label.length), 0)
    : 0;
  const left = horizontal || scaleLabelLength > 8
    ? Math.min(width * 0.42, Math.max(42, Math.max(longLabel, scaleLabelLength) * base * 0.43 + base * 2.8))
    : 38;
  const right = publication.legendPosition === 'right' ? Math.min(130, width * 0.3) : (hasSecondaryY ? 40 : 10);
  const titleSpace = publication.showTitle ? base * 1.8 : 2;
  const legendSpace = legendEnabled && (publication.legendPosition === 'bottom' || publication.legendPosition === 'top') ? legendRows * (base + 5) + 3 : 0;
  const top = 7 + titleSpace + (publication.legendPosition === 'top' ? legendSpace : 0);
  const xScaleLabelLength = Number(xOptions?.ticks?.stepSize) === 1
    ? Array.from({ length: 20 }, (_, index) => compactSeriesLabel(tickLabel(xOptions, index))).reduce((max, label) => Math.max(max, label.length), 0)
    : 0;
  const angledTickSpace = xScaleLabelLength > 10 ? Math.min(35, xScaleLabelLength * base * 0.22) : 0;
  const bottom = 29 + angledTickSpace + (publication.legendPosition === 'bottom' ? legendSpace : 0);
  const plotWidth = Math.max(20, width - left - right);
  const plotHeight = Math.max(20, height - top - bottom);
  const isCartesian = config.type === 'line' || config.type === 'scatter';
  const points = datasets.flatMap((dataset) => (dataset.data || []).map((entry: any, index: number) => ({
    x: numeric(entry?.x) ?? (isCartesian && labels.length ? index : numeric(entry) ?? index),
    y: numeric(entry?.y) ?? numeric(entry) ?? 0,
  })));
  const primaryBarValues = datasets.filter((dataset) => dataset.yAxisID !== 'y1').flatMap((dataset) => (dataset.data || []).map((entry: any) => numeric(entry?.y) ?? numeric(entry) ?? 0));
  const secondaryBarValues = datasets.filter((dataset) => dataset.yAxisID === 'y1').flatMap((dataset) => (dataset.data || []).map((entry: any) => numeric(entry?.y) ?? numeric(entry) ?? 0));
  const barValues = primaryBarValues.length ? primaryBarValues : datasets.flatMap((dataset) => (dataset.data || []).map((entry: any) => numeric(entry?.y) ?? numeric(entry) ?? 0));
  const [xMin, xMax] = horizontal
    ? niceRange(barValues, xOptions.beginAtZero ? 0 : xOptions.min, xOptions.max)
    : niceRange(points.map((point) => point.x), xOptions.min, xOptions.max);
  const [yMin, yMax] = horizontal
    ? [0, Math.max(1, labels.length)]
    : niceRange(config.type === 'bar' ? barValues : points.map((point) => point.y), yOptions.beginAtZero ? 0 : yOptions.min, yOptions.max);
  const [y1Min, y1Max] = niceRange(secondaryBarValues, y1Options?.beginAtZero ? 0 : y1Options?.min, y1Options?.max);
  const sx = (value: number) => left + ((value - xMin) / (xMax - xMin)) * plotWidth;
  const sy = (value: number) => yOptions.reverse
    ? top + ((value - yMin) / (yMax - yMin)) * plotHeight
    : top + plotHeight - ((value - yMin) / (yMax - yMin)) * plotHeight;
  const sy1 = (value: number) => top + plotHeight - ((value - y1Min) / (y1Max - y1Min)) * plotHeight;
  const elements: string[] = [];
  elements.push(`<rect width="${width}" height="${height}" fill="white"/>`);
  if (publication.showTitle) elements.push(`<text x="${width / 2}" y="${base + 3}" text-anchor="middle" font-size="${base * 1.12}" font-weight="bold">${esc(title)}</text>`);
  const ticks = 5;
  for (const value of tickValues(yMin, yMax, yOptions)) {
    if (!horizontal || config.type !== 'bar') {
      const y = sy(value);
      elements.push(`<line x1="${left}" y1="${y}" x2="${left + plotWidth}" y2="${y}" stroke="#d1d5db" stroke-width="0.35"/>`);
      elements.push(`<text x="${left - 4}" y="${y + base * 0.32}" text-anchor="end" font-size="${base * 0.88}">${esc(shorten(compactSeriesLabel(tickLabel(yOptions, value)), 28))}</text>`);
    }
  }
  if (config.type === 'bar') {
    const count = Math.max(1, labels.length);
    const seriesCount = Math.max(1, datasets.length);
    if (horizontal) {
      const row = plotHeight / count;
      const bar = row * 0.78 / seriesCount;
      const maxLabelChars = Math.max(10, Math.floor((left - 8) / (base * 0.43)));
      labels.forEach((label, labelIndex) => {
        const center = top + row * (labelIndex + 0.5);
        elements.push(`<text x="${left - 4}" y="${center + base * 0.3}" text-anchor="end" font-size="${base * 0.82}">${esc(shorten(compactSeriesLabel(label), maxLabelChars))}</text>`);
        datasets.forEach((dataset, datasetIndex) => {
          const value = numeric(dataset.data?.[labelIndex]) ?? 0;
          const x0 = sx(Math.max(0, xMin)); const x1 = sx(value);
          const y = center - row * 0.39 + datasetIndex * bar;
          elements.push(`<rect x="${Math.min(x0, x1)}" y="${y}" width="${Math.max(0.5, Math.abs(x1 - x0))}" height="${bar * 0.9}" fill="${color(dataset.backgroundColor, labelIndex)}" stroke="${color(dataset.borderColor, labelIndex, '#374151')}" stroke-width="0.5"/>`);
        });
      });
      for (let index = 0; index <= ticks; index += 1) {
        const value = xMin + (xMax - xMin) * index / ticks; const x = sx(value);
        elements.push(`<line x1="${x}" y1="${top}" x2="${x}" y2="${top + plotHeight}" stroke="#d1d5db" stroke-width="0.35"/>`);
        elements.push(`<text x="${x}" y="${top + plotHeight + base + 3}" text-anchor="middle" font-size="${base * 0.88}">${esc(value.toFixed(2).replace(/0+$/, '').replace(/\.$/, ''))}</text>`);
      }
    } else {
      const column = plotWidth / count;
      const bar = column * 0.78 / seriesCount;
      labels.forEach((label, labelIndex) => {
        const center = left + column * (labelIndex + 0.5);
        datasets.forEach((dataset, datasetIndex) => {
          const value = numeric(dataset.data?.[labelIndex]) ?? 0;
          const mapY = dataset.yAxisID === 'y1' ? sy1 : sy;
          const axisMin = dataset.yAxisID === 'y1' ? y1Min : yMin;
          const x = center - column * 0.39 + datasetIndex * bar; const y = mapY(value); const zero = mapY(Math.max(0, axisMin));
          elements.push(`<rect x="${x}" y="${Math.min(y, zero)}" width="${bar * 0.9}" height="${Math.max(0.5, Math.abs(zero - y))}" fill="${color(dataset.backgroundColor, labelIndex)}" stroke="${color(dataset.borderColor, labelIndex, '#374151')}" stroke-width="0.5"/>`);
        });
        elements.push(`<text transform="translate(${center},${top + plotHeight + 5}) rotate(${longLabel > 12 ? -32 : 0})" text-anchor="${longLabel > 12 ? 'end' : 'middle'}" font-size="${base * 0.8}">${esc(shorten(compactSeriesLabel(label), 26))}</text>`);
      });
    }
  } else {
    let previousLineData: Array<{ x: number; y: number; raw: any }> = [];
    for (const dataset of datasets) {
      const data = (dataset.data || []).filter(Boolean).map((entry: any, index: number) => ({ x: numeric(entry?.x) ?? index, y: numeric(entry?.y) ?? numeric(entry) ?? 0, raw: entry }));
      const stroke = color(dataset.borderColor ?? dataset.backgroundColor);
      const dash = Array.isArray(dataset.borderDash) && dataset.borderDash.length ? `stroke-dasharray="${dataset.borderDash.join(' ')}"` : '';
      if (config.type === 'line' && data.length > 1) {
        if (dataset.fill === '-1' && previousLineData.length) {
          const polygon = [...previousLineData, ...[...data].reverse()].map((point: any) => `${sx(point.x)},${sy(point.y)}`).join(' ');
          elements.push(`<polygon points="${polygon}" fill="${color(dataset.backgroundColor, 0, 'rgba(107,114,128,0.12)')}" stroke="none"/>`);
        }
        const path = data.map((point: any, index: number) => `${index ? 'L' : 'M'} ${sx(point.x)} ${sy(point.y)}`).join(' ');
        elements.push(`<path d="${path}" fill="none" stroke="${stroke}" stroke-width="${dataset.borderWidth || 1.3}" ${dash}/>`);
      }
      data.forEach((point: any, index: number) => {
        const fill = color(dataset.backgroundColor ?? dataset.borderColor, index);
        const rawRadius = Array.isArray(dataset.pointRadius) ? dataset.pointRadius[index] : dataset.pointRadius;
        const matrixRadius = point.raw?.probability !== undefined ? Math.max(5, Math.min(plotWidth / Math.max(1, xMax - xMin) * 0.34, plotHeight / Math.max(1, yMax - yMin) * 0.34)) : 0;
        const radius = matrixRadius || Math.max(config.type === 'scatter' ? 2.2 : 0, Math.min(8, typeof rawRadius === 'number' ? rawRadius : 2.2));
        if (radius > 0) elements.push(marker(String(dataset.pointStyle || 'circle'), sx(point.x), sy(point.y), radius, fill));
        if (point.raw?.probability !== undefined && datasets.length === 1) elements.push(`<text x="${sx(point.x)}" y="${sy(point.y) + base * 0.3}" text-anchor="middle" font-size="${base * 0.72}" fill="#111">${Math.round(point.raw.probability * 100)}%</text>`);
      });
      if (config.type === 'line') previousLineData = data;
    }
    for (const value of tickValues(xMin, xMax, xOptions)) {
      const x = sx(value);
      elements.push(`<line x1="${x}" y1="${top}" x2="${x}" y2="${top + plotHeight}" stroke="#d1d5db" stroke-width="0.35"/>`);
      const label = shorten(compactSeriesLabel(tickLabel(xOptions, value)), 18);
      const rotate = label.length > 10 ? -28 : 0;
      elements.push(`<text transform="translate(${x},${top + plotHeight + base + 3}) rotate(${rotate})" text-anchor="${rotate ? 'end' : 'middle'}" font-size="${base * 0.82}">${esc(label)}</text>`);
    }
  }
  elements.push(`<line x1="${left}" y1="${top}" x2="${left}" y2="${top + plotHeight}" stroke="#374151" stroke-width="0.65"/>`);
  elements.push(`<line x1="${left}" y1="${top + plotHeight}" x2="${left + plotWidth}" y2="${top + plotHeight}" stroke="#374151" stroke-width="0.65"/>`);
  if (hasSecondaryY) {
    elements.push(`<line x1="${left + plotWidth}" y1="${top}" x2="${left + plotWidth}" y2="${top + plotHeight}" stroke="#374151" stroke-width="0.65"/>`);
    for (const value of tickValues(y1Min, y1Max, y1Options)) {
      const y = sy1(value);
      elements.push(`<text x="${left + plotWidth + 4}" y="${y + base * 0.32}" text-anchor="start" font-size="${base * 0.82}">${esc(tickLabel(y1Options, value))}</text>`);
    }
  }
  const xTitle = compactAxisTitle(xOptions.title?.text);
  const yTitle = compactAxisTitle(yOptions.title?.text);
  const y1Title = compactAxisTitle(y1Options?.title?.text);
  if (xTitle) elements.push(`<text x="${left + plotWidth / 2}" y="${height - (publication.legendPosition === 'bottom' ? legendSpace : 0) - 3}" text-anchor="middle" font-size="${base}">${esc(xTitle)}</text>`);
  if (yTitle) elements.push(`<text transform="translate(${base},${top + plotHeight / 2}) rotate(-90)" text-anchor="middle" font-size="${base}">${esc(yTitle)}</text>`);
  if (hasSecondaryY && y1Title) elements.push(`<text transform="translate(${width - base * 0.4},${top + plotHeight / 2}) rotate(90)" text-anchor="middle" font-size="${base}">${esc(y1Title)}</text>`);

  if (legendItems.length) {
    let x = 9; let y = publication.legendPosition === 'top' ? 7 + titleSpace + base : height - legendSpace + base;
    legendItems.forEach((item, index) => {
      const dataset = legendDatasets[index] || {};
      const itemWidth = legendItemWidth(item);
      if (x + itemWidth > width - 8) { x = 9; y += base + 5; }
      const fill = color(dataset.borderColor ?? dataset.backgroundColor);
      elements.push(`<line x1="${x}" y1="${y - base * 0.25}" x2="${x + 10}" y2="${y - base * 0.25}" stroke="${fill}" stroke-width="1.4"/>`);
      elements.push(marker(String(dataset.pointStyle || 'circle'), x + 5, y - base * 0.25, 1.8, fill));
      const maxLegendChars = Math.max(8, Math.floor((itemWidth - 16) / (base * 0.42)));
      elements.push(`<text x="${x + 14}" y="${y}" font-size="${base * 0.82}">${esc(shorten(item, maxLegendChars))}</text>`);
      x += itemWidth;
    });
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${inches.width}in" height="${inches.height}in" viewBox="0 0 ${width} ${height}" font-family="${esc(font)}" fill="#111827">${elements.join('')}</svg>`;
}

function exportSvg(config: ChartConfiguration, title: string, settings: FigureExportSettings, publication: PublicationSettings): void {
  const svg = chartToSvg(config, title, settings, publication);
  download(new Blob([svg], { type: 'image/svg+xml;charset=utf-8' }), `${filenamePart(title)}.svg`);
}

function exportPdf(config: ChartConfiguration, title: string, settings: FigureExportSettings, publication: PublicationSettings): void {
  const svg = chartToSvg(config, title, settings, publication);
  const inches = figureInches(settings);
  const popup = window.open('', '_blank');
  if (!popup) throw new Error('Allow popups to open the vector PDF print dialog.');
  popup.opener = null;
  popup.document.write(`<title>${esc(title)}</title><style>@page{size:${inches.width}in ${inches.height}in;margin:0}html,body{margin:0;width:${inches.width}in;height:${inches.height}in}svg{display:block}</style>${svg}<script>window.onload=()=>setTimeout(()=>window.print(),100)<\/script>`);
  popup.document.close();
}

export async function exportFigure(
  config: ChartConfiguration,
  title: string,
  settings: FigureExportSettings,
  publication: PublicationSettings,
): Promise<void> {
  if (settings.format === 'png') return exportPng(config, title, settings);
  if (settings.format === 'pdf') return exportPdf(config, title, settings, publication);
  exportSvg(config, title, settings, publication);
}
