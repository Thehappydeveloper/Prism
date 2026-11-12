import type { ChartConfiguration } from 'chart.js';
import type { RunMetrics } from '../../../core/types';

export interface CompareChartContext {
  metrics: RunMetrics;
  aliases: Record<string, string>;
  options: Record<string, string | number | boolean>;
}

export interface PluginControlOption {
  label: string;
  value: string | number;
}

export interface PluginControl {
  key: string;
  label: string;
  type: 'select' | 'number' | 'checkbox';
  defaultValue: string | number | boolean;
  options?: PluginControlOption[];
  min?: number;
  max?: number;
  step?: number;
  help?: string;
}

export type PluginCategory = 'global' | 'fov' | 'bandwidth' | 'abr' | 'buffer' | 'qoe';
export type MetricDataset = 'qoe' | 'qoe_series' | 'fov_quality' | 'fov_prediction' | 'delta_fov'
  | 'stall' | 'component_bits' | 'bitrate_vs_quality' | 'buffer' | 'latency';

export interface ChartPlugin {
  id: string;
  title: string;
  category?: PluginCategory;
  description?: string;
  /** Compact metric families fetched for this plugin. */
  requiredDatasets: MetricDataset[];
  controls?: PluginControl[];
  /** Sort categorical predictor bars by the primary displayed metric. */
  sortDirection?: 'asc' | 'desc' | 'none';
  build: (ctx: CompareChartContext) => ChartConfiguration<any, any, any> | null;
}
