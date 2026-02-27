import type { ChartConfiguration } from 'chart.js';
import type { RunMetrics } from '../../../core/types';

export interface CompareChartContext {
  metrics: RunMetrics;
  aliases: Record<string, string>;
}

export interface ChartPlugin {
  id: string;
  title: string;
  description?: string;
  build: (ctx: CompareChartContext) => ChartConfiguration<any, any, any> | null;
}
