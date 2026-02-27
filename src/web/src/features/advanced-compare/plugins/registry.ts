import type { ChartPlugin } from './types';
import { avgEfficiencyPlugin } from './avgEfficiency';
import { bitrateQualityPlugin } from './bitrateQuality';
import { bufferStabilityPlugin } from './bufferStability';
import { componentBitratePlugin } from './componentBitrate';
import { fovQualityPlugin } from './fovQuality';
import { qoeCdfPlugin } from './qoeCdf';
import { qoeStallPlugin } from './qoeStall';
import { stallSummaryPlugin } from './stallSummary';
import { temporalDeltaPlugin } from './temporalDelta';

export const chartPluginRegistry: ChartPlugin[] = [
  qoeCdfPlugin,
  qoeStallPlugin,
  avgEfficiencyPlugin,
  fovQualityPlugin,
  temporalDeltaPlugin,
  stallSummaryPlugin,
  bufferStabilityPlugin,
  componentBitratePlugin,
  bitrateQualityPlugin,
];
