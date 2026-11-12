import type { ChartPlugin } from './types';
import { avgEfficiencyPlugin } from './avgEfficiency';
import { bitrateQualityPlugin } from './bitrateQuality';
import { bufferStabilityPlugin } from './bufferStability';
import { componentBitratePlugin } from './componentBitrate';
import { fovReliabilityGuaranteePlugin } from './fovReliabilityGuarantee';
import { fovBitrateWasteAbsolutePlugin } from './fovBitrateWasteAbsolute';
import { fovBitrateWasteRatiosPlugin } from './fovBitrateWasteRatios';
import { fovCatastrophicMissPlugin } from './fovCatastrophicMiss';
import { fovOverlapIouPlugin } from './fovOverlapIou';
import { fovPairwiseWinProbabilityPlugin } from './fovPairwiseWinProbability';
import { fovPredictionBitrateWastePlugin } from './fovPredictionBitrateWaste';
import { fovPredictionCalibrationPlugin } from './fovPredictionCalibration';
import { fovPredictionErrorPlugin } from './fovPredictionError';
import { fovQualityWastePlugin } from './fovQualityWaste';
import { fovTopKRecallPlugin } from './fovTopKRecall';
import { fovUsefulBitrateEfficiencyPlugin } from './fovUsefulBitrateEfficiency';
import { qoeCdfPlugin } from './qoeCdf';
import { qoeStallPlugin } from './qoeStall';
import { predictorLatencyPlugin } from './predictorLatency';
import { stallSummaryPlugin } from './stallSummary';
import { temporalDeltaPlugin } from './temporalDelta';

export const chartPluginRegistry: ChartPlugin[] = [
  qoeCdfPlugin,
  qoeStallPlugin,
  avgEfficiencyPlugin,
  fovReliabilityGuaranteePlugin,
  fovPairwiseWinProbabilityPlugin,
  predictorLatencyPlugin,
  fovPredictionErrorPlugin,
  fovCatastrophicMissPlugin,
  fovOverlapIouPlugin,
  fovTopKRecallPlugin,
  fovPredictionCalibrationPlugin,
  fovQualityWastePlugin,
  fovBitrateWasteAbsolutePlugin,
  fovBitrateWasteRatiosPlugin,
  fovPredictionBitrateWastePlugin,
  fovUsefulBitrateEfficiencyPlugin,
  temporalDeltaPlugin,
  stallSummaryPlugin,
  bufferStabilityPlugin,
  componentBitratePlugin,
  bitrateQualityPlugin,
];
