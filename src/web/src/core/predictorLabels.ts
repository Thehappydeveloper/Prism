/** Presentation names; persisted configs always retain implementation IDs. */
const FOV_PREDICTOR_LABELS: Record<string, string> = {
  ReachabilityRaycastPredictor: 'Argus',
};

export function fovPredictorLabel(identifier: string | null | undefined): string {
  const value = String(identifier || '');
  return FOV_PREDICTOR_LABELS[value] || value;
}
