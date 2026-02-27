import type { RunConfig, RunSummary } from '../../core/types';

export type RunStatusFilter = 'all' | 'queued' | 'running' | 'finished' | 'failed';
export type RunRuntimeFilter = 'all' | 'lt60' | '60to180' | 'gt180' | 'unfinished';

export interface RunRow {
  run: RunSummary;
  alias: string;
  runtimeSec: number | null;
  startedTs: number | null;
}

export function parseIsoTime(value: string | undefined): number | null {
  if (!value) return null;
  const ts = Date.parse(value);
  return Number.isFinite(ts) ? ts : null;
}

export function formatDateTime(value: string | undefined): string {
  const ts = parseIsoTime(value);
  if (ts === null) return value || '—';
  return new Date(ts).toLocaleString();
}

export function runtimeSeconds(run: RunSummary): number | null {
  const start = parseIsoTime(run.started);
  const end = parseIsoTime(run.finished);
  if (start === null || end === null) return null;
  return Math.max(0, (end - start) / 1000);
}

export function formatRuntime(sec: number | null): string {
  if (sec === null) return '—';
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const min = Math.floor(sec / 60);
  const rem = Math.round(sec % 60);
  return `${min}m ${rem}s`;
}

export function basenamePath(path: string | null | undefined): string {
  if (!path) return '—';
  const normalized = path.replace(/\\/g, '/');
  const parts = normalized.split('/');
  return parts[parts.length - 1] || path;
}

export function statusDotClass(status: string): string {
  return `status-dot ${status || 'unknown'}`;
}

export function buildRunRows(runs: RunSummary[], runAliases: Record<string, string>): RunRow[] {
  return runs.map((run) => ({
    run,
    alias: runAliases[run.run_id] || '',
    runtimeSec: runtimeSeconds(run),
    startedTs: parseIsoTime(run.started),
  }));
}

function runMatchesFilters(
  row: RunRow,
  runSearch: string,
  statusFilter: RunStatusFilter,
  runtimeFilter: RunRuntimeFilter,
  runConfigCache: Record<string, RunConfig | null>,
): boolean {
  const rowStatus = (row.run.status || '').toLowerCase();
  if (statusFilter !== 'all' && rowStatus !== statusFilter) return false;
  if (runtimeFilter === 'lt60' && (row.runtimeSec === null || row.runtimeSec >= 60)) return false;
  if (runtimeFilter === '60to180' && (row.runtimeSec === null || row.runtimeSec < 60 || row.runtimeSec >= 180)) return false;
  if (runtimeFilter === 'gt180' && (row.runtimeSec === null || row.runtimeSec < 180)) return false;
  if (runtimeFilter === 'unfinished' && row.runtimeSec !== null) return false;
  if (!runSearch.trim()) return true;
  const q = runSearch.trim().toLowerCase();
  const runId = (row.run.run_id || '').toLowerCase();
  if (runId.includes(q)) return true;
  const cfg = runConfigCache[row.run.run_id];
  const haystack = [
    runId,
    row.alias,
    rowStatus,
    cfg?.abr || '',
    cfg?.predictor || '',
    cfg?.network_trace || '',
    cfg?.fov_trace || '',
    basenamePath(cfg?.network_trace || ''),
    basenamePath(cfg?.fov_trace || ''),
  ]
    .join(' ')
    .toLowerCase()
    .replace(/[_\-/\\]+/g, ' ');
  const normalizedQuery = q.replace(/[_\-/\\]+/g, ' ');
  return haystack.includes(normalizedQuery);
}

export function filterAndSortRunRows(
  runRows: RunRow[],
  runSearch: string,
  statusFilter: RunStatusFilter,
  runtimeFilter: RunRuntimeFilter,
  runConfigCache: Record<string, RunConfig | null>,
): RunRow[] {
  return runRows
    .filter((row) => runMatchesFilters(row, runSearch, statusFilter, runtimeFilter, runConfigCache))
    .sort((a, b) => {
      const aTs = a.startedTs ?? 0;
      const bTs = b.startedTs ?? 0;
      return bTs - aTs;
    });
}
