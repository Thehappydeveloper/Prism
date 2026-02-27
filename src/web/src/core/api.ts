import type { AppOptions, DiagnosticsRequest, EventLog, NetworkTraceResponse, RunConfig, RunMetrics, RunSummary } from './types';

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return (await res.json()) as T;
}

export async function getOptions(): Promise<AppOptions> {
  return fetchJson<AppOptions>('/api/options');
}

export async function listRuns(): Promise<RunSummary[]> {
  return fetchJson<RunSummary[]>('/api/runs');
}

export async function startRun(payload: RunConfig): Promise<{ run_id: string; status: string }> {
  return fetchJson('/api/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function deleteRun(runId: string): Promise<{ status: string; run_id: string }> {
  return fetchJson(`/api/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' });
}

export async function getRunConfig(runId: string): Promise<RunConfig> {
  return fetchJson<RunConfig>(`/api/runs/${encodeURIComponent(runId)}/config`);
}

export async function getRunEvents(runId: string): Promise<EventLog> {
  return fetchJson<EventLog>(`/api/runs/${encodeURIComponent(runId)}/events`);
}

export async function getDiagnostics(payload: DiagnosticsRequest): Promise<RunMetrics> {
  return fetchJson<RunMetrics>('/api/diagnostics', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function getTrace(path: string): Promise<NetworkTraceResponse> {
  const encoded = encodeURIComponent(path);
  return fetchJson<NetworkTraceResponse>(`/api/trace?path=${encoded}`);
}

export function subscribeRuns(onRuns: (runs: RunSummary[]) => void, onError?: (error: unknown) => void): () => void {
  const source = new EventSource('/api/runs/stream');
  source.addEventListener('runs', (evt) => {
    try {
      const data = JSON.parse((evt as MessageEvent<string>).data) as RunSummary[];
      onRuns(data);
    } catch (error) {
      onError?.(error);
    }
  });
  source.onerror = (err) => {
    onError?.(err);
  };
  return () => source.close();
}
