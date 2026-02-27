const ALIASES_KEY = 'run_aliases';
const COMPARE_RUNS_KEY = 'selected_runs_for_compare';

export function loadRunAliases(): Record<string, string> {
  try {
    const parsed = JSON.parse(localStorage.getItem(ALIASES_KEY) || '{}');
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

export function saveRunAliases(aliases: Record<string, string>): void {
  localStorage.setItem(ALIASES_KEY, JSON.stringify(aliases));
}

export function loadSelectedCompareRuns(): string[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(COMPARE_RUNS_KEY) || '[]');
    return Array.isArray(parsed) ? parsed.filter((x) => typeof x === 'string') : [];
  } catch {
    return [];
  }
}

export function saveSelectedCompareRuns(runIds: string[]): void {
  localStorage.setItem(COMPARE_RUNS_KEY, JSON.stringify(runIds));
}
