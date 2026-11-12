export type TraceSortMode = 'name' | 'count';

export interface TraceGroupingOption {
  id: string;
  label: string;
}

export interface TraceFilterOption {
  id: string;
  label: string;
  values: string[];
}

export interface TraceGroup {
  name: string;
  traces: string[];
}

export interface TraceAnalysis {
  groupingOptions: TraceGroupingOption[];
  filterOptions: TraceFilterOption[];
  defaultGroupingId: string;
}

function normalizeFolders(path: string): string[] {
  const parts = path.split('/').filter(Boolean);
  return parts.slice(0, -1);
}

function commonPrefixLength(paths: string[][]): number {
  if (!paths.length) return 0;
  const shortest = Math.min(...paths.map((parts) => parts.length));
  let idx = 0;
  while (idx < shortest) {
    const value = paths[0]?.[idx];
    if (!value) break;
    if (paths.some((parts) => parts[idx] !== value)) break;
    idx += 1;
  }
  return idx;
}

function normalizeRoot(rootPath: string): string[] {
  return rootPath.split('/').filter(Boolean);
}

function relativeFolders(paths: string[], rootPath = ''): string[][] {
  const folders = paths.map(normalizeFolders);
  const rootParts = normalizeRoot(rootPath);
  const canUseRoot = rootParts.length > 0 && folders.every((parts) => rootParts.every((segment, idx) => parts[idx] === segment));
  if (canUseRoot) {
    return folders.map((parts) => parts.slice(rootParts.length));
  }
  const prefixLen = commonPrefixLength(folders);
  return folders.map((parts) => parts.slice(prefixLen));
}

function segmentLabel(levelIndex: number, sampleValue: string, isExample: boolean): string {
  if (!sampleValue) return `Subfolder ${levelIndex + 1}`;
  return isExample ? `Subfolder ${levelIndex + 1} (e.g. ${sampleValue})` : `Subfolder ${levelIndex + 1} (${sampleValue})`;
}

function uniqueSorted(values: Iterable<string>): string[] {
  return Array.from(new Set(Array.from(values).filter(Boolean))).sort((a, b) => a.localeCompare(b));
}

function groupNameFor(relative: string[], groupingId: string): string {
  if (!relative.length) return 'Ungrouped';
  if (groupingId === 'path') return relative.join(' / ');
  const [mode, rawIndex] = groupingId.split(':');
  const index = Number(rawIndex);
  if (!Number.isFinite(index) || index < 0) return relative[0] || 'Ungrouped';
  if (mode === 'pair') {
    const left = relative[index] || 'Other';
    const right = relative[index + 1] || 'Other';
    return left === right ? left : `${left} / ${right}`;
  }
  return relative[index] || 'Other';
}

export function analyzeTracePaths(paths: string[], rootPath = ''): TraceAnalysis {
  const relatives = relativeFolders(paths, rootPath);
  const maxDepth = relatives.reduce((max, parts) => Math.max(max, parts.length), 0);
  const groupingOptions: TraceGroupingOption[] = [];
  const filterOptions: TraceFilterOption[] = [];
  const samplePath = relatives.find((parts) => parts.length) || [];

  for (let index = 0; index < maxDepth; index += 1) {
    const values = uniqueSorted(relatives.map((parts) => parts[index] || ''));
    if (!values.length) continue;
    const sampleValue = samplePath[index] || values[0] || '';
    const isExample = values.length > 1;
    groupingOptions.push({ id: `level:${index}`, label: segmentLabel(index, sampleValue, isExample) });
    if (values.length > 1) {
      filterOptions.push({ id: `level:${index}`, label: segmentLabel(index, sampleValue, true), values });
    }
  }

  const defaultGroupingId = groupingOptions.find((option) => option.id === 'level:0')?.id
    || groupingOptions[0]?.id
    || 'level:0';

  return { groupingOptions, filterOptions, defaultGroupingId };
}

export function traceDisplayName(path: string): string {
  const parts = path.split('/').filter(Boolean);
  return parts.slice(-2).join('/') || path;
}

export function summarizeSingleTrace(path: string): string {
  return path || '';
}

export function summarizeMultiTrace(paths: string[], maxItems = 3): string {
  if (!paths.length) return '';
  const names = paths.map(traceDisplayName).sort((a, b) => a.localeCompare(b));
  if (names.length <= maxItems) return names.join(', ');
  return `${names.slice(0, maxItems).join(', ')} and ${names.length - maxItems} more`;
}

export function groupTraces(
  paths: string[],
  rootPath: string,
  groupingId: string,
  filterSelections: Record<string, string[]>,
  searchText: string,
  sortMode: TraceSortMode,
): TraceGroup[] {
  const relatives = relativeFolders(paths, rootPath);
  const needle = searchText.trim().toLowerCase();
  const groups = new Map<string, string[]>();

  paths.forEach((path, idx) => {
    const relative = relatives[idx] || [];
    const matchesFilters = Object.entries(filterSelections).every(([filterId, values]) => {
      if (!values.length) return true;
      const [, rawIndex] = filterId.split(':');
      const index = Number(rawIndex);
      const segment = relative[index] || '';
      return values.includes(segment);
    });
    if (!matchesFilters) return;

    const groupName = groupNameFor(relative, groupingId);
    const display = traceDisplayName(path).toLowerCase();
    if (needle && !groupName.toLowerCase().includes(needle) && !path.toLowerCase().includes(needle) && !display.includes(needle)) {
      return;
    }

    if (!groups.has(groupName)) groups.set(groupName, []);
    groups.get(groupName)!.push(path);
  });

  let out = Array.from(groups.entries()).map(([name, traces]) => ({
    name,
    traces: traces.slice().sort((a, b) => a.localeCompare(b)),
  }));

  out = out.sort((a, b) => {
    if (sortMode === 'count') return b.traces.length - a.traces.length || a.name.localeCompare(b.name);
    return a.name.localeCompare(b.name);
  });

  return out;
}
