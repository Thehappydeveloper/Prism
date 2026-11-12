import type { RunRow } from './runExplorerUtils';

export function selectVisibleRows(rows: RunRow[]): Set<string> {
  return new Set(rows.map((row) => row.run.run_id));
}

export function invertVisibleRowsSelection(selected: Set<string>, rows: RunRow[]): Set<string> {
  const next = new Set(selected);
  rows.forEach((row) => {
    const id = row.run.run_id;
    if (next.has(id)) next.delete(id);
    else next.add(id);
  });
  return next;
}

export function selectRangeRows(
  rows: RunRow[],
  anchorIndex: number,
  rowIndex: number,
  selected: Set<string>,
  additive: boolean,
): Set<string> {
  const lo = Math.min(anchorIndex, rowIndex);
  const hi = Math.max(anchorIndex, rowIndex);
  const next = additive ? new Set(selected) : new Set<string>();
  for (let i = lo; i <= hi; i += 1) {
    next.add(rows[i]!.run.run_id);
  }
  return next;
}
