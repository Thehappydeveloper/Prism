type CacheEntry<T> = {
  value: T;
  createdAt: number;
  lastUsedAt: number;
};

export class GraphCache<T> {
  private entries = new Map<string, CacheEntry<T>>();

  constructor(
    private readonly name: string,
    private readonly maxEntries = 32,
  ) {}

  get(key: string): T | null {
    const started = performance.now();
    const entry = this.entries.get(key);
    if (!entry) {
      console.debug(`[graph-cache] ${this.name} miss`, { key, lookupMs: Math.round(performance.now() - started) });
      return null;
    }
    entry.lastUsedAt = performance.now();
    console.debug(`[graph-cache] ${this.name} hit`, {
      key,
      ageMs: Math.round(performance.now() - entry.createdAt),
      lookupMs: Math.round(performance.now() - started),
    });
    return entry.value;
  }

  set(key: string, value: T): void {
    if (this.entries.size >= this.maxEntries && !this.entries.has(key)) {
      const oldest = Array.from(this.entries.entries()).sort((a, b) => a[1].lastUsedAt - b[1].lastUsedAt)[0];
      if (oldest) {
        this.entries.delete(oldest[0]);
        console.debug(`[graph-cache] ${this.name} invalidate`, { key: oldest[0] });
      }
    }
    const now = performance.now();
    this.entries.set(key, { value, createdAt: now, lastUsedAt: now });
  }

  clear(): void {
    this.entries.clear();
    console.debug(`[graph-cache] ${this.name} cleared`);
  }
}

export function stableStringify(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map((item) => stableStringify(item)).join(',')}]`;
  return `{${Object.keys(value as Record<string, unknown>)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${stableStringify((value as Record<string, unknown>)[key])}`)
    .join(',')}}`;
}

export function scheduleGraphWork<T>(work: () => T | Promise<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    const run = () => {
      Promise.resolve()
        .then(work)
        .then(resolve)
        .catch(reject);
    };
    const idle = (window as Window & { requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => number }).requestIdleCallback;
    if (idle) idle(run, { timeout: 250 });
    else window.setTimeout(run, 0);
  });
}

export function timed<T>(label: string, work: () => T): T {
  const start = performance.now();
  try {
    return work();
  } finally {
    console.debug(`[graph-timing] ${label}`, { elapsedMs: Math.round(performance.now() - start) });
  }
}
