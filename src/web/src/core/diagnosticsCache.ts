import type { RunMetrics } from './types';

const DATABASE_NAME = 'prism-advanced-compare';
const STORE_NAME = 'diagnostics-v1';
const MAX_ENTRIES = 12;

interface CacheEntry {
  created: number;
  metrics: RunMetrics;
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, 1);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) database.createObjectStore(STORE_NAME);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Could not open diagnostics cache'));
  });
}

export async function loadCachedDiagnostics(key: string): Promise<RunMetrics | null> {
  if (!key || typeof indexedDB === 'undefined') return null;
  try {
    const database = await openDatabase();
    return await new Promise((resolve, reject) => {
      const request = database.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).get(key);
      request.onsuccess = () => {
        const value = request.result as CacheEntry | RunMetrics | undefined;
        resolve(value && 'metrics' in value ? value.metrics : value || null);
      };
      request.onerror = () => reject(request.error);
    });
  } catch {
    return null;
  }
}

export async function saveCachedDiagnostics(key: string, metrics: RunMetrics): Promise<void> {
  if (!key || typeof indexedDB === 'undefined') return;
  try {
    const database = await openDatabase();
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      store.put({ created: Date.now(), metrics } satisfies CacheEntry, key);
      const entries: Array<{ key: IDBValidKey; created: number }> = [];
      const cursorRequest = store.openCursor();
      cursorRequest.onsuccess = () => {
        const cursor = cursorRequest.result;
        if (cursor) {
          entries.push({ key: cursor.key, created: Number((cursor.value as CacheEntry)?.created || 0) });
          cursor.continue();
          return;
        }
        entries.sort((a, b) => b.created - a.created);
        entries.slice(MAX_ENTRIES).forEach((entry) => store.delete(entry.key));
      };
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
    });
  } catch {
    // IndexedDB is an optional acceleration; the server cache remains authoritative.
  }
}
