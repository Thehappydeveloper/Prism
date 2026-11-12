import { writable } from 'svelte/store';

export type UIMode = 'dark' | 'clear';
export type AccentId = 'cyan' | 'rose' | 'lime' | 'amber' | 'violet';

export interface ThemeModeOption {
  id: UIMode;
  label: string;
}

export interface AccentOption {
  id: AccentId;
  label: string;
}

export interface ThemePrefs {
  mode: UIMode;
  accent: AccentId;
}

export const MODE_OPTIONS: ThemeModeOption[] = [
  { id: 'dark', label: 'Dark' },
  { id: 'clear', label: 'Clear' },
];

export const ACCENT_OPTIONS: AccentOption[] = [
  { id: 'cyan', label: 'Cyan' },
  { id: 'rose', label: 'Rose' },
  { id: 'lime', label: 'Lime' },
  { id: 'amber', label: 'Amber' },
  { id: 'violet', label: 'Violet' },
];

const STORAGE_KEY = 'prism_ui_theme_v2';
const DEFAULT_PREFS: ThemePrefs = { mode: 'dark', accent: 'cyan' };

function isMode(value: string | null): value is UIMode {
  return MODE_OPTIONS.some((option) => option.id === value);
}

function isAccent(value: string | null): value is AccentId {
  return ACCENT_OPTIONS.some((option) => option.id === value);
}

function normalize(raw: Partial<ThemePrefs> | null | undefined): ThemePrefs {
  const mode = isMode(raw?.mode || null) ? raw!.mode! : DEFAULT_PREFS.mode;
  const accent = isAccent(raw?.accent || null) ? raw!.accent! : DEFAULT_PREFS.accent;
  return { mode, accent };
}

function applyTheme(prefs: ThemePrefs): void {
  if (typeof document === 'undefined') return;
  document.documentElement.dataset.mode = prefs.mode;
  document.documentElement.dataset.accent = prefs.accent;
}

function storedPrefs(): ThemePrefs {
  if (typeof window === 'undefined') return DEFAULT_PREFS;
  try {
    const parsed = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '{}') as Partial<ThemePrefs>;
    return normalize(parsed);
  } catch {
    return DEFAULT_PREFS;
  }
}

export const themePrefs = writable<ThemePrefs>(DEFAULT_PREFS);

let hydrated = false;
themePrefs.subscribe((next) => {
  const normalized = normalize(next);
  applyTheme(normalized);
  if (!hydrated || typeof window === 'undefined') return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(normalized));
});

export function initializeTheme(): void {
  hydrated = true;
  themePrefs.set(storedPrefs());
}

export function setMode(next: string): void {
  themePrefs.update((current) => ({
    ...current,
    mode: isMode(next) ? next : DEFAULT_PREFS.mode,
  }));
}

export function setAccent(next: string): void {
  themePrefs.update((current) => ({
    ...current,
    accent: isAccent(next) ? next : DEFAULT_PREFS.accent,
  }));
}
