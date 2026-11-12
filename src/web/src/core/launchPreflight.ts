import type { LaunchPreflightMode } from './types';
const KEY = 'prism_launch_preflight_mode_v1';
export function loadLaunchPreflightMode(): LaunchPreflightMode {
  const value = typeof localStorage === 'undefined' ? 'issues' : localStorage.getItem(KEY);
  return value === 'always' || value === 'never' ? value : 'issues';
}
export function saveLaunchPreflightMode(mode: LaunchPreflightMode): void { localStorage.setItem(KEY, mode); }
