import { writable } from 'svelte/store';
import type { RouteId } from './types';

export const route = writable<RouteId>(resolveRoute(window.location.pathname));

export function resolveRoute(pathname: string): RouteId {
  const p = pathname.toLowerCase();
  if (p.endsWith('/static/index.html')) return 'control';
  if (p.endsWith('/static/aggregate.html')) return 'batch';
  if (p.endsWith('/static/compare.html')) return 'compare';
  if (p.endsWith('/static/home.html')) return 'home';
  return 'home';
}

export function gotoRoute(next: RouteId): void {
  const path = routeToPath(next);
  history.pushState({}, '', path);
  route.set(next);
}

export function routeToPath(next: RouteId): string {
  switch (next) {
    case 'control':
      return '/static/index.html';
    case 'batch':
      return '/static/aggregate.html';
    case 'compare':
      return '/static/compare.html';
    case 'home':
    default:
      return '/';
  }
}

window.addEventListener('popstate', () => {
  route.set(resolveRoute(window.location.pathname));
});
