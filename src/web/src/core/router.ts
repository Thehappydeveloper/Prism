import { writable } from 'svelte/store';
import type { RouteId } from './types';

export const route = writable<RouteId>(resolveRoute(window.location.pathname));

export function resolveRoute(pathname: string): RouteId {
  const p = pathname.toLowerCase();
  if (p.endsWith('/static/index.html')) return 'control';
  if (p.endsWith('/static/aggregate.html')) return 'batch';
  if (p.endsWith('/static/compare.html')) return 'compare';
  if (p.endsWith('/static/reports.html')) return 'reports';
  if (p.endsWith('/static/settings.html')) return 'settings';
  if (p.endsWith('/static/training.html')) return 'training';
  if (p.endsWith('/static/datasets.html')) return 'datasets';
  if (p.endsWith('/static/extra.html')) return 'extra';
  if (p.endsWith('/static/speed-acceleration.html')) return 'speedAcceleration';
  if (p.endsWith('/static/point-cloud-sequence-analyzer.html')) return 'pointCloudSequence';
  if (p.endsWith('/static/dataset-transformation.html')) return 'traceHeightNormalizer';
  if (p.endsWith('/static/trace-height-normalizer.html')) return 'traceHeightNormalizer';
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
    case 'reports':
      return '/static/reports.html';
    case 'settings':
      return '/static/settings.html';
    case 'training':
      return '/static/training.html';
    case 'datasets':
      return '/static/datasets.html';
    case 'extra':
      return '/static/extra.html';
    case 'speedAcceleration':
      return '/static/speed-acceleration.html';
    case 'pointCloudSequence':
      return '/static/point-cloud-sequence-analyzer.html';
    case 'traceHeightNormalizer':
      return '/static/dataset-transformation.html';
    case 'home':
    default:
      return '/';
  }
}

window.addEventListener('popstate', () => {
  route.set(resolveRoute(window.location.pathname));
});
