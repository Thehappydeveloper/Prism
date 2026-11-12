<script lang="ts">
  import { route } from './core/router';
  import type { RouteId } from './core/types';
  import HomePage from './pages/HomePage.svelte';
  import ControlCenterPage from './pages/ControlCenterPage.svelte';
  import BatchPlannerPage from './pages/BatchPlannerPage.svelte';
  import AdvancedComparePage from './pages/AdvancedComparePage.svelte';
  import ReportsPage from './pages/ReportsPage.svelte';
  import SettingsPage from './pages/SettingsPage.svelte';
  import TrainingPage from './pages/TrainingPage.svelte';
  import DatasetsPage from './pages/DatasetsPage.svelte';
  import ExtraPage from './pages/ExtraPage.svelte';
  import SpeedAccelerationAnalyzerPage from './pages/SpeedAccelerationAnalyzerPage.svelte';
  import PointCloudSequenceAnalyzerPage from './pages/PointCloudSequenceAnalyzerPage.svelte';
  import TraceHeightNormalizerPage from './pages/TraceHeightNormalizerPage.svelte';

  $: currentRoute = $route;
  let mountedRoutes: RouteId[] = [];

  $: if (currentRoute && !mountedRoutes.includes(currentRoute)) {
    mountedRoutes = [...mountedRoutes, currentRoute];
  }
</script>

{#if mountedRoutes.includes('home')}
  <div class:page-hidden={currentRoute !== 'home'}>
    <HomePage />
  </div>
{/if}

{#if mountedRoutes.includes('control')}
  <div class:page-hidden={currentRoute !== 'control'}>
    <ControlCenterPage />
  </div>
{/if}

{#if mountedRoutes.includes('batch')}
  <div class:page-hidden={currentRoute !== 'batch'}>
    <BatchPlannerPage />
  </div>
{/if}

{#if mountedRoutes.includes('compare')}
  <div class:page-hidden={currentRoute !== 'compare'}>
    <AdvancedComparePage />
  </div>
{/if}

{#if mountedRoutes.includes('reports')}
  <div class:page-hidden={currentRoute !== 'reports'}>
    <ReportsPage />
  </div>
{/if}

{#if mountedRoutes.includes('settings')}
  <div class:page-hidden={currentRoute !== 'settings'}>
    <SettingsPage />
  </div>
{/if}

{#if mountedRoutes.includes('training')}
  <div class:page-hidden={currentRoute !== 'training'}>
    <TrainingPage />
  </div>
{/if}

{#if mountedRoutes.includes('datasets')}
  <div class:page-hidden={currentRoute !== 'datasets'}>
    <DatasetsPage />
  </div>
{/if}

{#if mountedRoutes.includes('extra')}
  <div class:page-hidden={currentRoute !== 'extra'}>
    <ExtraPage />
  </div>
{/if}

{#if mountedRoutes.includes('speedAcceleration')}
  <div class:page-hidden={currentRoute !== 'speedAcceleration'}>
    <SpeedAccelerationAnalyzerPage />
  </div>
{/if}

{#if mountedRoutes.includes('pointCloudSequence')}
  <div class:page-hidden={currentRoute !== 'pointCloudSequence'}>
    <PointCloudSequenceAnalyzerPage />
  </div>
{/if}

{#if mountedRoutes.includes('traceHeightNormalizer')}
  <div class:page-hidden={currentRoute !== 'traceHeightNormalizer'}>
    <TraceHeightNormalizerPage />
  </div>
{/if}

<style>
  .page-hidden {
    display: none;
  }
</style>
