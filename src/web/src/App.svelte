<script lang="ts">
  import { route } from './core/router';
  import HomePage from './pages/HomePage.svelte';
  import ControlCenterPage from './pages/ControlCenterPage.svelte';
  import BatchPlannerPage from './pages/BatchPlannerPage.svelte';
  import AdvancedComparePage from './pages/AdvancedComparePage.svelte';

  $: currentRoute = $route;
</script>

{#if currentRoute === 'home'}
  <HomePage />
{:else if currentRoute === 'control'}
  <ControlCenterPage />
{:else if currentRoute === 'batch'}
  <BatchPlannerPage />
{:else if currentRoute === 'compare'}
  <AdvancedComparePage />
{:else}
  <HomePage />
{/if}
