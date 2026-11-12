<script lang="ts">
  import { ACCENT_OPTIONS, MODE_OPTIONS, setAccent, setMode, themePrefs } from '../core/theme';

  function onModeChange(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement | null;
    setMode(target?.value || '');
  }

  function onAccentChange(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement | null;
    setAccent(target?.value || '');
  }
</script>

<section class="theme-switcher" aria-label="Theme settings">
  <label>
    <span>Theme</span>
    <select value={$themePrefs.mode} on:change={onModeChange}>
      {#each MODE_OPTIONS as option}
        <option value={option.id}>{option.label}</option>
      {/each}
    </select>
  </label>

  <label>
    <span>Accent</span>
    <select value={$themePrefs.accent} on:change={onAccentChange}>
      {#each ACCENT_OPTIONS as option}
        <option value={option.id}>{option.label}</option>
      {/each}
    </select>
  </label>
</section>

<style>
  .theme-switcher {
    display: grid;
    grid-template-columns: repeat(2, minmax(140px, 1fr));
    gap: 0.55rem;
    padding: 0.65rem;
    border-radius: 12px;
    border: 1px solid var(--ghost-border);
    background: var(--soft-bg);
    width: min(100%, 460px);
  }

  .theme-switcher label {
    gap: 0.25rem;
  }

  .theme-switcher span {
    color: var(--text-subtle);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
  }

  .theme-switcher select {
    min-width: 0;
    width: 100%;
    padding: 0.42rem 0.55rem;
    font-size: 0.88rem;
    border-radius: 8px;
  }

  @media (max-width: 720px) {
    .theme-switcher {
      grid-template-columns: 1fr;
      width: 100%;
    }
  }
</style>
