import './styles/global.css';
import App from './App.svelte';
import { initializeTheme } from './core/theme';

initializeTheme();

const app = new App({
  target: document.getElementById('app') as HTMLElement,
});

export default app;
