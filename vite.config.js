import { defineConfig } from 'vite';

// `base` must match the repository name so that every asset (and public/models/human_heart.glb)
// resolves under https://<user>.github.io/heart/
export default defineConfig({
  base: '/heart/',
  build: {
    target: 'es2020',
    chunkSizeWarningLimit: 1500,
  },
});
