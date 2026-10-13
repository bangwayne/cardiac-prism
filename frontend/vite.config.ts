import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  assetsInclude: ['**/*.wasm'],
  plugins: [react()],
  optimizeDeps: {
    exclude: ['@cornerstonejs/core', '@cornerstonejs/tools'],
    include: [
      'globalthis', 'fast-deep-equal', 'xmlbuilder2', 'seedrandom', 'spark-md5', 'lodash.get',
      '@kitware/vtk.js/Filters/General/WindowedSincPolyDataFilter',
      '@kitware/vtk.js/Filters/Core/PolyDataNormals',
      '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow',
    ],
  },
  server: {
    proxy: { '/api': process.env.KPRISM_API_URL || 'http://127.0.0.1:8502' },
    headers: { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' },
  },
  build: { target: 'es2022', chunkSizeWarningLimit: 1800 },
  worker: { format: 'es' },
});
