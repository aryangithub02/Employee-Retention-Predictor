import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        // Backend runs on port 8000 (uvicorn). Proxy API requests to that server.
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
