import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  resolve: { alias: mode === 'benchmark' ? [{ find: 'react-dom/client', replacement: 'react-dom/profiling' }] : [] },
  server: { proxy: { '/api': 'http://127.0.0.1:18765' } },
  build: { sourcemap: false },
}))
