/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The bundle is served by falkor-chat's FastAPI process, mounted at /shop.
// `base` must match that mount point or every hashed asset URL 404s.
// See salesperson/README.md and docs/plans/salesperson-ui.md §4.2.
export default defineConfig({
  base: '/shop/',
  plugins: [react(), tailwindcss()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./vitest.setup.ts'],
    // Unit/component tests live beside the code they cover.
    // tests/e2e/** is Playwright's; Vitest must not try to run it.
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['node_modules/**', 'dist/**', 'tests/e2e/**'],
    // The scaffold ships no tests of its own (S5 owns no src/ file's content).
    // Drop this once S12a lands the first real test.
    passWithNoTests: true,
  },
})
