import { defineConfig, devices } from '@playwright/test'

// Scaffolded by S5; S12b owns the e2e suite under tests/e2e/ and may extend
// this file (add projects, fixtures, a webServer block).
//
// The storefront is served by falkor-chat's FastAPI process at /shop, so
// Playwright is pointed at a *running* server rather than starting one:
// bring it up with falkor-chat/scripts/start_demo.sh first.
// Override the target with SALESPERSON_E2E_BASE_URL.
const baseURL =
  process.env.SALESPERSON_E2E_BASE_URL ?? 'http://127.0.0.1:8000/shop/'

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? 'line' : 'list',
  use: {
    baseURL,
    trace: 'on-first-retry',
  },
  projects: [
    {
      // Mobile-first is the product requirement (§4.2, AC-4): the only
      // project is a mobile viewport, deliberately.
      name: 'mobile-chrome',
      use: { ...devices['Pixel 7'] },
    },
  ],
})
