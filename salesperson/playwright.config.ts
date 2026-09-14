import { defineConfig, devices } from '@playwright/test'

// Scaffolded by S5; S12b owns the e2e suite under tests/e2e/ and may extend
// this file (add projects, fixtures, a webServer block).
//
// The storefront is served by falkor-chat's FastAPI process at /shop, so
// Playwright is pointed at a *running* server rather than starting one.
// falkor-chat/scripts/start_demo.sh *will* bring it up, but that script is
// not built yet (plan step S11) — until then see "Bringing the stack up:
// manual until S11" in ./README.md.
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
    // Mobile-first is the product requirement (§4.2, AC-4) — every project
    // here is a mobile viewport, deliberately, at the two exact sizes S12b's
    // done-condition names (docs/plans/salesperson-ui.md §5.1's S12b row):
    // a small phone and a mid-size phone.
    {
      name: 'mobile-360x740',
      use: { ...devices['Pixel 7'], viewport: { width: 360, height: 740 } },
    },
    {
      name: 'mobile-390x844',
      use: { ...devices['Pixel 7'], viewport: { width: 390, height: 844 } },
    },
  ],
})
