// The mobile layout shell (docs/plans/salesperson-ui.md §5.1's S12b row):
// sticky header, bottom sheets, safe-area insets, no horizontal scroll —
// run under both `playwright.config.ts` mobile projects (360×740, 390×844).
// `/shop/api/*` is mocked via `page.route()` throughout: this suite verifies
// the *client's* mobile chrome, not a live falkor-chat backend (there is
// none in this environment — see `../../README.md`'s "Bringing the stack
// up: manual until S11"); the reset control's fetch-level contract against
// a real server is S12a's own `hooks.test.tsx` plus S15's load harness.
import { expect, test } from '@playwright/test'

async function hasHorizontalOverflow(page: import('@playwright/test').Page) {
  return page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  )
}

test.describe('mobile shell', () => {
  test('no horizontal overflow on the join screen', async ({ page }) => {
    await page.goto('')
    await expect(page.getByText('Storefront')).toBeVisible()
    expect(await hasHorizontalOverflow(page)).toBe(false)
  })

  test('each header icon opens its own sheet, with no overflow while open, and closes', async ({
    page,
  }) => {
    await page.goto('')

    for (const label of ['Browse catalog', 'Cart', 'Order status', 'Profile']) {
      const button = page.getByRole('button', { name: label, exact: true })
      await button.click()
      const dialog = page.getByRole('dialog')
      await expect(dialog).toBeVisible()
      expect(await hasHorizontalOverflow(page)).toBe(false)

      await page.getByRole('button', { name: 'Close' }).last().click()
      await expect(dialog).toBeHidden()
    }
  })

  test('each of the four sheets mounts its own real S14 panel, not a shared placeholder', async ({
    page,
  }) => {
    // S14 replaced S12b's seed placeholders with real, data-fetching panels
    // (src/views/{Cart,Order,Profile,Catalog}Panel.tsx) — there is no
    // longer any literal placeholder copy to assert on here (DEF-5). This
    // page has no participant session, so every panel's query is
    // `enabled: false` (`src/api/hooks.ts`) and each renders its own
    // permanent, panel-specific "Loading …" copy — real production output,
    // not a placeholder invented for this test, and distinct per panel, so
    // it still proves *which* panel mounted rather than just that a dialog
    // opened. Same precedent already applied at the unit-test tier,
    // `src/layout/Shell.test.tsx`'s "wires each of the four header
    // buttons..." test (S12b-testfix).
    await page.goto('')

    await page.getByRole('button', { name: 'Browse catalog', exact: true }).click()
    await expect(page.getByText(/loading the catalog/i)).toBeVisible()
    await page.getByRole('button', { name: 'Close' }).last().click()

    await page.getByRole('button', { name: 'Cart', exact: true }).click()
    await expect(page.getByText(/loading your cart/i)).toBeVisible()
    await page.getByRole('button', { name: 'Close' }).last().click()

    await page.getByRole('button', { name: 'Order status', exact: true }).click()
    await expect(page.getByText(/loading your order/i)).toBeVisible()
    await page.getByRole('button', { name: 'Close' }).last().click()

    await page.getByRole('button', { name: 'Profile', exact: true }).click()
    await expect(page.getByText(/loading your profile/i)).toBeVisible()
  })

  test('the profile sheet shows a join prompt, not a reset control, before joining', async ({
    page,
  }) => {
    await page.goto('')
    await page.getByRole('button', { name: 'Profile', exact: true }).click()
    await expect(page.getByText(/join the store to manage your session/i)).toBeVisible()
    await expect(page.getByRole('button', { name: 'Reset my session' })).toHaveCount(0)
  })

  test('AC-5 — the reset control requires confirmation, then resets and returns to the language step', async ({
    page,
  }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem(
        'salesperson.participant',
        JSON.stringify({
          participantId: 'p-e2e-1',
          token: 'tok-e2e',
          displayName: 'Ada',
          language: 'pt-BR',
        }),
      )
    })
    await page.route('**/shop/api/state', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          profile: { name: 'Ada', deliveryAddress: null },
          cart: { items: [], total: 0 },
          order: null,
          turn: { state: 'idle', queuePosition: 0, lastTurn: null },
        }),
      }),
    )
    await page.route('**/shop/api/messages*', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: '[]' }),
    )
    let resetCalled = false
    await page.route('**/shop/api/reset', async (route) => {
      resetCalled = true
      expect(route.request().method()).toBe('POST')
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ threadId: 't-1', language: 'pt-BR' }),
      })
    })

    await page.goto('')
    await page.getByRole('button', { name: 'Profile', exact: true }).click()
    await page.getByRole('button', { name: 'Reset my session' }).click()
    await expect(page.getByText(/cannot be undone/i)).toBeVisible()

    await page.getByRole('button', { name: 'Yes, reset' }).click()
    await expect.poll(() => resetCalled).toBe(true)
    // The sheet closes and the client returns to the language-only step
    // (previous language pre-selected) rather than the full join form.
    await expect(page.getByRole('dialog')).toBeHidden()
    await expect(page.getByRole('heading', { name: /choose your language/i })).toBeVisible()
    expect(await hasHorizontalOverflow(page)).toBe(false)
  })
})
