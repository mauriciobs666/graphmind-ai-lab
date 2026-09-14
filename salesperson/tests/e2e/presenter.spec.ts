// AC-5's presenter view (docs/plans/salesperson-ui.md §5.1's S12d row):
// key entry -> roster table (name + language only, §5.2's four-key
// projection, no activity data) -> reset-everyone behind a confirm step,
// including the `incomplete`/`unresolved` rendering. Own spec file —
// `./mobile-shell.spec.ts` is S12b's and is not touched here (§5.0's
// `tests/e2e/**` row: separate spec files, S12d owns this one).
// `/shop/api/*` is mocked via `page.route()` throughout, the same posture
// `mobile-shell.spec.ts` takes: this suite verifies the *client's*
// rendering contract, not a live falkor-chat backend.
import { expect, test } from '@playwright/test'

test.describe('presenter view', () => {
  test('key entry rejects a bad key without navigating, then accepts the right one and shows the roster', async ({
    page,
  }) => {
    let sessionAttempts = 0
    await page.route('**/shop/api/presenter/session', async (route) => {
      sessionAttempts += 1
      expect(route.request().method()).toBe('POST')
      const body = route.request().postDataJSON() as { key: string }
      if (body.key !== 'right-key') {
        await route.fulfill({
          status: 403,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'bad_key' }),
        })
        return
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ token: 'ptok-e2e' }),
      })
    })
    await page.route('**/shop/api/presenter/participants', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: '[]' }),
    )

    await page.goto('/presenter')
    await expect(page.getByRole('heading', { name: 'Presenter key' })).toBeVisible()

    await page.getByLabel('Presenter key').fill('wrong-key')
    await page.getByRole('button', { name: 'Enter' }).click()
    await expect(page.getByText('That key was not accepted.')).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Presenter key' })).toBeVisible()

    await page.getByLabel('Presenter key').fill('right-key')
    await page.getByRole('button', { name: 'Enter' }).click()
    await expect.poll(() => sessionAttempts).toBe(2)
    await expect(page.getByRole('heading', { name: 'Participants' })).toBeVisible()
    await expect(page.getByText('No one has joined yet.')).toBeVisible()
  })

  test('renders one row per participant with name and language only — no activity data', async ({
    page,
  }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem(
        'salesperson.presenter',
        JSON.stringify({ token: 'ptok-e2e' }),
      )
    })
    await page.route('**/shop/api/presenter/participants', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          { participantId: 'p-1', displayName: 'Ada', language: 'en', joinedAt: 1000 },
          { participantId: 'p-2', displayName: 'Grace', language: 'pt-BR', joinedAt: 2000 },
          // Understood (server-side) to carry cart items and a placed order
          // — the negative control for the four-key contract: the roster
          // must show nothing beyond name/language even for this row.
          { participantId: 'p-3', displayName: 'Marge', language: 'es', joinedAt: 3000 },
        ]),
      }),
    )

    await page.goto('/presenter')

    const rows = page.getByRole('row')
    await expect(rows).toHaveCount(4) // header + 3
    await expect(page.getByRole('columnheader', { name: 'Name' })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: 'Language' })).toBeVisible()
    for (const [name, language] of [
      ['Ada', 'en'],
      ['Grace', 'pt-BR'],
      ['Marge', 'es'],
    ]) {
      const row = page.getByRole('row', { name: new RegExp(name) })
      await expect(row).toContainText(name)
      await expect(row).toContainText(language)
    }
  })

  test('reset-everyone requires confirmation, then renders an incomplete sweep as a named list — not a clean sweep', async ({
    page,
  }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem(
        'salesperson.presenter',
        JSON.stringify({ token: 'ptok-e2e' }),
      )
    })
    await page.route('**/shop/api/presenter/participants', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          { participantId: 'p-1', displayName: 'Ada', language: 'en', joinedAt: 1000 },
        ]),
      }),
    )
    let resetCalled = false
    await page.route('**/shop/api/presenter/reset-all', async (route) => {
      resetCalled = true
      expect(route.request().method()).toBe('POST')
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          clearedParticipants: 0,
          incomplete: true,
          unresolved: ['p-stuck-1', 'p-stuck-2'],
        }),
      })
    })

    await page.goto('/presenter')
    await expect(page.getByText('Ada')).toBeVisible()

    await expect(page.getByText(/cannot be undone/i)).toHaveCount(0)
    await page.getByRole('button', { name: 'Reset everyone' }).click()
    await expect(
      page.getByText("This clears every participant's session for the whole demo. This cannot be undone."),
    ).toBeVisible()

    await page.getByRole('button', { name: 'Yes, reset everyone' }).click()
    await expect.poll(() => resetCalled).toBe(true)

    await expect(page.getByText(/reset incomplete/i)).toBeVisible()
    await expect(page.getByText('p-stuck-1')).toBeVisible()
    await expect(page.getByText('p-stuck-2')).toBeVisible()
    await expect(page.getByText(/reset complete/i)).toHaveCount(0)
  })
})
