import { expect, test } from '@playwright/test';

test('chat page renders', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('body')).toBeVisible();
});
