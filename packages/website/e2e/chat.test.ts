import { expect, test } from '@playwright/test';

test('chat page renders', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('body')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Three-LLM' })).toBeVisible();
  await expect(page.getByRole('contentinfo')).toContainText('Built with');
  await expect(page.getByRole('link', { name: 'Ben Houston' })).toHaveAttribute('href', 'https://ben3d.ca');
});
