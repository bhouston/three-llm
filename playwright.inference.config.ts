import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './packages/website/e2e',
  testMatch: 'inference.test.ts',
  workers: 1,
  retries: 0,
  reporter: [['list'], ['json', { outputFile: 'test-results/inference-report.json' }]],
  use: {
    ...devices['Desktop Chrome'],
    baseURL: 'http://127.0.0.1:4178',
    launchOptions: { args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] },
  },
  webServer: {
    command: 'node scripts/inference-test-server.mjs',
    url: 'http://127.0.0.1:4178',
    timeout: 30_000,
  },
});
