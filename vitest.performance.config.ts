import { mkdirSync, writeFileSync } from 'node:fs';
import { playwright } from '@vitest/browser-playwright';
import { defineConfig } from 'vitest/config';
export default defineConfig({
  test: {
    fileParallelism: false,
    include: ['packages/three-llm/src/test/*.benchmark.ts'],
    testTimeout: 120000,
    browser: {
      enabled: true,
      provider: playwright({
        launchOptions: { channel: 'chromium', args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] },
      }),
      headless: true,
      instances: [{ browser: 'chromium' }],
      commands: {
        saveBenchmark: (_context, name: string, data: unknown) => {
          if (!['normalization', 'submission'].includes(name)) throw new Error('Unknown benchmark');
          mkdirSync('profile-output', { recursive: true });
          writeFileSync(`profile-output/astra-${name}.json`, JSON.stringify(data, null, 2) + '\n');
        },
      },
    },
  },
});
