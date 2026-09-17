import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { playwright } from '@vitest/browser-playwright';
import { defineConfig } from 'vitest/config';

const rootDir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  test: {
    coverage: {
      provider: 'v8',
      thresholds: { lines: 59, statements: 58, branches: 58, functions: 52 },
      reporter: ['text', 'lcov', 'json-summary'],
      reportsDirectory: './coverage',
      include: ['packages/three-llm/src/**/*.ts'],
      exclude: ['**/*.test.ts', '**/*.d.ts', '**/index.ts', '**/test/**'],
    },
    projects: [
      {
        test: {
          name: 'library',
          include: ['packages/three-llm/src/**/*.test.ts'],
          exclude: ['**/e2e/**', '**/*.browser.test.ts', '**/*.checkpoint.test.ts'],
          environment: 'node',
        },
      },
      {
        test: {
          name: 'library-browser',
          include: ['packages/three-llm/src/**/*.browser.test.ts'],
          exclude: ['**/*.checkpoint.browser.test.ts'],
          browser: {
            enabled: true,
            provider: playwright({
              launchOptions: {
                channel: 'chromium',
                args: [
                  '--enable-unsafe-webgpu',
                  '--ignore-gpu-blocklist',
                  ...(process.env.WEBGPU_SOFTWARE ? ['--use-webgpu-adapter=swiftshader'] : []),
                ],
              },
            }),
            headless: true,
            instances: [{ browser: 'chromium' }],
          },
        },
        server: {
          proxy: {
            '/api/models': {
              target: 'https://storage.googleapis.com/three-llm',
              changeOrigin: true,
              rewrite: (requestPath) => requestPath.replace(/^\/api\/models/, ''),
            },
          },
        },
      },
      {
        test: {
          name: 'checkpoints',
          include: ['packages/three-llm/src/**/*.checkpoint.test.ts'],
          exclude: ['**/*.browser.test.ts'],
          environment: 'node',
        },
      },
      {
        test: {
          name: 'checkpoints-browser',
          include: ['packages/three-llm/src/**/*.checkpoint.browser.test.ts'],
          browser: {
            enabled: true,
            provider: playwright({
              launchOptions: {
                channel: 'chromium',
                args: [
                  '--enable-unsafe-webgpu',
                  '--ignore-gpu-blocklist',
                  ...(process.env.WEBGPU_SOFTWARE ? ['--use-webgpu-adapter=swiftshader'] : []),
                ],
              },
            }),
            headless: true,
            instances: [{ browser: 'chromium' }],
          },
        },
        server: {
          proxy: {
            '/api/models': {
              target: 'https://storage.googleapis.com/three-llm',
              changeOrigin: true,
              rewrite: (requestPath) => requestPath.replace(/^\/api\/models/, ''),
            },
          },
        },
      },
      {
        resolve: { alias: { '@': path.join(rootDir, 'packages/website/src') } },
        test: {
          name: 'website',
          include: ['packages/website/src/**/*.test.ts'],
          exclude: ['**/e2e/**'],
          environment: 'node',
        },
      },
    ],
  },
  resolve: {
    alias: {
      'three-llm': path.join(rootDir, 'packages/three-llm/src/index.ts'),
    },
  },
});
