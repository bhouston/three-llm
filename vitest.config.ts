import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { playwright } from '@vitest/browser-playwright';
import { defineConfig } from 'vitest/config';

const rootDir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: 'library',
          include: ['packages/three-llm/src/**/*.test.ts'],
          exclude: ['**/e2e/**', '**/*.browser.test.ts', '**/*.checkpoint.test.ts'],
          environment: 'node',
          coverage: {
            provider: 'v8',
            reporter: ['text', 'lcov'],
            reportsDirectory: './coverage',
            include: ['packages/three-llm/src/**/*.ts'],
            exclude: [
              '**/*.test.ts',
              '**/*.browser.test.ts',
              '**/*.checkpoint.test.ts',
              '**/*.d.ts',
              '**/index.ts',
              '**/test/**',
            ],
          },
        },
      },
      {
        test: {
          name: 'library-browser',
          include: ['packages/three-llm/src/**/*.browser.test.ts'],
          exclude: ['**/*.checkpoint.browser.test.ts'],
          browser: {
            enabled: true,
            provider: playwright(),
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
            provider: playwright(),
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
