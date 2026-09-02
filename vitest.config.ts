import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const rootDir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: 'library',
          include: ['packages/three-llm/src/**/*.test.ts'],
          exclude: ['**/e2e/**'],
          environment: 'node',
          coverage: {
            provider: 'v8',
            reporter: ['text', 'lcov'],
            reportsDirectory: './coverage',
            include: ['packages/three-llm/src/**/*.ts'],
            exclude: ['**/*.test.ts', '**/*.d.ts', '**/index.ts'],
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
