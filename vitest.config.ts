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
          include: ['packages/vgpu-llm/src/**/*.test.ts'],
          exclude: ['**/e2e/**', '**/*.gpu.test.ts', '**/*.checkpoint.test.ts'],
          environment: 'node',
          coverage: {
            provider: 'v8',
            reporter: ['text', 'lcov'],
            reportsDirectory: './coverage',
            include: ['packages/vgpu-llm/src/**/*.ts'],
            exclude: [
              '**/*.test.ts',
              '**/*.gpu.test.ts',
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
          // Real WGSL execution on a headless Dawn-backed device via
          // `vgpu/node` (Metal/Vulkan/D3D12) — no browser needed. Tests
          // `skip()` themselves on hosts without a usable GPU backend.
          name: 'library-gpu',
          include: ['packages/vgpu-llm/src/**/*.gpu.test.ts'],
          exclude: ['**/*.checkpoint.test.ts'],
          environment: 'node',
        },
      },
      {
        test: {
          name: 'checkpoints',
          include: ['packages/vgpu-llm/src/**/*.checkpoint.test.ts'],
          exclude: ['**/*.gpu.checkpoint.test.ts'],
          environment: 'node',
        },
      },
      {
        test: {
          name: 'checkpoints-gpu',
          include: ['packages/vgpu-llm/src/**/*.gpu.checkpoint.test.ts'],
          environment: 'node',
        },
      },
      {
        resolve: {
          alias: { '@': path.join(rootDir, 'packages/website/src') },
        },
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
      'vgpu-llm': path.join(rootDir, 'packages/vgpu-llm/src/index.ts'),
    },
  },
});
