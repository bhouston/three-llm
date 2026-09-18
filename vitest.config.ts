import type { IncomingMessage } from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { playwright } from '@vitest/browser-playwright';
import type { ProxyOptions } from 'vite';
import { defineConfig } from 'vitest/config';

const rootDir = path.dirname(fileURLToPath(import.meta.url));

// The real /api/models route (packages/website/src/routes/api/models/$.ts) turns
// ?part=&partSize= into a byte-range GCS read. This dev-server proxy talks to GCS
// directly, which ignores those params and returns the whole object every time —
// so concurrent chunk downloads overflow the shared target buffer. Translate them
// into a real Range header here so chunked downloads behave the same in tests.
function modelsProxy(): ProxyOptions {
  return {
    target: 'https://storage.googleapis.com/three-llm',
    changeOrigin: true,
    rewrite: (requestPath) => requestPath.replace(/^\/api\/models/, ''),
    configure(proxy) {
      proxy.on('proxyReq', (proxyReq, req: IncomingMessage) => {
        const url = new URL(req.url ?? '', 'http://localhost');
        const part = url.searchParams.get('part');
        const partSize = url.searchParams.get('partSize');
        if (part === null || partSize === null) return;

        const start = Number(part) * Number(partSize);
        const end = start + Number(partSize) - 1;
        if (Number.isSafeInteger(start) && Number.isSafeInteger(end)) {
          proxyReq.setHeader('Range', `bytes=${start}-${end}`);
        }
      });
    },
  };
}

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
            '/api/models': modelsProxy(),
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
        // Browser tests have no `process`; expose CI as a build-time constant so the
        // huge Phi-1.5 / Qwen3.5 checkpoints can be skipped there and still run locally.
        define: { 'import.meta.env.CI': JSON.stringify(!!process.env.CI) },
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
            '/api/models': modelsProxy(),
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
