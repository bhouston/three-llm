import { createRequire } from 'node:module';
import path from 'node:path';
import { expect, test } from '@playwright/test';

const libraryPath = path.resolve('packages/vgpu-llm');
const require = createRequire(path.join(libraryPath, 'package.json'));

test('browser executes FP32 linear inference with a partial workgroup', async ({ page }, testInfo) => {
  await page.goto('/');
  const result = await page.evaluate(
    async ({ library, vgpu }) => {
      const { init } = await import(/* @vite-ignore */ vgpu);
      const { LinearKernel } = await import(/* @vite-ignore */ `${library}/src/kernels/LinearKernel.ts`);
      const { uploadStorage, readFloat32 } = await import(/* @vite-ignore */ `${library}/src/gpu/device.ts`);
      // A missing adapter or shader validation error must fail this GPU test.
      const gpu = await init();
      try {
        const input = uploadStorage(gpu, new Float32Array([1, -2, 0.5]));
        const kernel = new LinearKernel(
          gpu,
          input,
          new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
          new Float32Array([1, 0, -1, 2, -2]),
          3,
          5,
          { precision: 'fp32', workgroupSize: 4 },
        );
        kernel.run();
        const output = Array.from(await readFloat32(kernel.outputBuffer));
        // Warm pipeline, completed GPU work, repeated samples; no universal
        // latency threshold because browser/adapter/driver scheduling varies.
        const samples = [];
        for (let sample = 0; sample < 5; sample++) {
          const start = performance.now();
          for (let i = 0; i < 20; i++) kernel.run();
          await readFloat32(kernel.outputBuffer);
          samples.push((performance.now() - start) / 20);
        }
        const adapter = gpu.device.adapterInfo;
        return {
          output,
          millisecondsPerDispatchIncludingReadback: samples,
          userAgent: navigator.userAgent,
          adapter: adapter
            ? {
                vendor: adapter.vendor,
                architecture: adapter.architecture,
                device: adapter.device,
                description: adapter.description,
              }
            : null,
        };
      } finally {
        gpu.dispose();
      }
    },
    { library: `/@fs${libraryPath}`, vgpu: `/@fs${require.resolve('vgpu')}` },
  );
  expect(result.output).toEqual([-4.5, -6, -7.5, -5, -9.5]);
  await testInfo.attach('fp32-linear-timing.json', {
    body: JSON.stringify(result, null, 2),
    contentType: 'application/json',
  });
});
