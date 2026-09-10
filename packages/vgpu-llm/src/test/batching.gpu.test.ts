import { expect, it, vi } from 'vitest';
import { allocStorage, makeCompute, readFloat32, uploadStorage, withComputeBatch, writeBuffer } from '../gpu/device.js';
import { LinearKernel } from '../kernels/LinearKernel.js';
import { DecoderGpuRunner } from '../decoder/DecoderGpuRunner.js';
import { QwenGpuRunner } from '../qwen/QwenGpuRunner.js';
import { createTinyLlama, createTinyQwenWeights } from './helpers.js';
import { createGpu } from './gpu.js';
import { expectClose } from './reference.js';

for (const family of ['llama', 'qwen']) {
  it(`${family}: one submission per token/chunk with identical logits`, async ({ skip }) => {
    const gpu = await createGpu(skip);
    try {
      const build = (batchCompute: boolean) =>
        family === 'llama'
          ? new DecoderGpuRunner(gpu, createTinyLlama(), { batchCompute, maxTokens: 16, prefillChunkSize: 3 })
          : new QwenGpuRunner(gpu, createTinyQwenWeights(), { batchCompute, maxTokens: 16, prefillChunkSize: 3 });
      const before = build(false),
        after = build(true);
      const submit = vi.spyOn(gpu.device.gpu.queue, 'submit');
      const tokens = [1, 2, 3, 1, 2, 3, 2];
      submit.mockClear();
      before.computeToken(tokens[0], 0);
      const baselineSubmits = submit.mock.calls.length;
      expect(baselineSubmits).toBeGreaterThan(10);
      submit.mockClear();
      after.computeToken(tokens[0], 0);
      expect(submit).toHaveBeenCalledTimes(1);
      expectClose(await after.readLogits(), await before.readLogits());
      before.resetCache();
      after.resetCache();
      await before.prefillTokens(tokens, 0, 6);
      submit.mockClear();
      await after.prefillTokens(tokens, 0, 6);
      expect(submit).toHaveBeenCalledTimes(2); // two chunks, GPU-owned cursor/position
      before.computeToken(tokens[6], 6);
      after.computeToken(tokens[6], 6);
      expectClose(await after.readLogits(), await before.readLogits());
      submit.mockRestore();
    } finally {
      gpu.dispose();
    }
  });
}
it('queue writes flush prior dispatches; nested batches preserve ordering', async ({ skip }) => {
  const gpu = await createGpu(skip);
  try {
    const input = uploadStorage(gpu, new Float32Array([2]));
    const a = new LinearKernel(gpu, input, new Float32Array([3]), null, 1, 1);
    const b = new LinearKernel(gpu, input, new Float32Array([5]), null, 1, 1);
    withComputeBatch(gpu, () => {
      a.run();
      writeBuffer(input, new Float32Array([7]));
      withComputeBatch(gpu, () => {
        b.run();
      });
    });
    expect(Array.from(await readFloat32(a.outputBuffer))).toEqual([6]);
    expect(Array.from(await readFloat32(b.outputBuffer))).toEqual([35]);
  } finally {
    gpu.dispose();
  }
});
it('failed batch abandons pending commands and the next batch still works', async ({ skip }) => {
  const gpu = await createGpu(skip);
  try {
    const input = uploadStorage(gpu, new Float32Array([2]));
    const kernel = new LinearKernel(gpu, input, new Float32Array([3]), null, 1, 1);
    expect(() =>
      withComputeBatch(gpu, () => {
        kernel.run();
        throw new Error('stop');
      }),
    ).toThrow('stop');
    expect(Array.from(await readFloat32(kernel.outputBuffer))).toEqual([0]);
    withComputeBatch(gpu, () => {
      kernel.run();
    });
    expect(Array.from(await readFloat32(kernel.outputBuffer))).toEqual([6]);
    const buffer = allocStorage(gpu, 1);
    const invalid = makeCompute(
      gpu,
      `
      @group(0) @binding(0) var<storage, read_write> a: array<f32>;
      @group(0) @binding(1) var<storage, read_write> b: array<f32>;
      @compute @workgroup_size(1) fn cs_main() { a[0] = b[0] + 1.; }
    `,
      { set: { a: buffer, b: buffer } },
    );
    expect(() => withComputeBatch(gpu, () => invalid.dispatch(1))).toThrow(/alias/);
  } finally {
    gpu.dispose();
  }
});
