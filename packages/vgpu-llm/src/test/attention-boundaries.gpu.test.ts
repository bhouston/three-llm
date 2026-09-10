import { expect, it } from 'vitest';
import { AttentionKernel } from '../kernels/AttentionKernel.js';
import { uploadStorage, writeBuffer, readFloat32 } from '../gpu/device.js';
import { causalAttention } from '../runtime/math.js';
import { createGpu } from './gpu.js';
import { fillSin } from './helpers.js';

for (const slidingWindow of [0, 7]) {
  it(`FP32 GQA survives context/workgroup boundaries and reset (window=${slidingWindow})`, async ({ skip }) => {
    const gpu = await createGpu(skip);
    try {
      const options = {
        headCount: 4,
        kvHeadCount: 2,
        headDim: 6,
        ropeTheta: 10000,
        rotaryDim: 4,
        slidingWindow,
        qNormWeight: new Float32Array([1, 0.8, 1.2, 0.7, 1, 1.1]),
        kNormWeight: new Float32Array([0.9, 1, 1.1, 1.2, 1, 0.8]),
      };
      const maxTokens = 67;
      const input = uploadStorage(gpu, new Float32Array(48));
      const kernel = new AttentionKernel(gpu, input, 24, 4, maxTokens, { ...options, workgroupSize: 64 });
      for (const seed of [0.3, 4.7]) {
        kernel.reset();
        const keyCache = new Float32Array(12 * maxTokens);
        const valueCache = new Float32Array(12 * maxTokens);
        for (let position = 0; position < maxTokens; position++) {
          const qkv = fillSin(new Float32Array(48), seed + position * 0.37);
          writeBuffer(input, qkv);
          kernel.run(position);
          const expected = causalAttention(qkv, { ...options, keyCache, valueCache, position });
          const actual = await readFloat32(kernel.outputBuffer);
          for (let i = 0; i < expected.length; i++) {
            expect(Number.isFinite(actual[i])).toBe(true);
            expect(Math.abs(actual[i]! - expected[i]!)).toBeLessThanOrEqual(1e-5 + 1e-4 * Math.abs(expected[i]!));
          }
        }
      }
    } finally {
      gpu.dispose();
    }
  });
}
