import { it } from 'vitest';
import { TSLRMSNorm } from '../tsl/TSLRMSNorm.js';
import { TSLNormalize } from '../tsl/TSLNormalize.js';
import { createRenderer, storageFromArray, readOutput } from './gpu.js';
import { expectClose } from './reference.js';
for (const [size, workgroupSize] of [
  [1, 1],
  [3, 3],
  [31, 7],
  [65, 64],
  [257, 63],
  [768, 64],
  [2048, 64],
  [4096, 64],
]) {
  it(`normalization size=${size}, workgroup=${workgroupSize} matches double precision`, async () => {
    const renderer = await createRenderer(() => {
      throw new Error('WebGPU required');
    });
    try {
      for (const offset of [0, 10000]) {
        const data = Float32Array.from({ length: size }, (_, i) => offset + Math.sin(i * 0.4) * 0.5);
        const input = storageFromArray(data);
        const weight = Float32Array.from({ length: size }, (_, i) => 0.8 + i / size);
        const bias = Float32Array.from({ length: size }, (_, i) => i * 0.01);
        const epsilon = 1e-5;
        const mean = Array.from(data).reduce((a, b) => a + b, 0) / size;
        const variance = Array.from(data).reduce((a, b) => a + (b - mean) ** 2, 0) / size;
        const rms = Math.sqrt(Array.from(data).reduce((a, b) => a + b * b, 0) / size + epsilon);
        const layer = new TSLNormalize(input.node, weight, bias, size, { workgroupSize, epsilon });
        layer.compute(renderer);
        expectClose(
          await readOutput(renderer, layer),
          Array.from(data, (x, i) => ((x - mean) / Math.sqrt(variance + epsilon)) * weight[i] + bias[i]),
          2e-5,
          2e-5,
        );
        for (const offsetWeight of [false, true]) {
          const norm = new TSLRMSNorm(input.node, weight, size, { workgroupSize, epsilon, offsetWeight });
          norm.compute(renderer);
          expectClose(
            await readOutput(renderer, norm),
            Array.from(data, (x, i) => (x / rms) * (weight[i] + Number(offsetWeight))),
            2e-5,
            2e-5,
          );
        }
      }
      const input = storageFromArray(new Float32Array(size).fill(10000));
      const layer = new TSLNormalize(input.node, new Float32Array(size).fill(1), null, size, { workgroupSize });
      layer.compute(renderer);
      expectClose(await readOutput(renderer, layer), new Float32Array(size), 1e-7, 0);
    } finally {
      renderer.dispose();
    }
  });
}
