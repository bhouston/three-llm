import { it } from 'vitest';
import { RMSNormKernel } from '../kernels/RMSNormKernel.js';
import { NormalizeKernel } from '../kernels/NormalizeKernel.js';
import { layerNorm, rmsNorm } from '../runtime/math.js';
import { readFloat32, uploadStorage } from '../gpu/device.js';
import { createGpu } from './gpu.js';
import { expectClose } from './reference.js';

for (const [size, workgroupSize] of [
  [1, 64],
  [3, 3],
  [65, 3],
  [257, 64],
  [768, 64],
  [2048, 64],
  [4096, 64],
]) {
  it(`cooperative normalization matches reference for H=${size}, WG=${workgroupSize}`, async ({ skip }) => {
    const gpu = await createGpu(skip);
    try {
      for (const kind of ['zero', 'constant', 'near-constant', 'varied']) {
        const values = Float32Array.from({ length: size }, (_, i) =>
          kind === 'zero'
            ? 0
            : kind === 'constant'
              ? 4
              : kind === 'near-constant'
                ? 4 + Math.sin(i) * 0.0001
                : Math.sin(i * 0.7) * 3,
        );
        const weight = Float32Array.from({ length: size }, (_, i) => Math.cos(i) * 0.2 + 0.9);
        const bias = Float32Array.from({ length: size }, (_, i) => Math.sin(i) * 0.1);
        const input = uploadStorage(gpu, values);
        const w = uploadStorage(gpu, weight);
        const b = uploadStorage(gpu, bias);
        const ln = new NormalizeKernel(gpu, input, w, b, size, { workgroupSize });
        ln.run();
        expectClose(
          await readFloat32(ln.outputBuffer),
          layerNorm(values, weight, bias),
          kind === 'near-constant' ? 2e-4 : 2e-5,
        );
        for (const offsetWeight of [false, true]) {
          const rms = new RMSNormKernel(gpu, input, w, size, { workgroupSize, offsetWeight });
          rms.run();
          expectClose(await readFloat32(rms.outputBuffer), rmsNorm(values, weight, 1e-5, offsetWeight));
        }
      }
    } finally {
      gpu.dispose();
    }
  });
}
