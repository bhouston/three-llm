import { mkdirSync, writeFileSync } from 'node:fs';
import { init } from 'vgpu/node';
import { it } from 'vitest';
import { NormalizeKernel } from '../kernels/NormalizeKernel.js';
import { RMSNormKernel } from '../kernels/RMSNormKernel.js';
import { BaselineNormalizeKernel } from './baselines/NormalizeKernel.js';
import { BaselineRMSNormKernel } from './baselines/RMSNormKernel.js';
import { readFloat32, uploadStorage, uploadWeightStorage, withComputeBatch } from '../gpu/device.js';
import { expectClose } from './reference.js';

const median = (values: number[]) => values.toSorted((left, right) => left - right)[Math.floor(values.length / 2)];

it('paired normalization benchmark (completed queue work, no universal latency gate)', async () => {
  const gpu = await init({ requiredFeatures: ['shader-f16', 'timestamp-query'] });
  try {
    const rows = [];
    const iterations = 128;
    for (const precision of ['fp32', 'fp16'] as const)
      for (const size of [768, 2048, 4096])
        for (const kind of ['rms', 'layer']) {
          const input = uploadStorage(
            gpu,
            Float32Array.from({ length: size }, (_, i) => Math.sin(i * 0.17)),
          );
          const w = uploadWeightStorage(gpu, new Float32Array(size).fill(1), precision);
          const b = uploadWeightStorage(gpu, new Float32Array(size), precision);
          const before =
            kind === 'rms'
              ? new BaselineRMSNormKernel(gpu, input, w, size, { precision })
              : new BaselineNormalizeKernel(gpu, input, w, b, size, { precision });
          const after =
            kind === 'rms'
              ? new RMSNormKernel(gpu, input, w, size, { precision })
              : new NormalizeKernel(gpu, input, w, b, size, { precision });
          before.run();
          after.run();
          expectClose(await readFloat32(after.outputBuffer), await readFloat32(before.outputBuffer));
          const querySet = gpu.device.gpu.createQuerySet({ type: 'timestamp', count: 4 });
          const resolved = gpu.device.gpu.createBuffer({ size: 256, usage: 512 | 4 });
          const staging = gpu.device.gpu.createBuffer({ size: 32, usage: 1 | 8 });
          const measurePair = async (reverse: boolean) => {
            const enqueue = (kernel: { run(): unknown }, offset: number) =>
              withComputeBatch(
                gpu,
                () => {
                  for (let i = 0; i < iterations; i++) kernel.run();
                },
                { querySet, beginningOfPassWriteIndex: offset, endOfPassWriteIndex: offset + 1 },
              );
            // Queue both sides before waiting, reducing power-state differences
            // caused by readback gaps. Query slots always identify before/after.
            if (reverse) {
              enqueue(after, 2);
              enqueue(before, 0);
            } else {
              enqueue(before, 0);
              enqueue(after, 2);
            }
            const encoder = gpu.device.gpu.createCommandEncoder();
            encoder.resolveQuerySet(querySet, 0, 4, resolved, 0);
            encoder.copyBufferToBuffer(resolved, 0, staging, 0, 32);
            gpu.device.gpu.queue.submit([encoder.finish()]);
            await staging.mapAsync(1);
            const times = new BigUint64Array(staging.getMappedRange());
            const durations = [Number(times[1] - times[0]), Number(times[3] - times[2])].map(
              (ns) => ns / 1e6 / iterations,
            );
            staging.unmap();
            return durations;
          };
          for (let i = 0; i < 5; i++) await measurePair(i % 2 === 0);
          const baseline: number[] = [],
            candidate: number[] = [];
          for (let i = 0; i < 31; i++) {
            const pair = await measurePair(i % 2 === 0);
            baseline.push(pair[0]);
            candidate.push(pair[1]);
          }
          querySet.destroy();
          resolved.destroy();
          staging.destroy();
          rows.push({
            kind,
            size,
            precision,
            baseline,
            candidate,
            medianSpeedup: median(baseline.map((value, i) => value / candidate[i])),
          });
        }
    const info = gpu.device.adapterInfo;
    const report = {
      adapter: info && {
        vendor: info.vendor,
        architecture: info.architecture,
        description: info.description,
        device: info.device,
      },
      node: process.version,
      iterations,
      metric:
        'GPU timestamp ms per dispatch, 128 dispatches per pass; warm, alternating paired order, one readback per pair',
      rows,
    };
    mkdirSync('profile-output', { recursive: true });
    writeFileSync('profile-output/astra-normalization.json', JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } finally {
    gpu.dispose();
  }
}, 120_000);
