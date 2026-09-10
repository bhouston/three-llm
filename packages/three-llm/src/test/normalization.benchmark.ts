import { commands } from 'vitest/browser';
import { it, expect } from 'vitest';
import { WebGPURenderer } from 'three/webgpu';
import { TSLRMSNorm } from '../tsl/TSLRMSNorm.js';
import { TSLNormalize } from '../tsl/TSLNormalize.js';
import { TSLRMSNorm as BaselineRMS } from './baselines/TSLRMSNorm.js';
import { TSLNormalize as BaselineLayer } from './baselines/TSLNormalize.js';
import { storageFromArray } from './gpu.js';
import { expectClose } from './reference.js';
const median = (x: number[]) => x.toSorted((a, b) => a - b)[Math.floor(x.length / 2)];
it('paired normalization GPU timestamps on the selected adapter', async () => {
  const adapter = await navigator.gpu.requestAdapter();
  expect(adapter).not.toBeNull();
  const renderer = new WebGPURenderer({ trackTimestamp: true });
  await renderer.init();
  expect(renderer.hasFeature('timestamp-query')).toBe(true);
  const results = [];
  try {
    for (const size of [768, 2048, 4096])
      for (const kind of ['rms', 'layer']) {
        const input = storageFromArray(Float32Array.from({ length: size }, (_, i) => Math.sin(i * 0.17) * 0.4));
        const weight = Float32Array.from({ length: size }, (_, i) => 1 + Math.cos(i * 0.23) * 0.1);
        const bias = new Float32Array(size);
        const before =
          kind === 'rms'
            ? new BaselineRMS(input.node, weight, size)
            : new BaselineLayer(input.node, weight, bias, size);
        const after =
          kind === 'rms' ? new TSLRMSNorm(input.node, weight, size) : new TSLNormalize(input.node, weight, bias, size);
        renderer.compute([before.computeNode, after.computeNode]);
        expectClose(
          new Float32Array(await renderer.getArrayBufferAsync(after.outputAttribute)),
          new Float32Array(await renderer.getArrayBufferAsync(before.outputAttribute)),
          1e-5,
          1e-5,
        );
        await renderer.resolveTimestampsAsync('compute');
        const nodes = [Array(128).fill(before.computeNode), Array(128).fill(after.computeNode)];
        const samples: number[][] = [];
        for (let round = -5; round < 31; round++) {
          const pair = [0, 0];
          for (const index of round % 2 === 0 ? [0, 1] : [1, 0]) {
            renderer.compute(nodes[index]);
            pair[index] = (await renderer.resolveTimestampsAsync('compute'))! / 128;
          }
          if (round >= 0) samples.push(pair);
        }
        results.push({
          kind,
          size,
          samples,
          beforeMs: median(samples.map((x) => x[0])),
          afterMs: median(samples.map((x) => x[1])),
          speedup: median(samples.map((x) => x[0] / x[1])),
        });
      }
    await commands.saveBenchmark('normalization', {
      adapter: { vendor: adapter!.info.vendor, architecture: adapter!.info.architecture },
      userAgent: navigator.userAgent,
      dispatches: 128,
      warmups: 5,
      pairs: 31,
      results,
    });
  } finally {
    renderer.dispose();
  }
});
