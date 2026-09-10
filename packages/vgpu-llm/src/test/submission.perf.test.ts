import { mkdirSync, writeFileSync } from 'node:fs';
import { it } from 'vitest';
import { init } from 'vgpu/node';
import { DecoderGpuRunner } from '../decoder/DecoderGpuRunner.js';
import { reference, referenceWeights, expectClose } from './reference.js';

const median = (values: number[]) => values.toSorted((left, right) => left - right)[Math.floor(values.length / 2)];

it('paired submission benchmark with identical FP32 token/chunk workloads', async () => {
  const gpu = await init();
  try {
    const before = new DecoderGpuRunner(gpu, referenceWeights(reference.cases[0].config), {
      maxTokens: 64,
      batchCompute: false,
      prefillChunkSize: 7,
    });
    const after = new DecoderGpuRunner(gpu, referenceWeights(reference.cases[0].config), {
      maxTokens: 64,
      batchCompute: true,
      prefillChunkSize: 7,
    });
    const tokens = reference.input_ids;
    const last = tokens.length - 1;
    for (const runner of [before, after]) {
      await runner.prefillTokens(tokens, 0, last);
      runner.computeToken(tokens[last], last);
    }
    expectClose(await after.readLogits(), await before.readLogits());
    const rows = [];
    for (const workload of ['decode', 'prefill']) {
      const measure = async (runner: DecoderGpuRunner) => {
        const start = performance.now();
        for (let i = 0; i < 8; i++) {
          if (workload === 'decode') runner.computeToken(tokens[last], last);
          else await runner.prefillTokens(tokens, 0, last);
        }
        await gpu.device.gpu.queue.onSubmittedWorkDone();
        return (performance.now() - start) / 8;
      };
      for (let i = 0; i < 3; i++) {
        await measure(before);
        await measure(after);
      }
      const baseline: number[] = [],
        candidate: number[] = [];
      for (let i = 0; i < 15; i++) {
        if (i % 2 === 0) {
          baseline.push(await measure(before));
          candidate.push(await measure(after));
        } else {
          candidate.push(await measure(after));
          baseline.push(await measure(before));
        }
      }
      rows.push({ workload, baseline, candidate, medianSpeedup: median(baseline) / median(candidate) });
    }
    const info = gpu.device.adapterInfo;
    const report = {
      adapter: info && { vendor: info.vendor, device: info.device, description: info.description },
      node: process.version,
      model: 'pinned synthetic Llama H32, 2 layers, FP32; not a production throughput benchmark',
      metric: 'completed wall ms per decode token or 18-token prefill (chunk size 7), 8 repetitions per sample',
      rows,
    };
    mkdirSync('profile-output', { recursive: true });
    writeFileSync('profile-output/astra-submission.json', JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } finally {
    gpu.dispose();
  }
}, 120_000);
