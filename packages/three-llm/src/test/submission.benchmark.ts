import { TSLGatedDeltaNet } from '../tsl/TSLGatedDeltaNet.js';
import { createTinyQwenWeights } from './helpers.js';
import { commands } from 'vitest/browser';
import { it, expect } from 'vitest';
import { TSLAttention } from '../tsl/TSLAttention.js';
import { createRenderer, storageFromArray, readOutput } from './gpu.js';
import { expectClose, reference, referenceWeights } from './reference.js';
import { DecoderTSLRunner } from '../decoder/DecoderTSLRunner.js';
const median = (x: number[]) => x.toSorted((a, b) => a - b)[Math.floor(x.length / 2)];
it('standalone attention and recurrence: separate versus batched submissions', async () => {
  const renderer = await createRenderer(() => {
    throw new Error('WebGPU required');
  });
  const queue = renderer.backend.device.queue;
  const adapter = await navigator.gpu.requestAdapter();
  try {
    const input = storageFromArray(Float32Array.from({ length: 96 }, (_, i) => Math.sin(i * 0.31) * 0.1));
    const before = new TSLAttention(input.node, 32, 4, 64, { ropeTheta: 10000 });
    const after = new TSLAttention(input.node, 32, 4, 64, { ropeTheta: 10000 });
    const runBefore = () => {
      before.setPosition(0);
      for (const node of before.computeNodes) renderer.compute(node);
    };
    const runAfter = () => after.compute(renderer, 0);
    runBefore();
    runAfter();
    expectClose(await readOutput(renderer, after), await readOutput(renderer, before));
    const samples: number[][] = [];
    const run = [runBefore, runAfter];
    for (let round = -3; round < 25; round++) {
      const pair = [0, 0];
      for (const index of round % 2 === 0 ? [0, 1] : [1, 0]) {
        await queue.onSubmittedWorkDone();
        const start = performance.now();
        for (let i = 0; i < 64; i++) run[index]();
        await queue.onSubmittedWorkDone();
        pair[index] = (performance.now() - start) / 64;
      }
      if (round >= 0) samples.push(pair);
    }
    const weights = createTinyQwenWeights();
    const deltaInput = storageFromArray(new Float32Array(weights.hiddenSize).fill(0.1));
    const options = {
      hiddenSize: weights.hiddenSize,
      numKHeads: weights.linearKeyHeads,
      numVHeads: weights.linearValueHeads,
      keyDim: weights.linearKeyDim,
      valueDim: weights.linearValueDim,
      kernelSize: weights.linearConvKernel,
    };
    const deltaBefore = new TSLGatedDeltaNet(deltaInput.node, weights.block(0).delta!, options);
    const deltaAfter = new TSLGatedDeltaNet(deltaInput.node, weights.block(0).delta!, options);
    const deltaRun = [
      () => {
        for (const node of deltaBefore.computeNodes) renderer.compute(node);
      },
      () => deltaAfter.compute(renderer),
    ];
    for (let i = 0; i < 8; i++) {
      deltaRun[0]();
      deltaRun[1]();
      expectClose(await readOutput(renderer, deltaAfter.outProj), await readOutput(renderer, deltaBefore.outProj));
    }
    const deltaSamples: number[][] = [];
    for (let round = -3; round < 25; round++) {
      const pair = [0, 0];
      for (const index of round % 2 === 0 ? [0, 1] : [1, 0]) {
        await queue.onSubmittedWorkDone();
        const start = performance.now();
        for (let i = 0; i < 64; i++) deltaRun[index]();
        await queue.onSubmittedWorkDone();
        pair[index] = (performance.now() - start) / 64;
      }
      if (round >= 0) deltaSamples.push(pair);
    }
    const runner = new DecoderTSLRunner(referenceWeights(reference.cases[0].config), {
      maxTokens: 64,
      prefillChunkSize: 7,
    });
    runner.computeToken(renderer, 1, 0);
    await queue.onSubmittedWorkDone();
    let submissions = 0;
    const submit = queue.submit.bind(queue);
    queue.submit = (buffers: GPUCommandBuffer[]) => {
      submissions++;
      submit(buffers);
    };
    let standaloneBefore = 0,
      standaloneAfter = 0,
      decode = 0,
      prefill = 0,
      deltaSeparate = 0,
      deltaBatched = 0;
    try {
      deltaRun[0]();
      deltaSeparate = submissions;
      submissions = 0;
      deltaRun[1]();
      deltaBatched = submissions;
      submissions = 0;
      runBefore();
      standaloneBefore = submissions;
      submissions = 0;
      runAfter();
      standaloneAfter = submissions;
      submissions = 0;
      runner.computeToken(renderer, 5, 1);
      decode = submissions;
      submissions = 0;
      await runner.prefillTokens(renderer, reference.input_ids, 0, 18);
      prefill = submissions;
    } finally {
      queue.submit = submit;
    }
    expect(deltaSeparate).toBe(11);
    expect(deltaBatched).toBe(1);
    expect(standaloneBefore).toBe(4);
    expect(standaloneAfter).toBe(1);
    expect(decode).toBe(1);
    expect(prefill).toBe(3);
    await commands.saveBenchmark('submission', {
      adapter: { vendor: adapter?.info.vendor, architecture: adapter?.info.architecture },
      userAgent: navigator.userAgent,
      repetitions: 64,
      warmups: 3,
      pairs: 25,
      samples,
      beforeMs: median(samples.map((x) => x[0])),
      afterMs: median(samples.map((x) => x[1])),
      speedup: median(samples.map((x) => x[0] / x[1])),
      delta: {
        samples: deltaSamples,
        beforeMs: median(deltaSamples.map((x) => x[0])),
        afterMs: median(deltaSamples.map((x) => x[1])),
        speedup: median(deltaSamples.map((x) => x[0] / x[1])),
      },
      submissions: { standaloneBefore, standaloneAfter, decode, prefill, deltaSeparate, deltaBatched },
    });
  } finally {
    renderer.dispose();
  }
});
