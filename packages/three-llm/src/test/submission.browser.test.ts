import { it, expect } from 'vitest';
import { TSLGatedDeltaNet } from '../tsl/TSLGatedDeltaNet.js';
import { TSLAttention } from '../tsl/TSLAttention.js';
import { DecoderTSLRunner } from '../decoder/DecoderTSLRunner.js';
import { createTinyQwenWeights } from './helpers.js';
import { createRenderer, storageFromArray, readOutput } from './gpu.js';
import { expectClose, reference, referenceWeights } from './reference.js';
it('standalone recurrent batching preserves state and submission counts', async () => {
  const renderer = await createRenderer(() => {
    throw new Error('WebGPU required');
  });
  const weights = createTinyQwenWeights();
  const values = new Float32Array(weights.hiddenSize);
  const input = storageFromArray(values);
  const options = {
    hiddenSize: weights.hiddenSize,
    numKHeads: weights.linearKeyHeads,
    numVHeads: weights.linearValueHeads,
    keyDim: weights.linearKeyDim,
    valueDim: weights.linearValueDim,
    kernelSize: weights.linearConvKernel,
  };
  const before = new TSLGatedDeltaNet(input.node, weights.block(0).delta!, options);
  const after = new TSLGatedDeltaNet(input.node, weights.block(0).delta!, options);
  for (let step = 0; step < 8; step++) {
    for (let i = 0; i < values.length; i++) values[i] = Math.sin(step * 0.37 + i * 0.13) * 0.2;
    input.attribute.needsUpdate = true;
    for (const node of before.computeNodes) renderer.compute(node);
    after.compute(renderer);
    expectClose(await readOutput(renderer, after.outProj), await readOutput(renderer, before.outProj));
  }
  const attention = new TSLAttention(storageFromArray(new Float32Array(24)).node, 8, 1, 64);
  const runner = new DecoderTSLRunner(referenceWeights(reference.cases[0].config), {
    maxTokens: 64,
    prefillChunkSize: 7,
  });
  runner.computeToken(renderer, 1, 0);
  attention.compute(renderer, 0);
  const queue = renderer.backend.device.queue;
  const submit = queue.submit.bind(queue);
  let count = 0;
  queue.submit = (buffers: GPUCommandBuffer[]) => {
    count++;
    submit(buffers);
  };
  try {
    after.compute(renderer);
    expect(count).toBe(1);
    count = 0;
    attention.compute(renderer, 1);
    expect(count).toBe(1);
    count = 0;
    runner.computeToken(renderer, 5, 1);
    expect(count).toBe(1);
    count = 0;
    await runner.prefillTokens(renderer, reference.input_ids, 0, 18);
    expect(count).toBe(3);
  } finally {
    queue.submit = submit;
  }
});
