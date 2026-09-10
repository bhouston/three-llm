import { expect, it, vi } from 'vitest';
import { DecoderGpuRunner } from '../decoder/DecoderGpuRunner.js';
import { DecoderCPURunner } from '../decoder/DecoderCPURunner.js';
import { QwenGpuRunner } from '../qwen/QwenGpuRunner.js';
import { QwenCPURunner } from '../qwen/QwenCPURunner.js';
import { createTinyLlama, createTinyQwenWeights } from './helpers.js';
import { createGpu } from './gpu.js';

for (const architecture of ['llama', 'qwen'] as const) {
  for (const topK of [1, 3]) {
    it(`${architecture} top-${topK} selects GPU candidates once per sample`, async ({ skip }) => {
      const gpu = await createGpu(skip);
      try {
        const options = { maxNewTokens: 3, temperature: topK === 1 ? 0 : 0.8, topK, random: () => 0.37 };
        const cpu =
          architecture === 'llama'
            ? new DecoderCPURunner(createTinyLlama(), { maxTokens: 8 })
            : new QwenCPURunner(createTinyQwenWeights(), { maxTokens: 8 });
        const expected = cpu.generate('hello', options);
        const runner =
          architecture === 'llama'
            ? new DecoderGpuRunner(gpu, createTinyLlama(), { maxTokens: 8 })
            : new QwenGpuRunner(gpu, createTinyQwenWeights(), { maxTokens: 8 });
        const run = vi.spyOn(runner.logitSampler, 'run');
        const sample = vi.spyOn(runner.logitSampler, 'sampleToken');
        const result = await runner.generate('hello', options);
        expect(result.generatedTokens).toEqual(expected.generatedTokens);
        expect(sample.mock.calls.length).toBeGreaterThan(0);
        expect(run.mock.calls.length).toBe(sample.mock.calls.length);
      } finally {
        gpu.dispose();
      }
    });
  }
}
