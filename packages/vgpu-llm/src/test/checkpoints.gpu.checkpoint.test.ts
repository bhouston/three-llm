import { describe, expect, it } from 'vitest';
import type { Gpu } from 'vgpu';

import { DecoderCPURunner } from '../decoder/DecoderCPURunner.js';
import { DecoderGpuRunner } from '../decoder/DecoderGpuRunner.js';
import { DecoderWeights } from '../decoder/DecoderWeights.js';
import { QwenCPURunner } from '../qwen/QwenCPURunner.js';
import { QwenGpuRunner } from '../qwen/QwenGpuRunner.js';
import { QwenWeights } from '../qwen/QwenWeights.js';
import {
  catalogEntry,
  checkpointRoot,
  GREEDY,
  GREEDY_SHORT,
  loadLocalCheckpoint,
  STORY_PROMPT,
} from './checkpoints.js';
import { createGpu } from './gpu.js';

async function withGpu(skip: () => never, run: (gpu: Gpu) => Promise<void> | void) {
  const gpu = await createGpu(skip);
  try {
    await run(gpu);
  } finally {
    gpu.dispose();
  }
}

async function expectDecoderGpuMatchesCpu(
  skip: () => never,
  gpu: Gpu,
  id: string,
  maxTokens = 32,
  timeoutOptions = GREEDY,
) {
  const weights = await loadLocalCheckpoint(skip, DecoderWeights, checkpointRoot(catalogEntry(id), 'node'));
  const cpu = new DecoderCPURunner(weights, { maxTokens }).generate(STORY_PROMPT, timeoutOptions);
  const gpuResult = await new DecoderGpuRunner(gpu, weights, { maxTokens }).generate(STORY_PROMPT, timeoutOptions);

  expect(cpu.text.startsWith(STORY_PROMPT)).toBe(true);
  expect(gpuResult.text).toBe(cpu.text);
  expect(gpuResult.generatedTokens).toEqual(cpu.generatedTokens);
}

describe('checkpoint GPU tests (vgpu/node, real WGSL execution)', () => {
  it('TinyStories greedy continuation matches the CPU runner', async ({ skip }) => {
    await withGpu(skip, (gpu) => expectDecoderGpuMatchesCpu(skip, gpu, 'tinystories', 128));
  }, 180_000);

  it('GPT-2 greedy continuation matches the CPU runner', async ({ skip }) => {
    await withGpu(skip, (gpu) => expectDecoderGpuMatchesCpu(skip, gpu, 'gpt2', 32));
  }, 180_000);

  it('SmolLM2 greedy continuation matches the CPU runner', async ({ skip }) => {
    await withGpu(skip, (gpu) => expectDecoderGpuMatchesCpu(skip, gpu, 'smollm2', 32));
  }, 180_000);

  it('Phi-1.5 greedy continuation matches the CPU runner', async ({ skip }) => {
    await withGpu(skip, (gpu) => expectDecoderGpuMatchesCpu(skip, gpu, 'phi-1.5', 32));
  }, 300_000);

  it('Qwen3.5 0.8B greedy continuation matches the CPU runner', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const weights = await loadLocalCheckpoint(
        skip,
        QwenWeights,
        checkpointRoot(catalogEntry('qwen3.5-0.8b'), 'node'),
      );
      const cpu = new QwenCPURunner(weights, { maxTokens: 32 }).generate(STORY_PROMPT, GREEDY_SHORT);
      const gpuResult = await new QwenGpuRunner(gpu, weights, { maxTokens: 32 }).generate(STORY_PROMPT, GREEDY_SHORT);

      expect(cpu.text.startsWith(STORY_PROMPT)).toBe(true);
      expect(gpuResult.text).toBe(cpu.text);
      expect(gpuResult.generatedTokens).toEqual(cpu.generatedTokens);
    });
  }, 300_000);
});
