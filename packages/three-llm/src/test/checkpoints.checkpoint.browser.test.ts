import { describe, expect, it } from 'vitest';
import type { WebGPURenderer } from 'three/webgpu';

import { DecoderCPURunner } from '../decoder/DecoderCPURunner.js';
import { DecoderTSLRunner } from '../decoder/DecoderTSLRunner.js';
import { DecoderWeights } from '../decoder/DecoderWeights.js';
import { QwenCPURunner } from '../qwen/QwenCPURunner.js';
import { QwenTSLRunner } from '../qwen/QwenTSLRunner.js';
import { QwenWeights } from '../qwen/QwenWeights.js';
import {
  catalogEntry,
  checkpointRoot,
  GREEDY,
  GREEDY_SHORT,
  loadLocalCheckpoint,
  STORY_PROMPT,
} from './checkpoints.js';
import { createRenderer } from './gpu.js';

async function withRenderer(skip: () => never, run: (renderer: WebGPURenderer) => Promise<void> | void) {
  const renderer = await createRenderer(skip);
  try {
    await run(renderer);
  } finally {
    renderer.dispose();
  }
}

async function expectDecoderGpuMatchesCpu(
  skip: () => never,
  renderer: WebGPURenderer,
  id: string,
  maxTokens = 32,
  timeoutOptions = GREEDY,
) {
  const weights = await loadLocalCheckpoint(skip, DecoderWeights, checkpointRoot(catalogEntry(id), 'browser'));
  const cpu = new DecoderCPURunner(weights, { maxTokens }).generate(STORY_PROMPT, timeoutOptions);
  const gpu = await new DecoderTSLRunner(weights, { maxTokens }).generate(renderer, STORY_PROMPT, timeoutOptions);

  expect(cpu.text.startsWith(STORY_PROMPT)).toBe(true);
  expect(gpu.text).toBe(cpu.text);
  expect(gpu.generatedTokens).toEqual(cpu.generatedTokens);
}

describe('checkpoint browser GPU tests', () => {
  it('TSL TinyStories greedy continuation matches the CPU runner', async ({ skip }) => {
    await withRenderer(skip, (renderer) => expectDecoderGpuMatchesCpu(skip, renderer, 'tinystories', 128));
  }, 180_000);

  it('TSL GPT-2 greedy continuation matches the CPU runner', async ({ skip }) => {
    await withRenderer(skip, (renderer) => expectDecoderGpuMatchesCpu(skip, renderer, 'gpt2', 32));
  }, 180_000);

  it('TSL SmolLM2 greedy continuation matches the CPU runner', async ({ skip }) => {
    await withRenderer(skip, (renderer) => expectDecoderGpuMatchesCpu(skip, renderer, 'smollm2', 32));
  }, 180_000);

  it('TSL Phi-1.5 greedy continuation matches the CPU runner', async ({ skip }) => {
    await withRenderer(skip, (renderer) => expectDecoderGpuMatchesCpu(skip, renderer, 'phi-1.5', 32));
  }, 300_000);

  it('TSL Qwen3.5 0.8B greedy continuation matches the CPU runner', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const weights = await loadLocalCheckpoint(
        skip,
        QwenWeights,
        checkpointRoot(catalogEntry('qwen3.5-0.8b'), 'browser'),
      );
      const cpu = new QwenCPURunner(weights, { maxTokens: 32 }).generate(STORY_PROMPT, GREEDY_SHORT);
      const gpu = await new QwenTSLRunner(weights, { maxTokens: 32 }).generate(renderer, STORY_PROMPT, GREEDY_SHORT);

      expect(cpu.text.startsWith(STORY_PROMPT)).toBe(true);
      expect(gpu.text).toBe(cpu.text);
      expect(gpu.generatedTokens).toEqual(cpu.generatedTokens);
    });
  }, 300_000);
});
