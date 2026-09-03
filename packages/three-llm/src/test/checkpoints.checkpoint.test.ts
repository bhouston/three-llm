import { describe, expect, it } from 'vitest';

import { DecoderCPURunner } from '../decoder/DecoderCPURunner.js';
import { DecoderWeights } from '../decoder/DecoderWeights.js';
import { architectureFor } from '../load/DecoderRecipe.js';
import { QwenCPURunner } from '../qwen/QwenCPURunner.js';
import { QwenWeights } from '../qwen/QwenWeights.js';
import {
  catalogEntry,
  checkpointRoot,
  GPT2_PARIS_GREEDY_TEXT,
  GPT2_STORY_GREEDY_TEXT,
  GREEDY,
  GREEDY_SHORT,
  loadLocalCheckpoint,
  PARIS_PROMPT,
  PHI15_GREEDY_TEXT,
  STORY_PROMPT,
} from './checkpoints.js';

describe('checkpoint CPU tests', () => {
  it('TinyStories greedy continuation stays on the prompt and reuses cache', async ({ skip }) => {
    const weights = await loadLocalCheckpoint(
      skip,
      DecoderWeights,
      checkpointRoot(catalogEntry('tinystories'), 'node'),
    );
    const prompt = STORY_PROMPT;
    const encoded = weights.tokenizer.encode(prompt);
    const prepared = weights.prepareGeneration(prompt, 8, 128);

    expect(prepared.inputTokens).toEqual(encoded);
    expect(prepared.newTokenBudget).toBe(8 - encoded.length);

    const short = new DecoderCPURunner(weights, { maxTokens: 8 }).generate(prompt, {
      maxNewTokens: 128,
      temperature: 0,
      topK: 1,
    });
    expect(short.text.startsWith(prompt)).toBe(true);
    expect(short.generatedTokens.length).toBe(8 - encoded.length);

    const runner = new DecoderCPURunner(weights, { maxTokens: 64 });
    const first = runner.generate(prompt, { maxNewTokens: 8, temperature: 0, topK: 1 });
    const continued = `${first.text} She`;
    const second = runner.generate(continued, { maxNewTokens: 8, temperature: 0, topK: 1 });
    const fresh = new DecoderCPURunner(weights, { maxTokens: 64 }).generate(continued, {
      maxNewTokens: 8,
      temperature: 0,
      topK: 1,
    });

    expect(second.text).toBe(fresh.text);
    expect(second.cachedPromptTokens).toBeGreaterThan(0);

    const edited = `${first.text} He`;
    const rewound = runner.generate(edited, { maxNewTokens: 8, temperature: 0, topK: 1 });
    const freshEdited = new DecoderCPURunner(weights, { maxTokens: 64 }).generate(edited, {
      maxNewTokens: 8,
      temperature: 0,
      topK: 1,
    });

    expect(rewound.text).toBe(freshEdited.text);
    expect(rewound.cachedPromptTokens).toBeGreaterThan(0);

    runner.resetCache();
    const afterReset = runner.generate(continued, { maxNewTokens: 8, temperature: 0, topK: 1 });
    expect(afterReset.cachedPromptTokens).toBe(0);
    expect(afterReset.text).toBe(fresh.text);

    const result = new DecoderCPURunner(weights).generate(prompt, {
      maxNewTokens: 24,
      temperature: 0,
      topK: 1,
    });
    expect(result.text).toBe(
      'Once upon a time, there was a little girl named Lily. She loved to play with her toys. One day, she saw a big,',
    );
  }, 180_000);

  it('GPT-2 greedy continuation stays on the prompt', async ({ skip }) => {
    const weights = await loadLocalCheckpoint(skip, DecoderWeights, checkpointRoot(catalogEntry('gpt2'), 'node'));
    const runner = new DecoderCPURunner(weights, { maxTokens: 32 });
    const story = runner.generate(STORY_PROMPT, GREEDY);
    const paris = runner.generate(PARIS_PROMPT, GREEDY);

    expect(architectureFor(weights.config)).toBe('gpt2');
    expect(weights.architecture).toBe('gpt2');
    expect(weights.hiddenSize).toBe(768);
    expect(weights.layerCount).toBe(12);
    expect(weights.headCount).toBe(12);
    expect(weights.vocabSize).toBe(50257);
    expect(weights.tokenizer.encode(STORY_PROMPT)).toEqual([7454, 2402, 257, 640, 11]);
    expect(weights.tokenizer.encode(PARIS_PROMPT)).toEqual([40313, 373, 4950, 287, 262, 2121, 13]);
    expect(story.text).toBe(GPT2_STORY_GREEDY_TEXT);
    expect(paris.text).toBe(GPT2_PARIS_GREEDY_TEXT);
  }, 120_000);

  it('SmolLM2 loads Llama-style weights and generates the expected greedy continuation', async ({ skip }) => {
    const weights = await loadLocalCheckpoint(skip, DecoderWeights, checkpointRoot(catalogEntry('smollm2'), 'node'));

    expect(architectureFor(weights.config)).toBe('llama');
    expect(weights.architecture).toBe('llama');
    expect(weights.hiddenSize).toBe(576);
    expect(weights.innerSize).toBe(1536);
    expect(weights.layerCount).toBe(30);
    expect(weights.headCount).toBe(9);
    expect(weights.kvHeadCount).toBe(3);
    expect(weights.headDim).toBe(64);
    expect(weights.qSize).toBe(576);
    expect(weights.kvSize).toBe(192);
    expect(weights.vocabSize).toBe(49152);
    expect(weights.ropeTheta).toBe(100000);
    expect(weights.mlpActivation).toBe('silu');
    expect(weights.endOfTextTokenId).toBe(0);

    const promptIds = weights.tokenizer.encode(STORY_PROMPT);
    expect(promptIds).toEqual([6403, 1980, 253, 655, 28]);
    expect(weights.tokenizer.decode(promptIds)).toBe(STORY_PROMPT);
    expect(weights.hasTensor('layers.0.self_attn.q_proj.weight')).toBe(true);

    const result = new DecoderCPURunner(weights, { maxTokens: 32 }).generate(STORY_PROMPT, GREEDY);
    expect(result.text.startsWith(STORY_PROMPT)).toBe(true);
    expect(result.generatedTokens.length).toBe(8);
    expect(result.text).toBe('Once upon a time, there was a little girl named Lily.');
  }, 180_000);

  it('Phi-1.5 loads Phi-style weights and generates the expected greedy continuation', async ({ skip }) => {
    const weights = await loadLocalCheckpoint(skip, DecoderWeights, checkpointRoot(catalogEntry('phi-1.5'), 'node'));

    expect(architectureFor(weights.config)).toBe('phi');
    expect(weights.architecture).toBe('phi');
    expect(weights.hiddenSize).toBe(2048);
    expect(weights.innerSize).toBe(8192);
    expect(weights.layerCount).toBe(24);
    expect(weights.headCount).toBe(32);
    expect(weights.kvHeadCount).toBe(32);
    expect(weights.headDim).toBe(64);
    expect(weights.qSize).toBe(2048);
    expect(weights.kvSize).toBe(2048);
    expect(weights.vocabSize).toBe(51200);
    expect(weights.ropeTheta).toBe(10000);
    expect(weights.rotaryDim).toBe(32);
    expect(weights.endOfTextTokenId).toBe(50256);

    const promptIds = weights.tokenizer.encode(STORY_PROMPT);
    expect(promptIds).toEqual([7454, 2402, 257, 640, 11]);
    expect(weights.tokenizer.decode(promptIds)).toBe(STORY_PROMPT);
    expect(weights.tensors['model.layers.0.self_attn.q_proj.weight']).toBeDefined();
    expect(weights.tensors['lm_head.weight']).toBeDefined();

    const result = new DecoderCPURunner(weights, { maxTokens: 32 }).generate(STORY_PROMPT, GREEDY);
    expect(result.text.startsWith(STORY_PROMPT)).toBe(true);
    expect(result.generatedTokens.length).toBe(8);
    expect(result.text).toBe(PHI15_GREEDY_TEXT);
  }, 300_000);

  it('Qwen3.5 0.8B loads hybrid weights and emits greedy tokens', async ({ skip }) => {
    const weights = await loadLocalCheckpoint(skip, QwenWeights, checkpointRoot(catalogEntry('qwen3.5-0.8b'), 'node'));

    expect(architectureFor(weights.config)).toBe('qwen3_5');
    expect(weights.architecture).toBe('qwen3_5');
    expect(weights.hiddenSize).toBe(1024);
    expect(weights.innerSize).toBe(3584);
    expect(weights.layerCount).toBe(24);
    expect(weights.headCount).toBe(8);
    expect(weights.kvHeadCount).toBe(2);
    expect(weights.headDim).toBe(256);
    expect(weights.qSize).toBe(2048);
    expect(weights.kvSize).toBe(512);
    expect(weights.vocabSize).toBe(248320);
    expect(weights.linearKeyHeads).toBe(16);
    expect(weights.linearValueHeads).toBe(16);
    expect(weights.linearKeyDim).toBe(128);
    expect(weights.rotaryDim).toBe(64);
    expect(weights.mlpActivation).toBe('silu');
    expect(weights.block(0).layerType).toBe('linear_attention');
    expect(weights.block(3).layerType).toBe('full_attention');
    expect(weights.hasTensor('layers.0.linear_attn.in_proj_qkv.weight')).toBe(true);
    expect(weights.hasTensor('layers.3.self_attn.q_norm.weight')).toBe(true);

    const promptIds = weights.tokenizer.encode(STORY_PROMPT);
    expect(promptIds).toEqual([12162, 5028, 264, 854, 11]);
    expect(weights.tokenizer.decode(promptIds)).toBe(STORY_PROMPT);

    const chat = weights.formatChat([{ role: 'user', text: 'Hi' }]);
    const chatIds = weights.tokenizer.encode(chat);
    expect(chatIds[0]).toBe(248045);
    expect(chatIds).toContain(248068);
    expect(chatIds).toContain(248069);
    expect(weights.tokenizer.decode(chatIds)).toBe(chat);
    expect(weights.stopTokenIds).toEqual([248044, 248046]);

    const result = new QwenCPURunner(weights, { maxTokens: 32 }).generate(STORY_PROMPT, GREEDY_SHORT);
    expect(result.text.startsWith(STORY_PROMPT)).toBe(true);
    expect(result.generatedTokens.length).toBeGreaterThan(0);
  }, 300_000);
});
