import { expect, it } from 'vitest';
import { generateAsync } from './generate.js';

it('aborted prefill does not advertise uncomputed tokens as cached', async () => {
  const signal = AbortSignal.abort();
  const runner = {
    maxTokens: 8,
    weights: {
      endOfTextTokenId: 0,
      tokenizer: { encode: () => [], decode: (ids: number[]) => ids.join(',') },
      prepareGeneration: () => ({ inputTokens: [1, 2], newTokenBudget: 1 }),
    },
    _cacheTokens: [] as number[],
    _cacheLogits: null as Float32Array | null,
  };
  let calls = 0;
  await generateAsync(
    runner,
    '',
    { signal, prefillMode: false },
    {
      rewindable: true,
      resetCache() {},
      computeToken() {
        calls++;
      },
      readLogits: async () => null,
    },
  );
  expect(calls).toBe(0);
  expect(runner._cacheTokens).toEqual([]);
  expect(runner._cacheLogits).toBeNull();
});

it.each([false, true])('zero-budget append invalidates old logits (batched prefill=%s)', async (prefillMode) => {
  const oldLogits = new Float32Array([0, 9, 1]);
  const runner = {
    maxTokens: 8,
    weights: {
      endOfTextTokenId: 0,
      tokenizer: { encode: () => [], decode: (ids: number[]) => ids.join(',') },
      prepareGeneration: () => ({ inputTokens: [1, 2], newTokenBudget: 0 }),
    },
    _cacheTokens: [1],
    _cacheLogits: oldLogits as Float32Array | null,
  };
  await generateAsync(
    runner,
    '',
    { prefillMode },
    {
      rewindable: true,
      resetCache() {},
      computeToken() {},
      prefillTokens: async () => {},
      readLogits: async () => null,
    },
  );
  expect(runner._cacheTokens).toEqual([1, 2]);
  expect(runner._cacheLogits).toBeNull();
});
