import { readFileSync } from 'node:fs';
import { DecoderWeights } from '../decoder/DecoderWeights.js';
import type { HuggingFaceConfig, TensorMap } from '../types.js';
import { expect } from 'vitest';

interface ReferenceCase {
  name: string;
  config: HuggingFaceConfig;
  logits: number[][];
  final_hidden: number[][];
  fp16_storage_logits: number[][];
  fp16_storage_final_hidden: number[][];
  rope: {
    positions: number[];
    inv_freq: number[];
    attention_factor: number;
    query: number[][];
    key: number[][];
    rotated_query: number[][];
    rotated_key: number[][];
  };
}
export const reference = JSON.parse(
  readFileSync(new URL('./fixtures/transformers-llama.json', import.meta.url), 'utf8'),
) as {
  metadata: Record<string, string>;
  input_ids: number[];
  cases: ReferenceCase[];
  tensors: Record<string, { shape: number[]; values: number[] }>;
};
export function referenceWeights(config: HuggingFaceConfig) {
  const tensors: TensorMap = {};
  for (const [name, tensor] of Object.entries(reference.tensors))
    tensors[name] = { name, shape: tensor.shape, dtype: 'F32', data: new Float32Array(tensor.values) };
  return new DecoderWeights(config, tensors, {
    endOfTextTokenId: 0,
    encode: () => reference.input_ids.slice(),
    decode: (ids) => ids.join(','),
  });
}
export function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, atol = 2e-5, rtol = 2e-4) {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Number.isFinite(actual[i]), `element ${i} is finite`).toBe(true);
    expect(Math.abs(actual[i] - expected[i]), `element ${i}: ${actual[i]} versus ${expected[i]}`).toBeLessThanOrEqual(
      atol + rtol * Math.abs(expected[i]),
    );
  }
}
