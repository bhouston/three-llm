import { expect, it } from 'vitest';
import { recipeFor } from '../load/DecoderRecipe.js';
import { DecoderCPURunner } from '../decoder/DecoderCPURunner.js';
import { applyRoPE } from './math.js';
import { ropeParameters } from './rope.js';
import { reference, referenceWeights, expectClose } from '../test/reference.js';

for (const fixture of reference.cases) {
  it(`${fixture.name}: CPU logits and RoPE match pinned Transformers`, () => {
    const weights = referenceWeights(fixture.config);
    const recipe = weights.recipe;
    const params = ropeParameters(8, 10000, recipe.ropeScaling);
    expectClose(params.invFreq, fixture.rope.inv_freq, 1e-8, 1e-6);
    expect(params.attentionFactor).toBeCloseTo(fixture.rope.attention_factor, 6);
    for (let i = 0; i < fixture.rope.positions.length; i++) {
      const options = { ropeScaling: recipe.ropeScaling };
      expectClose(
        applyRoPE(new Float32Array(fixture.rope.query[i]), 0, 8, fixture.rope.positions[i], 10000, options),
        fixture.rope.rotated_query[i],
      );
      expectClose(
        applyRoPE(new Float32Array(fixture.rope.key[i]), 0, 8, fixture.rope.positions[i], 10000, options),
        fixture.rope.rotated_key[i],
      );
    }
    const runner = new DecoderCPURunner(weights, { maxTokens: 64 });
    for (let i = 0; i < reference.input_ids.length; i++)
      expectClose(runner.forwardToken(reference.input_ids[i], i), fixture.logits[i]);
  });
}
it.each(['dynamic', 'longrope', 'unrecognized'])(
  'rejects unsupported %s RoPE instead of silently ignoring it',
  (rope_type) => {
    expect(() => recipeFor({ ...reference.cases[0].config, rope_scaling: { rope_type, factor: 4 } })).toThrow(
      /Unsupported RoPE/,
    );
  },
);
it('preserves explicit zero rotation and rejects malformed dimensions/scaling', () => {
  const base = reference.cases[0].config;
  expect(recipeFor({ ...base, rotary_dim: 0 }).rotaryDim).toBe(0);
  const value = new Float32Array([1, 2, 3, 4]);
  expect(applyRoPE(value, 0, 0, 8, 10000)).toEqual(value);
  for (const rotary_dim of [-2, 3, 10]) expect(() => recipeFor({ ...base, rotary_dim })).toThrow(/rotary dimension/);
  expect(() => recipeFor({ ...base, rope_scaling: { rope_type: 'yarn', factor: NaN } })).toThrow(/factor/);
  expect(() => recipeFor({ ...base, rope_scaling: { rope_type: 'llama3', factor: 4 } })).toThrow(/frequency range/);
});

it('pins the independent fixture and generator hashes', async () => {
  const { createHash } = await import('node:crypto');
  const { readFileSync } = await import('node:fs');
  const digest = (bytes: Buffer) => createHash('sha256').update(bytes).digest('hex');
  const fixtureURL = new URL('../test/fixtures/transformers-llama.json', import.meta.url);
  expect(digest(readFileSync(fixtureURL))).toBe(
    readFileSync(new URL('../test/fixtures/transformers-llama.sha256', import.meta.url), 'utf8').trim(),
  );
  expect(digest(readFileSync(new URL('../../../../scripts/reference/generate.py', import.meta.url)))).toBe(
    reference.metadata.generator_sha256,
  );
  expect(reference.metadata.transformers_commit).toBe('5f4ecf2d9f867a1255131d2461d75793c0cf1db2');
});
