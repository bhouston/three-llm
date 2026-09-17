import { it, expect } from 'vitest';
import { DecoderTSLRunner } from '../decoder/DecoderTSLRunner.js';
import { TSLAttention } from '../tsl/TSLAttention.js';
import { createRenderer, storageFromArray } from './gpu.js';
import { reference, referenceWeights, expectClose } from './reference.js';
for (const fixture of reference.cases) {
  it(`${fixture.name}: TSL logits match pinned Transformers tokenwise and chunked`, async () => {
    const renderer = await createRenderer(() => {
      throw new Error('WebGPU required for reference validation');
    });
    // SwiftShader's transcendental approximations accumulate across decoder layers.
    // Measured maxima over every fixture/token: logits 1.80e-4, hidden 7.52e-4.
    // Keep hardware tolerances unchanged and check greedy decisions on both adapters.
    const adapter = await navigator.gpu.requestAdapter();
    const software = adapter?.info.architecture === 'swiftshader';
    const logitAtol = software ? 2e-4 : 3e-5;
    const hiddenAtol = software ? 1e-3 : 3e-5;
    try {
      const runner = new DecoderTSLRunner(referenceWeights(fixture.config), {
        maxTokens: 64,
        prefillChunkSize: 7,
        logitChunkSize: 7,
      });
      for (let i = 0; i < reference.input_ids.length; i++) {
        runner.computeToken(renderer, reference.input_ids[i], i);
        const logits = await runner.readLogits(renderer);
        expectClose(logits, fixture.logits[i], logitAtol, 2e-4);
        expect(argmax(logits)).toBe(argmax(fixture.logits[i]));
        expectClose(
          new Float32Array(await renderer.getArrayBufferAsync(runner.finalNorm.outputAttribute)),
          fixture.final_hidden[i],
          hiddenAtol,
          2e-4,
        );
      }
      runner.resetCache();
      const last = reference.input_ids.length - 1;
      await runner.prefillTokens(renderer, reference.input_ids, 0, last);
      runner.computeToken(renderer, reference.input_ids[last], last);
      expectClose(await runner.readLogits(renderer), fixture.logits[last], logitAtol, 2e-4);
    } finally {
      renderer.dispose();
    }
  });
  it(`${fixture.name}: TSL rotated Q/K match pinned reference around context boundary`, async () => {
    const renderer = await createRenderer(() => {
      throw new Error('WebGPU required');
    });
    try {
      const values = new Float32Array(24);
      const input = storageFromArray(values);
      const layer = new TSLAttention(input.node, 8, 1, 64, {
        ropeTheta: 10000,
        ropeScaling: referenceWeights(fixture.config).recipe.ropeScaling,
      });
      for (let i = 0; i < fixture.rope.positions.length; i++) {
        const pos = fixture.rope.positions[i];
        values.set(fixture.rope.query[i]);
        values.set(fixture.rope.key[i], 8);
        input.attribute.needsUpdate = true;
        layer.compute(renderer, pos);
        expectClose(
          new Float32Array(await renderer.getArrayBufferAsync(layer.queryAttribute)),
          fixture.rope.rotated_query[i],
        );
        expectClose(
          new Float32Array(await renderer.getArrayBufferAsync(layer.keyCacheAttribute)).subarray(
            pos * 8,
            (pos + 1) * 8,
          ),
          fixture.rope.rotated_key[i],
        );
      }
      expect(renderer.backend.isWebGPUBackend).toBe(true);
    } finally {
      renderer.dispose();
    }
  });
}

function argmax(values: ArrayLike<number>) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}
