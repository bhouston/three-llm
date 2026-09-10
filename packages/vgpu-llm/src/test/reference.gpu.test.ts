import { it } from 'vitest';
import { DecoderGpuRunner } from '../decoder/DecoderGpuRunner.js';
import { readFloat32, uploadStorage, writeBuffer } from '../gpu/device.js';
import { AttentionKernel } from '../kernels/AttentionKernel.js';
import { createGpu, createGpuWithF16 } from './gpu.js';
import { reference, referenceWeights, expectClose } from './reference.js';

for (const fixture of reference.cases) {
  for (const precision of ['fp32', 'fp16'] as const) {
    it(`${fixture.name}: ${precision} GPU logits match pinned Transformers, tokenwise and chunked`, async ({
      skip,
    }) => {
      const gpu = await (precision === 'fp16' ? createGpuWithF16 : createGpu)(skip);
      try {
        const runner = new DecoderGpuRunner(gpu, referenceWeights(fixture.config), {
          maxTokens: 64,
          precision,
          prefillChunkSize: 7,
          logitChunkSize: 7,
        });
        const atol = precision === 'fp16' ? 2e-4 : 3e-5;
        const logits = precision === 'fp16' ? fixture.fp16_storage_logits : fixture.logits;
        const hidden = precision === 'fp16' ? fixture.fp16_storage_final_hidden : fixture.final_hidden;
        for (let i = 0; i < reference.input_ids.length; i++) {
          runner.computeToken(reference.input_ids[i], i);
          expectClose(await runner.readLogits(), logits[i], atol, 2e-4);
          expectClose(await readFloat32(runner.finalNorm.outputBuffer), hidden[i], atol, 2e-4);
        }
        runner.resetCache();
        const last = reference.input_ids.length - 1;
        await runner.prefillTokens(reference.input_ids, 0, last);
        runner.computeToken(reference.input_ids[last], last);
        expectClose(await runner.readLogits(), logits[last], atol, 2e-4);
      } finally {
        gpu.dispose();
      }
    });
  }
  it(`${fixture.name}: GPU rotated Q/K match independent reference around the context boundary`, async ({ skip }) => {
    const gpu = await createGpu(skip);
    try {
      const input = uploadStorage(gpu, new Float32Array(24));
      const kernel = new AttentionKernel(gpu, input, 8, 1, 64, {
        ropeTheta: 10000,
        ropeScaling: referenceWeights(fixture.config).recipe.ropeScaling,
      });
      for (let i = 0; i < fixture.rope.positions.length; i++) {
        const position = fixture.rope.positions[i];
        writeBuffer(input, new Float32Array([...fixture.rope.query[i], ...fixture.rope.key[i], ...Array(8).fill(0)]));
        kernel.run(position);
        expectClose(await readFloat32(kernel.queryBuffer), fixture.rope.rotated_query[i]);
        expectClose(
          (await readFloat32(kernel.keyCacheBuffer)).subarray(position * 8, (position + 1) * 8),
          fixture.rope.rotated_key[i],
        );
      }
    } finally {
      gpu.dispose();
    }
  });
}
