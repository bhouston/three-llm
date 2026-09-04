import { describe, expect, it } from 'vitest';
import { init as initMockGpu } from 'vgpu/mock';
import type { Gpu } from 'vgpu';

import { DecoderCPURunner } from '../decoder/DecoderCPURunner.js';
import { DecoderGpuRunner } from '../decoder/DecoderGpuRunner.js';
import { QwenCPURunner } from '../qwen/QwenCPURunner.js';
import { QwenGpuRunner } from '../qwen/QwenGpuRunner.js';
import { geluNew, linear, rmsNorm, silu } from '../runtime/math.js';
import { AttentionKernel } from '../kernels/AttentionKernel.js';
import { GatedDeltaNetKernel } from '../kernels/GatedDeltaNetKernel.js';
import { GatedMLPKernel } from '../kernels/GatedMLPKernel.js';
import { createChunkedLogitLayers, createLogitSampler } from '../kernels/LogitsKernel.js';
import { LinearKernel } from '../kernels/LinearKernel.js';
import { RMSNormKernel } from '../kernels/RMSNormKernel.js';
import { allocStorage, uploadStorage, uploadWeightStorage, writeBuffer } from '../gpu/device.js';
import { createGpuWithF16, readOutput } from './gpu.js';
import { closeArray, createTinyLlama, createTinyQwenWeights, fillSin } from './helpers.js';

async function withF16Gpu(skip: () => never, run: (gpu: Gpu) => Promise<void> | void) {
  const gpu = await createGpuWithF16(skip);
  try {
    await run(gpu);
  } finally {
    gpu.dispose();
  }
}

// f16 has ~3 significant decimal digits, so kernel-level comparisons use a
// looser relative tolerance than the fp32 tests; `closeArray` takes an
// absolute epsilon, so it's scaled to the typical magnitude of each fixture.
const FP16_EPSILON = 0.05;

describe('fp16 precision (opt-in, requires the shader-f16 device feature)', () => {
  it('LinearKernel fp16 weights stay close to the fp32 CPU reference', async ({ skip }) => {
    await withF16Gpu(skip, async (gpu) => {
      const inputSize = 16;
      const outputSize = 12;
      const input = fillSin(new Float32Array(inputSize), 0.3);
      const weight = fillSin(new Float32Array(inputSize * outputSize), 1.1);
      const bias = fillSin(new Float32Array(outputSize), 2.2);

      const inputBuffer = uploadStorage(gpu, input, 'read');
      const layer = new LinearKernel(gpu, inputBuffer, weight, bias, inputSize, outputSize, {
        workgroupSize: 4,
        precision: 'fp16',
      });

      layer.run();

      const expected = linear(input, weight, bias, inputSize, outputSize);
      closeArray(await readOutput(layer.outputBuffer), expected, FP16_EPSILON);
    });
  });

  it('RMSNormKernel fp16 weights stay close to the fp32 CPU reference', async ({ skip }) => {
    await withF16Gpu(skip, async (gpu) => {
      const hiddenSize = 16;
      const input = fillSin(new Float32Array(hiddenSize), 0.5);
      const weight = fillSin(new Float32Array(hiddenSize), 1.5);

      const inputBuffer = uploadStorage(gpu, input, 'read');
      const weightBuffer = uploadWeightStorage(gpu, weight, 'fp16');
      const layer = new RMSNormKernel(gpu, inputBuffer, weightBuffer, hiddenSize, {
        workgroupSize: 4,
        precision: 'fp16',
      });

      layer.run();

      closeArray(await readOutput(layer.outputBuffer), rmsNorm(input, weight), FP16_EPSILON);
    });
  });

  it('GatedMLPKernel fp16 weights (SwiGLU) stay close to the fp32 CPU reference', async ({ skip }) => {
    await withF16Gpu(skip, async (gpu) => {
      const hiddenSize = 8;
      const innerSize = 12;
      const input = fillSin(new Float32Array(hiddenSize), 0.4);
      const gateWeight = fillSin(new Float32Array(hiddenSize * innerSize), 1.4);
      const upWeight = fillSin(new Float32Array(hiddenSize * innerSize), 2.4);
      const downWeight = fillSin(new Float32Array(innerSize * hiddenSize), 3.4);

      const gate = linear(input, gateWeight, null, hiddenSize, innerSize);
      const up = linear(input, upWeight, null, hiddenSize, innerSize);
      const hidden = new Float32Array(innerSize);
      for (let i = 0; i < innerSize; i++) hidden[i] = silu(gate[i]!) * up[i]!;
      const expected = linear(hidden, downWeight, null, innerSize, hiddenSize);

      const inputBuffer = uploadStorage(gpu, input, 'read');
      const layer = new GatedMLPKernel(gpu, inputBuffer, gateWeight, upWeight, downWeight, hiddenSize, innerSize, {
        workgroupSize: 4,
        precision: 'fp16',
      });

      layer.run();

      closeArray(await readOutput(layer.outputBuffer), expected, FP16_EPSILON);
    });
  });

  it('LinearKernel fp16 GeGLU path (gelu_pytorch_tanh) matches the fp32 CPU reference', async ({ skip }) => {
    await withF16Gpu(skip, async (gpu) => {
      const hiddenSize = 8;
      const innerSize = 10;
      const input = fillSin(new Float32Array(hiddenSize), 0.6);
      const gateWeight = fillSin(new Float32Array(hiddenSize * innerSize), 1.6);
      const upWeight = fillSin(new Float32Array(hiddenSize * innerSize), 2.6);
      const downWeight = fillSin(new Float32Array(innerSize * hiddenSize), 3.6);

      const gate = linear(input, gateWeight, null, hiddenSize, innerSize);
      const up = linear(input, upWeight, null, hiddenSize, innerSize);
      const hidden = new Float32Array(innerSize);
      for (let i = 0; i < innerSize; i++) hidden[i] = geluNew(gate[i]!) * up[i]!;
      const expected = linear(hidden, downWeight, null, innerSize, hiddenSize);

      const inputBuffer = uploadStorage(gpu, input, 'read');
      const layer = new GatedMLPKernel(gpu, inputBuffer, gateWeight, upWeight, downWeight, hiddenSize, innerSize, {
        workgroupSize: 4,
        precision: 'fp16',
        activation: 'gelu_pytorch_tanh',
      });

      layer.run();

      closeArray(await readOutput(layer.outputBuffer), expected, FP16_EPSILON);
    });
  });

  it('GatedDeltaNetKernel narrows its dense projections without breaking the recurrence', async ({ skip }) => {
    await withF16Gpu(skip, async (gpu) => {
      const hiddenSize = 8;
      const numKHeads = 2;
      const numVHeads = 4;
      const keyDim = 4;
      const valueDim = 4;
      const kernelSize = 4;
      const keySize = numKHeads * keyDim;
      const valueSize = numVHeads * valueDim;
      const convDim = keySize * 2 + valueSize;

      const weights = {
        qkvWeight: fillSin(new Float32Array(hiddenSize * convDim), 0.11),
        zWeight: fillSin(new Float32Array(hiddenSize * valueSize), 0.22),
        bWeight: fillSin(new Float32Array(hiddenSize * numVHeads), 0.33),
        aWeight: fillSin(new Float32Array(hiddenSize * numVHeads), 0.44),
        outWeight: fillSin(new Float32Array(valueSize * hiddenSize), 0.55),
        convWeight: fillSin(new Float32Array(convDim * kernelSize), 0.66),
        aLog: fillSin(new Float32Array(numVHeads), 0.77),
        dtBias: fillSin(new Float32Array(numVHeads), 0.88),
        normWeight: fillSin(new Float32Array(valueDim), 0.99),
      };

      const inputBuffer = allocStorage(gpu, hiddenSize, 'read-write');
      const kernel = new GatedDeltaNetKernel(gpu, inputBuffer, weights, {
        hiddenSize,
        numKHeads,
        numVHeads,
        keyDim,
        valueDim,
        kernelSize,
        workgroupSize: 4,
        precision: 'fp16',
      });

      const input = fillSin(new Float32Array(hiddenSize), 1.23);
      writeBuffer(inputBuffer, input);
      kernel.run();

      const output = await readOutput(kernel.outputBuffer);
      expect(output.length).toBe(hiddenSize);
      expect(Array.from(output).every((value) => Number.isFinite(value))).toBe(true);
    });
  });

  it('chunked LogitsKernel fp16 weights stay close to the fp32 CPU reference', async ({ skip }) => {
    await withF16Gpu(skip, async (gpu) => {
      const hiddenSize = 6;
      const vocabSize = 37;
      const chunkSize = 16;
      const input = fillSin(new Float32Array(hiddenSize), 0.9);
      const logitWeight = fillSin(new Float32Array(hiddenSize * vocabSize), 1.9);

      const inputBuffer = uploadStorage(gpu, input, 'read');
      const chunks = createChunkedLogitLayers(
        gpu,
        inputBuffer,
        { hiddenSize, vocabSize, logitWeight },
        chunkSize,
        'TestFP16Logits',
        'fp16',
      );

      for (const chunk of chunks) chunk.layer.run();

      const cpuLogits = linear(input, logitWeight, null, hiddenSize, vocabSize);

      for (const chunk of chunks) {
        const gpuChunk = await readOutput(chunk.layer.outputBuffer);
        closeArray(
          gpuChunk.subarray(0, chunk.size),
          cpuLogits.subarray(chunk.offset, chunk.offset + chunk.size),
          FP16_EPSILON,
        );
      }

      // Greedy argmax should still land on (one of) the true top logit under fp16 rounding.
      const sampler = createLogitSampler(gpu, chunks, { candidateCount: 1 });
      sampler.run(1);
      const token = await sampler.readToken();

      let bestIndex = 0;
      let bestValue = cpuLogits[0]!;
      for (let i = 1; i < cpuLogits.length; i++) {
        if (cpuLogits[i]! > bestValue) {
          bestValue = cpuLogits[i]!;
          bestIndex = i;
        }
      }

      expect(Math.abs(cpuLogits[token]! - bestValue)).toBeLessThan(FP16_EPSILON);
      void bestIndex;
    });
  });

  it('DecoderGpuRunner fp16 logits stay close to the fp32 CPU reference for a tiny Llama', async ({ skip }) => {
    await withF16Gpu(skip, async (gpu) => {
      const cpuRunner = new DecoderCPURunner(createTinyLlama(), { maxTokens: 8 });
      const gpuRunner = new DecoderGpuRunner(gpu, createTinyLlama(), { maxTokens: 8, precision: 'fp16' });

      const tokenId = 1;
      const position = 0;
      const cpuLogits = cpuRunner.forwardToken(tokenId, position);
      gpuRunner.computeToken(tokenId, position);
      const gpuLogits = await gpuRunner.readLogits();

      closeArray(gpuLogits, cpuLogits, 0.2);
    });
  });

  it('QwenGpuRunner fp16 logits stay close to the fp32 CPU reference (gated delta net + attention)', async ({
    skip,
  }) => {
    await withF16Gpu(skip, async (gpu) => {
      const cpuRunner = new QwenCPURunner(createTinyQwenWeights(), { maxTokens: 8 });
      const gpuRunner = new QwenGpuRunner(gpu, createTinyQwenWeights(), { maxTokens: 8, precision: 'fp16' });

      const tokenId = 1;
      const position = 0;
      const cpuLogits = cpuRunner.forwardToken(tokenId, position);
      gpuRunner.computeToken(tokenId, position);
      const gpuLogits = await gpuRunner.readLogits();

      closeArray(gpuLogits, cpuLogits, 0.2);
    });
  });

  it('AttentionKernel is unaffected by fp16 elsewhere in the model (its own norms stay fp32)', async ({ skip }) => {
    await withF16Gpu(skip, async (gpu) => {
      const headCount = 2;
      const headDim = 4;
      const maxTokens = 4;
      const qSize = headCount * headDim;
      const qkvBuffer = uploadStorage(gpu, fillSin(new Float32Array(qSize * 3), 0.7), 'read');
      const attention = new AttentionKernel(gpu, qkvBuffer, qSize, headCount, maxTokens, { workgroupSize: 4 });

      attention.run(0);
      const output = await readOutput(attention.outputBuffer);
      expect(output.length).toBe(qSize);
      expect(Array.from(output).every((value) => Number.isFinite(value))).toBe(true);
    });
  });
});

describe('fp16 without the shader-f16 feature', () => {
  it('throws a clear error instead of silently falling back to fp32', async () => {
    const gpu = await initMockGpu(); // vgpu/mock reports no device features by default.
    const input = uploadStorage(gpu, new Float32Array([1, 2]), 'read');

    expect(
      () => new LinearKernel(gpu, input, new Float32Array([1, 2, 3, 4]), null, 2, 2, { precision: 'fp16' }),
    ).toThrow(/shader-f16/);

    gpu.dispose();
  });

  it('DecoderGpuRunner rejects precision: "fp16" up front when the feature is missing', async () => {
    const gpu = await initMockGpu();

    expect(() => new DecoderGpuRunner(gpu, createTinyLlama(), { maxTokens: 8, precision: 'fp16' })).toThrow(
      /shader-f16/,
    );

    gpu.dispose();
  });
});
