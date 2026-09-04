import { describe, expect, it } from 'vitest';
import type { Gpu } from 'vgpu';

import { DecoderCPURunner } from '../decoder/DecoderCPURunner.js';
import { DecoderGpuRunner } from '../decoder/DecoderGpuRunner.js';
import { QwenCPURunner } from '../qwen/QwenCPURunner.js';
import { QwenGpuRunner } from '../qwen/QwenGpuRunner.js';
import { geluNew, layerNorm, linear, rmsNorm, silu } from '../runtime/math.js';
import { AddKernel } from '../kernels/AddKernel.js';
import { GatedMLPKernel } from '../kernels/GatedMLPKernel.js';
import { GELUKernel } from '../kernels/GELUKernel.js';
import { createLogitSampler } from '../kernels/LogitsKernel.js';
import { LinearKernel } from '../kernels/LinearKernel.js';
import { MLPKernel } from '../kernels/MLPKernel.js';
import { NormalizeKernel } from '../kernels/NormalizeKernel.js';
import { RMSNormKernel } from '../kernels/RMSNormKernel.js';
import { SiLUMulKernel } from '../kernels/SiLUMulKernel.js';
import { uploadStorage } from '../gpu/device.js';
import { assertAttentionSequence, assertCausalSequence, createGpu, readOutput } from './gpu.js';
import {
  closeArray,
  createTinyGemma,
  createTinyLlama,
  createTinyPhi,
  createTinyQwenWeights,
  fillSin,
} from './helpers.js';

async function withGpu(skip: () => never, run: (gpu: Gpu) => Promise<void> | void) {
  const gpu = await createGpu(skip);
  try {
    await run(gpu);
  } finally {
    gpu.dispose();
  }
}

function mapGelu(values: Float32Array) {
  const target = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) target[i] = geluNew(values[i]!);
  return target;
}

function cpuMLP(
  input: Float32Array,
  fcWeight: Float32Array,
  fcBias: Float32Array,
  projWeight: Float32Array,
  projBias: Float32Array,
  hiddenSize: number,
  innerSize: number,
) {
  return linear(
    mapGelu(linear(input, fcWeight, fcBias, hiddenSize, innerSize)),
    projWeight,
    projBias,
    innerSize,
    hiddenSize,
  );
}

describe('GPU kernels (vgpu/node, real WGSL execution)', () => {
  it('LinearKernel matches CPU reference', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const input = uploadStorage(gpu, new Float32Array([1, 2]), 'read');
      const layer = new LinearKernel(gpu, input, new Float32Array([3, 4, 5, 6]), new Float32Array([7, 8]), 2, 2);

      layer.run();

      closeArray(await readOutput(layer.outputBuffer), new Float32Array([20, 24]), 1e-5);
    });
  });

  it('LinearKernel matches CPU reference without bias and for a non-square map', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const input = new Float32Array([1, -1, 0.5]);
      const weight = new Float32Array([1, 2, 3, 4, 5, 6]);
      const inputBuffer = uploadStorage(gpu, input, 'read');
      const layer = new LinearKernel(gpu, inputBuffer, weight, null, 3, 2, { workgroupSize: 2 });

      layer.run();

      closeArray(await readOutput(layer.outputBuffer), linear(input, weight, null, 3, 2), 1e-5);
    });
  });

  it('LogitSampler returns greedy and top-k candidates', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const first = uploadStorage(gpu, new Float32Array([1, 7, 7, -2]), 'read');
      const second = uploadStorage(gpu, new Float32Array([8, 3, 6]), 'read');
      const sampler = createLogitSampler(
        gpu,
        [
          { offset: 0, size: 4, layer: { outputBuffer: first } as never },
          { offset: 4, size: 3, layer: { outputBuffer: second } as never },
        ],
        { candidateCount: 3 },
      );

      sampler.run(3);

      expect(await sampler.readToken()).toBe(4);
      expect(await sampler.readCandidates(3)).toEqual([
        [4, 8],
        [1, 7],
        [2, 7],
      ]);
    });
  });

  it('LogitSampler applies softcap before ranking', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const logits = uploadStorage(gpu, new Float32Array([10, 2, -1]), 'read');
      const sampler = createLogitSampler(gpu, [{ offset: 0, size: 3, layer: { outputBuffer: logits } as never }], {
        candidateCount: 2,
        logitSoftcap: 1,
      });

      sampler.run(2);

      const candidates = await sampler.readCandidates(2);
      expect(candidates[0]![0]).toBe(0);
      expect(candidates[0]![1]).toBeLessThanOrEqual(1);
      expect(candidates[1]![0]).toBe(1);
    });
  });

  it('NormalizeKernel matches CPU reference', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const input = new Float32Array([1, 2, 3]);
      const weight = new Float32Array([1, 1, 1]);
      const bias = new Float32Array([0, 0, 0]);
      const inputBuffer = uploadStorage(gpu, input, 'read');
      const weightBuffer = uploadStorage(gpu, weight, 'read');
      const biasBuffer = uploadStorage(gpu, bias, 'read');
      const layer = new NormalizeKernel(gpu, inputBuffer, weightBuffer, biasBuffer, 3, { workgroupSize: 3 });

      layer.run();

      closeArray(await readOutput(layer.outputBuffer), layerNorm(input, weight, bias), 1e-5);
    });
  });

  it('NormalizeKernel matches CPU reference with affine parameters', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const input = new Float32Array([1, 2, 3, -1]);
      const weight = new Float32Array([2, 0.5, 1, 1.5]);
      const bias = new Float32Array([0.1, -0.2, 0.3, 0]);
      const inputBuffer = uploadStorage(gpu, input, 'read');
      const weightBuffer = uploadStorage(gpu, weight, 'read');
      const biasBuffer = uploadStorage(gpu, bias, 'read');
      const layer = new NormalizeKernel(gpu, inputBuffer, weightBuffer, biasBuffer, 4, {
        workgroupSize: 4,
        epsilon: 1e-5,
      });

      layer.run();

      closeArray(await readOutput(layer.outputBuffer), layerNorm(input, weight, bias), 1e-5);
    });
  });

  it('GELUKernel matches CPU gelu_new', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const input = new Float32Array([-2, -1, 0, 0.5, 1, 2, 3]);
      const inputBuffer = uploadStorage(gpu, input, 'read');
      const layer = new GELUKernel(gpu, inputBuffer, input.length, { workgroupSize: input.length });

      layer.run();

      const output = await readOutput(layer.outputBuffer);
      closeArray(output, mapGelu(input), 1e-5);
      expect(Math.abs(output[4]! - 0.84119199)).toBeLessThan(1e-5);
    });
  });

  it('AddKernel matches element-wise CPU add', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const a = uploadStorage(gpu, new Float32Array([1, 2, 3, -4]), 'read');
      const b = uploadStorage(gpu, new Float32Array([4, -1, 0.5, 4]), 'read');
      const layer = new AddKernel(gpu, a, b, 4, { workgroupSize: 4 });

      layer.run();

      closeArray(await readOutput(layer.outputBuffer), new Float32Array([5, 1, 3.5, 0]), 1e-5);
    });
  });

  it('MLPKernel matches CPU dense -> gelu_new -> dense', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const input = new Float32Array([1, -1]);
      const fcWeight = new Float32Array([0.5, -0.25, 1, 0.75, 0.5, -1]);
      const fcBias = new Float32Array([0.1, 0, -0.2]);
      const projWeight = new Float32Array([1, 0, 0, 1, 0.5, -0.5]);
      const projBias = new Float32Array([0, 0.25]);
      const inputBuffer = uploadStorage(gpu, input, 'read');
      const layer = new MLPKernel(gpu, inputBuffer, fcWeight, fcBias, projWeight, projBias, 2, 3, { workgroupSize: 3 });

      layer.run();

      closeArray(
        await readOutput(layer.outputBuffer),
        cpuMLP(input, fcWeight, fcBias, projWeight, projBias, 2, 3),
        1e-4,
      );
    });
  });

  it('AttentionKernel matches CPU reference for a one-token and two-token pass', async ({ skip }) => {
    await withGpu(skip, (gpu) =>
      assertAttentionSequence(
        gpu,
        4,
        2,
        4,
        [
          [1, 0, 0, 1, 1, 0, 0, 1, 2, 3, 4, 5],
          [0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1],
        ],
        4,
      ),
    );
  });

  it('AttentionKernel matches CPU reference over a longer cached sequence', async ({ skip }) => {
    await withGpu(skip, (gpu) =>
      assertAttentionSequence(
        gpu,
        4,
        2,
        8,
        [
          [1, 0, 0, 1, 1, 0, 0, 1, 2, 3, 4, 5],
          [0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1],
          [0.5, -1, 2, 0, 0.25, 0.5, -0.5, 1, 0, 2, -1, 3],
          [-2, 1, 0.5, 0.5, 1, -1, 0, 0.25, 4, 0, 1, -2],
        ],
        64,
      ),
    );
  });

  it('AttentionKernel matches CPU reference for GPT-2-sized heads', async ({ skip }) => {
    await withGpu(skip, (gpu) => {
      const hiddenSize = 128;
      const sequence: Float32Array[] = [];

      for (let position = 0; position < 4; position++) {
        const qkv = new Float32Array(hiddenSize * 3);
        for (let i = 0; i < qkv.length; i++) qkv[i] = Math.sin(position * 19.1 + i * 0.17) * 0.35;
        sequence.push(qkv);
      }

      return assertAttentionSequence(gpu, hiddenSize, 2, 8, sequence, 64, 2e-4);
    });
  });

  it('RMSNormKernel matches CPU rmsNorm', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const input = new Float32Array([1, 2, 3, -1]);
      const weight = new Float32Array([2, 0.5, 1, 1.5]);
      const inputBuffer = uploadStorage(gpu, input, 'read');
      const weightBuffer = uploadStorage(gpu, weight, 'read');
      const layer = new RMSNormKernel(gpu, inputBuffer, weightBuffer, 4, { workgroupSize: 4 });

      layer.run();

      closeArray(await readOutput(layer.outputBuffer), rmsNorm(input, weight), 1e-5);
    });
  });

  it('SiLUMulKernel matches silu(gate) * up', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const gate = new Float32Array([-1, 0, 1, 2]);
      const up = new Float32Array([0.5, -2, 3, 0.25]);
      const expected = new Float32Array(gate.map((value, i) => silu(value) * up[i]!));
      const gateBuffer = uploadStorage(gpu, gate, 'read');
      const upBuffer = uploadStorage(gpu, up, 'read');
      const layer = new SiLUMulKernel(gpu, gateBuffer, upBuffer, 4, { workgroupSize: 4 });

      layer.run();

      closeArray(await readOutput(layer.outputBuffer), expected, 1e-5);
    });
  });

  it('GatedMLPKernel matches CPU SwiGLU', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const input = new Float32Array([1, -0.5]);
      const gateWeight = new Float32Array([0.5, -0.25, 1, 0.75, 0.5, -1]);
      const upWeight = new Float32Array([1, 0, 0, 1, 0.5, -0.5]);
      const downWeight = new Float32Array([1, 0, 0, 0.25, -0.5, 0.5]);
      const gate = linear(input, gateWeight, null, 2, 3);
      const up = linear(input, upWeight, null, 2, 3);
      const hidden = new Float32Array(3);
      for (let i = 0; i < 3; i++) hidden[i] = silu(gate[i]!) * up[i]!;
      const expected = linear(hidden, downWeight, null, 3, 2);
      const inputBuffer = uploadStorage(gpu, input, 'read');
      const layer = new GatedMLPKernel(gpu, inputBuffer, gateWeight, upWeight, downWeight, 2, 3, { workgroupSize: 3 });

      layer.run();

      closeArray(await readOutput(layer.outputBuffer), expected, 1e-4);
    });
  });

  it('GatedMLPKernel matches CPU GeGLU', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const input = new Float32Array([1, -0.5]);
      const gateWeight = new Float32Array([0.5, -0.25, 1, 0.75, 0.5, -1]);
      const upWeight = new Float32Array([1, 0, 0, 1, 0.5, -0.5]);
      const downWeight = new Float32Array([1, 0, 0, 0.25, -0.5, 0.5]);
      const gate = linear(input, gateWeight, null, 2, 3);
      const up = linear(input, upWeight, null, 2, 3);
      const hidden = new Float32Array(3);
      for (let i = 0; i < 3; i++) hidden[i] = geluNew(gate[i]!) * up[i]!;
      const expected = linear(hidden, downWeight, null, 3, 2);
      const inputBuffer = uploadStorage(gpu, input, 'read');
      const layer = new GatedMLPKernel(gpu, inputBuffer, gateWeight, upWeight, downWeight, 2, 3, {
        activation: 'gelu_pytorch_tanh',
        workgroupSize: 3,
      });

      layer.run();

      closeArray(await readOutput(layer.outputBuffer), expected, 1e-4);
    });
  });

  it('AttentionKernel matches GQA without RoPE', async ({ skip }) => {
    await withGpu(skip, (gpu) =>
      assertCausalSequence(gpu, [fillSin(new Float32Array(16), 0.3), fillSin(new Float32Array(16), 1.1)], {
        headCount: 4,
        kvHeadCount: 2,
        headDim: 2,
        maxTokens: 4,
        workgroupSize: 16,
      }),
    );
  });

  it('AttentionKernel matches RoPE multi-head attention', async ({ skip }) => {
    await withGpu(skip, (gpu) =>
      assertCausalSequence(
        gpu,
        [fillSin(new Float32Array(12), 0.2), fillSin(new Float32Array(12), 0.8), fillSin(new Float32Array(12), 1.4)],
        { headCount: 2, kvHeadCount: 2, headDim: 2, maxTokens: 8, ropeTheta: 10000, workgroupSize: 16 },
      ),
    );
  });

  it('AttentionKernel matches grouped-query RoPE', async ({ skip }) => {
    await withGpu(skip, (gpu) =>
      assertCausalSequence(gpu, [fillSin(new Float32Array(16), 0.5), fillSin(new Float32Array(16), 1.5)], {
        headCount: 4,
        kvHeadCount: 2,
        headDim: 2,
        maxTokens: 4,
        ropeTheta: 10000,
        rotaryDim: 2,
        workgroupSize: 16,
      }),
    );
  });

  it('AttentionKernel matches partial RoPE and an explicit attention scale', async ({ skip }) => {
    await withGpu(skip, (gpu) =>
      assertCausalSequence(gpu, [fillSin(new Float32Array(24), 0.6), fillSin(new Float32Array(24), 1.6)], {
        headCount: 2,
        kvHeadCount: 2,
        headDim: 4,
        maxTokens: 4,
        ropeTheta: 10000,
        rotaryDim: 2,
        attnScale: 0.25,
        workgroupSize: 16,
      }),
    );
  });

  it('AttentionKernel matches sliding-window GQA', async ({ skip }) => {
    await withGpu(skip, (gpu) =>
      assertCausalSequence(
        gpu,
        [fillSin(new Float32Array(16), 0.4), fillSin(new Float32Array(16), 1.4), fillSin(new Float32Array(16), 2.4)],
        { headCount: 4, kvHeadCount: 2, headDim: 2, maxTokens: 8, slidingWindow: 2, workgroupSize: 16 },
      ),
    );
  });

  it('AttentionKernel matches QK-norm and RoPE', async ({ skip }) => {
    await withGpu(skip, (gpu) =>
      assertCausalSequence(gpu, [fillSin(new Float32Array(16), 0.7), fillSin(new Float32Array(16), 1.7)], {
        headCount: 2,
        kvHeadCount: 1,
        headDim: 4,
        maxTokens: 4,
        ropeTheta: 10000,
        qNormWeight: new Float32Array([0.5, -0.25, 0.1, 0]),
        kNormWeight: new Float32Array([0.2, 0.3, -0.1, 0.4]),
        offsetRMSNorm: true,
        rmsEpsilon: 1e-6,
        workgroupSize: 16,
      }),
    );
  });
});

describe('GPU runners', () => {
  it('tiny Llama greedy GPU matches CPU', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const weights = createTinyLlama();
      const options = { maxNewTokens: 4, temperature: 0, topK: 1 };
      const cpu = new DecoderCPURunner(weights, { maxTokens: 8 }).generate('hello', options);
      const gpuResult = await new DecoderGpuRunner(gpu, weights, { maxTokens: 8 }).generate('hello', options);

      expect(gpuResult.generatedTokens).toEqual(cpu.generatedTokens);
      expect(gpuResult.text).toBe(cpu.text);
    });
  });

  it('tiny Phi greedy GPU matches CPU', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const weights = createTinyPhi();
      const options = { maxNewTokens: 4, temperature: 0, topK: 1 };
      const cpu = new DecoderCPURunner(weights, { maxTokens: 8 }).generate('hello', options);
      const gpuResult = await new DecoderGpuRunner(gpu, weights, { maxTokens: 8 }).generate('hello', options);

      expect(gpuResult.generatedTokens).toEqual(cpu.generatedTokens);
      expect(gpuResult.text).toBe(cpu.text);
    });
  });

  it('tiny Gemma 3 greedy GPU matches CPU', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const weights = createTinyGemma();
      const options = { maxNewTokens: 4, temperature: 0, topK: 1 };
      const cpu = new DecoderCPURunner(weights, { maxTokens: 8 }).generate('hello', options);
      const gpuResult = await new DecoderGpuRunner(gpu, weights, { maxTokens: 8 }).generate('hello', options);

      expect(weights.block(0).slidingWindow).toBe(2);
      expect(weights.block(1).slidingWindow).toBe(0);
      expect(gpuResult.generatedTokens).toEqual(cpu.generatedTokens);
      expect(gpuResult.text).toBe(cpu.text);
    });
  });

  it('tiny Qwen3.5 greedy GPU matches CPU', async ({ skip }) => {
    await withGpu(skip, async (gpu) => {
      const weights = createTinyQwenWeights();
      const options = { maxNewTokens: 4, temperature: 0, topK: 1 };
      const cpu = new QwenCPURunner(weights, { maxTokens: 8 }).generate('hello', options);
      const gpuResult = await new QwenGpuRunner(gpu, weights, { maxTokens: 8 }).generate('hello', options);

      expect(weights.block(0).layerType).toBe('linear_attention');
      expect(weights.block(1).layerType).toBe('full_attention');
      expect(gpuResult.generatedTokens).toEqual(cpu.generatedTokens);
      expect(gpuResult.text).toBe(cpu.text);
    });
  });
});
