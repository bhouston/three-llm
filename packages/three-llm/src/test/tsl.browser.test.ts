import { describe, expect, it } from 'vitest';
import type { WebGPURenderer } from 'three/webgpu';

import { DecoderCPURunner } from '../decoder/DecoderCPURunner.js';
import { DecoderTSLRunner } from '../decoder/DecoderTSLRunner.js';
import { QwenCPURunner } from '../qwen/QwenCPURunner.js';
import { QwenTSLRunner } from '../qwen/QwenTSLRunner.js';
import { geluNew, layerNorm, linear, rmsNorm, silu } from '../runtime/math.js';
import { TSLAdd } from '../tsl/TSLAdd.js';
import { TSLGatedMLP } from '../tsl/TSLGatedMLP.js';
import { TSLGELU } from '../tsl/TSLGELU.js';
import { createLogitSampler } from '../tsl/TSLLogits.js';
import { TSLLinear } from '../tsl/TSLLinear.js';
import { TSLMLP } from '../tsl/TSLMLP.js';
import { TSLNormalize } from '../tsl/TSLNormalize.js';
import { TSLRMSNorm } from '../tsl/TSLRMSNorm.js';
import { TSLSiLUMul } from '../tsl/TSLSiLUMul.js';
import {
  closeArray,
  createTinyGemma,
  createTinyLlama,
  createTinyPhi,
  createTinyQwenWeights,
  fillSin,
} from './helpers.js';
import { assertAttentionSequence, assertCausalSequence, createRenderer, readOutput, storageFromArray } from './gpu.js';

async function withRenderer(skip: () => never, run: (renderer: WebGPURenderer) => Promise<void> | void) {
  const renderer = await createRenderer(skip);
  try {
    await run(renderer);
  } finally {
    renderer.dispose();
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

describe('TSL kernels', () => {
  it('TSLLinear matches CPU reference', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const input = storageFromArray(new Float32Array([1, 2]));
      const layer = new TSLLinear(input.node, new Float32Array([3, 4, 5, 6]), new Float32Array([7, 8]), 2, 2);

      layer.compute(renderer);

      closeArray(await readOutput(renderer, layer), new Float32Array([20, 24]), 1e-5);
    });
  });

  it('TSLLinear matches CPU reference without bias and for a non-square map', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const input = new Float32Array([1, -1, 0.5]);
      const weight = new Float32Array([1, 2, 3, 4, 5, 6]);
      const { node } = storageFromArray(input);
      const layer = new TSLLinear(node, weight, null, 3, 2, { workgroupSize: 2 });

      layer.compute(renderer);

      closeArray(await readOutput(renderer, layer), new Float32Array([0.5, 1]), 1e-5);
    });
  });

  it('TSL logit sampler returns greedy and top-k candidates', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const first = storageFromArray(new Float32Array([1, 7, 7, -2]));
      const second = storageFromArray(new Float32Array([8, 3, 6]));
      const sampler = createLogitSampler(
        [
          { offset: 0, size: 4, layer: { outputNode: first.node } },
          { offset: 4, size: 3, layer: { outputNode: second.node } },
        ],
        { candidateCount: 3 },
      );

      renderer.compute(sampler.computeNodesFor(3));

      expect(await sampler.readToken(renderer)).toBe(4);
      expect(await sampler.readCandidates(renderer, 3)).toEqual([
        [4, 8],
        [1, 7],
        [2, 7],
      ]);
    });
  });

  it('TSL logit sampler applies softcap before ranking', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const logits = storageFromArray(new Float32Array([10, 2, -1]));
      const sampler = createLogitSampler([{ offset: 0, size: 3, layer: { outputNode: logits.node } }], {
        candidateCount: 2,
        logitSoftcap: 1,
      });

      renderer.compute(sampler.computeNodesFor(2));

      const candidates = await sampler.readCandidates(renderer, 2);
      expect(candidates[0]![0]).toBe(0);
      expect(candidates[0]![1]).toBeLessThanOrEqual(1);
      expect(candidates[1]![0]).toBe(1);
    });
  });

  it('TSLNormalize matches CPU reference', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const input = new Float32Array([1, 2, 3]);
      const { node } = storageFromArray(input);
      const layer = new TSLNormalize(node, new Float32Array([1, 1, 1]), new Float32Array([0, 0, 0]), 3, {
        workgroupSize: 3,
      });

      layer.compute(renderer);

      closeArray(
        await readOutput(renderer, layer),
        layerNorm(input, new Float32Array([1, 1, 1]), new Float32Array([0, 0, 0])),
        1e-5,
      );
    });
  });

  it('TSLNormalize matches CPU reference with affine parameters', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const input = new Float32Array([1, 2, 3, -1]);
      const weight = new Float32Array([2, 0.5, 1, 1.5]);
      const bias = new Float32Array([0.1, -0.2, 0.3, 0]);
      const { node } = storageFromArray(input);
      const layer = new TSLNormalize(node, weight, bias, 4, { workgroupSize: 4, epsilon: 1e-5 });

      layer.compute(renderer);

      closeArray(await readOutput(renderer, layer), layerNorm(input, weight, bias), 1e-5);
    });
  });

  it('TSLGELU matches CPU gelu_new', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const input = new Float32Array([-2, -1, 0, 0.5, 1, 2, 3]);
      const { node } = storageFromArray(input);
      const layer = new TSLGELU(node, input.length, { workgroupSize: input.length });

      layer.compute(renderer);

      const output = await readOutput(renderer, layer);
      closeArray(output, mapGelu(input), 1e-5);
      expect(Math.abs(output[4]! - 0.84119199)).toBeLessThan(1e-5);
    });
  });

  it('TSLAdd matches element-wise CPU add', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const a = new Float32Array([1, 2, 3, -4]);
      const b = new Float32Array([4, -1, 0.5, 4]);
      const layer = new TSLAdd(storageFromArray(a).node, storageFromArray(b).node, 4, { workgroupSize: 4 });

      layer.compute(renderer);

      closeArray(await readOutput(renderer, layer), new Float32Array([5, 1, 3.5, 0]), 1e-5);
    });
  });

  it('TSLMLP matches CPU dense -> gelu_new -> dense', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const input = new Float32Array([1, -1]);
      const fcWeight = new Float32Array([0.5, -0.25, 1, 0.75, 0.5, -1]);
      const fcBias = new Float32Array([0.1, 0, -0.2]);
      const projWeight = new Float32Array([1, 0, 0, 1, 0.5, -0.5]);
      const projBias = new Float32Array([0, 0.25]);
      const layer = new TSLMLP(storageFromArray(input).node, fcWeight, fcBias, projWeight, projBias, 2, 3, {
        workgroupSize: 3,
      });

      layer.compute(renderer);

      closeArray(
        await readOutput(renderer, layer.proj),
        cpuMLP(input, fcWeight, fcBias, projWeight, projBias, 2, 3),
        1e-4,
      );
    });
  });

  it('TSLAttention matches CPU reference for a one-token and two-token pass', async ({ skip }) => {
    await withRenderer(skip, (renderer) =>
      assertAttentionSequence(
        renderer,
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

  it('TSLAttention matches CPU reference over a longer cached sequence', async ({ skip }) => {
    await withRenderer(skip, (renderer) =>
      assertAttentionSequence(
        renderer,
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

  it('TSLAttention matches CPU reference for GPT-2-sized heads', async ({ skip }) => {
    await withRenderer(skip, (renderer) => {
      const hiddenSize = 128;
      const sequence: Float32Array[] = [];

      for (let position = 0; position < 4; position++) {
        const qkv = new Float32Array(hiddenSize * 3);
        for (let i = 0; i < qkv.length; i++) qkv[i] = Math.sin(position * 19.1 + i * 0.17) * 0.35;
        sequence.push(qkv);
      }

      return assertAttentionSequence(renderer, hiddenSize, 2, 8, sequence, 64, 2e-4);
    });
  });

  it('TSLRMSNorm matches CPU rmsNorm', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const input = new Float32Array([1, 2, 3, -1]);
      const weight = new Float32Array([2, 0.5, 1, 1.5]);
      const layer = new TSLRMSNorm(storageFromArray(input).node, weight, 4, { workgroupSize: 4 });

      layer.compute(renderer);

      closeArray(await readOutput(renderer, layer), rmsNorm(input, weight), 1e-5);
    });
  });

  it('TSLSiLUMul matches silu(gate) * up', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const gate = new Float32Array([-1, 0, 1, 2]);
      const up = new Float32Array([0.5, -2, 3, 0.25]);
      const expected = new Float32Array(gate.map((value, i) => silu(value) * up[i]!));
      const layer = new TSLSiLUMul(storageFromArray(gate).node, storageFromArray(up).node, 4, { workgroupSize: 4 });

      layer.compute(renderer);

      closeArray(await readOutput(renderer, layer), expected, 1e-5);
    });
  });

  it('TSLGatedMLP matches CPU SwiGLU', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const input = new Float32Array([1, -0.5]);
      const gateWeight = new Float32Array([0.5, -0.25, 1, 0.75, 0.5, -1]);
      const upWeight = new Float32Array([1, 0, 0, 1, 0.5, -0.5]);
      const downWeight = new Float32Array([1, 0, 0, 0.25, -0.5, 0.5]);
      const gate = linear(input, gateWeight, null, 2, 3);
      const up = linear(input, upWeight, null, 2, 3);
      const hidden = new Float32Array(3);
      for (let i = 0; i < 3; i++) hidden[i] = silu(gate[i]!) * up[i]!;
      const expected = linear(hidden, downWeight, null, 3, 2);
      const layer = new TSLGatedMLP(storageFromArray(input).node, gateWeight, upWeight, downWeight, 2, 3, {
        workgroupSize: 3,
      });

      layer.compute(renderer);

      closeArray(await readOutput(renderer, layer.down), expected, 1e-4);
    });
  });

  it('TSLGatedMLP matches CPU GeGLU', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const input = new Float32Array([1, -0.5]);
      const gateWeight = new Float32Array([0.5, -0.25, 1, 0.75, 0.5, -1]);
      const upWeight = new Float32Array([1, 0, 0, 1, 0.5, -0.5]);
      const downWeight = new Float32Array([1, 0, 0, 0.25, -0.5, 0.5]);
      const gate = linear(input, gateWeight, null, 2, 3);
      const up = linear(input, upWeight, null, 2, 3);
      const hidden = new Float32Array(3);
      for (let i = 0; i < 3; i++) hidden[i] = geluNew(gate[i]!) * up[i]!;
      const expected = linear(hidden, downWeight, null, 3, 2);
      const layer = new TSLGatedMLP(storageFromArray(input).node, gateWeight, upWeight, downWeight, 2, 3, {
        activation: 'gelu_pytorch_tanh',
        workgroupSize: 3,
      });

      layer.compute(renderer);

      closeArray(await readOutput(renderer, layer.down), expected, 1e-4);
    });
  });

  it('TSLAttention matches GQA without RoPE', async ({ skip }) => {
    await withRenderer(skip, (renderer) =>
      assertCausalSequence(renderer, [fillSin(new Float32Array(16), 0.3), fillSin(new Float32Array(16), 1.1)], {
        headCount: 4,
        kvHeadCount: 2,
        headDim: 2,
        maxTokens: 4,
        workgroupSize: 16,
      }),
    );
  });

  it('TSLAttention matches RoPE multi-head attention', async ({ skip }) => {
    await withRenderer(skip, (renderer) =>
      assertCausalSequence(
        renderer,
        [fillSin(new Float32Array(12), 0.2), fillSin(new Float32Array(12), 0.8), fillSin(new Float32Array(12), 1.4)],
        { headCount: 2, kvHeadCount: 2, headDim: 2, maxTokens: 8, ropeTheta: 10000, workgroupSize: 16 },
      ),
    );
  });

  it('TSLAttention matches grouped-query RoPE', async ({ skip }) => {
    await withRenderer(skip, (renderer) =>
      assertCausalSequence(renderer, [fillSin(new Float32Array(16), 0.5), fillSin(new Float32Array(16), 1.5)], {
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

  it('TSLAttention matches partial RoPE and an explicit attention scale', async ({ skip }) => {
    await withRenderer(skip, (renderer) =>
      assertCausalSequence(renderer, [fillSin(new Float32Array(24), 0.6), fillSin(new Float32Array(24), 1.6)], {
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

  it('TSLAttention matches sliding-window GQA', async ({ skip }) => {
    await withRenderer(skip, (renderer) =>
      assertCausalSequence(
        renderer,
        [fillSin(new Float32Array(16), 0.4), fillSin(new Float32Array(16), 1.4), fillSin(new Float32Array(16), 2.4)],
        { headCount: 4, kvHeadCount: 2, headDim: 2, maxTokens: 8, slidingWindow: 2, workgroupSize: 16 },
      ),
    );
  });

  it('TSLAttention matches QK-norm and RoPE', async ({ skip }) => {
    await withRenderer(skip, (renderer) =>
      assertCausalSequence(renderer, [fillSin(new Float32Array(16), 0.7), fillSin(new Float32Array(16), 1.7)], {
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

describe('TSL runners', () => {
  it('tiny Llama greedy GPU matches CPU', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const weights = createTinyLlama();
      const options = { maxNewTokens: 4, temperature: 0, topK: 1 };
      const cpu = new DecoderCPURunner(weights, { maxTokens: 8 }).generate('hello', options);
      const gpu = await new DecoderTSLRunner(weights, { maxTokens: 8 }).generate(renderer, 'hello', options);

      expect(gpu.generatedTokens).toEqual(cpu.generatedTokens);
      expect(gpu.text).toBe(cpu.text);
    });
  });

  it('tiny Phi greedy GPU matches CPU', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const weights = createTinyPhi();
      const options = { maxNewTokens: 4, temperature: 0, topK: 1 };
      const cpu = new DecoderCPURunner(weights, { maxTokens: 8 }).generate('hello', options);
      const gpu = await new DecoderTSLRunner(weights, { maxTokens: 8 }).generate(renderer, 'hello', options);

      expect(gpu.generatedTokens).toEqual(cpu.generatedTokens);
      expect(gpu.text).toBe(cpu.text);
    });
  });

  it('tiny Gemma 3 greedy GPU matches CPU', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const weights = createTinyGemma();
      const options = { maxNewTokens: 4, temperature: 0, topK: 1 };
      const cpu = new DecoderCPURunner(weights, { maxTokens: 8 }).generate('hello', options);
      const gpu = await new DecoderTSLRunner(weights, { maxTokens: 8 }).generate(renderer, 'hello', options);

      expect(weights.block(0).slidingWindow).toBe(2);
      expect(weights.block(1).slidingWindow).toBe(0);
      expect(gpu.generatedTokens).toEqual(cpu.generatedTokens);
      expect(gpu.text).toBe(cpu.text);
    });
  });

  it('tiny Qwen3.5 greedy GPU matches CPU', async ({ skip }) => {
    await withRenderer(skip, async (renderer) => {
      const weights = createTinyQwenWeights();
      const options = { maxNewTokens: 4, temperature: 0, topK: 1 };
      const cpu = new QwenCPURunner(weights, { maxTokens: 8 }).generate('hello', options);
      const gpu = await new QwenTSLRunner(weights, { maxTokens: 8 }).generate(renderer, 'hello', options);

      expect(weights.block(0).layerType).toBe('linear_attention');
      expect(weights.block(1).layerType).toBe('full_attention');
      expect(gpu.generatedTokens).toEqual(cpu.generatedTokens);
      expect(gpu.text).toBe(cpu.text);
    });
  });
});
