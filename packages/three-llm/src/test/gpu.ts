import { StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import { storage } from 'three/tsl';
import { expect } from 'vitest';

import { closeArray } from './helpers.js';
import { causalAttention, softmax } from '../runtime/math.js';
import { TSLAttention } from '../tsl/TSLAttention.js';
import type { Renderer, TslNode } from '../types.js';

export function storageFromArray(array: Float32Array) {
  const attribute = new StorageBufferAttribute(array, 1);
  return {
    attribute,
    node: storage(attribute, 'float', array.length).toReadOnly(),
  };
}

export async function createRenderer(skip: () => never): Promise<WebGPURenderer> {
  const gpu = typeof navigator === 'undefined' ? undefined : (navigator as Navigator & { gpu?: any }).gpu;
  if (!gpu) {
    skip();
  }

  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    skip();
  }

  const renderer = new WebGPURenderer({ antialias: false });

  try {
    await renderer.init();
  } catch {
    renderer.dispose();
    skip();
  }

  return renderer;
}

export async function readOutput(renderer: Renderer, layer: { outputAttribute: StorageBufferAttribute }) {
  return new Float32Array(await renderer.getArrayBufferAsync(layer.outputAttribute));
}

function cpuAttention(
  qkv: Float32Array,
  hiddenSize: number,
  headCount: number,
  keyCache: Float32Array,
  valueCache: Float32Array,
  position: number,
) {
  const headSize = hiddenSize / headCount;
  const scale = 1 / Math.sqrt(headSize);

  for (let dim = 0; dim < hiddenSize; dim++) {
    keyCache[position * hiddenSize + dim] = qkv[hiddenSize + dim]!;
    valueCache[position * hiddenSize + dim] = qkv[hiddenSize * 2 + dim]!;
  }

  const output = new Float32Array(hiddenSize);

  for (let head = 0; head < headCount; head++) {
    const headOffset = head * headSize;
    const scores = new Float32Array(position + 1);

    for (let token = 0; token <= position; token++) {
      let dot = 0;
      for (let i = 0; i < headSize; i++) {
        dot += qkv[headOffset + i]! * keyCache[token * hiddenSize + headOffset + i]!;
      }
      scores[token] = dot * scale;
    }

    const weights = softmax(scores);

    for (let i = 0; i < headSize; i++) {
      let sum = 0;
      for (let token = 0; token <= position; token++) {
        sum += weights[token]! * valueCache[token * hiddenSize + headOffset + i]!;
      }
      output[headOffset + i] = sum;
    }
  }

  return output;
}

function cpuAttentionScores(
  qkv: Float32Array,
  hiddenSize: number,
  headCount: number,
  keyCache: Float32Array,
  position: number,
  maxTokens: number,
) {
  const headSize = hiddenSize / headCount;
  const scale = 1 / Math.sqrt(headSize);
  const scores = new Float32Array(headCount * maxTokens);

  for (let head = 0; head < headCount; head++) {
    const headOffset = head * headSize;

    for (let token = 0; token <= position; token++) {
      let dot = 0;
      for (let i = 0; i < headSize; i++) {
        dot += qkv[headOffset + i]! * keyCache[token * hiddenSize + headOffset + i]!;
      }
      scores[head * maxTokens + token] = dot * scale;
    }
  }

  return scores;
}

export async function assertAttentionSequence(
  renderer: Renderer,
  hiddenSize: number,
  headCount: number,
  maxTokens: number,
  sequence: Array<Float32Array | number[]>,
  workgroupSize: number,
  epsilon = 1e-4,
) {
  const qkvBuffer = new Float32Array(hiddenSize * 3);
  const { attribute, node } = storageFromArray(qkvBuffer);
  const layer = new TSLAttention(node, hiddenSize, headCount, maxTokens, { workgroupSize });
  const keyCache = new Float32Array(hiddenSize * maxTokens);
  const valueCache = new Float32Array(hiddenSize * maxTokens);

  for (let position = 0; position < sequence.length; position++) {
    qkvBuffer.set(sequence[position]!);
    attribute.needsUpdate = true;
    layer.compute(renderer, position);

    const expected = cpuAttention(qkvBuffer.slice(), hiddenSize, headCount, keyCache, valueCache, position);
    const gpuOutput = await readOutput(renderer, layer);
    const gpuScores = new Float32Array(await renderer.getArrayBufferAsync(layer.scoreAttribute));
    const expectedScores = cpuAttentionScores(qkvBuffer, hiddenSize, headCount, keyCache, position, maxTokens);

    closeArray(gpuOutput, expected, epsilon);

    for (let head = 0; head < headCount; head++) {
      for (let token = 0; token <= position; token++) {
        const index = head * maxTokens + token;
        expect(Math.abs(gpuScores[index]! - expectedScores[index]!)).toBeLessThanOrEqual(epsilon);
      }
    }
  }
}

interface CausalSequenceOptions {
  headCount: number;
  headDim: number;
  kvHeadCount?: number;
  maxTokens: number;
  ropeTheta?: number;
  rotaryDim?: number;
  slidingWindow?: number;
  attnScale?: number;
  qNormWeight?: Float32Array;
  kNormWeight?: Float32Array;
  rmsEpsilon?: number;
  offsetRMSNorm?: boolean;
  workgroupSize: number;
}

export async function assertCausalSequence(
  renderer: Renderer,
  sequence: Array<Float32Array | number[]>,
  options: CausalSequenceOptions,
  epsilon = 1e-4,
) {
  const { headCount, headDim, maxTokens, workgroupSize } = options;
  const kvHeadCount = options.kvHeadCount || headCount;
  const qSize = headCount * headDim;
  const kvSize = kvHeadCount * headDim;
  const qkvBuffer = new Float32Array(qSize + 2 * kvSize);
  const { attribute, node } = storageFromArray(qkvBuffer);
  const layer = new TSLAttention(node, qSize, headCount, maxTokens, {
    headDim,
    kvHeadCount,
    ropeTheta: options.ropeTheta || 0,
    rotaryDim: options.rotaryDim,
    slidingWindow: options.slidingWindow || 0,
    attnScale: options.attnScale,
    qNormWeight: options.qNormWeight,
    kNormWeight: options.kNormWeight,
    rmsEpsilon: options.rmsEpsilon,
    offsetRMSNorm: options.offsetRMSNorm,
    workgroupSize,
  });
  const keyCache = new Float32Array(kvSize * maxTokens);
  const valueCache = new Float32Array(kvSize * maxTokens);

  for (let position = 0; position < sequence.length; position++) {
    qkvBuffer.set(sequence[position]!);
    attribute.needsUpdate = true;
    layer.compute(renderer, position);

    const expected = causalAttention(qkvBuffer.slice(), {
      headCount,
      headDim,
      kvHeadCount,
      position,
      keyCache,
      valueCache,
      ropeTheta: options.ropeTheta || 0,
      rotaryDim: options.rotaryDim !== undefined ? options.rotaryDim : headDim,
      slidingWindow: options.slidingWindow || 0,
      attnScale: options.attnScale,
      qNormWeight: options.qNormWeight || null,
      kNormWeight: options.kNormWeight || null,
      rmsEpsilon: options.rmsEpsilon,
      offsetRMSNorm: options.offsetRMSNorm === true,
    });

    closeArray(await readOutput(renderer, layer), expected, epsilon);
  }
}

export type { TslNode };
