import { ropeParameters } from '../runtime/rope.js';
import { allocStorage, makeCompute, uploadStorage, workgroupCount, writeBuffer } from '../gpu/device.js';
import type { Compute, Gpu, StorageBuffer } from '../gpu/device.js';
import type { AttentionKernelOptions } from '../types.js';

/**
 * Builds the WGSL body for one head element (query or key), matching the
 * CPU reference's `applyRoPE` + optional packed-head RMSNorm exactly:
 * norm (if `normBinding` given) is applied over the full head, THEN RoPE
 * rotates pairs `(i, i + rotaryDim/2)` of the *normalized* values.
 *
 * Assumes `headOffset: u32`, `localDim: u32`, and `position: u32` are in
 * scope at the call site, and reads the packed QKV buffer `qkv`.
 */
function headValueExpr(
  fnName: string,
  cfg: {
    headDim: number;
    ropeTheta: number;
    rotaryDim: number;
    ropePairCount: number;
    attentionFactor: number;
    rmsEpsilon: number;
    offsetRMSNorm: boolean;
    normBinding: string | null;
  },
): string {
  const { headDim, ropeTheta, rotaryDim, ropePairCount, attentionFactor, rmsEpsilon, offsetRMSNorm, normBinding } = cfg;
  const hasRope = ropeTheta > 0 && rotaryDim > 0;

  if (!normBinding && !hasRope) {
    return `
      fn ${fnName}(headOffset: u32, localDim: u32, tokenPos: u32) -> f32 {
        return qkv[headOffset + localDim];
      }
    `;
  }

  const half = Math.floor(rotaryDim / 2);

  if (!normBinding) {
    // Pure RoPE, no norm: rotate raw QKV values in place.
    return `
      fn ${fnName}(headOffset: u32, localDim: u32, tokenPos: u32) -> f32 {
        let x = qkv[headOffset + localDim];
        if (localDim >= ${rotaryDim}u) { return x; }
        let half = ${half}u;
        let freqIndex = localDim % half;
        let partnerIndex = select(headOffset + localDim - half, headOffset + localDim + half, localDim < half);
        let partnerRaw = qkv[partnerIndex];
        let partner = select(partnerRaw, -partnerRaw, localDim < half);
        let angle = select(0.0, f32(tokenPos) * ropeFreq[freqIndex], freqIndex < ${ropePairCount}u);
        return (x * cos(angle) + partner * sin(angle)) * ${attentionFactor};
      }
    `;
  }

  // RMSNorm over the full head, then (optionally) RoPE on the normalized values.
  const normScaleExpr = (idx: string) => (offsetRMSNorm ? `(${normBinding}[${idx}] + 1.0)` : `${normBinding}[${idx}]`);
  const ropeTail = hasRope
    ? `
        if (localDim >= ${rotaryDim}u) { return x; }
        let half = ${half}u;
        let freqIndex = localDim % half;
        let partnerLocal = select(localDim - half, localDim + half, localDim < half);
        let partnerRaw = qkv[headOffset + partnerLocal];
        let partnerScale = ${normScaleExpr('partnerLocal')};
        let partnerScaled = partnerRaw * invRms * partnerScale;
        let partner = select(partnerScaled, -partnerScaled, localDim < half);
        let angle = select(0.0, f32(tokenPos) * ropeFreq[freqIndex], freqIndex < ${ropePairCount}u);
        return (x * cos(angle) + partner * sin(angle)) * ${attentionFactor};
    `
    : `
        return x;
    `;

  return `
    fn ${fnName}(headOffset: u32, localDim: u32, tokenPos: u32) -> f32 {
      var sumSquares: f32 = 0.0;
      for (var i: u32 = 0u; i < ${headDim}u; i = i + 1u) {
        let value = qkv[headOffset + i];
        sumSquares = sumSquares + value * value;
      }
      let invRms = inverseSqrt(sumSquares / ${headDim}.0 + ${rmsEpsilon});
      let nScale = ${normScaleExpr('localDim')};
      let x = qkv[headOffset + localDim] * invRms * nScale;
      ${ropeTail}
    }
  `;
}

/**
 * One-token causal self-attention for decode.
 *
 * Past keys/values stay in a cache. Each step writes the new K/V, then
 * materializes scaled Q·K scores once per (head, token), then applies the
 * same two-pass softmax-and-value mix as the CPU reference.
 *
 * GPT-2 uses packed QKV with one key/value head per query head. Llama-style
 * models can pass grouped-query (`kvHeadCount`) and rotary (`ropeTheta`)
 * options without changing the packed layout: `[Q, K, V]`.
 */
class AttentionKernel {
  hiddenSize: number;
  headCount: number;
  headDim: number;
  kvHeadCount: number;
  qSize: number;
  kvSize: number;
  maxTokens: number;
  workgroupSize: number;
  slidingWindow: number;
  attnScale: number;
  sharedKV: boolean;

  keyCacheBuffer: StorageBuffer;
  valueCacheBuffer: StorageBuffer;
  scoreBuffer: StorageBuffer;
  queryBuffer: StorageBuffer;
  outputBuffer: StorageBuffer;
  positionBuffer: StorageBuffer;
  private ownsPosition: boolean;

  private copyPass: Compute | null;
  private queryPass: Compute;
  private scorePass: Compute;
  private softmaxPass: Compute;
  private copyWorkgroups: number;
  private queryWorkgroups: number;
  private scoreWorkgroups: number;
  private softmaxWorkgroups: number;

  constructor(
    gpu: Gpu,
    qkvBuffer: StorageBuffer,
    hiddenSize: number,
    headCount: number,
    maxTokens: number,
    options: AttentionKernelOptions = {},
  ) {
    this.hiddenSize = hiddenSize;
    this.headCount = headCount;
    this.headDim = options.headDim || hiddenSize / headCount;
    this.kvHeadCount = options.kvHeadCount || headCount;
    this.qSize = this.headCount * this.headDim;
    this.kvSize = this.kvHeadCount * this.headDim;
    this.maxTokens = maxTokens;
    this.workgroupSize = options.workgroupSize || 64;
    this.slidingWindow = options.slidingWindow || 0;
    this.attnScale = options.attnScale !== undefined ? options.attnScale : 1 / Math.sqrt(this.headDim);
    this.sharedKV = options.sharedAttention !== undefined;

    const ropeTheta = options.ropeTheta || 0;
    const rotaryDim = options.rotaryDim !== undefined ? options.rotaryDim : this.headDim;
    const ropeFreqDim = options.ropeFreqDim || rotaryDim;
    const ropePairCount = options.ropePairCount !== undefined ? options.ropePairCount : rotaryDim / 2;
    if (!Number.isInteger(rotaryDim) || rotaryDim < 0 || rotaryDim > this.headDim || rotaryDim % 2 !== 0)
      throw new Error('Invalid rotary dimension.');
    const rope =
      ropeTheta > 0 && rotaryDim > 0
        ? ropeParameters(ropeFreqDim, ropeTheta, options.ropeScaling ?? options.yarn)
        : null;
    const ropeBuffer = rope ? uploadStorage(gpu, rope.invFreq) : null;
    const rmsEpsilon = options.rmsEpsilon || 1e-6;
    const offsetRMSNorm = options.offsetRMSNorm === true;
    const vNorm = options.vNorm === true;
    const wg = this.workgroupSize;
    const name = options.name;

    const qNormBuffer = options.qNormWeight ? uploadStorage(gpu, options.qNormWeight, 'read') : null;
    const kNormBuffer = options.kNormWeight ? uploadStorage(gpu, options.kNormWeight, 'read') : null;

    if (options.sharedAttention) {
      this.keyCacheBuffer = options.sharedAttention.keyCacheBuffer;
      this.valueCacheBuffer = options.sharedAttention.valueCacheBuffer;
    } else {
      this.keyCacheBuffer = allocStorage(gpu, this.kvSize * maxTokens);
      this.valueCacheBuffer = allocStorage(gpu, this.kvSize * maxTokens);
    }

    this.scoreBuffer = allocStorage(gpu, headCount * maxTokens);
    this.queryBuffer = allocStorage(gpu, this.qSize);
    this.outputBuffer = allocStorage(gpu, this.qSize);

    if (options.positionBuffer) {
      this.positionBuffer = options.positionBuffer;
      this.ownsPosition = false;
    } else {
      this.positionBuffer = allocStorage(gpu, 1);
      this.ownsPosition = true;
    }

    const headValueCommon = {
      headDim: this.headDim,
      ropeTheta,
      rotaryDim,
      ropePairCount,
      attentionFactor: rope?.attentionFactor ?? 1,
      rmsEpsilon,
      offsetRMSNorm,
    };
    const keyValueFn = headValueExpr('key_value', { ...headValueCommon, normBinding: kNormBuffer ? 'kNorm' : null });
    const queryValueFn = headValueExpr('query_value', {
      ...headValueCommon,
      normBinding: qNormBuffer ? 'qNorm' : null,
    });

    // --- copy: write this step's (rotated/normalized) key and value into the cache ---
    if (this.sharedKV) {
      this.copyPass = null;
      this.copyWorkgroups = 0;
    } else {
      const vNormBlock = vNorm
        ? `
          let valueHead = (dim / ${this.headDim}u) * ${this.headDim}u + ${this.qSize + this.kvSize}u;
          var vSumSquares: f32 = 0.0;
          for (var i: u32 = 0u; i < ${this.headDim}u; i = i + 1u) {
            let sample = qkv[valueHead + i];
            vSumSquares = vSumSquares + sample * sample;
          }
          value = value * inverseSqrt(vSumSquares / ${this.headDim}.0 + ${rmsEpsilon});
        `
        : '';

      const source = `
        ${keyValueFn}

        @group(0) @binding(0) var<storage, read> qkv: array<f32>;
        @group(0) @binding(1) var<storage, read> position: array<u32>;
        @group(0) @binding(2) var<storage, read_write> keyCache: array<f32>;
        @group(0) @binding(3) var<storage, read_write> valueCache: array<f32>;
        ${kNormBuffer ? '@group(0) @binding(4) var<storage, read> kNorm: array<f32>;' : ''}
        ${ropeBuffer ? '@group(0) @binding(5) var<storage, read> ropeFreq: array<f32>;' : ''}

        @compute @workgroup_size(${wg})
        fn cs_main(@builtin(global_invocation_id) id: vec3u) {
          let dim = id.x;
          if (dim >= ${this.kvSize}u) { return; }

          let pos = position[0];
          let cacheOffset = pos * ${this.kvSize}u + dim;
          let headOffset = (dim / ${this.headDim}u) * ${this.headDim}u + ${this.qSize}u;
          let localDim = dim % ${this.headDim}u;
          let key = key_value(headOffset, localDim, pos);
          var value = qkv[${this.qSize + this.kvSize}u + dim];
          ${vNormBlock}

          keyCache[cacheOffset] = key;
          valueCache[cacheOffset] = value;
        }
      `;

      const set: Record<string, unknown> = {
        qkv: qkvBuffer,
        position: this.positionBuffer,
        keyCache: this.keyCacheBuffer,
        valueCache: this.valueCacheBuffer,
      };
      if (kNormBuffer) set.kNorm = kNormBuffer;
      if (ropeBuffer) set.ropeFreq = ropeBuffer;

      this.copyPass = makeCompute(gpu, source, { label: name ? `${name}CopyKV` : 'LLMAttentionCopyKV', set });
      this.copyWorkgroups = workgroupCount(this.kvSize, wg);
    }

    // --- query: materialize this step's (rotated/normalized) query vector ---
    {
      const set: Record<string, unknown> = { qkv: qkvBuffer, position: this.positionBuffer, query: this.queryBuffer };
      if (qNormBuffer) set.qNorm = qNormBuffer;
      if (ropeBuffer) set.ropeFreq = ropeBuffer;

      const source = `
        ${queryValueFn}

        @group(0) @binding(0) var<storage, read> qkv: array<f32>;
        @group(0) @binding(1) var<storage, read> position: array<u32>;
        @group(0) @binding(2) var<storage, read_write> query: array<f32>;
        ${qNormBuffer ? '@group(0) @binding(3) var<storage, read> qNorm: array<f32>;' : ''}
        ${ropeBuffer ? '@group(0) @binding(4) var<storage, read> ropeFreq: array<f32>;' : ''}

        @compute @workgroup_size(${wg})
        fn cs_main(@builtin(global_invocation_id) id: vec3u) {
          let dim = id.x;
          if (dim >= ${this.qSize}u) { return; }

          let pos = position[0];
          let headOffset = (dim / ${this.headDim}u) * ${this.headDim}u;
          let localDim = dim % ${this.headDim}u;
          query[dim] = query_value(headOffset, localDim, pos);
        }
      `;

      this.queryPass = makeCompute(gpu, source, { label: name ? `${name}Query` : 'LLMAttentionQuery', set });
      this.queryWorkgroups = workgroupCount(this.qSize, wg);
    }

    // --- score: scaled dot product of query against the causal (or sliding-window) key range ---
    {
      const scoreCount = headCount * maxTokens;
      const windowStartExpr =
        this.slidingWindow > 0
          ? `select(pos + 1u - ${this.slidingWindow}u, 0u, pos + 1u < ${this.slidingWindow}u)`
          : `0u`;

      const source = `
        @group(0) @binding(0) var<storage, read> query: array<f32>;
        @group(0) @binding(1) var<storage, read> keyCache: array<f32>;
        @group(0) @binding(2) var<storage, read_write> scores: array<f32>;
        @group(0) @binding(3) var<storage, read> position: array<u32>;

        @compute @workgroup_size(${wg})
        fn cs_main(@builtin(global_invocation_id) id: vec3u) {
          let index = id.x;
          if (index >= ${scoreCount}u) { return; }

          let pos = position[0];
          let head = index / ${maxTokens}u;
          let token = index % ${maxTokens}u;
          let windowStart = ${windowStartExpr};
          if (token < windowStart || token >= pos + 1u) { return; }

          let qOffset = head * ${this.headDim}u;
          let kvHead = (head * ${this.kvHeadCount}u) / ${headCount}u;
          let kvOffset = kvHead * ${this.headDim}u;

          var score: f32 = 0.0;
          for (var i: u32 = 0u; i < ${this.headDim}u; i = i + 1u) {
            score = score + query[qOffset + i] * keyCache[token * ${this.kvSize}u + kvOffset + i];
          }
          scores[index] = score * ${this.attnScale};
        }
      `;

      this.scorePass = makeCompute(gpu, source, {
        label: name ? `${name}Scores` : 'LLMAttentionScores',
        set: {
          query: this.queryBuffer,
          keyCache: this.keyCacheBuffer,
          scores: this.scoreBuffer,
          position: this.positionBuffer,
        },
      });
      this.scoreWorkgroups = workgroupCount(scoreCount, wg);
    }

    // --- softmax + value mix (+ optional output gate) ---
    {
      const gateBuffer = options.gateBuffer || null;
      const windowStartExpr =
        this.slidingWindow > 0
          ? `select(pos + 1u - ${this.slidingWindow}u, 0u, pos + 1u < ${this.slidingWindow}u)`
          : `0u`;
      const outputExpr = gateBuffer ? `(value / denominator) * (1.0 / (1.0 + exp(-gate[dim])))` : `value / denominator`;

      const source = `
        @group(0) @binding(0) var<storage, read> scores: array<f32>;
        @group(0) @binding(1) var<storage, read> valueCache: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;
        @group(0) @binding(3) var<storage, read> position: array<u32>;
        ${gateBuffer ? '@group(0) @binding(4) var<storage, read> gate: array<f32>;' : ''}

        @compute @workgroup_size(${wg})
        fn cs_main(@builtin(global_invocation_id) id: vec3u) {
          let dim = id.x;
          if (dim >= ${this.qSize}u) { return; }

          let pos = position[0];
          let head = dim / ${this.headDim}u;
          let localDim = dim % ${this.headDim}u;
          let kvHead = (head * ${this.kvHeadCount}u) / ${headCount}u;
          let kvOffset = kvHead * ${this.headDim}u;
          let scoreOffset = head * ${maxTokens}u;
          let windowStart = ${windowStartExpr};

          var maxScore: f32 = -3.4028234663852886e38;
          for (var token = windowStart; token < pos + 1u; token = token + 1u) {
            let score = scores[scoreOffset + token];
            if (score > maxScore) { maxScore = score; }
          }

          var denominator: f32 = 0.0;
          var value: f32 = 0.0;
          for (var token = windowStart; token < pos + 1u; token = token + 1u) {
            let probability = exp(scores[scoreOffset + token] - maxScore);
            let v = valueCache[token * ${this.kvSize}u + kvOffset + localDim];
            denominator = denominator + probability;
            value = value + probability * v;
          }

          output[dim] = ${outputExpr};
        }
      `;

      const set: Record<string, unknown> = {
        scores: this.scoreBuffer,
        valueCache: this.valueCacheBuffer,
        output: this.outputBuffer,
        position: this.positionBuffer,
      };
      if (gateBuffer) set.gate = gateBuffer;

      this.softmaxPass = makeCompute(gpu, source, { label: name || 'LLMAttention', set });
      this.softmaxWorkgroups = workgroupCount(this.qSize, wg);
    }
  }

  setPosition(position: number): void {
    writeBuffer(this.positionBuffer, new Uint32Array([position]));
  }

  reset(): void {
    if (this.sharedKV) return;
    writeBuffer(this.keyCacheBuffer, new Float32Array(this.kvSize * this.maxTokens));
    writeBuffer(this.valueCacheBuffer, new Float32Array(this.kvSize * this.maxTokens));
  }

  run(position: number): StorageBuffer {
    if (this.ownsPosition) this.setPosition(position);
    if (this.copyPass) this.copyPass.dispatch(this.copyWorkgroups);
    this.queryPass.dispatch(this.queryWorkgroups);
    this.scorePass.dispatch(this.scoreWorkgroups);
    this.softmaxPass.dispatch(this.softmaxWorkgroups);
    return this.outputBuffer;
  }
}

export { AttentionKernel };
