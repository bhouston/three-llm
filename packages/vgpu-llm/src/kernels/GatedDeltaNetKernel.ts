import { allocStorage, makeCompute, uploadStorage, workgroupCount, writeBuffer } from '../gpu/device.js';
import type { Compute, Gpu, StorageBuffer } from '../gpu/device.js';
import { LinearKernel } from './LinearKernel.js';
import type { KernelOptions, QwenDeltaWeights } from '../types.js';

interface GatedDeltaNetOptions extends KernelOptions {
  hiddenSize?: number;
  numKHeads?: number;
  numVHeads?: number;
  keyDim?: number;
  valueDim?: number;
  kernelSize?: number;
  epsilon?: number;
}

/**
 * One-token Gated DeltaNet decode: causal conv, recurrent delta rule, gated RMSNorm.
 */
class GatedDeltaNetKernel {
  hiddenSize: number;
  numKHeads: number;
  numVHeads: number;
  keyDim: number;
  valueDim: number;
  kernelSize: number;
  workgroupSize: number;
  epsilon: number;
  keySize: number;
  valueSize: number;
  convDim: number;
  stateSize: number;
  repeat: number;

  qkv: LinearKernel;
  zProj: LinearKernel;
  bProj: LinearKernel;
  aProj: LinearKernel;
  outProj: LinearKernel;

  convStateBuffer: StorageBuffer;
  convOutBuffer: StorageBuffer;
  queryBuffer: StorageBuffer;
  keyBuffer: StorageBuffer;
  valueBuffer: StorageBuffer;
  recurrentBuffer: StorageBuffer;
  mixedBuffer: StorageBuffer;
  decayBuffer: StorageBuffer;
  betaBuffer: StorageBuffer;
  outputBuffer: StorageBuffer;

  convPass: Compute;
  preparePass: Compute;
  normQKPass: Compute;
  decayPass: Compute;
  deltaPass: Compute;
  normPass: Compute;

  convWorkgroups: number;
  prepareWorkgroups: number;
  normQKWorkgroups: number;
  decayWorkgroups: number;
  deltaWorkgroups: number;
  normWorkgroups: number;

  constructor(gpu: Gpu, inputBuffer: StorageBuffer, weights: QwenDeltaWeights, options: GatedDeltaNetOptions = {}) {
    this.hiddenSize = options.hiddenSize as number;
    this.numKHeads = options.numKHeads as number;
    this.numVHeads = options.numVHeads as number;
    this.keyDim = options.keyDim as number;
    this.valueDim = options.valueDim as number;
    this.kernelSize = options.kernelSize || 4;
    this.workgroupSize = options.workgroupSize || 64;
    this.epsilon = options.epsilon || 1e-6;
    this.keySize = this.numKHeads * this.keyDim;
    this.valueSize = this.numVHeads * this.valueDim;
    this.convDim = this.keySize * 2 + this.valueSize;
    this.stateSize = this.numVHeads * this.keyDim * this.valueDim;
    this.repeat = this.numVHeads / this.numKHeads;

    const name = options.name;
    const wg = this.workgroupSize;

    this.qkv = new LinearKernel(gpu, inputBuffer, weights.qkvWeight, null, this.hiddenSize, this.convDim, {
      name: name ? `${name}QKV` : 'LLMDeltaQKV',
      workgroupSize: wg,
    });
    this.zProj = new LinearKernel(gpu, inputBuffer, weights.zWeight, null, this.hiddenSize, this.valueSize, {
      name: name ? `${name}Z` : 'LLMDeltaZ',
      workgroupSize: wg,
    });
    this.bProj = new LinearKernel(gpu, inputBuffer, weights.bWeight, null, this.hiddenSize, this.numVHeads, {
      name: name ? `${name}B` : 'LLMDeltaB',
      workgroupSize: wg,
    });
    this.aProj = new LinearKernel(gpu, inputBuffer, weights.aWeight, null, this.hiddenSize, this.numVHeads, {
      name: name ? `${name}A` : 'LLMDeltaA',
      workgroupSize: wg,
    });

    this.convStateBuffer = allocStorage(gpu, this.convDim * this.kernelSize);
    this.convOutBuffer = allocStorage(gpu, this.convDim);
    this.queryBuffer = allocStorage(gpu, this.numVHeads * this.keyDim);
    this.keyBuffer = allocStorage(gpu, this.numVHeads * this.keyDim);
    this.valueBuffer = allocStorage(gpu, this.valueSize);
    this.recurrentBuffer = allocStorage(gpu, this.stateSize);
    this.mixedBuffer = allocStorage(gpu, this.valueSize);
    this.decayBuffer = allocStorage(gpu, this.numVHeads);
    this.betaBuffer = allocStorage(gpu, this.numVHeads);

    const normWeightBuffer = uploadStorage(gpu, weights.normWeight, 'read');
    const aLogBuffer = uploadStorage(gpu, weights.aLog, 'read');
    const dtBiasBuffer = uploadStorage(gpu, weights.dtBias, 'read');
    const convWeightBuffer = uploadStorage(gpu, weights.convWeight, 'read');

    this.outProj = new LinearKernel(gpu, this.mixedBuffer, weights.outWeight, null, this.valueSize, this.hiddenSize, {
      name: name ? `${name}Out` : 'LLMDeltaOut',
      workgroupSize: wg,
    });
    this.outputBuffer = this.outProj.outputBuffer;

    // --- conv: depthwise causal convolution over a per-channel ring buffer ---
    this.convPass = makeCompute(
      gpu,
      `
      @group(0) @binding(0) var<storage, read> qkvOut: array<f32>;
      @group(0) @binding(1) var<storage, read> convWeight: array<f32>;
      @group(0) @binding(2) var<storage, read_write> convState: array<f32>;
      @group(0) @binding(3) var<storage, read_write> convOut: array<f32>;

      @compute @workgroup_size(${wg})
      fn cs_main(@builtin(global_invocation_id) id: vec3u) {
        let channel = id.x;
        if (channel >= ${this.convDim}u) { return; }

        let stateOffset = channel * ${this.kernelSize}u;
        let weightOffset = channel * ${this.kernelSize}u;
        let input = qkvOut[channel];

        var sum: f32 = 0.0;
        for (var i: u32 = 1u; i < ${this.kernelSize}u; i = i + 1u) {
          sum = sum + convWeight[weightOffset + i - 1u] * convState[stateOffset + i];
        }
        sum = sum + convWeight[weightOffset + ${this.kernelSize - 1}u] * input;
        convOut[channel] = sum / (1.0 + exp(-sum));

        for (var i: u32 = 0u; i < ${this.kernelSize - 1}u; i = i + 1u) {
          convState[stateOffset + i] = convState[stateOffset + i + 1u];
        }
        convState[stateOffset + ${this.kernelSize - 1}u] = input;
      }
    `,
      {
        label: name ? `${name}Conv` : 'LLMDeltaConv',
        set: {
          qkvOut: this.qkv.outputBuffer,
          convWeight: convWeightBuffer,
          convState: this.convStateBuffer,
          convOut: this.convOutBuffer,
        },
      },
    );
    this.convWorkgroups = workgroupCount(this.convDim, wg);

    // --- prepare: split conv output into per-(v-head) query/key (repeated from k-heads) and value ---
    const prepared = this.numVHeads * this.keyDim;
    this.preparePass = makeCompute(
      gpu,
      `
      @group(0) @binding(0) var<storage, read> convOut: array<f32>;
      @group(0) @binding(1) var<storage, read_write> query: array<f32>;
      @group(0) @binding(2) var<storage, read_write> key: array<f32>;
      @group(0) @binding(3) var<storage, read_write> value: array<f32>;

      @compute @workgroup_size(${wg})
      fn cs_main(@builtin(global_invocation_id) id: vec3u) {
        let index = id.x;

        if (index < ${prepared}u) {
          let vHead = index / ${this.keyDim}u;
          let local = index % ${this.keyDim}u;
          let kHead = vHead / ${this.repeat}u;
          let source = kHead * ${this.keyDim}u + local;
          query[index] = convOut[source];
          key[index] = convOut[${this.keySize}u + source];
        }

        if (index < ${this.valueSize}u) {
          value[index] = convOut[${this.keySize * 2}u + index];
        }
      }
    `,
      {
        label: name ? `${name}Prepare` : 'LLMDeltaPrepare',
        set: { convOut: this.convOutBuffer, query: this.queryBuffer, key: this.keyBuffer, value: this.valueBuffer },
      },
    );
    this.prepareWorkgroups = workgroupCount(Math.max(prepared, this.valueSize), wg);

    // --- normQK: L2-normalize query (scaled by 1/sqrt(keyDim)) and key per v-head ---
    this.normQKPass = makeCompute(
      gpu,
      `
      @group(0) @binding(0) var<storage, read_write> query: array<f32>;
      @group(0) @binding(1) var<storage, read_write> key: array<f32>;

      @compute @workgroup_size(${wg})
      fn cs_main(@builtin(global_invocation_id) id: vec3u) {
        let head = id.x;
        if (head >= ${this.numVHeads}u) { return; }

        let offset = head * ${this.keyDim}u;
        var qSum: f32 = 0.0;
        var kSum: f32 = 0.0;
        for (var i: u32 = 0u; i < ${this.keyDim}u; i = i + 1u) {
          let qv = query[offset + i];
          let kv = key[offset + i];
          qSum = qSum + qv * qv;
          kSum = kSum + kv * kv;
        }

        let qInv = inverseSqrt(qSum + 1e-6) / ${Math.sqrt(this.keyDim)};
        let kInv = inverseSqrt(kSum + 1e-6);

        for (var i: u32 = 0u; i < ${this.keyDim}u; i = i + 1u) {
          query[offset + i] = query[offset + i] * qInv;
          key[offset + i] = key[offset + i] * kInv;
        }
      }
    `,
      { label: name ? `${name}NormQK` : 'LLMDeltaNormQK', set: { query: this.queryBuffer, key: this.keyBuffer } },
    );
    this.normQKWorkgroups = workgroupCount(this.numVHeads, wg);

    // --- decay/beta gates ---
    this.decayPass = makeCompute(
      gpu,
      `
      @group(0) @binding(0) var<storage, read> aOut: array<f32>;
      @group(0) @binding(1) var<storage, read> bOut: array<f32>;
      @group(0) @binding(2) var<storage, read> aLog: array<f32>;
      @group(0) @binding(3) var<storage, read> dtBias: array<f32>;
      @group(0) @binding(4) var<storage, read_write> decay: array<f32>;
      @group(0) @binding(5) var<storage, read_write> beta: array<f32>;

      @compute @workgroup_size(${wg})
      fn cs_main(@builtin(global_invocation_id) id: vec3u) {
        let head = id.x;
        if (head >= ${this.numVHeads}u) { return; }

        let a = aOut[head] + dtBias[head];
        let softplus = select(log(1.0 + exp(a)), a, a > 20.0);
        decay[head] = exp(-exp(aLog[head]) * softplus);
        beta[head] = 1.0 / (1.0 + exp(-bOut[head]));
      }
    `,
      {
        label: name ? `${name}Decay` : 'LLMDeltaDecay',
        set: {
          aOut: this.aProj.outputBuffer,
          bOut: this.bProj.outputBuffer,
          aLog: aLogBuffer,
          dtBias: dtBiasBuffer,
          decay: this.decayBuffer,
          beta: this.betaBuffer,
        },
      },
    );
    this.decayWorkgroups = workgroupCount(this.numVHeads, wg);

    // --- delta rule: recurrent state update + query readout, one thread per (head, value-dim) ---
    const deltaCount = this.numVHeads * this.valueDim;
    this.deltaPass = makeCompute(
      gpu,
      `
      @group(0) @binding(0) var<storage, read> query: array<f32>;
      @group(0) @binding(1) var<storage, read> key: array<f32>;
      @group(0) @binding(2) var<storage, read> value: array<f32>;
      @group(0) @binding(3) var<storage, read> decay: array<f32>;
      @group(0) @binding(4) var<storage, read> beta: array<f32>;
      @group(0) @binding(5) var<storage, read_write> state: array<f32>;
      @group(0) @binding(6) var<storage, read_write> mixed: array<f32>;

      @compute @workgroup_size(${wg})
      fn cs_main(@builtin(global_invocation_id) id: vec3u) {
        let index = id.x;
        if (index >= ${deltaCount}u) { return; }

        let head = index / ${this.valueDim}u;
        let v = index % ${this.valueDim}u;
        let decayH = decay[head];
        let betaH = beta[head];
        let qOff = head * ${this.keyDim}u;
        let stateOff = head * ${this.keyDim}u * ${this.valueDim}u;

        var kvMem: f32 = 0.0;
        for (var k: u32 = 0u; k < ${this.keyDim}u; k = k + 1u) {
          let sIndex = stateOff + k * ${this.valueDim}u + v;
          state[sIndex] = state[sIndex] * decayH;
          kvMem = kvMem + state[sIndex] * key[qOff + k];
        }

        let delta = (value[head * ${this.valueDim}u + v] - kvMem) * betaH;
        var mixedValue: f32 = 0.0;
        for (var k: u32 = 0u; k < ${this.keyDim}u; k = k + 1u) {
          let sIndex = stateOff + k * ${this.valueDim}u + v;
          state[sIndex] = state[sIndex] + key[qOff + k] * delta;
          mixedValue = mixedValue + state[sIndex] * query[qOff + k];
        }

        mixed[index] = mixedValue;
      }
    `,
      {
        label: name ? `${name}Delta` : 'LLMDeltaRule',
        set: {
          query: this.queryBuffer,
          key: this.keyBuffer,
          value: this.valueBuffer,
          decay: this.decayBuffer,
          beta: this.betaBuffer,
          state: this.recurrentBuffer,
          mixed: this.mixedBuffer,
        },
      },
    );
    this.deltaWorkgroups = workgroupCount(deltaCount, wg);

    // --- gated RMSNorm: mixed * rms(mixed) * normWeight * silu(z) ---
    this.normPass = makeCompute(
      gpu,
      `
      @group(0) @binding(0) var<storage, read_write> mixed: array<f32>;
      @group(0) @binding(1) var<storage, read> z: array<f32>;
      @group(0) @binding(2) var<storage, read> normWeight: array<f32>;

      @compute @workgroup_size(${wg})
      fn cs_main(@builtin(global_invocation_id) id: vec3u) {
        let head = id.x;
        if (head >= ${this.numVHeads}u) { return; }

        let offset = head * ${this.valueDim}u;
        var sumSquares: f32 = 0.0;
        for (var i: u32 = 0u; i < ${this.valueDim}u; i = i + 1u) {
          let value = mixed[offset + i];
          sumSquares = sumSquares + value * value;
        }
        let invRms = inverseSqrt(sumSquares / ${this.valueDim}.0 + ${this.epsilon});

        for (var i: u32 = 0u; i < ${this.valueDim}u; i = i + 1u) {
          let index = offset + i;
          let zv = z[index];
          let silu = zv / (1.0 + exp(-zv));
          mixed[index] = mixed[index] * invRms * normWeight[i] * silu;
        }
      }
    `,
      {
        label: name ? `${name}Norm` : 'LLMDeltaGatedNorm',
        set: { mixed: this.mixedBuffer, z: this.zProj.outputBuffer, normWeight: normWeightBuffer },
      },
    );
    this.normWorkgroups = workgroupCount(this.numVHeads, wg);
  }

  reset(): void {
    writeBuffer(this.convStateBuffer, new Float32Array(this.convDim * this.kernelSize));
    writeBuffer(this.recurrentBuffer, new Float32Array(this.stateSize));
  }

  run(): StorageBuffer {
    this.qkv.run();
    this.zProj.run();
    this.bProj.run();
    this.aProj.run();
    this.convPass.dispatch(this.convWorkgroups);
    this.preparePass.dispatch(this.prepareWorkgroups);
    this.normQKPass.dispatch(this.normQKWorkgroups);
    this.decayPass.dispatch(this.decayWorkgroups);
    this.deltaPass.dispatch(this.deltaWorkgroups);
    this.normPass.dispatch(this.normWorkgroups);
    this.outProj.run();
    return this.outputBuffer;
  }
}

export { GatedDeltaNetKernel };
