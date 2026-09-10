import {
  allocStorage,
  gpuMemoryBytes,
  makeCompute,
  requireShaderF16,
  uploadWeightStorage,
  workgroupCount,
  writeBuffer,
  withComputeBatch,
} from '../gpu/device.js';
import type { Compute, Gpu, StorageBuffer } from '../gpu/device.js';
import { generateAsync } from '../runtime/generate.js';
import { AddKernel } from '../kernels/AddKernel.js';
import { AttentionKernel } from '../kernels/AttentionKernel.js';
import { ConcatKernel } from '../kernels/ConcatKernel.js';
import { GatedDeltaNetKernel } from '../kernels/GatedDeltaNetKernel.js';
import { GatedMLPKernel } from '../kernels/GatedMLPKernel.js';
import { LinearKernel } from '../kernels/LinearKernel.js';
import {
  createChunkedLogitLayers,
  createLogitSampler,
  LogitSampler,
  readChunkedLogits,
} from '../kernels/LogitsKernel.js';
import type { LogitChunk } from '../kernels/LogitsKernel.js';
import { RMSNormKernel } from '../kernels/RMSNormKernel.js';
import { SplitHeadGateKernel } from '../kernels/SplitHeadGateKernel.js';
import { QwenWeights } from './QwenWeights.js';
import type {
  GenerateOptions,
  GenerationResult,
  LoaderOptions,
  Precision,
  RunnerOptions,
  SampleOptions,
} from '../types.js';

interface FullAttentionMixer {
  qGate: LinearKernel;
  split: SplitHeadGateKernel;
  kv: LinearKernel;
  packed: ConcatKernel;
  attention: AttentionKernel;
  attnProj: LinearKernel;
  outputBuffer: StorageBuffer;
  run: () => void;
}

type QwenMixer = GatedDeltaNetKernel | FullAttentionMixer;

interface QwenLayer {
  ln1: RMSNormKernel;
  mixer: QwenMixer;
  addAttention: AddKernel;
  ln2: RMSNormKernel;
  mlp: GatedMLPKernel;
  addMLP: AddKernel;
  layerType?: string;
}

/**
 * Qwen3.5 text generation runner: alternating full causal attention and
 * Gated DeltaNet linear-attention layers, built on `vgpu` compute passes.
 */
class QwenGpuRunner {
  gpu: Gpu;
  weights: QwenWeights;
  maxTokens: number;
  workgroupSize: number;
  logitChunkSize: number;
  prefillChunkSize: number;
  batchCompute: boolean;
  hiddenSize: number;
  precision: Precision;
  /** Approximate total GPU storage bytes allocated for this runner (weights, KV caches, activations). */
  gpuMemoryBytes: number;

  embeddingBuffer: StorageBuffer;
  embeddingScratch: Float32Array;
  positionBuffer: StorageBuffer;
  prefillCursorBuffer: StorageBuffer;
  prefillEmbeddingBuffer: StorageBuffer;
  prefillEmbeddingScratch: Float32Array;

  private prefillCopyPass: Compute;
  private prefillAdvancePass: Compute;
  private prefillCopyWorkgroups: number;

  layers: QwenLayer[];
  finalNorm: RMSNormKernel;
  logits: LogitChunk[];
  logitSampler: LogitSampler;
  _cacheTokens?: number[];
  _cacheLogits?: Float32Array | null;

  constructor(gpu: Gpu, weights: QwenWeights, options: RunnerOptions = {}) {
    this.gpu = gpu;
    this.weights = weights;
    this.maxTokens = Math.min(options.maxTokens || weights.contextLimit(), weights.contextLimit());
    this.workgroupSize = options.workgroupSize || 64;
    this.logitChunkSize = options.logitChunkSize || 8192;
    this.batchCompute = options.batchCompute !== false;
    this.prefillChunkSize = options.prefillChunkSize || 32;
    this.hiddenSize = weights.hiddenSize;
    this.precision = options.precision || 'fp32';
    if (this.precision === 'fp16') requireShaderF16(gpu, 'QwenGpuRunner');

    this.embeddingScratch = new Float32Array(this.hiddenSize);
    this.embeddingBuffer = allocStorage(gpu, this.hiddenSize);
    this.positionBuffer = allocStorage(gpu, 1);
    this.prefillCursorBuffer = allocStorage(gpu, 1);
    this.prefillEmbeddingScratch = new Float32Array(this.prefillChunkSize * this.hiddenSize);
    this.prefillEmbeddingBuffer = allocStorage(gpu, this.prefillEmbeddingScratch.length);

    const wg = this.workgroupSize;
    this.prefillCopyPass = makeCompute(
      gpu,
      `
        @group(0) @binding(0) var<storage, read> prefillEmbeddings: array<f32>;
        @group(0) @binding(1) var<storage, read> cursor: array<u32>;
        @group(0) @binding(2) var<storage, read_write> embedding: array<f32>;

        @compute @workgroup_size(${wg})
        fn cs_main(@builtin(global_invocation_id) id: vec3u) {
          let dim = id.x;
          if (dim >= ${this.hiddenSize}u) { return; }
          let offset = cursor[0] * ${this.hiddenSize}u + dim;
          embedding[dim] = prefillEmbeddings[offset];
        }
      `,
      {
        label: 'QwenPrefillCopy',
        set: {
          prefillEmbeddings: this.prefillEmbeddingBuffer,
          cursor: this.prefillCursorBuffer,
          embedding: this.embeddingBuffer,
        },
      },
    );
    this.prefillCopyWorkgroups = workgroupCount(this.hiddenSize, wg);

    this.prefillAdvancePass = makeCompute(
      gpu,
      `
        @group(0) @binding(0) var<storage, read_write> cursor: array<u32>;
        @group(0) @binding(1) var<storage, read_write> position: array<u32>;

        @compute @workgroup_size(1)
        fn cs_main() {
          cursor[0] = cursor[0] + 1u;
          position[0] = position[0] + 1u;
        }
      `,
      { label: 'QwenPrefillAdvance', set: { cursor: this.prefillCursorBuffer, position: this.positionBuffer } },
    );

    this.layers = [];

    let currentBuffer = this.embeddingBuffer;

    for (let i = 0; i < weights.layerCount; i++) {
      const block = weights.block(i);
      const name = `QwenLayer${i}`;
      const ln1 = new RMSNormKernel(
        gpu,
        currentBuffer,
        uploadWeightStorage(gpu, block.ln1Weight!, this.precision),
        this.hiddenSize,
        {
          epsilon: weights.rmsNormEps,
          offsetWeight: true,
          name: `${name}LN1`,
          workgroupSize: this.workgroupSize,
          precision: this.precision,
        },
      );

      let mixer: QwenMixer;

      if (block.layerType === 'linear_attention') {
        mixer = new GatedDeltaNetKernel(gpu, ln1.outputBuffer, block.delta!, {
          name: `${name}Delta`,
          hiddenSize: this.hiddenSize,
          numKHeads: weights.linearKeyHeads,
          numVHeads: weights.linearValueHeads,
          keyDim: weights.linearKeyDim,
          valueDim: weights.linearValueDim,
          kernelSize: weights.linearConvKernel,
          epsilon: weights.rmsNormEps,
          workgroupSize: this.workgroupSize,
          precision: this.precision,
        });
      } else {
        const qGate = new LinearKernel(
          gpu,
          ln1.outputBuffer,
          block.qGateWeight!,
          null,
          this.hiddenSize,
          weights.qSize * 2,
          {
            name: `${name}QGate`,
            workgroupSize: this.workgroupSize,
            precision: this.precision,
          },
        );
        const split = new SplitHeadGateKernel(gpu, qGate.outputBuffer, weights.headCount, weights.headDim, {
          name: `${name}Split`,
          workgroupSize: this.workgroupSize,
        });
        const kv = new LinearKernel(
          gpu,
          ln1.outputBuffer,
          block.attnQKVWeight,
          null,
          this.hiddenSize,
          2 * weights.kvSize,
          {
            name: `${name}KV`,
            workgroupSize: this.workgroupSize,
            precision: this.precision,
          },
        );
        const packed = new ConcatKernel(
          gpu,
          [
            { buffer: split.queryBuffer, size: weights.qSize },
            { buffer: kv.outputBuffer, size: 2 * weights.kvSize },
          ],
          { name: `${name}Pack`, workgroupSize: this.workgroupSize },
        );
        const attention = new AttentionKernel(
          gpu,
          packed.outputBuffer,
          this.hiddenSize,
          weights.headCount,
          this.maxTokens,
          {
            name: `${name}Attention`,
            workgroupSize: this.workgroupSize,
            headDim: weights.headDim,
            kvHeadCount: weights.kvHeadCount,
            ropeTheta: weights.ropeTheta,
            rotaryDim: weights.rotaryDim,
            attnScale: weights.attnScale,
            qNormWeight: block.qNormWeight,
            kNormWeight: block.kNormWeight,
            rmsEpsilon: weights.rmsNormEps,
            offsetRMSNorm: true,
            gateBuffer: split.gateBuffer,
            positionBuffer: this.positionBuffer,
          },
        );
        const attnProj = new LinearKernel(
          gpu,
          attention.outputBuffer,
          block.attnProjWeight,
          null,
          weights.qSize,
          this.hiddenSize,
          {
            name: `${name}AttnProj`,
            workgroupSize: this.workgroupSize,
            precision: this.precision,
          },
        );

        mixer = {
          qGate,
          split,
          kv,
          packed,
          attention,
          attnProj,
          outputBuffer: attnProj.outputBuffer,
          run: () => {
            qGate.run();
            split.run();
            kv.run();
            packed.run();
            attention.run(0);
            attnProj.run();
          },
        };
      }

      const addAttention = new AddKernel(gpu, currentBuffer, mixer.outputBuffer, this.hiddenSize, {
        name: `${name}AddAttention`,
        workgroupSize: this.workgroupSize,
      });
      const ln2 = new RMSNormKernel(
        gpu,
        addAttention.outputBuffer,
        uploadWeightStorage(gpu, block.ln2Weight!, this.precision),
        this.hiddenSize,
        {
          epsilon: weights.rmsNormEps,
          offsetWeight: true,
          name: `${name}LN2`,
          workgroupSize: this.workgroupSize,
          precision: this.precision,
        },
      );
      const mlp = new GatedMLPKernel(
        gpu,
        ln2.outputBuffer,
        block.mlpGateWeight!,
        block.mlpUpWeight!,
        block.mlpDownWeight!,
        this.hiddenSize,
        weights.innerSize,
        {
          name: `${name}MLP`,
          workgroupSize: this.workgroupSize,
          activation: weights.mlpActivation,
          precision: this.precision,
        },
      );
      const addMLP = new AddKernel(gpu, addAttention.outputBuffer, mlp.outputBuffer, this.hiddenSize, {
        name: `${name}AddMLP`,
        workgroupSize: this.workgroupSize,
      });

      this.layers.push({ ln1, mixer, addAttention, ln2, mlp, addMLP, layerType: block.layerType });
      currentBuffer = addMLP.outputBuffer;
    }

    this.finalNorm = new RMSNormKernel(
      gpu,
      currentBuffer,
      uploadWeightStorage(gpu, weights.outputNormWeight!, this.precision),
      this.hiddenSize,
      {
        epsilon: weights.rmsNormEps,
        offsetWeight: true,
        name: 'QwenFinalNorm',
        workgroupSize: this.workgroupSize,
        precision: this.precision,
      },
    );
    this.logits = createChunkedLogitLayers(
      gpu,
      this.finalNorm.outputBuffer,
      weights,
      this.logitChunkSize,
      'QwenLogits',
      this.precision,
    );
    weights.logitWeight = null;
    this.logitSampler = createLogitSampler(gpu, this.logits, {
      candidateCount: options.logitCandidateCount || 8,
      name: 'QwenLogits',
    });

    this.weights.releaseCheckpointTensors();
    this.weights.releaseUnpackedWeightArrays();

    this.gpuMemoryBytes = gpuMemoryBytes(gpu);
  }

  static async fromURL(gpu: Gpu, baseURL: string, options: LoaderOptions & RunnerOptions = {}) {
    return new this(gpu, await QwenWeights.fromURL(baseURL, options), options);
  }

  private runLayer(layer: QwenLayer): void {
    layer.ln1.run();
    layer.mixer.run();
    layer.addAttention.run();
    layer.ln2.run();
    layer.mlp.run();
    layer.addMLP.run();
  }

  private runForward(includeLogits: boolean): void {
    for (const layer of this.layers) this.runLayer(layer);

    if (includeLogits) {
      this.finalNorm.run();
      for (const chunk of this.logits) chunk.layer.run();
    }
  }

  setPosition(position: number): void {
    writeBuffer(this.positionBuffer, new Uint32Array([position]));
  }

  computeToken(tokenId: number, position: number, computeLogits = true, sampleCandidateCount = 0): void {
    this.weights.embedding(tokenId, position, this.embeddingScratch);
    writeBuffer(this.embeddingBuffer, this.embeddingScratch);
    this.setPosition(position);

    const work = () => {
      this.runForward(computeLogits);
      if (computeLogits && sampleCandidateCount > 0) this.logitSampler.run(sampleCandidateCount);
    };
    if (this.batchCompute) withComputeBatch(this.gpu, work);
    else work();
  }

  async prefillTokens(
    inputTokens: number[],
    start: number,
    end: number,
    onProgress?: (n: number) => void | Promise<void>,
  ): Promise<void> {
    for (let offset = start; offset < end; offset += this.prefillChunkSize) {
      const count = Math.min(this.prefillChunkSize, end - offset);

      for (let i = 0; i < count; i++) {
        this.weights.embedding(
          inputTokens[offset + i]!,
          offset + i,
          this.prefillEmbeddingScratch.subarray(i * this.hiddenSize, (i + 1) * this.hiddenSize),
        );
      }

      writeBuffer(this.prefillEmbeddingBuffer, this.prefillEmbeddingScratch);
      writeBuffer(this.prefillCursorBuffer, new Uint32Array([0]));
      this.setPosition(offset);

      const work = () => {
        for (let i = 0; i < count; i++) {
          this.prefillCopyPass.dispatch(this.prefillCopyWorkgroups);
          this.runForward(false);
          this.prefillAdvancePass.dispatch(1);
        }
      };
      if (this.batchCompute) withComputeBatch(this.gpu, work);
      else work();

      if (onProgress) await onProgress(offset + count);
    }
  }

  async readLogits(): Promise<Float32Array> {
    return readChunkedLogits(this.logits, this.weights.vocabSize);
  }

  async sampleToken(candidateCount: number, options: SampleOptions): Promise<number> {
    return this.logitSampler.sampleToken(candidateCount, options, this.batchCompute);
  }

  resetCaches(): void {
    for (const layer of this.layers) {
      if (layer.layerType === 'linear_attention') (layer.mixer as GatedDeltaNetKernel).reset();
      else (layer.mixer as FullAttentionMixer).attention.reset();
    }
  }

  resetCache(): void {
    this._cacheTokens = [];
    this._cacheLogits = null;
    this.resetCaches();
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerationResult> {
    return generateAsync(this, prompt, options, {
      rewindable: false,
      resetCache: () => this.resetCache(),
      computeToken: (tokenId, position, computeLogits, sampleCandidateCount) =>
        this.computeToken(tokenId, position, computeLogits, sampleCandidateCount),
      prefillTokens: (inputTokens, start, end, onProgress) => this.prefillTokens(inputTokens, start, end, onProgress),
      readLogits: () => this.readLogits(),
      sampleToken: (candidateCount, sampleOptions) => this.sampleToken(candidateCount, sampleOptions),
      maxGpuCandidateCount: this.logitSampler.candidateCount,
    });
  }
}

export { QwenGpuRunner };
