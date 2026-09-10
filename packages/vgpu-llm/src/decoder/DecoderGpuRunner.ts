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
import { DecoderWeights } from './DecoderWeights.js';
import { generateAsync } from '../runtime/generate.js';
import { logitSoftcap } from '../runtime/math.js';
import { AddKernel } from '../kernels/AddKernel.js';
import { AttentionKernel } from '../kernels/AttentionKernel.js';
import { GatedMLPKernel } from '../kernels/GatedMLPKernel.js';
import { LinearKernel } from '../kernels/LinearKernel.js';
import {
  createChunkedLogitLayers,
  createLogitSampler,
  LogitSampler,
  readChunkedLogits,
} from '../kernels/LogitsKernel.js';
import type { LogitChunk } from '../kernels/LogitsKernel.js';
import { MLPKernel } from '../kernels/MLPKernel.js';
import { NormalizeKernel } from '../kernels/NormalizeKernel.js';
import { RMSNormKernel } from '../kernels/RMSNormKernel.js';
import type {
  DecoderBlock,
  DecoderRecipe,
  GenerateOptions,
  GenerationResult,
  LoaderOptions,
  Precision,
  RunnerOptions,
  SampleOptions,
} from '../types.js';

type NormKernel = NormalizeKernel | RMSNormKernel;
type MlpKernel = MLPKernel | GatedMLPKernel;

interface DecoderLayer {
  kind: 'parallel' | 'gemma' | 'sequential';
  outputBuffer: StorageBuffer;
  attention: AttentionKernel;
  ln?: NormKernel;
  ln1?: NormKernel;
  ln2?: NormKernel;
  qkv: LinearKernel;
  attnProj: LinearKernel;
  mlp: MlpKernel;
  addAttention: AddKernel;
  addMLP: AddKernel;
  postAttnNorm?: NormKernel;
  preMlp?: NormKernel;
  postMlpNorm?: NormKernel;
}

/**
 * Parameterized GPU decoder for GPT-2, Llama-family, Phi, and Gemma 3, built
 * on `vgpu` compute passes.
 */
class DecoderGpuRunner {
  gpu: Gpu;
  weights: DecoderWeights;
  recipe: DecoderRecipe;
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

  layers: DecoderLayer[];
  finalNorm: NormKernel;
  logits: LogitChunk[];
  logitSampler: LogitSampler;
  _cacheTokens?: number[];
  _cacheLogits?: Float32Array | null;

  constructor(gpu: Gpu, weights: DecoderWeights, options: RunnerOptions = {}) {
    this.gpu = gpu;
    this.weights = weights;
    this.recipe = weights.recipe;
    this.maxTokens = Math.min(options.maxTokens || weights.contextLimit(), weights.contextLimit());
    this.workgroupSize = options.workgroupSize || 64;
    this.logitChunkSize = options.logitChunkSize || 8192;
    this.batchCompute = options.batchCompute !== false;
    this.prefillChunkSize = options.prefillChunkSize || 32;
    this.hiddenSize = weights.hiddenSize;
    this.precision = options.precision || 'fp32';
    if (this.precision === 'fp16') requireShaderF16(gpu, `${weights.architecture}DecoderGpuRunner`);

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
        label: `${weights.architecture}PrefillCopy`,
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
      {
        label: `${weights.architecture}PrefillAdvance`,
        set: { cursor: this.prefillCursorBuffer, position: this.positionBuffer },
      },
    );

    this.layers = [];

    let currentBuffer = this.embeddingBuffer;

    for (let i = 0; i < weights.layerCount; i++) {
      const built = this.buildLayer(weights.block(i), i, currentBuffer);
      this.layers.push(built);
      currentBuffer = built.outputBuffer;
    }

    this.finalNorm = this.buildFinalNorm(currentBuffer);
    this.logits = createChunkedLogitLayers(
      gpu,
      this.finalNorm.outputBuffer,
      weights,
      this.logitChunkSize,
      `${weights.architecture}Logits`,
      this.precision,
    );
    weights.logitWeight = null;
    this.logitSampler = createLogitSampler(gpu, this.logits, {
      candidateCount: options.logitCandidateCount || 8,
      logitSoftcap: this.recipe.finalLogitSoftcap,
      name: `${weights.architecture}Logits`,
    });

    // Every weight this decoder needs has already been uploaded to the GPU
    // (each kernel's constructor writes its buffers eagerly), so the CPU
    // shadow copies can be freed immediately.
    this.weights.releaseCheckpointTensors();
    this.weights.releaseUnpackedWeightArrays();

    this.gpuMemoryBytes = gpuMemoryBytes(gpu);
  }

  static async fromURL(gpu: Gpu, baseURL: string, options?: LoaderOptions & RunnerOptions) {
    return new this(gpu, await DecoderWeights.fromURL(baseURL, options), options);
  }

  buildNorm(
    inputBuffer: StorageBuffer,
    weight: Float32Array | null | undefined,
    bias: Float32Array | null | undefined,
    name: string,
  ): NormKernel {
    if (this.recipe.norm === 'layer_norm') {
      const weightBuffer = uploadWeightStorage(this.gpu, weight!, this.precision, 'read');
      const biasBuffer = uploadWeightStorage(
        this.gpu,
        bias ?? new Float32Array(this.hiddenSize),
        this.precision,
        'read',
      );
      return new NormalizeKernel(this.gpu, inputBuffer, weightBuffer, biasBuffer, this.hiddenSize, {
        epsilon: this.recipe.normEps,
        name,
        workgroupSize: this.workgroupSize,
        precision: this.precision,
      });
    }

    const weightBuffer = uploadWeightStorage(this.gpu, weight!, this.precision, 'read');
    return new RMSNormKernel(this.gpu, inputBuffer, weightBuffer, this.hiddenSize, {
      epsilon: this.recipe.normEps,
      offsetWeight: this.recipe.norm === 'rms_offset',
      name,
      workgroupSize: this.workgroupSize,
      precision: this.precision,
    });
  }

  buildAttention(qkvBuffer: StorageBuffer, block: DecoderBlock, name: string): AttentionKernel {
    const { weights, recipe } = this;

    return new AttentionKernel(this.gpu, qkvBuffer, this.hiddenSize, weights.headCount, this.maxTokens, {
      name: `${name}Attention`,
      workgroupSize: this.workgroupSize,
      headDim: weights.headDim,
      kvHeadCount: weights.kvHeadCount,
      ropeTheta: block.ropeTheta !== undefined ? block.ropeTheta : recipe.ropeTheta,
      rotaryDim: recipe.rotaryDim ?? weights.headDim,
      yarn: block.yarn,
      ropeScaling: block.ropeScaling,
      slidingWindow: block.slidingWindow || 0,
      attnScale: recipe.attnScale,
      qNormWeight: block.qNormWeight,
      kNormWeight: block.kNormWeight,
      rmsEpsilon: recipe.normEps,
      offsetRMSNorm: recipe.norm === 'rms_offset',
      positionBuffer: this.positionBuffer,
    });
  }

  buildLayer(block: DecoderBlock, index: number, residualBuffer: StorageBuffer): DecoderLayer {
    const { weights, recipe } = this;
    const name = `${weights.architecture}Layer${index}`;

    if (recipe.residual === 'parallel') {
      const ln = this.buildNorm(residualBuffer, block.lnWeight, block.lnBias, `${name}LN`);
      const qkv = new LinearKernel(
        this.gpu,
        ln.outputBuffer,
        block.attnQKVWeight,
        block.attnQKVBias ?? null,
        this.hiddenSize,
        weights.qSize + 2 * weights.kvSize,
        {
          name: `${name}QKV`,
          workgroupSize: this.workgroupSize,
          precision: this.precision,
        },
      );
      const attention = this.buildAttention(qkv.outputBuffer, block, name);
      const attnProj = new LinearKernel(
        this.gpu,
        attention.outputBuffer,
        block.attnProjWeight,
        block.attnProjBias ?? null,
        weights.qSize,
        this.hiddenSize,
        {
          name: `${name}AttnProj`,
          workgroupSize: this.workgroupSize,
          precision: this.precision,
        },
      );
      const mlp = new MLPKernel(
        this.gpu,
        ln.outputBuffer,
        block.mlpFCWeight!,
        block.mlpFCBias,
        block.mlpProjWeight!,
        block.mlpProjBias,
        this.hiddenSize,
        weights.innerSize,
        {
          name: `${name}MLP`,
          workgroupSize: this.workgroupSize,
          precision: this.precision,
        },
      );
      const addAttention = new AddKernel(this.gpu, residualBuffer, attnProj.outputBuffer, this.hiddenSize, {
        name: `${name}AddAttention`,
        workgroupSize: this.workgroupSize,
      });
      const addMLP = new AddKernel(this.gpu, addAttention.outputBuffer, mlp.outputBuffer, this.hiddenSize, {
        name: `${name}AddMLP`,
        workgroupSize: this.workgroupSize,
      });

      return {
        kind: 'parallel',
        ln,
        qkv,
        attention,
        attnProj,
        mlp,
        addAttention,
        addMLP,
        outputBuffer: addMLP.outputBuffer,
      };
    }

    const ln1 = this.buildNorm(residualBuffer, block.ln1Weight, block.ln1Bias || null, `${name}LN1`);
    const qkvOut = recipe.architecture === 'gpt2' ? this.hiddenSize * 3 : weights.qSize + 2 * weights.kvSize;
    const qkv = new LinearKernel(
      this.gpu,
      ln1.outputBuffer,
      block.attnQKVWeight,
      block.attnQKVBias || null,
      this.hiddenSize,
      qkvOut,
      {
        name: `${name}QKV`,
        workgroupSize: this.workgroupSize,
        precision: this.precision,
      },
    );
    const attention = this.buildAttention(qkv.outputBuffer, block, name);
    const attnIn = recipe.architecture === 'gpt2' ? this.hiddenSize : weights.qSize;
    const attnProj = new LinearKernel(
      this.gpu,
      attention.outputBuffer,
      block.attnProjWeight,
      block.attnProjBias || null,
      attnIn,
      this.hiddenSize,
      {
        name: `${name}AttnProj`,
        workgroupSize: this.workgroupSize,
        precision: this.precision,
      },
    );

    if (recipe.postNorms) {
      const postAttnNorm = this.buildNorm(attnProj.outputBuffer, block.postAttnNormWeight, null, `${name}PostAttn`);
      const addAttention = new AddKernel(this.gpu, residualBuffer, postAttnNorm.outputBuffer, this.hiddenSize, {
        name: `${name}AddAttention`,
        workgroupSize: this.workgroupSize,
      });
      const preMlp = this.buildNorm(addAttention.outputBuffer, block.preMlpNormWeight, null, `${name}PreMLP`);
      const mlp = new GatedMLPKernel(
        this.gpu,
        preMlp.outputBuffer,
        block.mlpGateWeight!,
        block.mlpUpWeight!,
        block.mlpDownWeight!,
        this.hiddenSize,
        weights.innerSize,
        {
          name: `${name}MLP`,
          workgroupSize: this.workgroupSize,
          activation: recipe.mlpActivation,
          precision: this.precision,
        },
      );
      const postMlpNorm = this.buildNorm(mlp.outputBuffer, block.postMlpNormWeight, null, `${name}PostMLP`);
      const addMLP = new AddKernel(this.gpu, addAttention.outputBuffer, postMlpNorm.outputBuffer, this.hiddenSize, {
        name: `${name}AddMLP`,
        workgroupSize: this.workgroupSize,
      });

      return {
        kind: 'gemma',
        ln1,
        qkv,
        attention,
        attnProj,
        postAttnNorm,
        addAttention,
        preMlp,
        mlp,
        postMlpNorm,
        addMLP,
        outputBuffer: addMLP.outputBuffer,
      };
    }

    const addAttention = new AddKernel(this.gpu, residualBuffer, attnProj.outputBuffer, this.hiddenSize, {
      name: `${name}AddAttention`,
      workgroupSize: this.workgroupSize,
    });
    const ln2 = this.buildNorm(addAttention.outputBuffer, block.ln2Weight, block.ln2Bias || null, `${name}LN2`);
    const mlp: MlpKernel =
      recipe.mlp === 'dense_gelu'
        ? new MLPKernel(
            this.gpu,
            ln2.outputBuffer,
            block.mlpFCWeight!,
            block.mlpFCBias,
            block.mlpProjWeight!,
            block.mlpProjBias,
            this.hiddenSize,
            weights.innerSize,
            {
              name: `${name}MLP`,
              workgroupSize: this.workgroupSize,
              precision: this.precision,
            },
          )
        : new GatedMLPKernel(
            this.gpu,
            ln2.outputBuffer,
            block.mlpGateWeight!,
            block.mlpUpWeight!,
            block.mlpDownWeight!,
            this.hiddenSize,
            weights.innerSize,
            {
              name: `${name}MLP`,
              workgroupSize: this.workgroupSize,
              activation: recipe.mlpActivation,
              precision: this.precision,
            },
          );
    const addMLP = new AddKernel(this.gpu, addAttention.outputBuffer, mlp.outputBuffer, this.hiddenSize, {
      name: `${name}AddMLP`,
      workgroupSize: this.workgroupSize,
    });

    return {
      kind: 'sequential',
      ln1,
      qkv,
      attention,
      attnProj,
      addAttention,
      ln2,
      mlp,
      addMLP,
      outputBuffer: addMLP.outputBuffer,
    };
  }

  buildFinalNorm(inputBuffer: StorageBuffer): NormKernel {
    return this.buildNorm(
      inputBuffer,
      this.weights.outputNormWeight,
      this.weights.outputNormBias,
      `${this.weights.architecture}FinalNorm`,
    );
  }

  private runLayer(layer: DecoderLayer): void {
    if (layer.kind === 'parallel') {
      layer.ln!.run();
      layer.qkv.run();
      layer.attention.run(0);
      layer.attnProj.run();
      layer.mlp.run();
      layer.addAttention.run();
      layer.addMLP.run();
    } else if (layer.kind === 'gemma') {
      layer.ln1!.run();
      layer.qkv.run();
      layer.attention.run(0);
      layer.attnProj.run();
      layer.postAttnNorm!.run();
      layer.addAttention.run();
      layer.preMlp!.run();
      layer.mlp.run();
      layer.postMlpNorm!.run();
      layer.addMLP.run();
    } else {
      layer.ln1!.run();
      layer.qkv.run();
      layer.attention.run(0);
      layer.attnProj.run();
      layer.addAttention.run();
      layer.ln2!.run();
      layer.mlp.run();
      layer.addMLP.run();
    }
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
    const logits = await readChunkedLogits(this.logits, this.weights.vocabSize);
    return logitSoftcap(logits, this.recipe.finalLogitSoftcap);
  }

  async sampleToken(candidateCount: number, options: SampleOptions): Promise<number> {
    return this.logitSampler.sampleToken(candidateCount, options, this.batchCompute);
  }

  resetCache(): void {
    this._cacheTokens = [];
    this._cacheLogits = null;

    for (const layer of this.layers) layer.attention.reset();
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerationResult> {
    return generateAsync(this, prompt, options, {
      rewindable: true,
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

export { DecoderGpuRunner };
