import { allocStorage, makeCompute, readFloat32, readUint32, workgroupCount } from '../gpu/device.js';
import type { Compute, Gpu, StorageBuffer } from '../gpu/device.js';
import { LinearKernel } from './LinearKernel.js';
import { sampleTopKCandidates } from '../runtime/math.js';
import type { KernelOptions, SampleOptions } from '../types.js';

const LOWEST_FLOAT = -3.4028234663852886e38;

interface LogitWeights {
  hiddenSize: number;
  vocabSize: number;
  logitWeight: Float32Array | null;
}

export interface LogitChunk {
  offset: number;
  size: number;
  layer: LinearKernel;
}

interface LogitSamplerOptions extends KernelOptions {
  candidateCount?: number;
  logitSoftcap?: number;
}

function applySoftcap(expr: string, cap: number | undefined): string {
  if (cap === null || cap === undefined) return expr;
  return `(${cap} * tanh((${expr}) / ${cap}))`;
}

/**
 * Slices the vocab-projection weight matrix into `chunkSize`-wide dense
 * layers so no single kernel needs a `vocabSize`-long weight upload/output
 * in one pass. Same layout as `LinearKernel`: `[hiddenSize, chunkWidth]`.
 */
function createChunkedLogitLayers(
  gpu: Gpu,
  inputBuffer: StorageBuffer,
  weights: LogitWeights,
  chunkSize: number,
  name: string,
): LogitChunk[] {
  const logits: LogitChunk[] = [];
  const hiddenSize = weights.hiddenSize;

  for (let offset = 0; offset < weights.vocabSize; offset += chunkSize) {
    const size = Math.min(chunkSize, weights.vocabSize - offset);
    const chunkWeight = new Float32Array(hiddenSize * size);

    for (let i = 0; i < hiddenSize; i++) {
      const sourceOffset = i * weights.vocabSize + offset;
      chunkWeight.set(weights.logitWeight!.subarray(sourceOffset, sourceOffset + size), i * size);
    }

    logits.push({
      offset,
      size,
      layer: new LinearKernel(gpu, inputBuffer, chunkWeight, null, hiddenSize, size, {
        name: `${name}${offset}`,
        workgroupSize: 256,
      }),
    });
  }

  return logits;
}

async function readChunkedLogits(chunks: LogitChunk[], vocabSize: number): Promise<Float32Array> {
  const logits = new Float32Array(vocabSize);
  const values = await Promise.all(chunks.map((chunk) => readFloat32(chunk.layer.outputBuffer)));

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i]!;
    logits.set(values[i]!.subarray(0, chunk.size), chunk.offset);
  }

  return logits;
}

/**
 * GPU top-K (without replacement) over chunked logits, avoiding a full
 * `vocabSize`-length readback for greedy/low-temperature sampling.
 *
 * Three levels, mirroring the CPU top-K semantics (ties broken by lower
 * token id): (1) per-chunk parallel-reduction max per workgroup, (2) a
 * single-thread merge of a chunk's partial maxima into that chunk's next
 * unclaimed candidate, repeated per rank, (3) a single-thread merge of each
 * chunk's next candidate into the next global rank.
 */
class LogitSampler {
  chunks: LogitChunk[];
  candidateCount: number;
  workgroupSize: number;
  logitSoftcap?: number;
  chunkCandidateCount: number;
  partialOffsets: number[];
  partialCount: number;

  partialTokenBuffer: StorageBuffer;
  partialScoreBuffer: StorageBuffer;
  chunkCandidateTokenBuffer: StorageBuffer;
  chunkCandidateScoreBuffer: StorageBuffer;
  candidateTokenBuffer: StorageBuffer;
  candidateScoreBuffer: StorageBuffer;

  partialMaxPasses: Array<{ pass: Compute; workgroups: number }>;
  greedyMergePass: Compute;
  /** [rank][chunkIndex] chunk-candidate passes, plus one global-merge pass per rank at the end. */
  candidateLevels: Array<{ chunkPasses: Compute[]; globalPass: Compute }>;

  constructor(gpu: Gpu, chunks: LogitChunk[], options: LogitSamplerOptions = {}) {
    this.chunks = chunks;
    this.candidateCount = Math.max(1, options.candidateCount || 8);
    this.workgroupSize = options.workgroupSize || 256;
    this.logitSoftcap = options.logitSoftcap;
    this.chunkCandidateCount = chunks.length * this.candidateCount;
    this.partialOffsets = [];
    this.partialCount = 0;

    for (const chunk of chunks) {
      this.partialOffsets.push(this.partialCount);
      this.partialCount += Math.ceil(chunk.size / this.workgroupSize);
    }

    this.partialTokenBuffer = allocStorage(gpu, this.partialCount);
    this.partialScoreBuffer = allocStorage(gpu, this.partialCount);
    this.chunkCandidateTokenBuffer = allocStorage(gpu, this.chunkCandidateCount);
    this.chunkCandidateScoreBuffer = allocStorage(gpu, this.chunkCandidateCount);
    this.candidateTokenBuffer = allocStorage(gpu, this.candidateCount);
    this.candidateScoreBuffer = allocStorage(gpu, this.candidateCount);

    const name = options.name;

    this.partialMaxPasses = chunks.map((chunk, chunkIndex) =>
      this.buildChunkPartialMaxPass(
        gpu,
        chunk,
        chunkIndex,
        name ? `${name}Chunk${chunkIndex}PartialMax` : `LLMLogitChunk${chunkIndex}PartialMax`,
      ),
    );
    this.greedyMergePass = this.buildGreedyMergePass(gpu, name ? `${name}Greedy` : 'LLMLogitGreedy');

    this.candidateLevels = [];

    for (let rank = 0; rank < this.candidateCount; rank++) {
      const chunkPasses = chunks.map((chunk, chunkIndex) =>
        this.buildChunkCandidatePass(
          gpu,
          chunk,
          chunkIndex,
          rank,
          name ? `${name}Chunk${chunkIndex}Candidate${rank}` : `LLMLogitChunk${chunkIndex}Candidate${rank}`,
        ),
      );
      const globalPass = this.buildGlobalCandidatePass(
        gpu,
        rank,
        name ? `${name}Candidate${rank}` : `LLMLogitCandidate${rank}`,
      );
      this.candidateLevels.push({ chunkPasses, globalPass });
    }
  }

  private chunkCandidateIndex(chunkIndex: number, rank: number): number {
    return chunkIndex * this.candidateCount + rank;
  }

  private buildChunkPartialMaxPass(gpu: Gpu, chunk: LogitChunk, chunkIndex: number, name: string) {
    const wg = this.workgroupSize;
    const scoreExpr = applySoftcap('logits[tokenIndex]', this.logitSoftcap);
    const strides: number[] = [];

    for (let stride = wg / 2; stride >= 1; stride /= 2) strides.push(stride);

    const reduction = strides
      .map(
        (stride) => `
        if (localIndex < ${stride}u) {
          let otherIndex = localIndex + ${stride}u;
          let otherScore = localScores[otherIndex];
          let otherToken = localTokens[otherIndex];
          let bestScore = localScores[localIndex];
          let bestToken = localTokens[localIndex];
          if (otherScore > bestScore || (otherScore == bestScore && otherToken < bestToken)) {
            localScores[localIndex] = otherScore;
            localTokens[localIndex] = otherToken;
          }
        }
        workgroupBarrier();`,
      )
      .join('\n');

    const source = `
      var<workgroup> localTokens: array<u32, ${wg}>;
      var<workgroup> localScores: array<f32, ${wg}>;

      @group(0) @binding(0) var<storage, read> logits: array<f32>;
      @group(0) @binding(1) var<storage, read_write> partialTokens: array<u32>;
      @group(0) @binding(2) var<storage, read_write> partialScores: array<f32>;

      @compute @workgroup_size(${wg})
      fn cs_main(
        @builtin(global_invocation_id) id: vec3u,
        @builtin(local_invocation_index) localIndex: u32,
        @builtin(workgroup_id) wgId: vec3u,
      ) {
        let tokenIndex = id.x;
        let globalToken = ${chunk.offset}u + tokenIndex;
        var score: f32 = ${LOWEST_FLOAT};
        if (tokenIndex < ${chunk.size}u) {
          score = ${scoreExpr};
        }

        localTokens[localIndex] = globalToken;
        localScores[localIndex] = score;
        workgroupBarrier();
        ${reduction}

        if (localIndex == 0u) {
          let partialIndex = ${this.partialOffsets[chunkIndex]}u + wgId.x;
          partialTokens[partialIndex] = localTokens[0];
          partialScores[partialIndex] = localScores[0];
        }
      }
    `;

    const pass = makeCompute(gpu, source, {
      label: name,
      set: {
        logits: chunk.layer.outputBuffer,
        partialTokens: this.partialTokenBuffer,
        partialScores: this.partialScoreBuffer,
      },
    });

    return { pass, workgroups: workgroupCount(chunk.size, wg) };
  }

  private buildGreedyMergePass(gpu: Gpu, name: string) {
    const source = `
      @group(0) @binding(0) var<storage, read> partialTokens: array<u32>;
      @group(0) @binding(1) var<storage, read> partialScores: array<f32>;
      @group(0) @binding(2) var<storage, read_write> candidateTokens: array<u32>;
      @group(0) @binding(3) var<storage, read_write> candidateScores: array<f32>;

      @compute @workgroup_size(1)
      fn cs_main() {
        var bestToken: u32 = 0u;
        var bestScore: f32 = ${LOWEST_FLOAT};

        for (var i: u32 = 0u; i < ${this.partialCount}u; i = i + 1u) {
          let tokenId = partialTokens[i];
          let score = partialScores[i];
          if (score > bestScore || (score == bestScore && tokenId < bestToken)) {
            bestToken = tokenId;
            bestScore = score;
          }
        }

        candidateTokens[0] = bestToken;
        candidateScores[0] = bestScore;
      }
    `;

    return makeCompute(gpu, source, {
      label: name,
      set: {
        partialTokens: this.partialTokenBuffer,
        partialScores: this.partialScoreBuffer,
        candidateTokens: this.candidateTokenBuffer,
        candidateScores: this.candidateScoreBuffer,
      },
    });
  }

  private buildChunkCandidatePass(gpu: Gpu, chunk: LogitChunk, chunkIndex: number, rank: number, name: string) {
    const candidateOffset = this.chunkCandidateIndex(chunkIndex, 0);
    const scoreExpr = applySoftcap('logits[i]', this.logitSoftcap);
    const excludeChecks = Array.from(
      { length: rank },
      (_, previousRank) =>
        `if (chunkCandidateTokens[${candidateOffset + previousRank}u] == tokenId) { selected = 1u; }`,
    ).join('\n          ');

    const source = `
      @group(0) @binding(0) var<storage, read> logits: array<f32>;
      @group(0) @binding(1) var<storage, read_write> chunkCandidateTokens: array<u32>;
      @group(0) @binding(2) var<storage, read_write> chunkCandidateScores: array<f32>;

      @compute @workgroup_size(1)
      fn cs_main() {
        var bestToken: u32 = 0u;
        var bestScore: f32 = ${LOWEST_FLOAT};

        for (var i: u32 = 0u; i < ${chunk.size}u; i = i + 1u) {
          let tokenId = ${chunk.offset}u + i;
          let score = ${scoreExpr};
          var selected: u32 = 0u;
          ${excludeChecks}
          if (selected == 0u && (score > bestScore || (score == bestScore && tokenId < bestToken))) {
            bestToken = tokenId;
            bestScore = score;
          }
        }

        chunkCandidateTokens[${candidateOffset + rank}u] = bestToken;
        chunkCandidateScores[${candidateOffset + rank}u] = bestScore;
      }
    `;

    return makeCompute(gpu, source, {
      label: name,
      set: {
        logits: chunk.layer.outputBuffer,
        chunkCandidateTokens: this.chunkCandidateTokenBuffer,
        chunkCandidateScores: this.chunkCandidateScoreBuffer,
      },
    });
  }

  private buildGlobalCandidatePass(gpu: Gpu, rank: number, name: string) {
    const branches: string[] = [];

    for (let chunkIndex = 0; chunkIndex < this.chunks.length; chunkIndex++) {
      for (let localRank = 0; localRank <= rank; localRank++) {
        const candidateIndex = this.chunkCandidateIndex(chunkIndex, localRank);
        const excludeChecks = Array.from(
          { length: rank },
          (_, previousRank) => `if (candidateTokens[${previousRank}u] == tokenId) { selected = 1u; }`,
        ).join('\n          ');

        branches.push(`
        {
          let tokenId = chunkCandidateTokens[${candidateIndex}u];
          let score = chunkCandidateScores[${candidateIndex}u];
          var selected: u32 = 0u;
          ${excludeChecks}
          if (selected == 0u && (score > bestScore || (score == bestScore && tokenId < bestToken))) {
            bestToken = tokenId;
            bestScore = score;
          }
        }`);
      }
    }

    const source = `
      @group(0) @binding(0) var<storage, read> chunkCandidateTokens: array<u32>;
      @group(0) @binding(1) var<storage, read> chunkCandidateScores: array<f32>;
      @group(0) @binding(2) var<storage, read_write> candidateTokens: array<u32>;
      @group(0) @binding(3) var<storage, read_write> candidateScores: array<f32>;

      @compute @workgroup_size(1)
      fn cs_main() {
        var bestToken: u32 = 0u;
        var bestScore: f32 = ${LOWEST_FLOAT};
        ${branches.join('\n')}
        candidateTokens[${rank}u] = bestToken;
        candidateScores[${rank}u] = bestScore;
      }
    `;

    return makeCompute(gpu, source, {
      label: name,
      set: {
        chunkCandidateTokens: this.chunkCandidateTokenBuffer,
        chunkCandidateScores: this.chunkCandidateScoreBuffer,
        candidateTokens: this.candidateTokenBuffer,
        candidateScores: this.candidateScoreBuffer,
      },
    });
  }

  /** Runs the passes needed to fill the first `count` (or 1, for greedy) candidates. */
  run(count: number): void {
    if (count <= 1) {
      for (const { pass, workgroups } of this.partialMaxPasses) pass.dispatch(workgroups);
      this.greedyMergePass.dispatch(1);
      return;
    }

    const candidateCount = Math.min(Math.max(1, count), this.candidateCount);

    for (let rank = 0; rank < candidateCount; rank++) {
      const level = this.candidateLevels[rank]!;
      for (const chunkPass of level.chunkPasses) chunkPass.dispatch(1);
      level.globalPass.dispatch(1);
    }
  }

  async readToken(): Promise<number> {
    return (await readUint32(this.candidateTokenBuffer))[0]!;
  }

  async readCandidates(count: number): Promise<Array<[number, number]>> {
    const candidateCount = Math.min(Math.max(1, count), this.candidateCount);
    const tokens = await readUint32(this.candidateTokenBuffer);
    const scores = await readFloat32(this.candidateScoreBuffer);
    const candidates: Array<[number, number]> = [];

    for (let i = 0; i < candidateCount; i++) candidates.push([tokens[i]!, scores[i]!]);

    return candidates;
  }

  async sampleToken(count: number, options: SampleOptions): Promise<number> {
    this.run(count);

    if (count <= 1 || options.temperature! <= 0) return this.readToken();

    return sampleTopKCandidates(await this.readCandidates(count), options);
  }
}

function createLogitSampler(gpu: Gpu, chunks: LogitChunk[], options?: LogitSamplerOptions): LogitSampler {
  return new LogitSampler(gpu, chunks, options);
}

export { createChunkedLogitLayers, createLogitSampler, LogitSampler, readChunkedLogits };
