import {
  allocStorage,
  makeCompute,
  requireShaderF16,
  wgslEnableDirective,
  wgslScalarType,
  workgroupCount,
} from '../gpu/device.js';
import type { Compute, Gpu, StorageBuffer } from '../gpu/device.js';
import type { KernelOptions } from '../types.js';

interface RMSNormOptions extends KernelOptions {
  epsilon?: number;
  offsetWeight?: boolean;
}

/**
 * RMS normalization for a single hidden vector.
 *
 * Llama scales by `weight`. Gemma scales by `1 + weight`.
 *
 * `weightBuffer` must already be uploaded at `options.precision` (default
 * `fp32`) — e.g. via `uploadWeightStorage` — since this kernel only picks
 * the matching WGSL element type for it; it does not own the upload.
 */
class RMSNormKernel {
  hiddenSize: number;
  epsilon: number;
  offsetWeight: boolean;
  outputBuffer: StorageBuffer;
  weightBuffer: StorageBuffer;
  pass: Compute;
  workgroups: number;

  constructor(
    gpu: Gpu,
    inputBuffer: StorageBuffer,
    weightBuffer: StorageBuffer,
    hiddenSize: number,
    options: RMSNormOptions = {},
  ) {
    this.hiddenSize = hiddenSize;
    this.epsilon = options.epsilon || 1e-5;
    this.offsetWeight = options.offsetWeight === true;
    this.weightBuffer = weightBuffer;
    this.outputBuffer = allocStorage(gpu, hiddenSize);

    const precision = options.precision || 'fp32';
    if (precision === 'fp16') requireShaderF16(gpu, options.name || 'LLMRMSNorm');

    const weightType = wgslScalarType(precision);
    const workgroupSize = options.workgroupSize || 64;
    const scaleExpr = this.offsetWeight ? 'f32(weight[index]) + 1.0' : 'f32(weight[index])';
    const source = `
      ${wgslEnableDirective(precision)}
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read> weight: array<${weightType}>;
      @group(0) @binding(2) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(${workgroupSize})
      fn cs_main(@builtin(global_invocation_id) id: vec3u) {
        let index = id.x;
        if (index >= ${hiddenSize}u) { return; }

        var sumSquares: f32 = 0.0;
        for (var i: u32 = 0u; i < ${hiddenSize}u; i = i + 1u) {
          let value = input[i];
          sumSquares = sumSquares + value * value;
        }

        let invRms = inverseSqrt(sumSquares / f32(${hiddenSize}u) + ${this.epsilon});
        let scale = ${scaleExpr};
        output[index] = input[index] * invRms * scale;
      }
    `;

    this.pass = makeCompute(gpu, source, {
      label: options.name || 'LLMRMSNorm',
      set: { input: inputBuffer, weight: weightBuffer, output: this.outputBuffer },
    });
    this.workgroups = workgroupCount(hiddenSize, workgroupSize);
  }

  run(): StorageBuffer {
    this.pass.dispatch(this.workgroups);
    return this.outputBuffer;
  }
}

export { RMSNormKernel };
