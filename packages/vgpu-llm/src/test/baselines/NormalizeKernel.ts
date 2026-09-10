// Frozen pre-reduction implementation for paired performance/correctness comparisons.
import {
  allocStorage,
  makeCompute,
  requireShaderF16,
  wgslEnableDirective,
  wgslScalarType,
  workgroupCount,
} from '../../gpu/device.js';
import type { Compute, Gpu, StorageBuffer } from '../../gpu/device.js';
import type { KernelOptions } from '../../types.js';

interface NormalizeOptions extends KernelOptions {
  epsilon?: number;
}

/**
 * Layer normalization for a single hidden vector.
 *
 * `weightBuffer`/`biasBuffer` must already be uploaded at `options.precision`
 * (default `fp32`) — e.g. via `uploadWeightStorage` — since this kernel only
 * picks the matching WGSL element type for them; it does not own the upload.
 */
class BaselineNormalizeKernel {
  hiddenSize: number;
  epsilon: number;
  outputBuffer: StorageBuffer;
  weightBuffer: StorageBuffer;
  biasBuffer: StorageBuffer;
  pass: Compute;
  workgroups: number;

  constructor(
    gpu: Gpu,
    inputBuffer: StorageBuffer,
    weightBuffer: StorageBuffer,
    biasBuffer: StorageBuffer,
    hiddenSize: number,
    options: NormalizeOptions = {},
  ) {
    this.hiddenSize = hiddenSize;
    this.epsilon = options.epsilon || 1e-5;
    this.weightBuffer = weightBuffer;
    this.biasBuffer = biasBuffer;
    this.outputBuffer = allocStorage(gpu, hiddenSize);

    const precision = options.precision || 'fp32';
    if (precision === 'fp16') requireShaderF16(gpu, options.name || 'LLMLayerNorm');

    const weightType = wgslScalarType(precision);
    const workgroupSize = options.workgroupSize || 64;
    const source = `
      ${wgslEnableDirective(precision)}
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read> weight: array<${weightType}>;
      @group(0) @binding(2) var<storage, read> bias: array<${weightType}>;
      @group(0) @binding(3) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(${workgroupSize})
      fn cs_main(@builtin(global_invocation_id) id: vec3u) {
        let index = id.x;
        if (index >= ${hiddenSize}u) { return; }

        var mean: f32 = 0.0;
        for (var i: u32 = 0u; i < ${hiddenSize}u; i = i + 1u) {
          mean = mean + input[i];
        }
        mean = mean / f32(${hiddenSize}u);

        var variance: f32 = 0.0;
        for (var i: u32 = 0u; i < ${hiddenSize}u; i = i + 1u) {
          let delta = input[i] - mean;
          variance = variance + delta * delta;
        }
        variance = variance / f32(${hiddenSize}u);

        let value = (input[index] - mean) * inverseSqrt(variance + ${this.epsilon}) * f32(weight[index]) + f32(bias[index]);
        output[index] = value;
      }
    `;

    this.pass = makeCompute(gpu, source, {
      label: options.name || 'LLMLayerNorm',
      set: { input: inputBuffer, weight: weightBuffer, bias: biasBuffer, output: this.outputBuffer },
    });
    this.workgroups = workgroupCount(hiddenSize, workgroupSize);
  }

  run(): StorageBuffer {
    this.pass.dispatch(this.workgroups);
    return this.outputBuffer;
  }
}

export { BaselineNormalizeKernel };
