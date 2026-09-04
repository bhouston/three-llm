import { allocStorage, makeCompute, uploadStorage, workgroupCount } from '../gpu/device.js';
import type { Compute, Gpu, StorageBuffer } from '../gpu/device.js';
import type { KernelOptions } from '../types.js';

/**
 * One-token dense layer implemented as a compute pass.
 *
 * Weight layout is `[inputSize, outputSize]`, matching GPT-2 Conv1D tensors
 * stored by Hugging Face.
 */
class LinearKernel {
  inputSize: number;
  outputSize: number;
  weightBuffer: StorageBuffer;
  biasBuffer: StorageBuffer;
  outputBuffer: StorageBuffer;
  pass: Compute;
  workgroups: number;

  constructor(
    gpu: Gpu,
    inputBuffer: StorageBuffer,
    weightArray: Float32Array,
    biasArray: Float32Array | null,
    inputSize: number,
    outputSize: number,
    options: KernelOptions = {},
  ) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;

    this.weightBuffer = uploadStorage(gpu, weightArray, 'read');
    this.biasBuffer = uploadStorage(gpu, biasArray || new Float32Array(outputSize), 'read');
    this.outputBuffer = allocStorage(gpu, outputSize);

    const workgroupSize = options.workgroupSize || 64;
    const source = `
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read> weight: array<f32>;
      @group(0) @binding(2) var<storage, read> bias: array<f32>;
      @group(0) @binding(3) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(${workgroupSize})
      fn cs_main(@builtin(global_invocation_id) id: vec3u) {
        let outputIndex = id.x;
        if (outputIndex >= ${outputSize}u) { return; }

        var sum: f32 = bias[outputIndex];
        for (var i: u32 = 0u; i < ${inputSize}u; i = i + 1u) {
          sum = sum + input[i] * weight[i * ${outputSize}u + outputIndex];
        }
        output[outputIndex] = sum;
      }
    `;

    this.pass = makeCompute(gpu, source, {
      label: options.name || 'LLMLinear',
      set: { input: inputBuffer, weight: this.weightBuffer, bias: this.biasBuffer, output: this.outputBuffer },
    });
    this.workgroups = workgroupCount(outputSize, workgroupSize);
  }

  run(): StorageBuffer {
    this.pass.dispatch(this.workgroups);
    return this.outputBuffer;
  }
}

export { LinearKernel };
