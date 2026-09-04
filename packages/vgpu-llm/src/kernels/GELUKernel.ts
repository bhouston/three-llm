import { allocStorage, makeCompute, workgroupCount } from '../gpu/device.js';
import type { Compute, Gpu, StorageBuffer } from '../gpu/device.js';
import type { KernelOptions } from '../types.js';

/**
 * GPT-2 `gelu_new` activation: `0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715x^3)))`.
 */
class GELUKernel {
  size: number;
  outputBuffer: StorageBuffer;
  pass: Compute;
  workgroups: number;

  constructor(gpu: Gpu, inputBuffer: StorageBuffer, size: number, options: KernelOptions = {}) {
    this.size = size;
    this.outputBuffer = allocStorage(gpu, size);

    const workgroupSize = options.workgroupSize || 64;
    const sqrt2OverPi = Math.sqrt(2 / Math.PI);
    const source = `
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(${workgroupSize})
      fn cs_main(@builtin(global_invocation_id) id: vec3u) {
        let i = id.x;
        if (i >= ${size}u) { return; }
        let x = input[i];
        let cubic = x * x * x * 0.044715 + x;
        let inner = clamp(cubic * ${sqrt2OverPi}, -10.0, 10.0);
        output[i] = x * 0.5 * (tanh(inner) + 1.0);
      }
    `;

    this.pass = makeCompute(gpu, source, {
      label: options.name || 'LLMGELU',
      set: { input: inputBuffer, output: this.outputBuffer },
    });
    this.workgroups = workgroupCount(size, workgroupSize);
  }

  run(): StorageBuffer {
    this.pass.dispatch(this.workgroups);
    return this.outputBuffer;
  }
}

export { GELUKernel };
