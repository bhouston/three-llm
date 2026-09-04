import { allocStorage, makeCompute, workgroupCount } from '../gpu/device.js';
import type { Compute, Gpu, StorageBuffer } from '../gpu/device.js';
import type { KernelOptions } from '../types.js';

/**
 * Element-wise multiply for gated MLPs.
 */
class MulKernel {
  size: number;
  outputBuffer: StorageBuffer;
  pass: Compute;
  workgroups: number;

  constructor(gpu: Gpu, aBuffer: StorageBuffer, bBuffer: StorageBuffer, size: number, options: KernelOptions = {}) {
    this.size = size;
    this.outputBuffer = allocStorage(gpu, size);

    const workgroupSize = options.workgroupSize || 64;
    const source = `
      @group(0) @binding(0) var<storage, read> a: array<f32>;
      @group(0) @binding(1) var<storage, read> b: array<f32>;
      @group(0) @binding(2) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(${workgroupSize})
      fn cs_main(@builtin(global_invocation_id) id: vec3u) {
        let i = id.x;
        if (i >= ${size}u) { return; }
        output[i] = a[i] * b[i];
      }
    `;

    this.pass = makeCompute(gpu, source, {
      label: options.name || 'LLMMul',
      set: { a: aBuffer, b: bBuffer, output: this.outputBuffer },
    });
    this.workgroups = workgroupCount(size, workgroupSize);
  }

  run(): StorageBuffer {
    this.pass.dispatch(this.workgroups);
    return this.outputBuffer;
  }
}

export { MulKernel };
