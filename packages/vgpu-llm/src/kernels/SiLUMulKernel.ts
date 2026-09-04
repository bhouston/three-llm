import { allocStorage, makeCompute, workgroupCount } from '../gpu/device.js';
import type { Compute, Gpu, StorageBuffer } from '../gpu/device.js';
import type { KernelOptions } from '../types.js';

/**
 * Element-wise `silu(gate) * up` used by SwiGLU MLPs.
 */
class SiLUMulKernel {
  size: number;
  outputBuffer: StorageBuffer;
  pass: Compute;
  workgroups: number;

  constructor(gpu: Gpu, gateBuffer: StorageBuffer, upBuffer: StorageBuffer, size: number, options: KernelOptions = {}) {
    this.size = size;
    this.outputBuffer = allocStorage(gpu, size);

    const workgroupSize = options.workgroupSize || 64;
    const source = `
      @group(0) @binding(0) var<storage, read> gateBuf: array<f32>;
      @group(0) @binding(1) var<storage, read> upBuf: array<f32>;
      @group(0) @binding(2) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(${workgroupSize})
      fn cs_main(@builtin(global_invocation_id) id: vec3u) {
        let i = id.x;
        if (i >= ${size}u) { return; }
        let x = gateBuf[i];
        let silu = x / (1.0 + exp(-x));
        output[i] = silu * upBuf[i];
      }
    `;

    this.pass = makeCompute(gpu, source, {
      label: options.name || 'LLMSiLUMul',
      set: { gateBuf: gateBuffer, upBuf: upBuffer, output: this.outputBuffer },
    });
    this.workgroups = workgroupCount(size, workgroupSize);
  }

  run(): StorageBuffer {
    this.pass.dispatch(this.workgroups);
    return this.outputBuffer;
  }
}

export { SiLUMulKernel };
