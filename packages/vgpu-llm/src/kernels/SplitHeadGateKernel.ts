import { allocStorage, makeCompute, workgroupCount } from '../gpu/device.js';
import type { Compute, Gpu, StorageBuffer } from '../gpu/device.js';
import type { KernelOptions } from '../types.js';

/**
 * Split packed `[q_h, gate_h]` heads into separate query and gate vectors.
 */
class SplitHeadGateKernel {
  qSize: number;
  queryBuffer: StorageBuffer;
  gateBuffer: StorageBuffer;
  pass: Compute;
  workgroups: number;

  constructor(gpu: Gpu, packedBuffer: StorageBuffer, headCount: number, headDim: number, options: KernelOptions = {}) {
    this.qSize = headCount * headDim;
    this.queryBuffer = allocStorage(gpu, this.qSize);
    this.gateBuffer = allocStorage(gpu, this.qSize);

    const workgroupSize = options.workgroupSize || 64;
    const packedWidth = headDim * 2;
    const source = `
      @group(0) @binding(0) var<storage, read> packed: array<f32>;
      @group(0) @binding(1) var<storage, read_write> query: array<f32>;
      @group(0) @binding(2) var<storage, read_write> gate: array<f32>;

      @compute @workgroup_size(${workgroupSize})
      fn cs_main(@builtin(global_invocation_id) id: vec3u) {
        let index = id.x;
        if (index >= ${this.qSize}u) { return; }

        let head = index / ${headDim}u;
        let local = index % ${headDim}u;
        let packedOffset = head * ${packedWidth}u;

        query[index] = packed[packedOffset + local];
        gate[index] = packed[packedOffset + ${headDim}u + local];
      }
    `;

    this.pass = makeCompute(gpu, source, {
      label: options.name || 'LLMSplitHeadGate',
      set: { packed: packedBuffer, query: this.queryBuffer, gate: this.gateBuffer },
    });
    this.workgroups = workgroupCount(this.qSize, workgroupSize);
  }

  run(): StorageBuffer {
    this.pass.dispatch(this.workgroups);
    return this.queryBuffer;
  }
}

export { SplitHeadGateKernel };
