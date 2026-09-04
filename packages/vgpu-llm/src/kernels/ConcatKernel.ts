import { allocStorage, makeCompute, workgroupCount } from '../gpu/device.js';
import type { Compute, Gpu, StorageBuffer } from '../gpu/device.js';
import type { KernelOptions } from '../types.js';

export interface ConcatPart {
  buffer: StorageBuffer;
  size: number;
}

/**
 * Concatenate several 1D storage vectors into one buffer.
 */
class ConcatKernel {
  parts: ConcatPart[];
  size: number;
  outputBuffer: StorageBuffer;
  pass: Compute;
  workgroups: number;

  constructor(gpu: Gpu, parts: ConcatPart[], options: KernelOptions = {}) {
    this.parts = parts;
    this.size = parts.reduce((sum, part) => sum + part.size, 0);
    this.outputBuffer = allocStorage(gpu, this.size);

    const workgroupSize = options.workgroupSize || 64;
    const bindings: string[] = [];
    const branches: string[] = [];
    const set: Record<string, unknown> = { output: this.outputBuffer };
    let offset = 0;

    parts.forEach((part, i) => {
      const name = `part${i}`;
      bindings.push(`@group(0) @binding(${i + 1}) var<storage, read> ${name}: array<f32>;`);
      branches.push(
        `if (index >= ${offset}u && index < ${offset + part.size}u) { output[index] = ${name}[index - ${offset}u]; }`,
      );
      set[name] = part.buffer;
      offset += part.size;
    });

    const source = `
      @group(0) @binding(0) var<storage, read_write> output: array<f32>;
      ${bindings.join('\n      ')}

      @compute @workgroup_size(${workgroupSize})
      fn cs_main(@builtin(global_invocation_id) id: vec3u) {
        let index = id.x;
        if (index >= ${this.size}u) { return; }
        ${branches.join('\n        ')}
      }
    `;

    this.pass = makeCompute(gpu, source, { label: options.name || 'LLMConcat', set });
    this.workgroups = workgroupCount(this.size, workgroupSize);
  }

  run(): StorageBuffer {
    this.pass.dispatch(this.workgroups);
    return this.outputBuffer;
  }
}

export { ConcatKernel };
