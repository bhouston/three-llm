import type { Compute, Gpu } from 'vgpu';

interface Batch {
  timestampWrites?: GPUComputePassTimestampWrites;
  encoder?: GPUCommandEncoder;
  pass?: GPUComputePassEncoder;
}
const batches = new WeakMap<Gpu, Batch>();

/** Flush before a queue write/read: host updates must not overtake encoded work. */
export function flushComputeBatch(gpu: Gpu): void {
  const batch = batches.get(gpu);
  if (!batch?.encoder) return;
  batch.pass!.end();
  gpu.device.gpu.queue.submit([batch.encoder.finish()]);
  batch.encoder = undefined;
  batch.pass = undefined;
}

/** Synchronous scope only. Nested scopes share the outer command encoder. */
export function withComputeBatch(gpu: Gpu, work: () => void, timestampWrites?: GPUComputePassTimestampWrites): void {
  if (batches.has(gpu)) {
    work();
    return;
  }
  batches.set(gpu, { timestampWrites });
  try {
    work();
    flushComputeBatch(gpu);
  } finally {
    // On a synchronous exception, abandon commands not yet submitted.
    batches.delete(gpu);
  }
}

// vgpu 0.4 has no public encode API. Keep its pipeline/binding bridge isolated
// here and retain the public dispatch fallback when the shape is unavailable.
// We never patch the GPU queue or vgpu prototypes. Binding construction and
// ownership remain with vgpu. Remove this bridge when vgpu exposes encoding.
interface Encodable {
  pipeline: GPUComputePipeline;
  setCore: {
    bindGroups(): readonly { group: number; bindGroup: GPUBindGroup; offsets: readonly number[] }[];
    bindingState(name: string): { resource: GPUBufferBinding } | undefined;
  };
  reflection: { bindings: readonly { name: string; kind: string; addressSpace?: string; access?: string }[] };
}

export function batchableCompute(gpu: Gpu, original: Compute): Compute {
  const bridge = original as Compute & Partial<Encodable>;
  return {
    set(values) {
      flushComputeBatch(gpu);
      original.set(values);
      return this;
    },
    dispatch(
      x:
        | number
        | { indirect: import('vgpu').StorageBuffer | { buffer: import('vgpu').StorageBuffer; offset?: number } },
      y?: number,
      z?: number,
    ) {
      const batch = batches.get(gpu);
      if (
        !batch ||
        typeof x !== 'number' ||
        !bridge.pipeline ||
        !bridge.setCore?.bindGroups ||
        !bridge.setCore?.bindingState ||
        !Array.isArray(bridge.reflection?.bindings)
      ) {
        flushComputeBatch(gpu);
        if (typeof x === 'number') original.dispatch(x, y, z);
        else original.dispatch(x);
        return;
      }
      // Preserve vgpu's writable-storage alias rejection before encoding.
      const seen = new Map<GPUBuffer, boolean>();
      for (const binding of bridge.reflection!.bindings) {
        if (binding.kind !== 'buffer' || binding.addressSpace !== 'storage') continue;
        const state = bridge.setCore.bindingState(binding.name);
        if (!state) continue;
        const buffer = state.resource.buffer;
        const writable = binding.access !== 'read';
        if (seen.has(buffer) && (writable || seen.get(buffer)))
          throw new Error('Batched compute cannot alias writable storage.');
        seen.set(buffer, writable);
      }
      const groups = bridge.setCore.bindGroups();
      if (!batch.encoder) {
        batch.encoder = gpu.device.gpu.createCommandEncoder({ label: 'LLMComputeBatch' });
        batch.pass = batch.encoder.beginComputePass({
          label: 'LLMComputeBatch',
          timestampWrites: batch.timestampWrites,
        });
      }
      batch.pass!.setPipeline(bridge.pipeline);
      for (const binding of groups) batch.pass!.setBindGroup(binding.group, binding.bindGroup, binding.offsets);
      batch.pass!.dispatchWorkgroups(x, y ?? 1, z ?? 1);
    },
  };
}
