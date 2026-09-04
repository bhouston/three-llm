import { compute, storage } from 'vgpu';
import type { Compute, Gpu, StorageBuffer } from 'vgpu';

/**
 * A single WebGPU context, acquired once per runner via `vgpu`'s `init()`
 * (browser), `vgpu/node`'s `init()` (headless), or `vgpu/mock`'s `init()`
 * (deterministic, GPU-free tests).
 */
export type { Compute, Gpu, StorageBuffer };

/**
 * Allocates a GPU storage buffer sized for `length` 32-bit elements
 * (`f32`, `u32`, or `i32`) and leaves it zero-initialized.
 */
function allocStorage(gpu: Gpu, length: number, access: 'read' | 'read-write' = 'read-write'): StorageBuffer {
  return storage(gpu, Math.max(length, 1) * 4, access);
}

/**
 * Allocates a storage buffer and immediately uploads `data`. The caller's
 * typed array is copied to the GPU by `write()`; vgpu keeps no reference to
 * it afterward, so dropping the caller's reference is enough to free the
 * CPU-side copy.
 */
function uploadStorage(
  gpu: Gpu,
  data: Float32Array | Uint32Array | Int32Array,
  access: 'read' | 'read-write' = 'read',
): StorageBuffer {
  const buffer = allocStorage(gpu, data.length, access);
  writeBuffer(buffer, data);
  return buffer;
}

/**
 * `StorageBuffer.write()` types its parameter as `BufferSource` against a
 * DOM lib that (as of TS 7 / recent `@webgpu/types`) doesn't structurally
 * match the `ArrayBufferLike`-backed typed arrays this package allocates.
 * The values are real `ArrayBuffer`-backed views; this centralizes the cast.
 */
function writeBuffer(buffer: StorageBuffer, data: Float32Array | Uint32Array | Int32Array): void {
  buffer.write(data as unknown as BufferSource);
}

/** Reads a storage buffer back as a `Float32Array`. */
async function readFloat32(buffer: StorageBuffer): Promise<Float32Array> {
  return new Float32Array(await buffer.read());
}

/** Reads a storage buffer back as a `Uint32Array`. */
async function readUint32(buffer: StorageBuffer): Promise<Uint32Array> {
  return new Uint32Array(await buffer.read());
}

/**
 * Builds a `Compute` pass bound once at construction time (matching how
 * every kernel in this package creates its pipeline eagerly and only calls
 * `set()`/`dispatch()` per token).
 */
function makeCompute(gpu: Gpu, source: string, opts: Parameters<typeof compute>[2] = {}): Compute {
  return compute(gpu, source, opts);
}

/** Ceil-divides `total` work items across a fixed WGSL `@workgroup_size(wg)`. */
function workgroupCount(total: number, workgroupSize: number): number {
  return Math.max(1, Math.ceil(total / workgroupSize));
}

export { allocStorage, makeCompute, readFloat32, readUint32, uploadStorage, workgroupCount, writeBuffer };
