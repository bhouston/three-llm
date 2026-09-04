import { compute, storage } from 'vgpu';
import type { Compute, Gpu, StorageBuffer } from 'vgpu';
import type { Precision } from '../types.js';

/**
 * A single WebGPU context, acquired once per runner via `vgpu`'s `init()`
 * (browser), `vgpu/node`'s `init()` (headless), or `vgpu/mock`'s `init()`
 * (deterministic, GPU-free tests).
 */
export type { Compute, Gpu, StorageBuffer };
export type { Precision };

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
function writeBuffer(buffer: StorageBuffer, data: Float32Array | Uint32Array | Int32Array | Uint16Array): void {
  buffer.write(data as unknown as BufferSource);
}

/**
 * Converts one IEEE 754 `f32` value to the bit pattern of the nearest `f16`
 * value (round-to-nearest, ties round up; correct subnormal handling),
 * returned as a `u16`. This is the standard bit-twiddling `f32`->`f16`
 * algorithm (as used by, e.g., three.js's `DataUtils.toHalfFloat`).
 */
const _f32ToF16Buffer = new Float32Array(1);
const _f32ToF16View = new Int32Array(_f32ToF16Buffer.buffer);

function float32ToFloat16Bits(value: number): number {
  _f32ToF16Buffer[0] = value;
  const x = _f32ToF16View[0]!;

  let bits = (x >> 16) & 0x8000; // Sign.
  const m = (x >> 12) & 0x07ff; // Mantissa, pre-rounding.
  const e = (x >> 23) & 0xff; // Biased f32 exponent.

  if (e < 103) return bits; // Underflows to zero (even when rounded).

  if (e > 142) {
    // Overflow -> infinity, or the input was already infinity/NaN. Model
    // weights are never NaN in practice, so exact NaN-payload fidelity
    // isn't worth preserving here.
    bits |= 0x7c00;
    if (e === 255 && (x & 0x007fffff) !== 0) bits |= 0x0200; // Keep NaN a NaN, not infinity.
    return bits;
  }

  if (e < 113) {
    // Subnormal f16 result.
    const shifted = m | 0x0800;
    bits |= (shifted >> (114 - e)) + ((shifted >> (113 - e)) & 1);
    return bits;
  }

  bits |= ((e - 112) << 10) | (m >> 1);
  bits += m & 1; // Round to nearest; exact ties round up.
  return bits;
}

/** Packs a `Float32Array` into `f16` bit patterns (see `float32ToFloat16Bits`). */
function packFloat16(source: Float32Array): Uint16Array {
  const target = new Uint16Array(source.length);
  for (let i = 0; i < source.length; i++) target[i] = float32ToFloat16Bits(source[i]!);
  return target;
}

/** Whether `gpu`'s device was created with the `shader-f16` feature. */
function hasShaderF16(gpu: Gpu): boolean {
  return gpu.device.features.has('shader-f16');
}

/** Throws a clear, actionable error if `gpu` lacks the `shader-f16` feature. */
function requireShaderF16(gpu: Gpu, context: string): void {
  if (hasShaderF16(gpu)) return;

  throw new Error(
    `${context}: precision "fp16" requires the "shader-f16" device feature. ` +
      `Initialize the Gpu with it, e.g. \`init({ requiredFeatures: ['shader-f16'] })\`, ` +
      `or omit the "precision" option to use "fp32".`,
  );
}

/**
 * Uploads `data` as a weight storage buffer at the requested `precision`:
 * `fp32` uploads it unchanged (4 bytes/element); `fp16` rounds it to `f16`
 * bit patterns first (2 bytes/element, half the memory and upload/read
 * bandwidth). Compute still happens in `f32` — kernels read `fp16` storage
 * through `f32(...)` and accumulate at full precision; only the weights
 * themselves are narrowed. Callers pick the matching WGSL element type via
 * `wgslScalarType(precision)`.
 */
function uploadWeightStorage(
  gpu: Gpu,
  data: Float32Array,
  precision: Precision,
  access: 'read' | 'read-write' = 'read',
): StorageBuffer {
  if (precision === 'fp32') return uploadStorage(gpu, data, access);

  const bits = packFloat16(data);
  // WebGPU's queue.writeBuffer requires a size that's a multiple of 4 bytes;
  // `f16` is 2 bytes/element, so an odd element count needs one element of
  // zero padding. The padding element is never read: kernels index up to
  // the real (unpadded) length, not the buffer's allocated size.
  const paddedLength = bits.length % 2 === 0 ? bits.length : bits.length + 1;
  const padded = paddedLength === bits.length ? bits : new Uint16Array(paddedLength);
  if (padded !== bits) padded.set(bits);

  const buffer = storage(gpu, Math.max(paddedLength, 2) * 2, access);
  writeBuffer(buffer, padded);
  return buffer;
}

/** The WGSL scalar type backing a weight buffer uploaded at `precision`. */
function wgslScalarType(precision: Precision): 'f32' | 'f16' {
  return precision === 'fp16' ? 'f16' : 'f32';
}

/** The `enable f16;` directive a WGSL module needs, as its first line, to declare `array<f16>` bindings. */
function wgslEnableDirective(precision: Precision): string {
  return precision === 'fp16' ? 'enable f16;\n' : '';
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

export {
  allocStorage,
  float32ToFloat16Bits,
  hasShaderF16,
  makeCompute,
  packFloat16,
  readFloat32,
  readUint32,
  requireShaderF16,
  uploadStorage,
  uploadWeightStorage,
  wgslEnableDirective,
  wgslScalarType,
  workgroupCount,
  writeBuffer,
};
