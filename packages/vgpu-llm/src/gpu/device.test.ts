import { describe, expect, it } from 'vitest';

import { float32ToFloat16Bits, packFloat16, wgslEnableDirective, wgslScalarType } from './device.js';

/** Decodes an `f16` bit pattern back to `f32`, mirroring `load/tensors.ts`'s `float16ToFloat32`. */
function float16BitsToFloat32(bits: number): number {
  const sign = (bits >> 15) & 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;

  if (exponent === 0) {
    if (fraction === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
  }

  if (exponent === 31) return fraction ? NaN : sign ? -Infinity : Infinity;

  return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

describe('float32ToFloat16Bits', () => {
  it('round-trips exactly representable values', () => {
    for (const value of [0, 1, -1, 2, -2, 0.5, -0.5, 1.5, 3.75, 65504, -65504]) {
      expect(float16BitsToFloat32(float32ToFloat16Bits(value))).toBe(value);
    }
  });

  it('rounds to the nearest f16 value within relative tolerance', () => {
    for (const value of [3.14158, -2.71827, 0.001, 12345.6789, 1e-3, 100.25]) {
      const roundTripped = float16BitsToFloat32(float32ToFloat16Bits(value));
      const relativeError = Math.abs(roundTripped - value) / Math.abs(value);
      expect(relativeError).toBeLessThan(0.01); // f16 has ~3 significant decimal digits.
    }
  });

  it('handles zero and signed zero', () => {
    expect(float32ToFloat16Bits(0)).toBe(0x0000);
    expect(float32ToFloat16Bits(-0)).toBe(0x8000);
  });

  it('saturates overflow to signed infinity', () => {
    expect(float32ToFloat16Bits(1e10)).toBe(0x7c00);
    expect(float32ToFloat16Bits(-1e10)).toBe(0xfc00);
    expect(float32ToFloat16Bits(Infinity)).toBe(0x7c00);
    expect(float32ToFloat16Bits(-Infinity)).toBe(0xfc00);
  });

  it('flushes tiny values to zero, preserving sign', () => {
    expect(float32ToFloat16Bits(1e-9)).toBe(0x0000);
    expect(float32ToFloat16Bits(-1e-9)).toBe(0x8000);
  });

  it('produces correct subnormal f16 values', () => {
    // The smallest positive f16 subnormal is 2^-24.
    const smallestSubnormal = Math.pow(2, -24);
    expect(float32ToFloat16Bits(smallestSubnormal)).toBe(0x0001);
    expect(float16BitsToFloat32(0x0001)).toBeCloseTo(smallestSubnormal, 30);
  });

  it('rounds an exact tie consistently upward', () => {
    // 1.0 + 2^-11 is exactly halfway between two f16 values around 1.0.
    const halfway = 1 + Math.pow(2, -11);
    const bits = float32ToFloat16Bits(halfway);
    expect(float16BitsToFloat32(bits)).toBe(1 + Math.pow(2, -10));
  });

  it('keeps NaN a NaN', () => {
    expect(Number.isNaN(float16BitsToFloat32(float32ToFloat16Bits(NaN)))).toBe(true);
  });
});

describe('packFloat16', () => {
  it('packs a Float32Array element-wise', () => {
    const source = new Float32Array([1, -2.5, 0, 3.14158]);
    const packed = packFloat16(source);

    expect(packed).toBeInstanceOf(Uint16Array);
    expect(packed.length).toBe(source.length);

    for (let i = 0; i < source.length; i++) {
      expect(packed[i]).toBe(float32ToFloat16Bits(source[i]!));
    }
  });

  it('produces values close to the CPU f32 source for realistic weight magnitudes', () => {
    const source = new Float32Array(1000);
    let seed = 7;

    for (let i = 0; i < source.length; i++) {
      seed = (seed * 9301 + 49297) % 233280;
      source[i] = ((seed / 233280) * 2 - 1) * 0.1; // Typical small transformer-weight range.
    }

    const packed = packFloat16(source);

    for (let i = 0; i < source.length; i++) {
      const decoded = float16BitsToFloat32(packed[i]!);
      expect(Math.abs(decoded - source[i]!)).toBeLessThan(0.001);
    }
  });
});

describe('wgslScalarType', () => {
  it('maps fp32 -> f32 and fp16 -> f16', () => {
    expect(wgslScalarType('fp32')).toBe('f32');
    expect(wgslScalarType('fp16')).toBe('f16');
  });
});

describe('wgslEnableDirective', () => {
  it('emits the f16 enable directive only for fp16', () => {
    expect(wgslEnableDirective('fp32')).toBe('');
    expect(wgslEnableDirective('fp16')).toBe('enable f16;\n');
  });
});
