import type { RopeScalingConfig, YarnRoPEConfig } from '../types.js';

/** Static RoPE parameters, matching Transformers 4.51.3 (pinned fixtures in test/fixtures). */
export function ropeParameters(dim: number, theta: number, scaling?: RopeScalingConfig | YarnRoPEConfig) {
  if (!Number.isInteger(dim) || dim < 0 || dim % 2 !== 0)
    throw new Error('RoPE dimension must be a non-negative even integer.');
  if (!Number.isFinite(theta) || theta <= 1) throw new Error('RoPE theta must be finite and greater than one.');
  const invFreq = new Float32Array(dim / 2);
  let attentionFactor = 1;
  const type = scaling && ('type' in scaling ? scaling.type : 'yarn');
  const yarn = type === 'yarn' ? (scaling as YarnRoPEConfig) : undefined;
  let low = 0;
  let high = 0;
  if (scaling && (!Number.isFinite(scaling.factor) || scaling.factor < 1))
    throw new Error('RoPE scaling factor must be finite and at least one.');
  if (yarn) {
    if (![yarn.originalContextLength, yarn.betaFast, yarn.betaSlow].every(Number.isFinite))
      throw new Error('Invalid YaRN context or beta range.');
    if (!(yarn.originalContextLength > 0 && yarn.betaFast > 0 && yarn.betaSlow > 0 && yarn.betaFast >= yarn.betaSlow))
      throw new Error('Invalid YaRN context or beta range.');
    attentionFactor = yarn.attentionFactor ?? 1 + 0.1 * Math.log(yarn.factor);
    if (!Number.isFinite(attentionFactor) || attentionFactor <= 0) throw new Error('Invalid YaRN attention factor.');
    const correction = (rotations: number) =>
      (dim * Math.log(yarn.originalContextLength / (rotations * 2 * Math.PI))) / (2 * Math.log(theta));
    low = Math.max(0, Math.floor(correction(yarn.betaFast)));
    high = Math.min(dim - 1, Math.ceil(correction(yarn.betaSlow)));
    if (low === high) high += 0.001;
  }
  for (let i = 0; i < invFreq.length; i++) {
    let frequency = 1 / Math.pow(theta, (2 * i) / dim);
    if (type === 'linear') frequency /= scaling!.factor;
    if (yarn) {
      const ramp = Math.min(1, Math.max(0, (i - low) / (high - low)));
      frequency *= 1 - ramp + ramp / yarn.factor;
    }
    if (scaling && 'type' in scaling && scaling.type === 'llama3') {
      const { originalContextLength, lowFreqFactor, highFreqFactor, factor } = scaling;
      if (![originalContextLength, lowFreqFactor, highFreqFactor].every(Number.isFinite))
        throw new Error('Invalid Llama 3 RoPE frequency range.');
      if (!(originalContextLength > 0 && lowFreqFactor > 0 && highFreqFactor > lowFreqFactor))
        throw new Error('Invalid Llama 3 RoPE frequency range.');
      const wavelength = (2 * Math.PI) / frequency;
      if (wavelength > originalContextLength / lowFreqFactor) frequency /= factor;
      else if (wavelength >= originalContextLength / highFreqFactor) {
        const smooth = (originalContextLength / wavelength - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
        frequency *= (1 - smooth) / factor + smooth;
      }
    }
    invFreq[i] = frequency;
  }
  return { invFreq, attentionFactor };
}
