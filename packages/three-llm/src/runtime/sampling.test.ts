import { expect, it } from 'vitest';
import { gpuCandidateCount } from './generate.js';
import { sampleTopK } from './math.js';

it.each([0, -1, 1.9])('normalizes topK=%s consistently for CPU and GPU sampling', (topK) => {
  const options = { topK, temperature: 1, random: () => 0.99 };
  expect(gpuCandidateCount(options, 8)).toBe(1);
  expect(sampleTopK(new Float32Array([1, 3, 2]), options)).toBe(1);
});
