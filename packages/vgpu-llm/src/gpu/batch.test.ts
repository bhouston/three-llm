import { expect, it, vi } from 'vitest';
import { init } from 'vgpu/mock';
import { batchableCompute, withComputeBatch } from './batch.js';

it('falls back to public dispatch when vgpu does not expose encodable handles', async () => {
  const gpu = await init();
  try {
    const dispatch = vi.fn();
    const original = {
      dispatch,
      set() {
        return this;
      },
    };
    const pass = batchableCompute(gpu, original);
    withComputeBatch(gpu, () => pass.dispatch(2, 3, 4));
    expect(dispatch).toHaveBeenCalledExactlyOnceWith(2, 3, 4);
  } finally {
    gpu.dispose();
  }
});
