import { describe, expect, it } from 'vitest';

import { modelsObjectUrl, objectPathFromSplat } from './gcs-models';

describe('objectPathFromSplat', () => {
  it('accepts nested model object paths', () => {
    expect(objectPathFromSplat('gpt2/config.json')).toBe('gpt2/config.json');
    expect(objectPathFromSplat('phi-1.5/model.safetensors-1-of-2.safetensors')).toBe(
      'phi-1.5/model.safetensors-1-of-2.safetensors',
    );
  });

  it('rejects empty, traversal, and malformed paths', () => {
    expect(objectPathFromSplat(undefined)).toBeUndefined();
    expect(objectPathFromSplat('')).toBeUndefined();
    expect(objectPathFromSplat('../secret')).toBeUndefined();
    expect(objectPathFromSplat('gpt2/../config.json')).toBeUndefined();
    expect(objectPathFromSplat('gpt2//config.json')).toBeUndefined();
    expect(objectPathFromSplat('%E0%A4%A')).toBeUndefined();
  });
});

describe('modelsObjectUrl', () => {
  it('maps object names onto the public bucket URL', () => {
    expect(modelsObjectUrl('gpt2/config.json')).toBe('https://storage.googleapis.com/three-llm/gpt2/config.json');
  });
});
