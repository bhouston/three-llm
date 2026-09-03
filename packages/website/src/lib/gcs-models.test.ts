import { describe, expect, it } from 'vitest';

import {
  DEFAULT_MODEL_CHUNK_BYTES,
  MAX_MODEL_CHUNK_BYTES,
  modelChunkRequestFromURL,
  modelDirectoryFromSplat,
  modelsObjectUrl,
  objectPathFromSplat,
} from './gcs-models';

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

describe('modelDirectoryFromSplat', () => {
  it('accepts a single model directory segment', () => {
    expect(modelDirectoryFromSplat('smollm2-135m')).toBe('smollm2-135m');
  });

  it('rejects nested object paths and traversal', () => {
    expect(modelDirectoryFromSplat('smollm2-135m/model.safetensors')).toBeUndefined();
    expect(modelDirectoryFromSplat('../secret')).toBeUndefined();
  });
});

describe('modelChunkRequestFromURL', () => {
  it('builds byte ranges from part query parameters', () => {
    const url = new URL('https://example.test/api/models/gpt2/model.safetensors?part=2&partSize=24MB');

    expect(modelChunkRequestFromURL(url)).toEqual({
      kind: 'chunk',
      part: 2,
      partSize: 24 * 1024 * 1024,
      start: 48 * 1024 * 1024,
      end: 72 * 1024 * 1024 - 1,
    });
  });

  it('supports shardId aliases and a default chunk size', () => {
    const url = new URL('https://example.test/api/models/gpt2/model.safetensors?shardId=1');

    expect(modelChunkRequestFromURL(url)).toEqual({
      kind: 'chunk',
      part: 1,
      partSize: DEFAULT_MODEL_CHUNK_BYTES,
      start: DEFAULT_MODEL_CHUNK_BYTES,
      end: DEFAULT_MODEL_CHUNK_BYTES * 2 - 1,
    });
  });

  it('rejects invalid or oversized chunks', () => {
    expect(modelChunkRequestFromURL(new URL('https://example.test/model'))).toEqual({ kind: 'none' });
    expect(modelChunkRequestFromURL(new URL('https://example.test/model?part=-1&partSize=1'))).toMatchObject({
      kind: 'invalid',
    });
    expect(
      modelChunkRequestFromURL(new URL(`https://example.test/model?part=0&partSize=${MAX_MODEL_CHUNK_BYTES + 1}`)),
    ).toMatchObject({
      kind: 'invalid',
    });
  });
});
