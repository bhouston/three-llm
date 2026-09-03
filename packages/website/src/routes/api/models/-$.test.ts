import { describe, expect, it } from 'vitest';

import { MODELS_CACHE_CONTROL, MODELS_CDN_CACHE_CONTROL } from '@/lib/gcs-models';

import { ifNoneMatchMatches, notModifiedHeaders, responseHeaders } from './$';

describe('responseHeaders', () => {
  it('sets cacheable headers for virtual model chunks', () => {
    const headers = responseHeaders(
      {
        contentType: 'application/octet-stream',
        etag: '"source-etag"',
        lastModified: '2026-09-03T16:00:00.000Z',
        size: 30 * 1024 * 1024,
      },
      24 * 1024 * 1024,
      { start: 0, end: 24 * 1024 * 1024 - 1 },
      { part: 0, partSize: 24 * 1024 * 1024 },
    );

    expect(headers.get('content-type')).toBe('application/octet-stream');
    expect(headers.get('content-length')).toBe(String(24 * 1024 * 1024));
    expect(headers.get('accept-ranges')).toBe('bytes');
    expect(headers.get('cache-control')).toBe(MODELS_CACHE_CONTROL);
    expect(headers.get('cdn-cache-control')).toBe(MODELS_CDN_CACHE_CONTROL);
    expect(headers.get('cloudflare-cdn-cache-control')).toBe(MODELS_CDN_CACHE_CONTROL);
    expect(headers.get('etag')).toBe('W/"source-etag-part-0-size-25165824"');
    expect(headers.get('last-modified')).toBe('Thu, 03 Sep 2026 16:00:00 GMT');
    expect(headers.get('content-range')).toBeNull();
    expect(headers.get('x-model-content-range')).toBe('bytes 0-25165823/31457280');
  });

  it('sets Content-Range for real range responses', () => {
    const headers = responseHeaders(
      {
        contentType: 'application/octet-stream',
        size: 100,
      },
      10,
      { start: 10, end: 19 },
      undefined,
    );

    expect(headers.get('content-length')).toBe('10');
    expect(headers.get('content-range')).toBe('bytes 10-19/100');
  });

  it('sets validator headers for 304 responses without body headers', () => {
    const headers = notModifiedHeaders(
      {
        contentType: 'application/octet-stream',
        etag: '"source-etag"',
        lastModified: '2026-09-03T16:00:00.000Z',
        size: 30 * 1024 * 1024,
      },
      { part: 0, partSize: 24 * 1024 * 1024 },
    );

    expect(headers.get('cache-control')).toBe(MODELS_CACHE_CONTROL);
    expect(headers.get('cdn-cache-control')).toBe(MODELS_CDN_CACHE_CONTROL);
    expect(headers.get('cloudflare-cdn-cache-control')).toBe(MODELS_CDN_CACHE_CONTROL);
    expect(headers.get('etag')).toBe('W/"source-etag-part-0-size-25165824"');
    expect(headers.get('last-modified')).toBe('Thu, 03 Sep 2026 16:00:00 GMT');
    expect(headers.get('content-type')).toBeNull();
    expect(headers.get('content-length')).toBeNull();
    expect(headers.get('content-range')).toBeNull();
  });
});

describe('ifNoneMatchMatches', () => {
  it('matches wildcard, list, and weak-equivalent ETags', () => {
    expect(ifNoneMatchMatches('*', '"etag"')).toBe(true);
    expect(ifNoneMatchMatches('"other", W/"etag"', '"etag"')).toBe(true);
    expect(ifNoneMatchMatches('"other", "etag"', 'W/"etag"')).toBe(true);
  });

  it('ignores missing or non-matching validators', () => {
    expect(ifNoneMatchMatches(null, '"etag"')).toBe(false);
    expect(ifNoneMatchMatches('"etag"', null)).toBe(false);
    expect(ifNoneMatchMatches('"other"', '"etag"')).toBe(false);
  });
});
