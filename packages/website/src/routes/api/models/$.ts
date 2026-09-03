import { Readable } from 'node:stream';
import { Storage } from '@google-cloud/storage';
import { createFileRoute } from '@tanstack/react-router';

import {
  MODELS_BUCKET,
  MODELS_CACHE_CONTROL,
  MODELS_CDN_CACHE_CONTROL,
  modelChunkRequestFromURL,
  objectPathFromSplat,
} from '@/lib/gcs-models';

const storage = new Storage();
const bucket = storage.bucket(MODELS_BUCKET);

interface ObjectMetadata {
  contentType: string;
  etag?: string;
  lastModified?: string;
  size: number;
}

interface ByteRange {
  start: number;
  end: number;
}

interface ChunkIdentity {
  part: number;
  partSize: number;
}

async function modelObjectMetadata(objectPath: string): Promise<ObjectMetadata | undefined> {
  try {
    const [metadata] = await bucket.file(objectPath).getMetadata();
    const size = Number(metadata.size);
    if (Number.isSafeInteger(size) === false || size < 0) return undefined;

    return {
      contentType: metadata.contentType || 'application/octet-stream',
      etag: metadata.etag,
      lastModified: metadata.updated,
      size,
    };
  } catch (error) {
    const code = (error as { code?: unknown }).code;
    if (code === 403 || code === 404) return undefined;
    throw error;
  }
}

function parseRangeHeader(range: string | null, total: number): ByteRange | undefined {
  if (range === null) return undefined;

  const match = range.match(/^bytes=(\d*)-(\d*)$/);
  if (match === null) return undefined;

  const [, startText, endText] = match;
  if (startText === '' && endText === '') return undefined;

  if (startText === '') {
    const suffixLength = Number(endText);
    if (Number.isSafeInteger(suffixLength) === false || suffixLength <= 0) return undefined;

    return {
      start: Math.max(0, total - suffixLength),
      end: total - 1,
    };
  }

  const start = Number(startText);
  const end = endText === '' ? total - 1 : Number(endText);
  if (Number.isSafeInteger(start) === false || Number.isSafeInteger(end) === false || start < 0 || end < start) {
    return undefined;
  }

  if (start >= total) return undefined;
  return { start, end: Math.min(end, total - 1) };
}

function etagValue(metadata: ObjectMetadata, chunk: ChunkIdentity | undefined): string | undefined {
  if (!metadata.etag) return undefined;
  if (!chunk) return metadata.etag;

  const source = metadata.etag.replace(/^W\//, '').replaceAll(/["\\]/g, '');
  return `W/"${source}-part-${chunk.part}-size-${chunk.partSize}"`;
}

function httpDate(value: string | undefined): string | undefined {
  if (value === undefined) return undefined;

  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) return undefined;

  return new Date(timestamp).toUTCString();
}

function setCacheValidatorHeaders(headers: Headers, metadata: ObjectMetadata, chunk: ChunkIdentity | undefined): void {
  headers.set('accept-ranges', 'bytes');
  headers.set('Cache-Control', MODELS_CACHE_CONTROL);
  headers.set('CDN-Cache-Control', MODELS_CDN_CACHE_CONTROL);
  headers.set('Cloudflare-CDN-Cache-Control', MODELS_CDN_CACHE_CONTROL);

  const etag = etagValue(metadata, chunk);
  if (etag) headers.set('etag', etag);
  const lastModified = httpDate(metadata.lastModified);
  if (lastModified) headers.set('last-modified', lastModified);
}

function weakETagValue(etag: string): string {
  return etag.trim().replace(/^W\//i, '');
}

export function ifNoneMatchMatches(ifNoneMatch: string | null, etag: string | null): boolean {
  if (ifNoneMatch === null || etag === null) return false;

  return ifNoneMatch
    .split(',')
    .map((value) => value.trim())
    .some((value) => value === '*' || weakETagValue(value) === weakETagValue(etag));
}

export function responseHeaders(
  metadata: ObjectMetadata,
  length: number,
  range: ByteRange | undefined,
  chunk: ChunkIdentity | undefined,
): Headers {
  const headers = new Headers();
  const virtualChunk = chunk !== undefined;

  headers.set('content-type', metadata.contentType);
  headers.set('content-length', String(length));
  setCacheValidatorHeaders(headers, metadata, chunk);

  if (range) {
    const contentRange = `bytes ${range.start}-${range.end}/${metadata.size}`;

    if (virtualChunk) headers.set('x-model-content-range', contentRange);
    else headers.set('content-range', contentRange);
  }

  return headers;
}

export function notModifiedHeaders(metadata: ObjectMetadata, chunk: ChunkIdentity | undefined): Headers {
  const headers = new Headers();
  setCacheValidatorHeaders(headers, metadata, chunk);
  return headers;
}

function streamObject(objectPath: string, range: ByteRange | undefined): ReadableStream {
  const stream = bucket.file(objectPath).createReadStream(range);
  return Readable.toWeb(stream) as ReadableStream;
}

async function proxyModel(request: Request, splat: string | undefined): Promise<Response> {
  const objectPath = objectPathFromSplat(splat);
  if (!objectPath) {
    return new Response('Not found', { status: 404 });
  }

  const url = new URL(request.url);
  const chunk = modelChunkRequestFromURL(url);
  if (chunk.kind === 'invalid') {
    return new Response(chunk.reason, { status: 400 });
  }

  const virtualChunk = chunk.kind === 'chunk';
  const chunkIdentity = virtualChunk ? { part: chunk.part, partSize: chunk.partSize } : undefined;
  const metadata = await modelObjectMetadata(objectPath);
  if (metadata === undefined) {
    return new Response('Not found', { status: 404 });
  }

  let range: ByteRange | undefined;
  if (virtualChunk) {
    if (chunk.start >= metadata.size) return new Response('Not found', { status: 404 });
    range = { start: chunk.start, end: Math.min(chunk.end, metadata.size - 1) };
  } else {
    range = parseRangeHeader(request.headers.get('Range'), metadata.size);
    if (request.headers.has('Range') && range === undefined) {
      return new Response(null, {
        status: 416,
        headers: {
          'Content-Range': `bytes */${metadata.size}`,
        },
      });
    }
  }

  const length = range ? range.end - range.start + 1 : metadata.size;
  const headers = responseHeaders(metadata, length, range, chunkIdentity);
  if (ifNoneMatchMatches(request.headers.get('If-None-Match'), headers.get('etag'))) {
    return new Response(null, {
      status: 304,
      headers: notModifiedHeaders(metadata, chunkIdentity),
    });
  }

  const responseBody = request.method === 'HEAD' ? null : streamObject(objectPath, range);

  return new Response(responseBody, {
    status: virtualChunk ? 200 : range ? 206 : 200,
    headers,
  });
}

export const Route = createFileRoute('/api/models/$')({
  server: {
    handlers: {
      GET: async ({ params, request }) => proxyModel(request, params._splat),
      HEAD: async ({ params, request }) => proxyModel(request, params._splat),
    },
  },
});
