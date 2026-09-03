export const MODELS_BUCKET = 'three-llm';
export const MODELS_CACHE_CONTROL = 'public, max-age=31536000, s-maxage=31536000, immutable';
export const MODELS_CDN_CACHE_CONTROL = 'public, max-age=31536000, immutable';
export const DEFAULT_MODEL_CHUNK_BYTES = 24 * 1024 * 1024;
export const MAX_MODEL_CHUNK_BYTES = 24 * 1024 * 1024;
export const MODEL_DETAILS_CACHE_CONTROL = 'public, max-age=3600, s-maxage=86400';

export interface ModelSafetensorFile {
  name: string;
  size: number;
}

export interface ModelDetailsResponse {
  model: string;
  files: ModelSafetensorFile[];
}

export type ModelChunkRequest =
  | { kind: 'none' }
  | { kind: 'invalid'; reason: string }
  | { kind: 'chunk'; part: number; partSize: number; start: number; end: number };

export function objectPathFromSplat(splat: string | undefined): string | undefined {
  if (splat === undefined || splat.trim() === '') return undefined;

  let decoded: string;
  try {
    decoded = decodeURIComponent(splat);
  } catch {
    return undefined;
  }

  const normalized = decoded.replaceAll('\\', '/').replace(/^\/+/, '');
  if (
    normalized === '' ||
    normalized.split('/').some((segment) => segment === '' || segment === '.' || segment === '..')
  ) {
    return undefined;
  }

  return normalized;
}

export function modelDirectoryFromSplat(splat: string | undefined): string | undefined {
  const objectPath = objectPathFromSplat(splat);
  if (objectPath === undefined || objectPath.includes('/')) return undefined;

  return objectPath;
}

export function modelsObjectUrl(objectPath: string): string {
  return `https://storage.googleapis.com/${MODELS_BUCKET}/${objectPath}`;
}

function parsePositiveInteger(value: string | null): number | undefined {
  if (value === null || /^\d+$/.test(value) === false) return undefined;

  const parsed = Number(value);
  if (Number.isSafeInteger(parsed) === false || parsed < 0) return undefined;
  return parsed;
}

function parseChunkSize(value: string | null): number | undefined {
  if (value === null || value.trim() === '') return DEFAULT_MODEL_CHUNK_BYTES;

  const match = value.trim().match(/^(\d+)(?:\s*(b|bytes?|mib|mb))?$/i);
  if (match === null) return undefined;

  const amount = Number(match[1]);
  if (Number.isSafeInteger(amount) === false || amount <= 0) return undefined;

  const unit = match[2]?.toLowerCase();
  const multiplier = unit === 'mb' || unit === 'mib' ? 1024 * 1024 : 1;
  const bytes = amount * multiplier;

  if (Number.isSafeInteger(bytes) === false || bytes <= 0 || bytes > MAX_MODEL_CHUNK_BYTES) return undefined;
  return bytes;
}

export function modelChunkRequestFromURL(url: URL): ModelChunkRequest {
  const partParam = url.searchParams.get('part') ?? url.searchParams.get('shardId');
  const sizeParam = url.searchParams.get('partSize') ?? url.searchParams.get('shardSize');

  if (partParam === null && sizeParam === null) return { kind: 'none' };

  const part = parsePositiveInteger(partParam);
  if (part === undefined) return { kind: 'invalid', reason: 'Invalid chunk part.' };

  const partSize = parseChunkSize(sizeParam);
  if (partSize === undefined) return { kind: 'invalid', reason: 'Invalid chunk size.' };

  const start = part * partSize;
  const end = start + partSize - 1;
  if (Number.isSafeInteger(start) === false || Number.isSafeInteger(end) === false) {
    return { kind: 'invalid', reason: 'Chunk range is too large.' };
  }

  return { kind: 'chunk', part, partSize, start, end };
}
