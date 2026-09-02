export const MODELS_BUCKET = 'three-llm';
export const MODELS_CACHE_CONTROL = 'public, max-age=31536000, immutable';

export function objectPathFromSplat(splat: string | undefined): string | undefined {
  if (splat === undefined || splat.trim() === '') return undefined;

  let decoded: string;
  try {
    decoded = decodeURIComponent(splat);
  } catch {
    return undefined;
  }

  const normalized = decoded.replaceAll('\\', '/').replace(/^\/+/, '');
  if (normalized === '' || normalized.split('/').some((segment) => segment === '' || segment === '.' || segment === '..')) {
    return undefined;
  }

  return normalized;
}

export function modelsObjectUrl(objectPath: string): string {
  return `https://storage.googleapis.com/${MODELS_BUCKET}/${objectPath}`;
}
