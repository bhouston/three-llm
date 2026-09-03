import { Storage } from '@google-cloud/storage';
import { createFileRoute } from '@tanstack/react-router';

import {
  MODEL_DETAILS_CACHE_CONTROL,
  MODELS_BUCKET,
  type ModelDetailsResponse,
  type ModelSafetensorFile,
  modelDirectoryFromSplat,
} from '@/lib/gcs-models';

const storage = new Storage();
const bucket = storage.bucket(MODELS_BUCKET);

function parseObjectSize(size: unknown): number | undefined {
  const parsed = typeof size === 'number' ? size : typeof size === 'string' ? Number(size) : Number.NaN;
  if (Number.isSafeInteger(parsed) === false || parsed < 0) return undefined;

  return parsed;
}

async function safetensorFileSize(file: ReturnType<typeof bucket.file>): Promise<number | undefined> {
  const size = parseObjectSize(file.metadata.size);
  if (size !== undefined) return size;

  const [metadata] = await file.getMetadata();
  return parseObjectSize(metadata.size);
}

async function listModelSafetensors(model: string): Promise<ModelSafetensorFile[]> {
  const prefix = `${model}/`;
  const [files] = await bucket.getFiles({ prefix });
  const safetensorFiles = files.filter((file) => {
    const name = file.name.slice(prefix.length);
    return file.name.startsWith(prefix) && name !== '' && name.includes('/') === false && name.endsWith('.safetensors');
  });

  const sizes = await Promise.all(safetensorFiles.map((file) => safetensorFileSize(file)));
  return safetensorFiles
    .map((file, index) => {
      const size = sizes[index];
      if (size === undefined) return undefined;

      return {
        name: file.name.slice(prefix.length),
        size,
      };
    })
    .filter((file): file is ModelSafetensorFile => file !== undefined)
    .toSorted((a, b) => a.name.localeCompare(b.name));
}

function jsonResponse(body: ModelDetailsResponse, status = 200): Response {
  return Response.json(body, {
    status,
    headers: {
      'Cache-Control': MODEL_DETAILS_CACHE_CONTROL,
    },
  });
}

async function modelDetails(splat: string | undefined): Promise<Response> {
  const model = modelDirectoryFromSplat(splat);
  if (!model) return new Response('Not found', { status: 404 });

  try {
    const files = await listModelSafetensors(model);
    if (files.length === 0) return new Response('Not found', { status: 404 });

    return jsonResponse({ model, files });
  } catch (error) {
    const code = (error as { code?: unknown }).code;
    if (code === 403 || code === 404) return new Response('Not found', { status: 404 });
    throw error;
  }
}

export const Route = createFileRoute('/api/model-details/$')({
  server: {
    handlers: {
      GET: async ({ params }) => modelDetails(params._splat),
    },
  },
});
