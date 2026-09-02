import { createFileRoute } from '@tanstack/react-router';

import { MODELS_CACHE_CONTROL, modelsObjectUrl, objectPathFromSplat } from '@/lib/gcs-models';

const FORWARDED_HEADERS = ['content-type', 'content-length', 'content-range', 'accept-ranges', 'etag', 'last-modified'];

async function proxyModel(request: Request, splat: string | undefined): Promise<Response> {
  const objectPath = objectPathFromSplat(splat);
  if (!objectPath) {
    return new Response('Not found', { status: 404 });
  }

  const headers = new Headers();
  const range = request.headers.get('Range');
  if (range) headers.set('Range', range);

  const upstream = await fetch(modelsObjectUrl(objectPath), {
    method: request.method === 'HEAD' ? 'HEAD' : 'GET',
    headers,
  });

  if (upstream.status === 404 || upstream.status === 403) {
    return new Response('Not found', { status: 404 });
  }

  const responseHeaders = new Headers();
  for (const name of FORWARDED_HEADERS) {
    const value = upstream.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }
  responseHeaders.set('Cache-Control', MODELS_CACHE_CONTROL);

  return new Response(request.method === 'HEAD' ? null : upstream.body, {
    status: upstream.status,
    headers: responseHeaders,
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
