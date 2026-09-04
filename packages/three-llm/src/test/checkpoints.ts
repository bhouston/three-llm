import { MODELS_BUCKET_URL, MODEL_CATALOG } from '../catalog.js';
import type { ModelCatalogEntry } from '../types.js';

const TEST_CHECKPOINTS: ModelCatalogEntry[] = [
  {
    id: 'gpt2',
    name: 'GPT-2 124M',
    url: 'https://huggingface.co/openai-community/gpt2/resolve/main/',
    localUrl: '/api/models/gpt2/',
    prompt: 'Once upon a time,',
    note: 'Classic dense GPT-2 used in checkpoint tests, not listed in the demo catalog.',
    sizeHint: '548 MB',
  },
  {
    id: 'phi-1.5',
    name: 'Phi-1.5 1.3B',
    url: 'https://huggingface.co/microsoft/phi-1_5/resolve/main/',
    localUrl: '/api/models/phi-1.5/',
    prompt: 'Once upon a time,',
    note: 'Microsoft Phi-1.5 used in checkpoint tests, not listed in the demo catalog.',
    sizeHint: '2.8 GB',
  },
];

export const STORY_PROMPT = 'Once upon a time,';
export const PARIS_PROMPT = 'Paris was beautiful in the fall.';
export const GPT2_STORY_GREEDY_TEXT = 'Once upon a time, the world was a place of great beauty';
export const GPT2_PARIS_GREEDY_TEXT = 'Paris was beautiful in the fall.\n\n"I was in the middle';
export const GREEDY = { maxNewTokens: 8, temperature: 0, topK: 1 };
export const GREEDY_SHORT = { maxNewTokens: 4, temperature: 0, topK: 1 };
export const PHI15_GREEDY_TEXT = 'Once upon a time, in a small town called Sunnyville,';

const localCheckpoints = new Map<string, Promise<unknown> | null>();

export function catalogEntry(id: string): ModelCatalogEntry {
  const entry =
    MODEL_CATALOG.find((candidate) => candidate.id === id) ??
    TEST_CHECKPOINTS.find((candidate) => candidate.id === id);
  if (!entry) throw new Error(`Missing catalog entry "${id}"`);
  return entry;
}

export function checkpointRoot(entry: ModelCatalogEntry, mode: 'node' | 'browser') {
  const root = entry.localUrl || entry.url;
  if (mode === 'browser' && root.startsWith(MODELS_BUCKET_URL)) {
    return root.replace(MODELS_BUCKET_URL, '/api/models');
  }
  return root;
}

function isCapacityError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  return /allocation failed|out of memory|not enough memory|device lost|resource creation failed/i.test(message);
}

async function localCheckpointReady(root: string) {
  try {
    const configResponse = await fetch(`${root}config.json`);
    if (configResponse.ok === false) return false;

    const modelResponse = await fetch(`${root}model.safetensors`, { method: 'HEAD' });
    const indexResponse = modelResponse.ok
      ? new Response(null, { status: 404 })
      : await fetch(`${root}model.safetensors.index.json`);

    return modelResponse.ok || indexResponse.ok;
  } catch {
    return false;
  }
}

interface Loader<T> {
  fromURL(root: string): Promise<T>;
}

export async function loadLocalCheckpoint<T>(skip: () => never, LoaderClass: Loader<T>, root: string): Promise<T> {
  if (localCheckpoints.has(root)) {
    const loaded = localCheckpoints.get(root);
    if (loaded === null) skip();
    return (await loaded) as T;
  }

  if ((await localCheckpointReady(root)) === false) {
    localCheckpoints.set(root, null);
    skip();
  }

  const loaded = LoaderClass.fromURL(root);
  localCheckpoints.set(root, loaded);

  try {
    return await loaded;
  } catch (error) {
    if (isCapacityError(error)) {
      localCheckpoints.set(root, null);
      skip();
    }

    localCheckpoints.delete(root);
    throw error;
  }
}
