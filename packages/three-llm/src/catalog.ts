import { fetchResource } from './load/tensors.js';
import type { ModelCatalogEntry } from './types.js';

export const MODELS_BUCKET_URL = 'https://storage.googleapis.com/three-llm';

export const DEFAULT_MODEL_ID = 'smollm2';
export const DESKTOP_RECOMMENDED_MODEL_ID = 'qwen3.5-0.8b';
export const MOBILE_RECOMMENDED_MODEL_ID = 'smollm2';
export const MOBILE_MODEL_MAX_BYTES = 500_000_000;
export const MEDIUM_MODEL_MAX_BYTES = 2_000_000_000;

export type CatalogWeightClass = 'small' | 'medium' | 'large';

export const MODEL_CATALOG: ModelCatalogEntry[] = [
  {
    id: 'tinystories',
    name: 'TinyStories GPT-2 3M',
    url: 'https://huggingface.co/segestic/Tinystories-gpt-0.1-3m/resolve/main/',
    localUrl: '/api/models/tinystories-gpt2-0.1-3m/',
    prompt: 'Once upon a time,',
    note: "Children's stories. Dense GPT-2 at ~3.7M parameters, a few megabytes from Hugging Face.",
    sizeHint: '15 MB',
  },
  {
    id: 'smollm2',
    name: 'SmolLM2 135M',
    url: 'https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/',
    localUrl: '/api/models/smollm2-135m/',
    prompt: 'Once upon a time,',
    note: 'Llama-style: RMSNorm, RoPE, grouped-query attention, SwiGLU. ~270 MB BF16 from Hugging Face.',
    sizeHint: '269 MB',
  },
  {
    id: 'qwen3.5-0.8b',
    name: 'Qwen3.5 0.8B',
    url: 'https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/',
    localUrl: '/api/models/qwen3.5-0.8b/',
    prompt: 'Once upon a time,',
    note: 'Qwen3.5 0.8B hybrid: Gated DeltaNet linear attention plus gated full attention. About 1.8 GB BF16. Text-only decode; vision tensors are skipped. Thinking is off by default so replies skip the <think> block.',
    sizeHint: '1.7 GB',
  },
  {
    id: 'gemma-3-1b-it',
    name: 'Gemma 3 1B IT',
    url: 'https://huggingface.co/google/gemma-3-1b-it/resolve/main/',
    localUrl: '/api/models/gemma-3-1b-it/',
    prompt: 'Write a short friendly introduction to WebGPU.',
    note: 'Gemma 3 instruction-tuned text decoder with alternating local and full attention. About 1.9 GB BF16. Text-only inference; multimodal inputs are not supported.',
    sizeHint: '1.9 GB',
  },
];

export function catalogSizeBytes(sizeHint: string): number {
  const match = /^(\d+(?:\.\d+)?) (MB|GB)$/.exec(sizeHint);
  if (!match) throw new Error(`Invalid catalog size hint: ${sizeHint}`);
  const value = Number(match[1]);
  return match[2] === 'GB' ? value * 1_000_000_000 : value * 1_000_000;
}

export function catalogWeightClass(entry: ModelCatalogEntry): CatalogWeightClass {
  const bytes = catalogSizeBytes(entry.sizeHint);
  if (bytes <= MOBILE_MODEL_MAX_BYTES) return 'small';
  if (bytes < MEDIUM_MODEL_MAX_BYTES) return 'medium';
  return 'large';
}

export function isMobileCatalogModel(entry: ModelCatalogEntry): boolean {
  return catalogSizeBytes(entry.sizeHint) <= MOBILE_MODEL_MAX_BYTES;
}

export function catalogLabel(entry: ModelCatalogEntry): string {
  const badge = entry.badge ? ` (${entry.badge})` : '';
  return `${entry.name}${badge} [${entry.sizeHint}]`;
}

export async function resolveModelURL(entry: ModelCatalogEntry): Promise<string> {
  if (entry.localUrl) {
    try {
      const response = await fetchResource(`${entry.localUrl}config.json`, undefined, 'MODEL_CATALOG');
      if (response.ok) return entry.localUrl;
    } catch {
      // Fall through to the remote Hugging Face URL.
    }
  }

  return entry.url;
}
