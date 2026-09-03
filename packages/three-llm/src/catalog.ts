import type { ModelCatalogEntry } from './types.js';

export const MODELS_BUCKET_URL = 'https://storage.googleapis.com/three-llm';

export const DEFAULT_MODEL_ID = 'smollm2';

export const MODEL_CATALOG: ModelCatalogEntry[] = [
  {
    id: 'tinystories',
    name: 'TinyStories GPT-2 3M',
    url: 'https://huggingface.co/segestic/Tinystories-gpt-0.1-3m/resolve/main/',
    localUrl: `${MODELS_BUCKET_URL}/tinystories-gpt2-0.1-3m/`,
    prompt: 'Once upon a time,',
    note: "Children's stories. Dense GPT-2 at ~3.7M parameters, a few megabytes from Hugging Face.",
    sizeHint: '15 MB',
  },
  {
    id: 'gpt2',
    name: 'GPT-2 124M',
    url: 'https://huggingface.co/openai-community/gpt2/resolve/main/',
    localUrl: `${MODELS_BUCKET_URL}/gpt2/`,
    prompt: 'Once upon a time,',
    note: 'Classic dense GPT-2. About 500 MB of float32 weights from Hugging Face.',
    sizeHint: '548 MB',
  },
  {
    id: 'smollm2',
    name: 'SmolLM2 135M',
    url: 'https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/',
    localUrl: `${MODELS_BUCKET_URL}/smollm2-135m/`,
    prompt: 'Once upon a time,',
    note: 'Llama-style: RMSNorm, RoPE, grouped-query attention, SwiGLU. ~270 MB BF16 from Hugging Face.',
    sizeHint: '269 MB',
  },
  {
    id: 'qwen3.5-0.8b',
    name: 'Qwen3.5 0.8B',
    url: 'https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/',
    localUrl: `${MODELS_BUCKET_URL}/qwen3.5-0.8b/`,
    prompt: 'Once upon a time,',
    note: 'Qwen3.5 0.8B hybrid: Gated DeltaNet linear attention plus gated full attention. About 1.8 GB BF16. Text-only decode; vision tensors are skipped. Thinking is off by default so replies skip the <think> block.',
    sizeHint: '1.7 GB',
    badge: 'Best Results',
  },
  {
    id: 'phi-1.5',
    name: 'Phi-1.5 1.3B',
    url: 'https://huggingface.co/microsoft/phi-1_5/resolve/main/',
    localUrl: `${MODELS_BUCKET_URL}/phi-1.5/`,
    prompt: 'Once upon a time,',
    note: 'Microsoft Phi-1.5 (LayerNorm, partial RoPE, parallel attention + MLP). About 2.8 GB FP16 from Hugging Face.',
    sizeHint: '2.8 GB',
  },
];

export function catalogLabel(entry: ModelCatalogEntry): string {
  const badge = entry.badge ? ` (${entry.badge})` : '';
  return `${entry.name}${badge} [${entry.sizeHint}]`;
}

export async function resolveModelURL(entry: ModelCatalogEntry): Promise<string> {
  if (entry.localUrl) {
    try {
      const response = await fetch(`${entry.localUrl}config.json`);
      if (response.ok) return entry.localUrl;
    } catch {
      // Fall through to the remote Hugging Face URL.
    }
  }

  return entry.url;
}
