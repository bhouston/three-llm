import { GPT2Tokenizer, QWEN_TOKEN_PATTERN } from './GPT2Tokenizer.js';
import { architectureFor, recipeFor } from './DecoderRecipe.js';
import { convertAllTensors, createProgress, detectLanguagePrefix, fetchJSON, isEmbeddingTensorName, isolateEmbeddingTensors, unwrapTextConfig } from './tensors.js';
import { loadSafetensorsModel } from './SafeTensorsLoader.js';
import { UnigramTokenizer } from './UnigramTokenizer.js';
import type {
  Architecture,
  DecoderRecipe,
  HFModelBundle,
  HuggingFaceConfig,
  LoaderOptions,
  TensorMap,
  Tokenizer,
} from '../types.js';

/**
 * Shared Hugging Face directory load: config, tokenizer, safetensors shards.
 *
 */

function normalizeRoot(baseURL: string): string {
  return baseURL.endsWith('/') ? baseURL : `${baseURL}/`;
}

function detectPrefix(tensors: TensorMap, architecture: Architecture): string {
  if (architecture === 'gpt2') {
    return tensors['transformer.wte.weight'] !== undefined ? 'transformer.' : '';
  }

  return detectLanguagePrefix(tensors);
}

async function loadTokenizer(
  root: string,
  recipe: DecoderRecipe,
  config: HuggingFaceConfig,
  options: LoaderOptions,
  report: (message: string) => Promise<void>,
): Promise<Tokenizer> {
  if (recipe.tokenizer === 'unigram') {
    await report('Loading tokenizer.json (this file is large)');
    const tokenizer = await UnigramTokenizer.fromURL(root, options);
    const text = unwrapTextConfig(config);

    if (text.bos_token_id !== undefined) tokenizer.bosTokenId = text.bos_token_id;
    if (text.eos_token_id !== undefined) {
      tokenizer.eosTokenId = text.eos_token_id;
      tokenizer.endOfTextTokenId = Array.isArray(text.eos_token_id) ? text.eos_token_id[0] : text.eos_token_id;
    }

    return tokenizer;
  }

  if (recipe.tokenizer === 'qwen') {
    await report('Loading vocab.json');
    let addedTokens: Array<{ id: number; content: string }> = [];

    try {
      const tokenizerConfig = await fetchJSON<{ added_tokens_decoder?: Record<string, { content: string }> }>(
        `${root}tokenizer_config.json`,
        'HFModelBundle',
      );
      const decoder = tokenizerConfig.added_tokens_decoder || {};
      addedTokens = Object.keys(decoder).map((id) => ({
        id: Number(id),
        content: decoder[id].content,
      }));
    } catch {
      await report('tokenizer_config.json missing added tokens; chat specials may BPE-split');
    }

    return GPT2Tokenizer.fromURLs(`${root}vocab.json`, `${root}merges.txt`, {
      tokenPattern: QWEN_TOKEN_PATTERN,
      endOfTextToken: '<|endoftext|>',
      addedTokens,
    });
  }

  await report('Loading vocab.json');
  return GPT2Tokenizer.fromURLs(`${root}vocab.json`, `${root}merges.txt`);
}

async function loadHFModelBundle(baseURL: string, options: LoaderOptions = {}): Promise<HFModelBundle> {
  const root = normalizeRoot(baseURL);
  const report = createProgress(options.label || 'HFModelBundle', options.onProgress);

  await report('Loading config.json');
  const rawConfig = await fetchJSON<HuggingFaceConfig>(`${root}config.json`, options.label || 'HFModelBundle');
  const architecture = architectureFor(rawConfig);
  const recipe = recipeFor(rawConfig);
  const text = unwrapTextConfig(rawConfig);
  await report(`Using ${architecture} map for model_type "${rawConfig.model_type}"`);
  await report(`${recipe.layerCount} layers, hidden ${recipe.hiddenSize}, vocab ${recipe.vocabSize}`);

  const tokenizer = await loadTokenizer(root, recipe, rawConfig, options, report);
  const tensors = await loadSafetensorsModel(root, {
    onProgress: options.onProgress,
    label: options.label || 'HFModelBundle',
    keepTensor: recipe.keepTensor || options.keepTensor,
  });
  isolateEmbeddingTensors(tensors);
  await convertAllTensors(tensors, options.onProgress, options.label || 'HFModelBundle', isEmbeddingTensorName);

  return {
    root,
    rawConfig,
    config: text,
    architecture,
    recipe,
    tokenizer,
    tensors,
    prefix: detectPrefix(tensors, architecture),
  };
}

export { detectPrefix, loadHFModelBundle, normalizeRoot };
