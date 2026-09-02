import { MODEL_CATALOG } from './catalog.js';
import { DecoderCPURunner } from './DecoderCPURunner.js';
import { architectureFor } from './DecoderRecipe.js';
import { DecoderTSLRunner } from './DecoderTSLRunner.js';
import { DecoderWeights } from './DecoderWeights.js';
import { loadHFModelBundle, normalizeRoot } from './HFModelBundle.js';
import { createProgress } from './LLMTensors.js';
import { QwenCPURunner } from './QwenCPURunner.js';
import { QwenTSLRunner } from './QwenTSLRunner.js';
import { QwenWeights } from './QwenWeights.js';
import type { LoaderOptions, RunnerOptions } from './types.js';

async function loadWeights(baseURL: string, options: LoaderOptions = {}) {
  const report = createProgress('LLMFactory', options.onProgress);

  try {
    const bundle = await loadHFModelBundle(baseURL, { ...options, label: 'LLMFactory' });
    await report(`Using ${bundle.architecture} loader for model_type "${bundle.rawConfig.model_type}"`);

    if (bundle.recipe.graph === 'qwen35') return QwenWeights.fromBundle(bundle, options);

    return DecoderWeights.fromBundle(bundle, options);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`LLMFactory: failed to load "${normalizeRoot(baseURL)}": ${message}`);
  }
}

async function createTSLRunner(baseURL: string, options: RunnerOptions = {}) {
  const report = createProgress('LLMFactory', options.onProgress);
  const weights = await loadWeights(baseURL, options);
  await report(`Building ${weights.architecture} GPU runner (${weights.layerCount} layers, vocab ${weights.vocabSize})...`);

  const runner =
    weights.recipe.graph === 'qwen35'
      ? new QwenTSLRunner(weights as InstanceType<typeof QwenWeights>, options)
      : new DecoderTSLRunner(weights as InstanceType<typeof DecoderWeights>, options);

  await report('GPU runner ready');
  return runner;
}

async function createCPURunner(baseURL: string, options: RunnerOptions = {}) {
  const weights = await loadWeights(baseURL, options);

  if (weights.recipe.graph === 'qwen35') return new QwenCPURunner(weights as InstanceType<typeof QwenWeights>, options);

  return new DecoderCPURunner(weights as InstanceType<typeof DecoderWeights>, options);
}

export { MODEL_CATALOG, architectureFor, createCPURunner, createTSLRunner, loadWeights };
