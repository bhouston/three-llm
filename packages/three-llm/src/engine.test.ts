import { describe, expect, it } from 'vitest';

import { architectureFor, recipeFor } from './load/DecoderRecipe.js';
import { GPT2Tokenizer } from './load/GPT2Tokenizer.js';
import { generateSync, planPromptCache, prepareGenerationFromTokens, sharedPrefixLength } from './runtime/generate.js';
import {
  applyRoPE,
  geluNew,
  layerNorm,
  linear,
  logitSoftcap,
  rmsNorm,
  sampleTopK,
  silu,
  softmax,
  splitHeadGate,
} from './runtime/math.js';
import { bfloat16ToFloat32, convertAllTensors, float16ToFloat32, tensorToFloat32 } from './load/tensors.js';
import { catalogLabel, DEFAULT_MODEL_ID, MODEL_CATALOG } from './catalog.js';
import { QwenWeights } from './qwen/QwenWeights.js';
import { parseSafeTensors, resolveSafetensorFiles } from './load/SafeTensorsLoader.js';
import { resolveTensor } from './load/TensorNameMap.js';
import { UnigramTokenizer } from './load/UnigramTokenizer.js';
import type { Tensor, TensorMap, Tokenizer } from './types.js';

function closeArray(actual: ArrayLike<number>, expected: ArrayLike<number>, epsilon: number) {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs(actual[i]! - expected[i]!)).toBeLessThanOrEqual(epsilon);
  }
}

function createSafeTensorsFixture(dtype: 'F32' | 'F16' = 'F32', values: number[] = [1, 2, 3, 4]) {
  const bytesPerElement = dtype === 'F32' ? 4 : 2;
  const header = {
    values: {
      dtype,
      shape: [2, 2],
      data_offsets: [0, values.length * bytesPerElement],
    },
  };
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const buffer = new ArrayBuffer(8 + headerBytes.length + values.length * bytesPerElement);
  const view = new DataView(buffer);

  view.setUint32(0, headerBytes.length, true);
  view.setUint32(4, 0, true);
  new Uint8Array(buffer, 8, headerBytes.length).set(headerBytes);

  for (let i = 0; i < values.length; i++) {
    if (dtype === 'F32') view.setFloat32(8 + headerBytes.length + i * 4, values[i]!, true);
    else view.setUint16(8 + headerBytes.length + i * 2, values[i]!, true);
  }

  return buffer;
}

function fillSin(array: Float32Array, seed: number) {
  for (let i = 0; i < array.length; i++) array[i] = Math.sin(seed + i * 0.17) * 0.35;
  return array;
}

function makeTensor(name: string, shape: number[], seed: number): Tensor {
  const data = fillSin(new Float32Array(shape.reduce((product, value) => product * value, 1)), seed);
  return { name, dtype: 'F32', shape, data };
}

function tinyTokenizer(): Tokenizer {
  return {
    endOfTextTokenId: 0,
    encode() {
      return [1, 2];
    },
    decode(ids) {
      return ids.join(',');
    },
  };
}

describe('MODEL_CATALOG', () => {
  it('loads public Hugging Face checkpoints and skips Gemma', () => {
    const ids = MODEL_CATALOG.map((entry) => entry.id);
    expect(ids).toEqual(['tinystories', 'gpt2', 'smollm2', 'qwen3.5-0.8b', 'phi-1.5']);
    expect(ids.some((id) => id.includes('gemma'))).toBe(false);
    expect(DEFAULT_MODEL_ID).toBe('smollm2');
    expect(ids).toContain(DEFAULT_MODEL_ID);
    for (const entry of MODEL_CATALOG) {
      expect(entry.url).toMatch(/^https:\/\/huggingface\.co\//);
      expect(entry.localUrl).toMatch(/^https:\/\/storage\.googleapis\.com\/three-llm\//);
      expect(entry.sizeHint).toMatch(/^\d+(\.\d+)? (MB|GB)$/);
      expect(catalogLabel(entry)).toContain(`[${entry.sizeHint}]`);
    }
    const qwen = MODEL_CATALOG.find((entry) => entry.id === 'qwen3.5-0.8b');
    expect(qwen?.badge).toBe('Best Results');
    expect(catalogLabel(qwen!)).toBe('Qwen3.5 0.8B (Best Results) [1.7 GB]');
  });
});

describe('SafeTensorsLoader', () => {
  it('uses a single-file checkpoint when model.safetensors exists', async () => {
    const originalFetch = globalThis.fetch;
    const requested: string[] = [];
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requested.push(`${init?.method ?? 'GET'} ${url}`);
      if (url.endsWith('model.safetensors') && init?.method === 'HEAD') {
        return new Response(null, { status: 200 });
      }
      return new Response('Not found', { status: 404 });
    }) as typeof fetch;

    try {
      await expect(resolveSafetensorFiles('https://example.test/gpt2/')).resolves.toEqual(['model.safetensors']);
      expect(requested).toEqual(['HEAD https://example.test/gpt2/model.safetensors']);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('reads the shard index when model.safetensors is absent', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('model.safetensors') && init?.method === 'HEAD') {
        return new Response(null, { status: 404 });
      }
      if (url.endsWith('model.safetensors.index.json')) {
        return Response.json({
          weight_map: {
            a: 'model.safetensors-1-of-2.safetensors',
            b: 'model.safetensors-2-of-2.safetensors',
            c: 'model.safetensors-1-of-2.safetensors',
          },
        });
      }
      return new Response('Not found', { status: 404 });
    }) as typeof fetch;

    try {
      await expect(resolveSafetensorFiles('https://example.test/phi-1.5/')).resolves.toEqual([
        'model.safetensors-1-of-2.safetensors',
        'model.safetensors-2-of-2.safetensors',
      ]);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('parses F32 tensors', () => {
    const parsed = parseSafeTensors(createSafeTensorsFixture());
    expect(parsed.tensors.values?.shape).toEqual([2, 2]);
    expect(parsed.tensors.values?.dtype).toBe('F32');
    expect(Array.from(parsed.tensors.values?.data as Float32Array)).toEqual([1, 2, 3, 4]);
  });

  it('parses F16 tensors and rejects truncated data', () => {
    const parsed = parseSafeTensors(createSafeTensorsFixture('F16', [0x3c00, 0x4000, 0x4200, 0x4400]));
    expect(parsed.tensors.values?.dtype).toBe('F16');
    expect(Array.from(parsed.tensors.values?.data as Uint16Array)).toEqual([0x3c00, 0x4000, 0x4200, 0x4400]);

    const truncated = createSafeTensorsFixture().slice(0, -1);
    expect(() => parseSafeTensors(truncated)).toThrow(/data extends beyond the file/);
    expect(() => parseSafeTensors(new ArrayBuffer(7))).toThrow(/too small to contain a header/);
  });
});

describe('GPT2Tokenizer', () => {
  it('keeps hash-character BPE merges', () => {
    const tokenizer = new GPT2Tokenizer(
      {
        a: 0,
        '#': 1,
        '##': 2,
        'a#': 3,
        '<|endoftext|>': 4,
      },
      ['#version: 0.2', '# #', 'a #'],
    );

    expect(tokenizer.bpe('##')).toBe('##');
    expect(tokenizer.bpe('a#')).toBe('a#');
    expect(tokenizer.encode('##')).toEqual([2]);
  });

  it('encodes added special tokens as whole ids', () => {
    const tokenizer = new GPT2Tokenizer(
      {
        a: 0,
        '<|endoftext|>': 1,
      },
      [],
      {
        addedTokens: [
          { id: 10, content: '<|im_start|>' },
          { id: 11, content: '<think>' },
          { id: 12, content: '</think>' },
        ],
      },
    );

    expect(tokenizer.encode('<|im_start|><think></think>')).toEqual([10, 11, 12]);
    expect(tokenizer.decode([10, 11, 12])).toBe('<|im_start|><think></think>');
  });
});

describe('math', () => {
  it('computes reference operations', () => {
    closeArray(
      linear(new Float32Array([1, 2]), new Float32Array([3, 4, 5, 6]), new Float32Array([7, 8]), 2, 2),
      new Float32Array([20, 24]),
      1e-6,
    );
    closeArray(
      linear(new Float32Array([1, 2]), new Float32Array([3, 4, 5, 6]), null, 2, 2),
      new Float32Array([13, 16]),
      1e-6,
    );
    closeArray(softmax(new Float32Array([1, 2, 3])), new Float32Array([0.09003057, 0.24472848, 0.66524094]), 1e-6);
    expect(Math.abs(geluNew(0))).toBeLessThan(1e-6);
    expect(Math.abs(geluNew(1) - 0.84119199)).toBeLessThan(1e-6);
    expect(sampleTopK(new Float32Array([1, 2, 3]), { topK: 1 })).toBe(2);
    expect(sampleTopK(new Float32Array([1, 4, 3]), { temperature: 0, topK: 40 })).toBe(1);
    expect(sampleTopK(new Float32Array([1, 2, 3, 4]), { topK: 2, temperature: 1, random: () => 0 })).toBe(3);
    expect(sampleTopK(new Float32Array([5, 4]), { temperature: 0, tokens: [0], repetitionPenalty: 2 })).toBe(1);
    expect(
      sampleTopK(new Float32Array([0, 0, 10, 1]), {
        temperature: 0,
        tokens: [0, 1, 2, 0, 1],
        noRepeatNgramSize: 3,
      }),
    ).toBe(3);

    const normalized = layerNorm(
      new Float32Array([1, 2, 3]),
      new Float32Array([1, 1, 1]),
      new Float32Array([0, 0, 0]),
      1e-5,
    );
    closeArray(normalized, new Float32Array([-1.2247356, 0, 1.2247356]), 1e-5);

    const rms = rmsNorm(new Float32Array([1, 2, 3]), new Float32Array([1, 1, 1]));
    const invRms = 1 / Math.sqrt(14 / 3 + 1e-5);
    closeArray(rms, new Float32Array([invRms, 2 * invRms, 3 * invRms]), 1e-5);

    expect(Math.abs(silu(0))).toBeLessThan(1e-6);
    expect(Math.abs(silu(1) - 0.731058578)).toBeLessThan(1e-6);

    const rope = applyRoPE(new Float32Array([1, 0, 0, 1]), 0, 4, 1, 10000);
    expect(Math.abs(rope[0]! - Math.cos(1))).toBeLessThan(1e-5);

    expect(architectureFor({ model_type: 'gpt2' })).toBe('gpt2');
    expect(architectureFor({ model_type: 'llama' })).toBe('llama');
    expect(architectureFor({ model_type: 'phi' })).toBe('phi');
    expect(architectureFor({ model_type: 'gemma3_text' })).toBe('gemma3');
    expect(architectureFor({ model_type: 'qwen3_5' })).toBe('qwen3_5');
    expect(architectureFor({ model_type: 'qwen3_5', text_config: { model_type: 'qwen3_5_text' } })).toBe('qwen3_5');
    expect(() => architectureFor({ model_type: 'gemma4' })).toThrow(/Unsupported model_type "gemma4"/);

    const mapped = resolveTensor(
      {
        'model.layers.0.self_attn.q_proj.weight': { name: 'q', dtype: 'F32', shape: [1], data: new Float32Array(1) },
      },
      'model.',
      'llama',
      'attn_q',
      0,
    );
    expect(mapped.name).toBe('q');

    const capped = logitSoftcap(new Float32Array([60, -60, 0]), 30);
    expect(Math.abs(capped[0]! - 30 * Math.tanh(2))).toBeLessThan(1e-5);

    const split = splitHeadGate(new Float32Array([1, 2, 3, 4]), 2, 1);
    closeArray(split.query, new Float32Array([1, 3]), 1e-6);
    closeArray(split.gate, new Float32Array([2, 4]), 1e-6);

    expect(Math.abs(float16ToFloat32(0x3c00) - 1)).toBeLessThan(1e-6);
    expect(Math.abs(bfloat16ToFloat32(0x3f80) - 1)).toBeLessThan(1e-6);
    expect(
      Math.abs(
        tensorToFloat32({
          name: 'w',
          dtype: 'BF16',
          shape: [1],
          data: new Uint16Array([0x4000]),
        })[0]! - 2,
      ),
    ).toBeLessThan(1e-6);
  });
});

describe('tensors', () => {
  it('converts BF16 tensors to F32 with progress', async () => {
    const tensors: TensorMap = {
      small: { name: 'small', dtype: 'BF16', shape: [2], data: new Uint16Array([0x3f80, 0x4000]) },
      left: { name: 'left', dtype: 'F32', shape: [1], data: new Float32Array([9]) },
    };
    const messages: string[] = [];
    const count = await convertAllTensors(tensors, (message) => messages.push(message), 'Test');

    expect(count).toBe(1);
    expect(tensors.small?.dtype).toBe('F32');
    expect(Math.abs((tensors.small?.data as Float32Array)[0]! - 1)).toBeLessThan(1e-6);
    expect(Math.abs((tensors.small?.data as Float32Array)[1]! - 2)).toBeLessThan(1e-6);
    expect(tensors.left?.dtype).toBe('F32');
    expect(messages.some((message) => message.includes('Converting BF16'))).toBe(true);
  });
});

describe('DecoderRecipe', () => {
  it('preserves architecture-specific decode semantics', () => {
    const gpt2 = recipeFor({
      model_type: 'gpt2',
      n_embd: 8,
      n_head: 2,
      n_layer: 1,
      vocab_size: 16,
    });
    const phi = recipeFor({
      model_type: 'phi',
      hidden_size: 8,
      intermediate_size: 16,
      num_hidden_layers: 1,
      num_attention_heads: 2,
      vocab_size: 16,
      partial_rotary_factor: 0.5,
    });
    const gemma = recipeFor({
      model_type: 'gemma3_text',
      hidden_size: 8,
      intermediate_size: 16,
      num_hidden_layers: 6,
      num_attention_heads: 2,
      num_key_value_heads: 1,
      head_dim: 4,
      vocab_size: 16,
    });
    const qwen = recipeFor({
      model_type: 'qwen3_5_text',
      hidden_size: 8,
      intermediate_size: 16,
      num_hidden_layers: 8,
      num_attention_heads: 2,
      num_key_value_heads: 1,
      head_dim: 4,
      vocab_size: 16,
      full_attention_interval: 4,
    });
    const mistral = recipeFor({
      model_type: 'mistral',
      hidden_size: 8,
      intermediate_size: 16,
      num_hidden_layers: 1,
      num_attention_heads: 2,
      vocab_size: 16,
      partial_rotary_factor: 0.5,
      sliding_window: 32,
    });

    expect(gpt2.position).toBe('learned');
    expect(gpt2.packedQKV).toBe(true);
    expect(phi.residual).toBe('parallel');
    expect(phi.rotaryDim).toBe(2);
    expect(gemma.norm).toBe('rms_offset');
    expect(gemma.layerTypes?.[5]).toBe('full_attention');
    expect(qwen.layerTypes?.[0]).toBe('linear_attention');
    expect(qwen.layerTypes?.[3]).toBe('full_attention');
    expect(mistral.rotaryDim).toBe(2);
    expect(mistral.slidingWindow).toBe(32);
    expect(() => architectureFor({ model_type: 'gemma2' })).toThrow(/Unsupported model_type "gemma2"/);
  });
});

describe('generate', () => {
  it('reuses a matching prompt cache prefix', () => {
    expect(sharedPrefixLength([1, 2, 3], [1, 2, 9])).toBe(2);
    expect(sharedPrefixLength([1, 2], [1, 2, 3])).toBe(2);

    const append = planPromptCache([1, 2, 3], new Float32Array([0]), [1, 2, 3, 4], true);
    expect(append.start).toBe(3);
    expect(append.reset).toBe(false);
    expect(append.logits).not.toBeNull();

    const rewind = planPromptCache([1, 2, 3, 9], new Float32Array([0]), [1, 2, 4], true);
    expect(rewind.start).toBe(2);
    expect(rewind.reset).toBe(false);

    const recurrent = planPromptCache([1, 2, 3, 9], new Float32Array([0]), [1, 2, 4], false);
    expect(recurrent.reset).toBe(true);
  });

  it('accepts an explicit zero-token budget', () => {
    const prepared = prepareGenerationFromTokens([1, 2], 8, 0, 0);
    let requestedNewTokens: number | undefined;
    const runner = {
      maxTokens: 8,
      weights: {
        endOfTextTokenId: 0,
        prepareGeneration(_prompt: string, _maxTokens: number, maxNewTokens: number) {
          requestedNewTokens = maxNewTokens;
          return { inputTokens: [1], newTokenBudget: maxNewTokens };
        },
        tokenizer: { decode: () => '' },
      },
    };

    generateSync(
      runner as never,
      'prompt',
      { maxNewTokens: 0 },
      {
        rewindable: true,
        resetCache() {},
        forwardToken: () => new Float32Array([0, 1]),
      },
    );

    expect(prepared.inputTokens).toEqual([1, 2]);
    expect(prepared.newTokenBudget).toBe(0);
    expect(requestedNewTokens).toBe(0);
  });
});

describe('UnigramTokenizer', () => {
  it('encodes Hugging Face BPE vocabs', () => {
    const tokenizer = new UnigramTokenizer(
      {
        model: {
          type: 'BPE',
          byte_fallback: true,
          unk_token: '<unk>',
          vocab: {
            '<unk>': 0,
            '<eos>': 1,
            '<bos>': 2,
            '▁': 3,
            h: 4,
            e: 5,
            l: 6,
            o: 7,
            he: 8,
            '▁he': 9,
            ll: 10,
            'o▁': 11,
          },
          merges: [
            ['h', 'e'],
            ['l', 'l'],
            ['▁', 'he'],
          ],
        },
      },
      { bos_token_id: 2, eos_token_id: 1, add_bos_token: true },
    );

    expect(tokenizer.useBpe).toBe(true);
    expect(tokenizer.encode('he')).toEqual([2, 8]);
    expect(tokenizer.decode([2, 9, 10])).toBe('hell');
  });

  it('round-trips metaspace text and prepends BOS', () => {
    const tokenizer = new UnigramTokenizer(
      {
        model: {
          type: 'Unigram',
          unk_id: 0,
          vocab: [
            ['<unk>', 0],
            ['<eos>', 0],
            ['<bos>', 0],
            ['▁hello', -1],
            ['▁world', -2],
            ['▁', -4],
          ],
        },
      },
      { bos_token_id: 2, eos_token_id: 1, add_bos_token: true },
    );

    expect(tokenizer.encode('hello world')).toEqual([2, 3, 4]);
    expect(tokenizer.decode([2, 3, 4])).toBe('hello world');
  });
});

describe('QwenWeights', () => {
  it('formats chat with thinking disabled by default', () => {
    const hidden = 8;
    const inner = 8;
    const heads = 2;
    const kvHeads = 1;
    const headDim = 4;
    const layers = 2;
    const vocab = 8;
    const qSize = heads * headDim;
    const kvSize = kvHeads * headDim;
    const linHeads = 2;
    const linDim = 4;
    const kernel = 4;
    const convDim = linHeads * linDim * 2 + linHeads * linDim;
    const tensors: TensorMap = {
      'model.language_model.embed_tokens.weight': makeTensor('embed', [vocab, hidden], 0.12),
      'model.language_model.norm.weight': makeTensor('norm', [hidden], 1.12),
    };

    const linearPrefix = 'model.language_model.layers.0';
    tensors[`${linearPrefix}.input_layernorm.weight`] = makeTensor('ln1', [hidden], 2.1);
    tensors[`${linearPrefix}.post_attention_layernorm.weight`] = makeTensor('ln2', [hidden], 2.2);
    tensors[`${linearPrefix}.mlp.gate_proj.weight`] = makeTensor('gate', [inner, hidden], 8.1);
    tensors[`${linearPrefix}.mlp.up_proj.weight`] = makeTensor('up', [inner, hidden], 8.2);
    tensors[`${linearPrefix}.mlp.down_proj.weight`] = makeTensor('down', [hidden, inner], 8.3);
    tensors[`${linearPrefix}.linear_attn.in_proj_qkv.weight`] = makeTensor('dqkv', [convDim, hidden], 3.1);
    tensors[`${linearPrefix}.linear_attn.in_proj_z.weight`] = makeTensor('dz', [linHeads * linDim, hidden], 3.2);
    tensors[`${linearPrefix}.linear_attn.in_proj_b.weight`] = makeTensor('db', [linHeads, hidden], 3.3);
    tensors[`${linearPrefix}.linear_attn.in_proj_a.weight`] = makeTensor('da', [linHeads, hidden], 3.4);
    tensors[`${linearPrefix}.linear_attn.out_proj.weight`] = makeTensor('do', [hidden, linHeads * linDim], 3.5);
    tensors[`${linearPrefix}.linear_attn.conv1d.weight`] = makeTensor('dc', [convDim, 1, kernel], 3.6);
    tensors[`${linearPrefix}.linear_attn.A_log`] = makeTensor('alog', [linHeads], 0.4);
    tensors[`${linearPrefix}.linear_attn.dt_bias`] = makeTensor('dt', [linHeads], 0.2);
    tensors[`${linearPrefix}.linear_attn.norm.weight`] = makeTensor('dn', [linDim], 1.05);

    const fullPrefix = 'model.language_model.layers.1';
    tensors[`${fullPrefix}.input_layernorm.weight`] = makeTensor('fln1', [hidden], 2.3);
    tensors[`${fullPrefix}.post_attention_layernorm.weight`] = makeTensor('fln2', [hidden], 2.4);
    tensors[`${fullPrefix}.mlp.gate_proj.weight`] = makeTensor('fgate', [inner, hidden], 8.4);
    tensors[`${fullPrefix}.mlp.up_proj.weight`] = makeTensor('fup', [inner, hidden], 8.5);
    tensors[`${fullPrefix}.mlp.down_proj.weight`] = makeTensor('fdown', [hidden, inner], 8.6);
    tensors[`${fullPrefix}.self_attn.q_proj.weight`] = makeTensor('q', [qSize * 2, hidden], 4.1);
    tensors[`${fullPrefix}.self_attn.k_proj.weight`] = makeTensor('k', [kvSize, hidden], 4.2);
    tensors[`${fullPrefix}.self_attn.v_proj.weight`] = makeTensor('v', [kvSize, hidden], 4.3);
    tensors[`${fullPrefix}.self_attn.o_proj.weight`] = makeTensor('o', [hidden, qSize], 4.4);
    tensors[`${fullPrefix}.self_attn.q_norm.weight`] = makeTensor('qn', [headDim], 0.3);
    tensors[`${fullPrefix}.self_attn.k_norm.weight`] = makeTensor('kn', [headDim], 0.4);

    const weights = new QwenWeights(
      {
        model_type: 'qwen3_5_text',
        hidden_size: hidden,
        intermediate_size: inner,
        num_hidden_layers: layers,
        num_attention_heads: heads,
        num_key_value_heads: kvHeads,
        head_dim: headDim,
        vocab_size: vocab,
        hidden_act: 'silu',
        rms_norm_eps: 1e-6,
        layer_types: ['linear_attention', 'full_attention'],
        linear_conv_kernel_dim: kernel,
        linear_key_head_dim: linDim,
        linear_value_head_dim: linDim,
        linear_num_key_heads: linHeads,
        linear_num_value_heads: linHeads,
        rope_parameters: { rope_theta: 10000, partial_rotary_factor: 0.5 },
        tie_word_embeddings: true,
        eos_token_id: 0,
        max_position_embeddings: 16,
      },
      tensors,
      tinyTokenizer(),
    );

    const messages = [{ role: 'user' as const, text: 'Hi' }];
    expect(weights.formatChat(messages)).toBe(
      '<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
    );
    expect(weights.formatChat(messages, { enableThinking: true })).toBe(
      '<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n',
    );
  });
});
