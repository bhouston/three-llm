import { DecoderWeights } from '../decoder/DecoderWeights.js';
import { QwenWeights } from '../qwen/QwenWeights.js';
import type { Tensor, TensorMap, Tokenizer } from '../types.js';
import { expect } from 'vitest';

export function closeArray(actual: ArrayLike<number>, expected: ArrayLike<number>, epsilon: number) {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs(actual[i]! - expected[i]!)).toBeLessThanOrEqual(epsilon);
  }
}

export function fillSin(array: Float32Array, seed: number) {
  for (let i = 0; i < array.length; i++) array[i] = Math.sin(seed + i * 0.17) * 0.35;
  return array;
}

export function makeTensor(name: string, shape: number[], seed: number): Tensor {
  const data = fillSin(new Float32Array(shape.reduce((product, value) => product * value, 1)), seed);
  return { name, dtype: 'F32', shape, data };
}

export function tinyTokenizer(eos = 0): Tokenizer {
  return {
    endOfTextTokenId: eos,
    encode() {
      return [1, 2];
    },
    decode(ids) {
      return ids.join(',');
    },
  };
}

export function createTinyLlama() {
  const hidden = 8;
  const inner = 16;
  const heads = 4;
  const kvHeads = 2;
  const headDim = 2;
  const layers = 2;
  const vocab = 8;
  const qSize = heads * headDim;
  const kvSize = kvHeads * headDim;
  const tensors: TensorMap = {
    'model.embed_tokens.weight': makeTensor('embed', [vocab, hidden], 0.1),
    'model.norm.weight': makeTensor('norm', [hidden], 1.1),
  };

  for (let layer = 0; layer < layers; layer++) {
    const p = `model.layers.${layer}`;
    tensors[`${p}.input_layernorm.weight`] = makeTensor('ln1', [hidden], 2 + layer);
    tensors[`${p}.post_attention_layernorm.weight`] = makeTensor('ln2', [hidden], 3 + layer);
    tensors[`${p}.self_attn.q_proj.weight`] = makeTensor('q', [qSize, hidden], 4 + layer);
    tensors[`${p}.self_attn.k_proj.weight`] = makeTensor('k', [kvSize, hidden], 5 + layer);
    tensors[`${p}.self_attn.v_proj.weight`] = makeTensor('v', [kvSize, hidden], 6 + layer);
    tensors[`${p}.self_attn.o_proj.weight`] = makeTensor('o', [hidden, qSize], 7 + layer);
    tensors[`${p}.mlp.gate_proj.weight`] = makeTensor('gate', [inner, hidden], 8 + layer);
    tensors[`${p}.mlp.up_proj.weight`] = makeTensor('up', [inner, hidden], 9 + layer);
    tensors[`${p}.mlp.down_proj.weight`] = makeTensor('down', [hidden, inner], 10 + layer);
  }

  return new DecoderWeights(
    {
      model_type: 'llama',
      hidden_size: hidden,
      intermediate_size: inner,
      num_hidden_layers: layers,
      num_attention_heads: heads,
      num_key_value_heads: kvHeads,
      vocab_size: vocab,
      rms_norm_eps: 1e-5,
      rope_theta: 10000,
      hidden_act: 'silu',
      tie_word_embeddings: true,
      eos_token_id: 0,
      max_position_embeddings: 16,
    },
    tensors,
    tinyTokenizer(),
  );
}

export function createTinyPhi() {
  const hidden = 8;
  const inner = 8;
  const heads = 2;
  const layers = 1;
  const vocab = 8;
  const tensors: TensorMap = {
    'model.embed_tokens.weight': makeTensor('embed', [vocab, hidden], 0.2),
    'model.final_layernorm.weight': makeTensor('fnw', [hidden], 1.2),
    'model.final_layernorm.bias': makeTensor('fnb', [hidden], 1.3),
    'lm_head.weight': makeTensor('lm', [vocab, hidden], 0.4),
  };
  const p = 'model.layers.0';
  tensors[`${p}.input_layernorm.weight`] = makeTensor('lnw', [hidden], 2.1);
  tensors[`${p}.input_layernorm.bias`] = makeTensor('lnb', [hidden], 2.2);
  tensors[`${p}.self_attn.q_proj.weight`] = makeTensor('q', [hidden, hidden], 3.1);
  tensors[`${p}.self_attn.k_proj.weight`] = makeTensor('k', [hidden, hidden], 3.2);
  tensors[`${p}.self_attn.v_proj.weight`] = makeTensor('v', [hidden, hidden], 3.3);
  tensors[`${p}.self_attn.q_proj.bias`] = makeTensor('qb', [hidden], 3.4);
  tensors[`${p}.self_attn.k_proj.bias`] = makeTensor('kb', [hidden], 3.5);
  tensors[`${p}.self_attn.v_proj.bias`] = makeTensor('vb', [hidden], 3.6);
  tensors[`${p}.self_attn.dense.weight`] = makeTensor('d', [hidden, hidden], 3.7);
  tensors[`${p}.self_attn.dense.bias`] = makeTensor('db', [hidden], 3.8);
  tensors[`${p}.mlp.fc1.weight`] = makeTensor('fc1', [inner, hidden], 4.1);
  tensors[`${p}.mlp.fc1.bias`] = makeTensor('fc1b', [inner], 4.2);
  tensors[`${p}.mlp.fc2.weight`] = makeTensor('fc2', [hidden, inner], 4.3);
  tensors[`${p}.mlp.fc2.bias`] = makeTensor('fc2b', [hidden], 4.4);

  return new DecoderWeights(
    {
      model_type: 'phi',
      hidden_size: hidden,
      intermediate_size: inner,
      num_hidden_layers: layers,
      num_attention_heads: heads,
      vocab_size: vocab,
      layer_norm_eps: 1e-5,
      rope_theta: 10000,
      partial_rotary_factor: 0.5,
      eos_token_id: 0,
      max_position_embeddings: 16,
    },
    tensors,
    tinyTokenizer(),
  );
}

export function createTinyGemma() {
  const hidden = 8;
  const inner = 16;
  const heads = 2;
  const kvHeads = 1;
  const headDim = 8;
  const layers = 2;
  const vocab = 8;
  const qSize = heads * headDim;
  const kvSize = kvHeads * headDim;
  const tensors: TensorMap = {
    'model.embed_tokens.weight': makeTensor('embed', [vocab, hidden], 0.15),
    'model.norm.weight': makeTensor('norm', [hidden], 1.15),
  };

  for (let layer = 0; layer < layers; layer++) {
    const p = `model.layers.${layer}`;
    tensors[`${p}.input_layernorm.weight`] = makeTensor('ln1', [hidden], 2.1 + layer);
    tensors[`${p}.post_attention_layernorm.weight`] = makeTensor('postA', [hidden], 2.2 + layer);
    tensors[`${p}.pre_feedforward_layernorm.weight`] = makeTensor('preM', [hidden], 2.3 + layer);
    tensors[`${p}.post_feedforward_layernorm.weight`] = makeTensor('postM', [hidden], 2.4 + layer);
    tensors[`${p}.self_attn.q_norm.weight`] = makeTensor('qn', [headDim], 2.5 + layer);
    tensors[`${p}.self_attn.k_norm.weight`] = makeTensor('kn', [headDim], 2.6 + layer);
    tensors[`${p}.self_attn.q_proj.weight`] = makeTensor('q', [qSize, hidden], 4 + layer);
    tensors[`${p}.self_attn.k_proj.weight`] = makeTensor('k', [kvSize, hidden], 5 + layer);
    tensors[`${p}.self_attn.v_proj.weight`] = makeTensor('v', [kvSize, hidden], 6 + layer);
    tensors[`${p}.self_attn.o_proj.weight`] = makeTensor('o', [hidden, qSize], 7 + layer);
    tensors[`${p}.mlp.gate_proj.weight`] = makeTensor('gate', [inner, hidden], 8 + layer);
    tensors[`${p}.mlp.up_proj.weight`] = makeTensor('up', [inner, hidden], 9 + layer);
    tensors[`${p}.mlp.down_proj.weight`] = makeTensor('down', [hidden, inner], 10 + layer);
  }

  return new DecoderWeights(
    {
      model_type: 'gemma3_text',
      hidden_size: hidden,
      intermediate_size: inner,
      num_hidden_layers: layers,
      num_attention_heads: heads,
      num_key_value_heads: kvHeads,
      head_dim: headDim,
      vocab_size: vocab,
      rms_norm_eps: 1e-6,
      rope_theta: 1000000,
      rope_local_base_freq: 10000,
      sliding_window: 2,
      layer_types: ['sliding_attention', 'full_attention'],
      hidden_activation: 'gelu_pytorch_tanh',
      query_pre_attn_scalar: headDim,
      eos_token_id: 0,
      max_position_embeddings: 16,
    },
    tensors,
    tinyTokenizer(),
  );
}

export function createTinyQwenWeights() {
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

  return new QwenWeights(
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
}
