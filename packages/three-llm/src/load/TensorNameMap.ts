/**
 * Canonical Hugging Face tensor aliases, in the same spirit as llama.cpp's
 * `gguf-py/gguf/tensor_mapping.py`: many checkpoint spellings, one lookup.
 *
 */

import type { Architecture, Tensor, TensorMap } from '../types.js';

type AliasList = string[];
type AliasTable = Record<string, AliasList>;

const GLOBAL_ALIASES: Record<Architecture, AliasTable> = {
	gpt2: {
		token_embd: [ 'wte.weight' ],
		pos_embd: [ 'wpe.weight' ],
		output: [ 'wte.weight' ],
		output_norm: [ 'ln_f.weight' ],
		output_norm_bias: [ 'ln_f.bias' ]
	},
	llama: {
		token_embd: [ 'embed_tokens.weight' ],
		output: [ 'lm_head.weight', 'embed_tokens.weight' ],
		output_norm: [ 'norm.weight' ]
	},
	gemma3: {
		token_embd: [ 'embed_tokens.weight' ],
		output: [ 'lm_head.weight', 'embed_tokens.weight' ],
		output_norm: [ 'norm.weight' ]
	},
	phi: {
		token_embd: [ 'embed_tokens.weight' ],
		output: [ 'lm_head.weight', 'embed_tokens.weight' ],
		output_norm: [ 'final_layernorm.weight' ],
		output_norm_bias: [ 'final_layernorm.bias' ]
	},
	qwen3_5: {
		token_embd: [ 'embed_tokens.weight' ],
		output: [ 'lm_head.weight', 'embed_tokens.weight' ],
		output_norm: [ 'norm.weight' ]
	}
};

const BLOCK_ALIASES: Record<Architecture, AliasTable> = {
	gpt2: {
		attn_norm: [ 'h.{bid}.ln_1.weight' ],
		attn_norm_bias: [ 'h.{bid}.ln_1.bias' ],
		ffn_norm: [ 'h.{bid}.ln_2.weight' ],
		ffn_norm_bias: [ 'h.{bid}.ln_2.bias' ],
		attn_qkv: [ 'h.{bid}.attn.c_attn.weight' ],
		attn_qkv_bias: [ 'h.{bid}.attn.c_attn.bias' ],
		attn_out: [ 'h.{bid}.attn.c_proj.weight' ],
		attn_out_bias: [ 'h.{bid}.attn.c_proj.bias' ],
		ffn_up: [ 'h.{bid}.mlp.c_fc.weight' ],
		ffn_up_bias: [ 'h.{bid}.mlp.c_fc.bias' ],
		ffn_down: [ 'h.{bid}.mlp.c_proj.weight' ],
		ffn_down_bias: [ 'h.{bid}.mlp.c_proj.bias' ]
	},
	llama: {
		attn_norm: [ 'layers.{bid}.input_layernorm.weight' ],
		ffn_norm: [ 'layers.{bid}.post_attention_layernorm.weight' ],
		attn_q: [ 'layers.{bid}.self_attn.q_proj.weight' ],
		attn_k: [ 'layers.{bid}.self_attn.k_proj.weight' ],
		attn_v: [ 'layers.{bid}.self_attn.v_proj.weight' ],
		attn_q_bias: [ 'layers.{bid}.self_attn.q_proj.bias' ],
		attn_k_bias: [ 'layers.{bid}.self_attn.k_proj.bias' ],
		attn_v_bias: [ 'layers.{bid}.self_attn.v_proj.bias' ],
		attn_q_norm: [ 'layers.{bid}.self_attn.q_norm.weight' ],
		attn_k_norm: [ 'layers.{bid}.self_attn.k_norm.weight' ],
		attn_out: [ 'layers.{bid}.self_attn.o_proj.weight' ],
		ffn_gate: [ 'layers.{bid}.mlp.gate_proj.weight' ],
		ffn_up: [ 'layers.{bid}.mlp.up_proj.weight' ],
		ffn_down: [ 'layers.{bid}.mlp.down_proj.weight' ]
	},
	gemma3: {
		attn_norm: [ 'layers.{bid}.input_layernorm.weight' ],
		post_attn_norm: [ 'layers.{bid}.post_attention_layernorm.weight' ],
		ffn_norm: [ 'layers.{bid}.pre_feedforward_layernorm.weight' ],
		post_ffn_norm: [ 'layers.{bid}.post_feedforward_layernorm.weight' ],
		attn_q_norm: [ 'layers.{bid}.self_attn.q_norm.weight' ],
		attn_k_norm: [ 'layers.{bid}.self_attn.k_norm.weight' ],
		attn_q: [ 'layers.{bid}.self_attn.q_proj.weight' ],
		attn_k: [ 'layers.{bid}.self_attn.k_proj.weight' ],
		attn_v: [ 'layers.{bid}.self_attn.v_proj.weight' ],
		attn_out: [ 'layers.{bid}.self_attn.o_proj.weight' ],
		ffn_gate: [ 'layers.{bid}.mlp.gate_proj.weight' ],
		ffn_up: [ 'layers.{bid}.mlp.up_proj.weight' ],
		ffn_down: [ 'layers.{bid}.mlp.down_proj.weight' ]
	},
	phi: {
		attn_norm: [ 'layers.{bid}.input_layernorm.weight' ],
		attn_norm_bias: [ 'layers.{bid}.input_layernorm.bias' ],
		attn_q: [ 'layers.{bid}.self_attn.q_proj.weight' ],
		attn_k: [ 'layers.{bid}.self_attn.k_proj.weight' ],
		attn_v: [ 'layers.{bid}.self_attn.v_proj.weight' ],
		attn_q_bias: [ 'layers.{bid}.self_attn.q_proj.bias' ],
		attn_k_bias: [ 'layers.{bid}.self_attn.k_proj.bias' ],
		attn_v_bias: [ 'layers.{bid}.self_attn.v_proj.bias' ],
		attn_out: [ 'layers.{bid}.self_attn.dense.weight' ],
		attn_out_bias: [ 'layers.{bid}.self_attn.dense.bias' ],
		ffn_up: [ 'layers.{bid}.mlp.fc1.weight' ],
		ffn_up_bias: [ 'layers.{bid}.mlp.fc1.bias' ],
		ffn_down: [ 'layers.{bid}.mlp.fc2.weight' ],
		ffn_down_bias: [ 'layers.{bid}.mlp.fc2.bias' ]
	},
	qwen3_5: {
		attn_norm: [ 'layers.{bid}.input_layernorm.weight' ],
		ffn_norm: [ 'layers.{bid}.post_attention_layernorm.weight' ],
		ffn_gate: [ 'layers.{bid}.mlp.gate_proj.weight' ],
		ffn_up: [ 'layers.{bid}.mlp.up_proj.weight' ],
		ffn_down: [ 'layers.{bid}.mlp.down_proj.weight' ],
		attn_q: [ 'layers.{bid}.self_attn.q_proj.weight' ],
		attn_k: [ 'layers.{bid}.self_attn.k_proj.weight' ],
		attn_v: [ 'layers.{bid}.self_attn.v_proj.weight' ],
		attn_out: [ 'layers.{bid}.self_attn.o_proj.weight' ],
		attn_q_norm: [ 'layers.{bid}.self_attn.q_norm.weight' ],
		attn_k_norm: [ 'layers.{bid}.self_attn.k_norm.weight' ],
		delta_qkv: [ 'layers.{bid}.linear_attn.in_proj_qkv.weight' ],
		delta_z: [ 'layers.{bid}.linear_attn.in_proj_z.weight' ],
		delta_b: [ 'layers.{bid}.linear_attn.in_proj_b.weight' ],
		delta_a: [ 'layers.{bid}.linear_attn.in_proj_a.weight' ],
		delta_out: [ 'layers.{bid}.linear_attn.out_proj.weight' ],
		delta_conv: [ 'layers.{bid}.linear_attn.conv1d.weight' ],
		delta_a_log: [ 'layers.{bid}.linear_attn.A_log' ],
		delta_dt_bias: [ 'layers.{bid}.linear_attn.dt_bias' ],
		delta_norm: [ 'layers.{bid}.linear_attn.norm.weight' ]
	}
};

function aliasesFor( architecture: Architecture, key: string, bid?: number ): string[] | null {

	const table = bid === undefined ? GLOBAL_ALIASES[ architecture ] : BLOCK_ALIASES[ architecture ];

	if ( table === undefined || table[ key ] === undefined ) return null;

	const aliases = table[ key ];

	if ( bid === undefined ) return aliases;

	return aliases.map( ( alias ) => alias.replaceAll( '{bid}', String( bid ) ) );

}

function candidateKeys( prefix: string, name: string ): string[] {

	if ( prefix === '' ) return [ name ];

	return [ `${ prefix }${ name }`, name ];

}

function findTensor( tensors: TensorMap, prefix: string, aliases: string[] ): Tensor | { missing: true; tried: string[] } {

	const tried: string[] = [];

	for ( let i = 0; i < aliases.length; i ++ ) {

		const keys = candidateKeys( prefix, aliases[ i ] );

		for ( let k = 0; k < keys.length; k ++ ) {

			tried.push( keys[ k ] );
			if ( tensors[ keys[ k ] ] !== undefined ) return tensors[ keys[ k ] ];

		}

	}

	return { missing: true, tried };

}

function hasMappedTensor( tensors: TensorMap, prefix: string, architecture: Architecture, key: string, bid?: number ): boolean {

	const aliases = aliasesFor( architecture, key, bid );
	if ( aliases === null ) {

		return false;

	}
	const found = findTensor( tensors, prefix, aliases );
	return ! ( 'missing' in found );

}

function resolveTensor( tensors: TensorMap, prefix: string, architecture: Architecture, key: string, bid?: number ): Tensor {

	const aliases = aliasesFor( architecture, key, bid );

	if ( aliases === null ) {

		throw new Error( `TensorNameMap: Unknown tensor "${ key }" for architecture "${ architecture }".` );

	}

	const found = findTensor( tensors, prefix, aliases );

	if ( 'missing' in found ) {

		throw new Error( `TensorNameMap: Missing "${ key }" (${ architecture }). Tried: ${ found.tried.join( ', ' ) }.` );

	}

	return found;

}

function keepQwenTensor( name: string ): boolean {

	return name.includes( 'language_model' ) && name.includes( 'visual' ) === false && name.startsWith( 'mtp.' ) === false;

}

export {
	BLOCK_ALIASES,
	GLOBAL_ALIASES,
	hasMappedTensor,
	keepQwenTensor,
	resolveTensor
};
