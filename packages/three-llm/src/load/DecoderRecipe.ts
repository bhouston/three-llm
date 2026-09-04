import { unwrapTextConfig } from './tensors.js';
import { keepQwenTensor } from './TensorNameMap.js';
import type { Architecture, ChatTemplateKind, DecoderRecipe, HuggingFaceConfig, TokenizerKind, YarnRoPEConfig } from '../types.js';

const DEFAULT_EXAMPLE_CONTEXT_LIMIT = 2048;

/**
 * Maps Hugging Face `config.json` onto a decode recipe: graph family plus
 * kernel flags. Close cousins share a recipe; hybrids (Qwen 3.5) stay plugins.
 *
 */

function architectureFor( config: HuggingFaceConfig ): Architecture {

	const text = unwrapTextConfig( config );
	const type = text.model_type;
	const parent = text._parent_model_type;

	if ( type === 'gpt2' ) return 'gpt2';
	if ( type === 'gemma3_text' || type === 'gemma3' ) return 'gemma3';
	if ( type === 'qwen3_5' || type === 'qwen3_5_text' || parent === 'qwen3_5' ) return 'qwen3_5';
	if ( type === 'llama' || type === 'mistral' || type === 'qwen2' || type === 'qwen3' || type === 'kanana2_tiny' || type === 'gemma' ) return 'llama';
	if ( type === 'phi' ) return 'phi';

	throw new Error( `LLMFactory: Unsupported model_type "${ type }".` );

}

function defaultGemmaLayerTypes( layerCount: number ): string[] {

	const types: string[] = [];

	for ( let i = 0; i < layerCount; i ++ ) {

		types.push( ( i + 1 ) % 6 === 0 ? 'full_attention' : 'sliding_attention' );

	}

	return types;

}

function defaultQwenLayerTypes( layerCount: number, fullAttentionInterval: number ): string[] {

	const types: string[] = [];

	for ( let i = 0; i < layerCount; i ++ ) {

		types.push( ( i + 1 ) % fullAttentionInterval === 0 ? 'full_attention' : 'linear_attention' );

	}

	return types;

}

function firstEos( value: number | number[] | undefined | null, fallback: number ): number {

	if ( Array.isArray( value ) ) return value[ 0 ];
	if ( value !== undefined && value !== null ) return value;
	return fallback;

}

function eosIds( value: number | number[] | undefined | null, fallback: number ): number[] {

	if ( Array.isArray( value ) ) return value.slice();
	if ( value !== undefined && value !== null ) return [ value ];
	return [ fallback ];

}

function modelName( config: HuggingFaceConfig ): string {

	const value = config._name_or_path || config.name_or_path || config.model_id || '';
	return typeof value === 'string' ? value.toLowerCase() : '';

}

function denseTokenizer( text: HuggingFaceConfig ): TokenizerKind {

	const type = text.model_type;
	const name = modelName( text );

	if ( type === 'qwen2' || type === 'qwen3' || name.includes( 'qwen' ) || name.includes( 'deepseek-r1-distill-qwen' ) ) return 'qwen';
	if ( type === 'kanana2_tiny' || name.includes( 'kanana' ) ) return 'llama3';
	if ( name.includes( 'smollm2' ) ) return 'smollm';
	return 'gpt2';

}

function denseChatTemplate( text: HuggingFaceConfig ): ChatTemplateKind | undefined {

	const type = text.model_type;
	const name = modelName( text );

	if ( type === 'qwen3' ) return 'qwen3';
	if ( name.includes( 'deepseek-r1-distill-qwen' ) ) return 'deepseek-r1';
	if ( type === 'qwen2' || name.includes( 'qwen2.5-coder' ) ) return 'qwen2';
	if ( type === 'kanana2_tiny' || name.includes( 'kanana' ) ) return 'kanana';
	if ( name.includes( 'smollm2' ) && name.includes( 'instruct' ) ) return 'smollm2';
	return undefined;

}

function yarnConfig( text: HuggingFaceConfig ): YarnRoPEConfig | undefined {

	const scaling = text.rope_scaling;
	const ropeType = scaling?.rope_type || scaling?.type;

	if ( scaling === undefined || ropeType !== 'yarn' || scaling.factor === undefined ) return undefined;

	return {
		factor: scaling.factor,
		originalContextLength: scaling.original_max_position_embeddings || 4096,
		betaFast: scaling.beta_fast || 32,
		betaSlow: scaling.beta_slow || 1,
		attentionFactor: scaling.attention_factor
	};

}

function recipeFor( config: HuggingFaceConfig ): DecoderRecipe {

	const architecture = architectureFor( config );
	const text = unwrapTextConfig( config );

	if ( architecture === 'gpt2' ) {

		const hiddenSize = text.n_embd as number;
		const headCount = text.n_head as number;

		return {
			architecture,
			graph: 'decoder',
			tokenizer: 'gpt2',
			hiddenSize,
			innerSize: text.n_inner || hiddenSize * 4,
			layerCount: text.n_layer as number,
			headCount,
			kvHeadCount: headCount,
			headDim: hiddenSize / headCount,
			vocabSize: text.vocab_size as number,
			contextLimit: text.n_positions || text.n_ctx || 1024,
			norm: 'layer_norm',
			normEps: text.layer_norm_epsilon || 1e-5,
			mlp: 'dense_gelu',
			residual: 'sequential',
			position: 'learned',
			packedQKV: true,
			transposeLinears: false,
			qkNorm: false,
			postNorms: false,
			embedScale: 1,
			attnScale: undefined,
			ropeTheta: 0,
			rotaryDim: 0,
			endOfTextTokenId: firstEos( text.eos_token_id, 50256 )
		};

	}

	if ( architecture === 'phi' ) {

		const hiddenSize = text.hidden_size as number;
		const headCount = text.num_attention_heads as number;
		const headDim = hiddenSize / headCount;

		return {
			architecture,
			graph: 'decoder',
			tokenizer: 'gpt2',
			hiddenSize,
			innerSize: text.intermediate_size as number,
			layerCount: text.num_hidden_layers as number,
			headCount,
			kvHeadCount: text.num_key_value_heads || headCount,
			headDim,
			vocabSize: text.vocab_size as number,
			contextLimit: text.max_position_embeddings || 2048,
			norm: 'layer_norm',
			normEps: text.layer_norm_eps || 1e-5,
			mlp: 'dense_gelu',
			residual: 'parallel',
			position: 'rope',
			packedQKV: false,
			transposeLinears: true,
			qkNorm: false,
			postNorms: false,
			embedScale: 1,
			attnScale: undefined,
			ropeTheta: text.rope_theta || 10000,
			rotaryDim: text.rotary_dim || Math.round( ( text.partial_rotary_factor || 0.5 ) * headDim ),
			endOfTextTokenId: firstEos( text.eos_token_id, 0 )
		};

	}

	if ( architecture === 'gemma3' ) {

		const hiddenSize = text.hidden_size as number;
		const headCount = text.num_attention_heads as number;
		const headDim = text.head_dim || ( hiddenSize / headCount );
		const layerCount = text.num_hidden_layers as number;

		return {
			architecture,
			graph: 'decoder',
			tokenizer: 'unigram',
			hiddenSize,
			innerSize: text.intermediate_size as number,
			layerCount,
			headCount,
			kvHeadCount: text.num_key_value_heads || headCount,
			headDim,
			vocabSize: text.vocab_size as number,
			contextLimit: Math.min( text.max_position_embeddings || DEFAULT_EXAMPLE_CONTEXT_LIMIT, DEFAULT_EXAMPLE_CONTEXT_LIMIT ),
			norm: 'rms_offset',
			normEps: text.rms_norm_eps || 1e-6,
			mlp: 'gated',
			mlpActivation: text.hidden_activation || text.hidden_act || 'gelu_pytorch_tanh',
			residual: 'sequential',
			position: 'rope',
			packedQKV: false,
			transposeLinears: true,
			qkNorm: true,
			postNorms: true,
			embedScale: Math.sqrt( hiddenSize ),
			attnScale: ( text.query_pre_attn_scalar || headDim ) ** - 0.5,
			globalRopeTheta: text.rope_theta || 1000000,
			localRopeTheta: text.rope_local_base_freq || 10000,
			rotaryDim: headDim,
			slidingWindow: text.sliding_window || 512,
			layerTypes: text.layer_types || defaultGemmaLayerTypes( layerCount ),
			finalLogitSoftcap: text.final_logit_softcapping,
			endOfTextTokenId: firstEos( text.eos_token_id, 1 ),
			stopTokenIds: eosIds( text.eos_token_id, 1 ),
			chatTemplate: 'gemma3'
		};

	}

	if ( architecture === 'qwen3_5' ) {

		const hiddenSize = text.hidden_size as number;
		const headCount = text.num_attention_heads as number;
		const headDim = text.head_dim || ( hiddenSize / headCount );
		const rope = text.rope_parameters || {};
		const partialRotaryFactor = rope.partial_rotary_factor ?? 0.25;
		const fullAttentionInterval = text.full_attention_interval || 4;
		const layerCount = text.num_hidden_layers as number;

		return {
			architecture,
			graph: 'qwen35',
			tokenizer: 'qwen',
			keepTensor: keepQwenTensor,
			hiddenSize,
			innerSize: text.intermediate_size as number,
			layerCount,
			headCount,
			kvHeadCount: text.num_key_value_heads || headCount,
			headDim,
			vocabSize: text.vocab_size as number,
			contextLimit: Math.min( text.max_position_embeddings || DEFAULT_EXAMPLE_CONTEXT_LIMIT, DEFAULT_EXAMPLE_CONTEXT_LIMIT ),
			norm: 'rms_offset',
			normEps: text.rms_norm_eps || 1e-6,
			mlp: 'gated',
			mlpActivation: text.hidden_act || 'silu',
			residual: 'sequential',
			position: 'rope',
			packedQKV: false,
			transposeLinears: true,
			qkNorm: true,
			postNorms: false,
			embedScale: 1,
			attnScale: headDim ** - 0.5,
			ropeTheta: rope.rope_theta || text.rope_theta || 10000000,
			rotaryDim: Math.floor( headDim * partialRotaryFactor ),
			layerTypes: text.layer_types && text.layer_types.length > 0
				? text.layer_types
				: defaultQwenLayerTypes( layerCount, fullAttentionInterval ),
			linearKeyDim: text.linear_key_head_dim || 128,
			linearValueDim: text.linear_value_head_dim || 128,
			linearKeyHeads: text.linear_num_key_heads || 16,
			linearValueHeads: text.linear_num_value_heads || 16,
			linearConvKernel: text.linear_conv_kernel_dim || 4,
			endOfTextTokenId: firstEos( text.eos_token_id, 248044 ),
			stopTokenIds: eosIds( text.eos_token_id, 248044 ),
			chatTemplate: 'qwen3_5'
		};

	}

	const hiddenSize = text.hidden_size as number;
	const headCount = text.num_attention_heads as number;
	const headDim = text.head_dim || ( hiddenSize / headCount );
	const modelType = text.model_type;
	const isQwen3 = modelType === 'qwen3';
	const isKanana = modelType === 'kanana2_tiny';
	const layerCount = text.num_hidden_layers as number;
	const denseLayerTypes = isKanana
		? ( text.layer_types && text.layer_types.length > 0
			? text.layer_types
			: Array.from( { length: layerCount }, ( _, index ) => ( ( index + 1 ) % 4 === 0 ? 'full_attention' : 'sliding_attention' ) ) )
		: text.layer_types;
	const slidingWindow = text.use_sliding_window === false
		? 0
		: ( text.sliding_window || ( isKanana ? 1024 : 0 ) );
	const yarn = isKanana ? yarnConfig( text ) : undefined;

	return {
		architecture: 'llama',
		graph: 'decoder',
		tokenizer: denseTokenizer( text ),
		hiddenSize,
		innerSize: text.intermediate_size as number,
		layerCount,
		headCount,
		kvHeadCount: text.num_key_value_heads || headCount,
		headDim,
		vocabSize: text.vocab_size as number,
		contextLimit: Math.min( text.max_position_embeddings || 2048, isKanana && yarn ? DEFAULT_EXAMPLE_CONTEXT_LIMIT : ( text.max_position_embeddings || 2048 ) ),
		norm: modelType === 'gemma' ? 'rms_offset' : 'rms',
		normEps: text.rms_norm_eps || 1e-5,
		mlp: 'gated',
		mlpActivation: text.hidden_act || ( modelType === 'gemma' ? 'gelu_pytorch_tanh' : 'silu' ),
		residual: 'sequential',
		position: 'rope',
		packedQKV: false,
		transposeLinears: true,
		qkNorm: isQwen3 || isKanana,
		postNorms: false,
		embedScale: modelType === 'gemma' ? Math.sqrt( hiddenSize ) : 1,
		attnScale: undefined,
		ropeTheta: text.rope_theta || 10000,
		rotaryDim: text.rotary_dim ?? ( text.partial_rotary_factor !== undefined
			? Math.round( text.partial_rotary_factor * headDim )
			: headDim ),
		globalRopeTheta: text.rope_theta || 10000,
		localRopeTheta: text.rope_local_base_freq || text.rope_theta || 10000,
		slidingWindow,
		layerTypes: denseLayerTypes,
		endOfTextTokenId: firstEos( text.eos_token_id, 0 ),
		stopTokenIds: eosIds( text.eos_token_id, 0 ),
		chatTemplate: denseChatTemplate( text ),
		yarn
	};

}

export { architectureFor, recipeFor };
