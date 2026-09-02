import { architectureFor, recipeFor } from './DecoderRecipe.js';
import { detectPrefix, loadHFModelBundle } from './HFModelBundle.js';
import { createProgress, packBiases, packProjections, prepareGeneration, tensorToFloat32, transpose2D, unwrapTextConfig } from './LLMTensors.js';
import { hasMappedTensor, resolveTensor } from './TensorNameMap.js';
import type {
	Architecture, DecoderBlock, DecoderRecipe, HFModelBundle, HuggingFaceConfig, LoaderOptions,
	PreparedGeneration, ProgressCallback, Tensor, TensorMap, Tokenizer
} from './types.js';

/**
 * Unpacks a Hugging Face causal LM into canonical block tensors using a
 * TensorNameMap. GPT-2, Llama-family, Phi, and Gemma 3 share this class.
 *
 */
class DecoderWeights {

	rawConfig: HuggingFaceConfig;
	config: HuggingFaceConfig;
	architecture: Architecture;
	recipe: DecoderRecipe;
	tensors: TensorMap;
	tokenizer: Tokenizer;
	tensorPrefix: string;
	hiddenSize: number;
	innerSize: number;
	layerCount: number;
	headCount: number;
	kvHeadCount: number;
	headDim: number;
	qSize: number;
	kvSize: number;
	vocabSize: number;
	ropeTheta: number;
	rotaryDim: number;
	rmsNormEps: number;
	layerNormEps: number;
	offsetRMSNorm: boolean;
	mlpActivation?: string;
	embedScale: number;
	attnScale?: number;
	globalRopeTheta?: number;
	localRopeTheta?: number;
	slidingWindow: number;
	layerTypes?: string[];
	endOfTextTokenId: number;
	_float32: Map<string, Float32Array>;
	logitWeight: Float32Array | null;
	outputNormWeight: Float32Array | null;
	outputNormBias: Float32Array | null;
	_blocks: DecoderBlock[];

	constructor( config: HuggingFaceConfig, tensors: TensorMap, tokenizer: Tokenizer, options: LoaderOptions = {} ) {

		this.rawConfig = options.rawConfig || config;
		this.config = unwrapTextConfig( config );
		this.architecture = architectureFor( this.rawConfig );
		this.recipe = recipeFor( this.rawConfig );
		this.tensors = tensors;
		this.tokenizer = tokenizer;
		this.tensorPrefix = options.prefix !== undefined ? options.prefix : detectPrefix( tensors, this.architecture );
		this.hiddenSize = this.recipe.hiddenSize;
		this.innerSize = this.recipe.innerSize;
		this.layerCount = this.recipe.layerCount;
		this.headCount = this.recipe.headCount;
		this.kvHeadCount = this.recipe.kvHeadCount;
		this.headDim = this.recipe.headDim;
		this.qSize = this.headCount * this.headDim;
		this.kvSize = this.kvHeadCount * this.headDim;
		this.vocabSize = this.recipe.vocabSize;
		this.ropeTheta = this.recipe.ropeTheta || 0;
		this.rotaryDim = this.recipe.rotaryDim || 0;
		this.rmsNormEps = this.recipe.normEps;
		this.layerNormEps = this.recipe.normEps;
		this.offsetRMSNorm = this.recipe.norm === 'rms_offset';
		this.mlpActivation = this.recipe.mlpActivation;
		this.embedScale = this.recipe.embedScale;
		this.attnScale = this.recipe.attnScale;
		this.globalRopeTheta = this.recipe.globalRopeTheta;
		this.localRopeTheta = this.recipe.localRopeTheta;
		this.slidingWindow = this.recipe.slidingWindow || 0;
		this.layerTypes = this.recipe.layerTypes;
		let endOfTextTokenId: number | number[] = this.config.eos_token_id ?? tokenizer.endOfTextTokenId ?? this.recipe.endOfTextTokenId ?? 0;

		if ( Array.isArray( endOfTextTokenId ) ) {

			endOfTextTokenId = endOfTextTokenId[ 0 ];

		}

		this.endOfTextTokenId = endOfTextTokenId;

		this._float32 = new Map();
		this.logitWeight = null;
		this.outputNormWeight = null;
		this.outputNormBias = null;
		this._blocks = [];

		if ( options.deferUnpack !== true ) this.unpackSync();

	}

	contextLimit(): number {

		return this.recipe.contextLimit;

	}

	prepareGeneration( prompt: string, maxTokens: number, maxNewTokens: number ): PreparedGeneration {

		return prepareGeneration( this.tokenizer, prompt, maxTokens, maxNewTokens, this.endOfTextTokenId );

	}

	static async fromURL( baseURL: string, options: LoaderOptions = {} ): Promise<DecoderWeights> {

		const bundle = await loadHFModelBundle( baseURL, options );
		return this.fromBundle( bundle, options );

	}

	static async fromBundle( bundle: HFModelBundle, options: LoaderOptions = {} ): Promise<DecoderWeights> {

		const weights = new this( bundle.rawConfig, bundle.tensors, bundle.tokenizer, {
			deferUnpack: true,
			prefix: bundle.prefix,
			rawConfig: bundle.rawConfig
		} );
		await weights.unpack( options.onProgress );
		return weights;

	}

	unpackSync(): void {

		this.logitWeight = this.loadOutputWeight();
		this.outputNormWeight = this.mappedFloat( 'output_norm' );
		this.outputNormBias = this.hasMapped( 'output_norm_bias' ) ? this.mappedFloat( 'output_norm_bias' ) : null;

		for ( let i = 0; i < this.layerCount; i ++ ) this._blocks[ i ] = this.createBlock( i );

	}

	async unpack( onProgress?: ProgressCallback ): Promise<void> {

		const report = createProgress( 'DecoderWeights', onProgress );
		await report( `Transposing output projection (${ this.vocabSize } x ${ this.hiddenSize }); UI may pause...` );
		this.logitWeight = this.loadOutputWeight();
		this.outputNormWeight = this.mappedFloat( 'output_norm' );
		this.outputNormBias = this.hasMapped( 'output_norm_bias' ) ? this.mappedFloat( 'output_norm_bias' ) : null;

		for ( let i = 0; i < this.layerCount; i ++ ) {

			this._blocks[ i ] = this.createBlock( i );
			await report( `Unpacked layer ${ i + 1 } / ${ this.layerCount }` );

		}

	}

	hasMapped( key: string, bid?: number ): boolean {

		return hasMappedTensor( this.tensors, this.tensorPrefix, this.architecture, key, bid );

	}

	hasTensor( name: string ): boolean {

		return this.tensors[ `${ this.tensorPrefix }${ name }` ] !== undefined || this.tensors[ name ] !== undefined;

	}

	mappedTensor( key: string, bid?: number ): Tensor {

		return resolveTensor( this.tensors, this.tensorPrefix, this.architecture, key, bid );

	}

	mappedFloat( key: string, bid?: number ): Float32Array {

		const cacheKey = bid === undefined ? key : `${ key }.${ bid }`;

		if ( this._float32.has( cacheKey ) ) return this._float32.get( cacheKey )!;

		const data = tensorToFloat32( this.mappedTensor( key, bid ) );
		this._float32.set( cacheKey, data );
		return data;

	}

	tensor( name: string, unprefixed = false ): Float32Array {

		const key = unprefixed ? name : `${ this.tensorPrefix }${ name }`;

		if ( this._float32.has( key ) ) return this._float32.get( key )!;

		const tensor = this.tensors[ key ] || this.tensors[ name ];

		if ( tensor === undefined ) {

			throw new Error( `DecoderWeights: Missing tensor "${ key }".` );

		}

		const data = tensorToFloat32( tensor );
		this._float32.set( key, data );
		return data;

	}

	linearMapped( key: string, bid: number, outFeatures: number, inFeatures: number ): Float32Array {

		const data = this.mappedFloat( key, bid );
		return this.recipe.transposeLinears ? transpose2D( data, outFeatures, inFeatures ) : data;

	}

	optionalBias( key: string, bid: number, size: number ): Float32Array {

		if ( this.hasMapped( key, bid ) === false ) return new Float32Array( size );

		return this.mappedFloat( key, bid );

	}

	loadOutputWeight(): Float32Array {

		const useUntiedHead = this.architecture !== 'gpt2'
			&& this.hasTensor( 'lm_head.weight' )
			&& this.config.tie_word_embeddings !== true;
		const source = useUntiedHead ? this.mappedFloat( 'output' ) : this.mappedFloat( 'token_embd' );
		return transpose2D( source, this.vocabSize, this.hiddenSize );

	}

	block( index: number ): DecoderBlock {

		return this._blocks[ index ];

	}

	createBlock( index: number ): DecoderBlock {

		const { architecture, recipe, hiddenSize, qSize, kvSize, innerSize } = this;

		if ( architecture === 'gpt2' ) {

			return {
				ln1Weight: this.mappedFloat( 'attn_norm', index ),
				ln1Bias: this.mappedFloat( 'attn_norm_bias', index ),
				ln2Weight: this.mappedFloat( 'ffn_norm', index ),
				ln2Bias: this.mappedFloat( 'ffn_norm_bias', index ),
				attnQKVWeight: this.mappedFloat( 'attn_qkv', index ),
				attnQKVBias: this.mappedFloat( 'attn_qkv_bias', index ),
				attnProjWeight: this.mappedFloat( 'attn_out', index ),
				attnProjBias: this.mappedFloat( 'attn_out_bias', index ),
				mlpFCWeight: this.mappedFloat( 'ffn_up', index ),
				mlpFCBias: this.mappedFloat( 'ffn_up_bias', index ),
				mlpProjWeight: this.mappedFloat( 'ffn_down', index ),
				mlpProjBias: this.mappedFloat( 'ffn_down_bias', index )
			};

		}

		if ( architecture === 'phi' ) {

			const q = this.linearMapped( 'attn_q', index, qSize, hiddenSize );
			const k = this.linearMapped( 'attn_k', index, kvSize, hiddenSize );
			const v = this.linearMapped( 'attn_v', index, kvSize, hiddenSize );
			const lnWeight = this.mappedFloat( 'attn_norm', index );
			const lnBias = this.mappedFloat( 'attn_norm_bias', index );

			return {
				lnWeight,
				lnBias,
				ln1Weight: lnWeight,
				ln1Bias: lnBias,
				attnQKVWeight: packProjections( [ q, k, v ], hiddenSize ),
				attnQKVBias: packBiases( [
					this.optionalBias( 'attn_q_bias', index, qSize ),
					this.optionalBias( 'attn_k_bias', index, kvSize ),
					this.optionalBias( 'attn_v_bias', index, kvSize )
				] ),
				attnProjWeight: this.linearMapped( 'attn_out', index, hiddenSize, qSize ),
				attnProjBias: this.optionalBias( 'attn_out_bias', index, hiddenSize ),
				mlpFCWeight: this.linearMapped( 'ffn_up', index, innerSize, hiddenSize ),
				mlpFCBias: this.optionalBias( 'ffn_up_bias', index, innerSize ),
				mlpProjWeight: this.linearMapped( 'ffn_down', index, hiddenSize, innerSize ),
				mlpProjBias: this.optionalBias( 'ffn_down_bias', index, hiddenSize )
			};

		}

		const q = this.linearMapped( 'attn_q', index, qSize, hiddenSize );
		const k = this.linearMapped( 'attn_k', index, kvSize, hiddenSize );
		const v = this.linearMapped( 'attn_v', index, kvSize, hiddenSize );
		const layerType = ( this.layerTypes && this.layerTypes[ index ] ) || 'full_attention';
		const ropeTheta = architecture === 'gemma3'
			? ( layerType === 'full_attention' ? this.globalRopeTheta : this.localRopeTheta )
			: this.ropeTheta;
		const slidingWindow = architecture === 'gemma3'
			? ( layerType === 'sliding_attention' ? this.slidingWindow : 0 )
			: ( recipe.slidingWindow || 0 );

		const block: DecoderBlock = {
			layerType,
			ropeTheta,
			slidingWindow,
			ln1Weight: this.mappedFloat( 'attn_norm', index ),
			attnQKVWeight: packProjections( [ q, k, v ], hiddenSize ),
			attnQKVBias: null,
			attnProjWeight: this.linearMapped( 'attn_out', index, hiddenSize, qSize ),
			attnProjBias: null,
			mlpGateWeight: this.linearMapped( 'ffn_gate', index, innerSize, hiddenSize ),
			mlpUpWeight: this.linearMapped( 'ffn_up', index, innerSize, hiddenSize ),
			mlpDownWeight: this.linearMapped( 'ffn_down', index, hiddenSize, innerSize )
		};

		if ( recipe.postNorms ) {

			block.postAttnNormWeight = this.mappedFloat( 'post_attn_norm', index );
			block.preMlpNormWeight = this.mappedFloat( 'ffn_norm', index );
			block.postMlpNormWeight = this.mappedFloat( 'post_ffn_norm', index );

		} else {

			block.ln2Weight = this.mappedFloat( 'ffn_norm', index );

		}

		if ( recipe.qkNorm ) {

			block.qNormWeight = this.mappedFloat( 'attn_q_norm', index );
			block.kNormWeight = this.mappedFloat( 'attn_k_norm', index );

		}

		return block;

	}

	embedding( tokenId: number, position: number, target: Float32Array<ArrayBufferLike> = new Float32Array( this.hiddenSize ) ): Float32Array {

		const tokenEmbedding = this.mappedFloat( 'token_embd' );
		const tokenOffset = tokenId * this.hiddenSize;

		if ( this.recipe.position === 'learned' ) {

			const positionEmbedding = this.mappedFloat( 'pos_embd' );
			const positionOffset = position * this.hiddenSize;

			for ( let i = 0; i < this.hiddenSize; i ++ ) {

				target[ i ] = tokenEmbedding[ tokenOffset + i ] + positionEmbedding[ positionOffset + i ];

			}

			return target;

		}

		if ( this.embedScale !== 1 ) {

			for ( let i = 0; i < this.hiddenSize; i ++ ) {

				target[ i ] = tokenEmbedding[ tokenOffset + i ] * this.embedScale;

			}

			return target;

		}

		target.set( tokenEmbedding.subarray( tokenOffset, tokenOffset + this.hiddenSize ) );
		return target;

	}

}

export { DecoderWeights };
