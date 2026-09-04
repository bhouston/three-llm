import { detectPrefix, loadHFModelBundle } from '../load/HFModelBundle.js';
import { packProjections, prepareGeneration, tensorToFloat32, transpose2D, unwrapTextConfig, createProgress, copyTensorRow, releaseBlockWeightArrays } from '../load/tensors.js';
import { resolveTensor } from '../load/TensorNameMap.js';
import { recipeFor } from '../load/DecoderRecipe.js';
import { formatChatTemplate, stopTokenIdsFor } from '../runtime/chatTemplates.js';
import type {
	Architecture, ChatMessage, DecoderBlock, DecoderRecipe, FormatChatOptions, HFModelBundle,
	HuggingFaceConfig, LoaderOptions, PreparedGeneration, ProgressCallback, Tensor, TensorMap, Tokenizer
} from '../types.js';

/**
 * Loads a Hugging Face Qwen3.5 text backbone (`qwen3_5` / `qwen3_5_text`).
 *
 * Hybrid decode: Gated DeltaNet linear-attention layers and gated full
 * attention with QK-norm and partial RoPE.
 *
 */
class QwenWeights {

	architecture: Architecture;
	config: HuggingFaceConfig;
	rawConfig: HuggingFaceConfig;
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
	layerTypes?: string[];
	rmsNormEps: number;
	offsetRMSNorm: boolean;
	mlpActivation?: string;
	linearKeyDim: number;
	linearValueDim: number;
	linearKeyHeads: number;
	linearValueHeads: number;
	linearConvKernel: number;
	ropeTheta?: number;
	partialRotaryFactor: number;
	rotaryDim: number;
	attnScale?: number;
	endOfTextTokenId: number;
	stopTokenIds: number[];
	_float32: Map<string, Float32Array>;
	_tokenEmbed: Tensor | null;
	logitWeight: Float32Array | null;
	outputNormWeight: Float32Array | null;
	_blocks: DecoderBlock[];

	constructor( config: HuggingFaceConfig, tensors: TensorMap, tokenizer: Tokenizer, options: LoaderOptions = {} ) {

		this.architecture = 'qwen3_5';
		this.config = unwrapTextConfig( config );
		this.rawConfig = config;
		this.recipe = recipeFor( config );
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
		this.layerTypes = this.recipe.layerTypes;
		this.rmsNormEps = this.recipe.normEps;
		this.offsetRMSNorm = true;
		this.mlpActivation = this.recipe.mlpActivation;
		this.linearKeyDim = this.recipe.linearKeyDim as number;
		this.linearValueDim = this.recipe.linearValueDim as number;
		this.linearKeyHeads = this.recipe.linearKeyHeads as number;
		this.linearValueHeads = this.recipe.linearValueHeads as number;
		this.linearConvKernel = this.recipe.linearConvKernel as number;
		this.ropeTheta = this.recipe.ropeTheta;
		this.partialRotaryFactor = this.config.rope_parameters?.partial_rotary_factor ?? 0.25;
		this.rotaryDim = this.recipe.rotaryDim;
		this.attnScale = this.recipe.attnScale;
		this.endOfTextTokenId = this.recipe.endOfTextTokenId ?? tokenizer.endOfTextTokenId ?? 248044;
		this.stopTokenIds = stopTokenIdsFor( this.recipe.chatTemplate, tokenizer, this.recipe.stopTokenIds || [ this.endOfTextTokenId ] );
		this._float32 = new Map();
		this._tokenEmbed = null;

		this.logitWeight = null;
		this.outputNormWeight = null;
		this._blocks = [];

		if ( options.deferUnpack !== true ) this.unpackSync();

	}

	contextLimit(): number {

		return this.recipe.contextLimit;

	}

	prepareGeneration( prompt: string, maxTokens: number, maxNewTokens: number ): PreparedGeneration {

		return prepareGeneration( this.tokenizer, prompt, maxTokens, maxNewTokens, this.endOfTextTokenId );

	}

	formatChat( messages: ChatMessage[], options: FormatChatOptions = {} ): string {

		return formatChatTemplate( this.recipe.chatTemplate || 'qwen3_5', messages, options );

	}

	static async fromURL( baseURL: string, options: LoaderOptions = {} ): Promise<QwenWeights> {

		const bundle = await loadHFModelBundle( baseURL, { ...options, label: 'QwenWeights' } );
		return this.fromBundle( bundle, options );

	}

	static async fromBundle( bundle: HFModelBundle, options: LoaderOptions = {} ): Promise<QwenWeights> {

		const weights = new this( bundle.rawConfig, bundle.tensors, bundle.tokenizer, {
			deferUnpack: true,
			prefix: bundle.prefix
		} );
		await weights.unpack( options.onProgress );
		return weights;

	}

	unpackSync(): void {

		this.captureEmbeddings();
		this.logitWeight = this.loadOutputWeight();
		this.outputNormWeight = this.mappedFloat( 'output_norm' );
		for ( let i = 0; i < this.layerCount; i ++ ) this._blocks[ i ] = this.createBlock( i );

	}

	async unpack( onProgress?: ProgressCallback ): Promise<void> {

		const report = createProgress( 'QwenWeights', onProgress );
		await report( `Transposing output projection (${ this.vocabSize } x ${ this.hiddenSize }); UI may pause...` );
		this.captureEmbeddings();
		this.logitWeight = this.loadOutputWeight();
		this.outputNormWeight = this.mappedFloat( 'output_norm' );

		for ( let i = 0; i < this.layerCount; i ++ ) {

			this._blocks[ i ] = this.createBlock( i );
			await report( `Unpacked layer ${ i + 1 } / ${ this.layerCount }` );

		}

	}

	hasTensor( name: string ): boolean {

		return this.tensors[ `${ this.tensorPrefix }${ name }` ] !== undefined || this.tensors[ name ] !== undefined;

	}

	mappedFloat( key: string, bid?: number ): Float32Array {

		const cacheKey = bid === undefined ? key : `${ key }.${ bid }`;
		if ( this._float32.has( cacheKey ) ) return this._float32.get( cacheKey )!;
		const data = tensorToFloat32( resolveTensor( this.tensors, this.tensorPrefix, 'qwen3_5', key, bid ) );
		this._float32.set( cacheKey, data );
		return data;

	}

	captureEmbeddings(): void {

		this._tokenEmbed = resolveTensor( this.tensors, this.tensorPrefix, 'qwen3_5', 'token_embd' );

	}

	tensor( name: string ): Float32Array {

		if ( this._float32.has( name ) ) return this._float32.get( name )!;

		const tensor = this.tensors[ `${ this.tensorPrefix }${ name }` ] || this.tensors[ name ];

		if ( tensor === undefined ) {

			throw new Error( `QwenWeights: Missing tensor "${ this.tensorPrefix }${ name }".` );

		}

		const data = tensorToFloat32( tensor );
		this._float32.set( name, data );
		return data;

	}

	linearMapped( key: string, bid: number, outFeatures: number, inFeatures: number ): Float32Array {

		const transposed = transpose2D( this.mappedFloat( key, bid ), outFeatures, inFeatures );
		this._float32.delete( `${ key }.${ bid }` );
		return transposed;

	}

	loadOutputWeight(): Float32Array {

		const useUntiedHead = this.hasTensor( 'lm_head.weight' ) && this.config.tie_word_embeddings !== true;
		const sourceKey = useUntiedHead ? 'output' : 'token_embd';
		const source = this.mappedFloat( sourceKey );
		const weight = transpose2D( source, this.vocabSize, this.hiddenSize );
		this._float32.delete( sourceKey );
		return weight;

	}

	block( index: number ): DecoderBlock {

		return this._blocks[ index ];

	}

	createBlock( index: number ): DecoderBlock {

		const layerType = ( this.layerTypes && this.layerTypes[ index ] ) || 'full_attention';
		const { hiddenSize, innerSize } = this;
		const block = {
			layerType,
			ln1Weight: this.mappedFloat( 'attn_norm', index ),
			ln2Weight: this.mappedFloat( 'ffn_norm', index ),
			mlpGateWeight: this.linearMapped( 'ffn_gate', index, innerSize, hiddenSize ),
			mlpUpWeight: this.linearMapped( 'ffn_up', index, innerSize, hiddenSize ),
			mlpDownWeight: this.linearMapped( 'ffn_down', index, hiddenSize, innerSize )
		} as DecoderBlock;

		if ( layerType === 'linear_attention' ) {

			const convDim = this.linearKeyHeads * this.linearKeyDim * 2 + this.linearValueHeads * this.linearValueDim;
			block.delta = {
				qkvWeight: this.linearMapped( 'delta_qkv', index, convDim, hiddenSize ),
				zWeight: this.linearMapped( 'delta_z', index, this.linearValueHeads * this.linearValueDim, hiddenSize ),
				bWeight: this.linearMapped( 'delta_b', index, this.linearValueHeads, hiddenSize ),
				aWeight: this.linearMapped( 'delta_a', index, this.linearValueHeads, hiddenSize ),
				outWeight: this.linearMapped( 'delta_out', index, hiddenSize, this.linearValueHeads * this.linearValueDim ),
				convWeight: this.mappedFloat( 'delta_conv', index ),
				aLog: this.mappedFloat( 'delta_a_log', index ),
				dtBias: this.mappedFloat( 'delta_dt_bias', index ),
				normWeight: this.mappedFloat( 'delta_norm', index )
			};

		} else {

			const qGate = this.linearMapped( 'attn_q', index, this.qSize * 2, hiddenSize );
			const k = this.linearMapped( 'attn_k', index, this.kvSize, hiddenSize );
			const v = this.linearMapped( 'attn_v', index, this.kvSize, hiddenSize );
			block.qGateWeight = qGate;
			block.attnQKVWeight = packProjections( [ k, v ], hiddenSize );
			block.attnProjWeight = this.linearMapped( 'attn_out', index, hiddenSize, this.qSize );
			block.qNormWeight = this.mappedFloat( 'attn_q_norm', index );
			block.kNormWeight = this.mappedFloat( 'attn_k_norm', index );

		}

		return block;

	}

	embedding( tokenId: number, _position: number, target: Float32Array<ArrayBufferLike> = new Float32Array( this.hiddenSize ) ): Float32Array {

		const tokenEmbedding = this._tokenEmbed || resolveTensor( this.tensors, this.tensorPrefix, 'qwen3_5', 'token_embd' );
		return copyTensorRow( tokenEmbedding, tokenId * this.hiddenSize, this.hiddenSize, target );

	}

	releaseCheckpointTensors(): void {

		const keep = new Set<Tensor>();
		if ( this._tokenEmbed ) keep.add( this._tokenEmbed );

		for ( const name of Object.keys( this.tensors ) ) {

			if ( keep.has( this.tensors[ name ] ) === false ) delete this.tensors[ name ];

		}

		this._float32.clear();

	}

	releaseUnpackedWeightArrays(): void {

		this.logitWeight = null;
		this.outputNormWeight = null;
		this._float32.clear();

		for ( const block of this._blocks ) releaseBlockWeightArrays( block );

	}

}

export { QwenWeights };
