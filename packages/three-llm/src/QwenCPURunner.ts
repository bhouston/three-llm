import {
	causalAttention,
	causalConv1dStep,
	gatedDeltaRuleStep,
	l2norm,
	linear,
	rmsNorm,
	rmsNormGated,
	sigmoid,
	silu,
	softplus,
	splitHeadGate
} from './LLMMath.js';
import { generateSync } from './LLMGenerate.js';
import type { QwenWeights } from './QwenWeights.js';
import type { DecoderBlock, GenerateOptions, GenerationResult, QwenDeltaWeights, RunnerOptions } from './types.js';

interface KvCache {
	key: Float32Array;
	value: Float32Array;
	conv?: undefined;
	state?: undefined;
}

interface DeltaCache {
	conv: Float32Array;
	state: Float32Array;
	key?: undefined;
	value?: undefined;
}

type LayerCache = KvCache | DeltaCache;

/**
 * CPU reference for Qwen3.5 hybrid decode.
 *
 */
class QwenCPURunner {

	weights: QwenWeights;
	maxTokens: number;
	hiddenSize: number;
	caches: LayerCache[];
	lastHidden?: Float32Array;
	_cacheTokens?: number[];
	_cacheLogits?: Float32Array | null;

	constructor( weights: QwenWeights, options: RunnerOptions = {} ) {

		this.weights = weights;
		this.maxTokens = Math.min( options.maxTokens || weights.contextLimit(), weights.contextLimit() );
		this.hiddenSize = weights.hiddenSize;
		this.caches = [];

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			const layerType = ( weights.layerTypes && weights.layerTypes[ i ] ) || 'full_attention';

			if ( layerType === 'linear_attention' ) {

				this.caches.push( {
					conv: new Float32Array( ( weights.linearKeyHeads * weights.linearKeyDim * 2 + weights.linearValueHeads * weights.linearValueDim ) * weights.linearConvKernel ),
					state: new Float32Array( weights.linearValueHeads * weights.linearKeyDim * weights.linearValueDim )
				} );

			} else {

				this.caches.push( {
					key: new Float32Array( weights.kvSize * this.maxTokens ),
					value: new Float32Array( weights.kvSize * this.maxTokens )
				} );

			}

		}

	}

	forwardToken( tokenId: number, position: number ): Float32Array {

		const { weights, hiddenSize } = this;
		const x = weights.embedding( tokenId, position );

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			const block = weights.block( i );
			const cache = this.caches[ i ];
			const norm1 = rmsNorm( x, block.ln1Weight ?? null, weights.rmsNormEps, true );
			const mixed = block.layerType === 'linear_attention'
				? this.linearAttention( norm1, block.delta!, cache as DeltaCache )
				: this.fullAttention( norm1, block, cache as KvCache, position );

			for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] += mixed[ dim ];

			const norm2 = rmsNorm( x, block.ln2Weight ?? null, weights.rmsNormEps, true );
			const gate = linear( norm2, block.mlpGateWeight!, null, hiddenSize, weights.innerSize );
			const up = linear( norm2, block.mlpUpWeight!, null, hiddenSize, weights.innerSize );
			const hidden = new Float32Array( weights.innerSize );

			for ( let dim = 0; dim < hidden.length; dim ++ ) hidden[ dim ] = silu( gate[ dim ] ) * up[ dim ];

			const mlpOut = linear( hidden, block.mlpDownWeight!, null, weights.innerSize, hiddenSize );

			for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] += mlpOut[ dim ];

		}

		const normed = rmsNorm( x, weights.outputNormWeight, weights.rmsNormEps, true );
		this.lastHidden = normed;
		return linear( normed, weights.logitWeight!, null, hiddenSize, weights.vocabSize );

	}

	fullAttention( input: Float32Array, block: DecoderBlock, cache: KvCache, position: number ): Float32Array {

		const { weights } = this;
		const packedQGate = linear( input, block.qGateWeight!, null, weights.hiddenSize, weights.qSize * 2 );
		const { query, gate } = splitHeadGate( packedQGate, weights.headCount, weights.headDim );
		const kv = linear( input, block.attnQKVWeight, null, weights.hiddenSize, 2 * weights.kvSize );
		const qkv = new Float32Array( weights.qSize + 2 * weights.kvSize );
		qkv.set( query, 0 );
		qkv.set( kv, weights.qSize );
		const attn = causalAttention( qkv, {
			headCount: weights.headCount,
			kvHeadCount: weights.kvHeadCount,
			headDim: weights.headDim,
			position,
			keyCache: cache.key,
			valueCache: cache.value,
			ropeTheta: weights.ropeTheta,
			rotaryDim: weights.rotaryDim,
			attnScale: weights.attnScale,
			qNormWeight: block.qNormWeight,
			kNormWeight: block.kNormWeight,
			rmsEpsilon: weights.rmsNormEps,
			offsetRMSNorm: true,
			outputGate: gate
		} );
		return linear( attn, block.attnProjWeight, null, weights.qSize, weights.hiddenSize );

	}

	linearAttention( input: Float32Array, delta: QwenDeltaWeights, cache: DeltaCache ): Float32Array {

		const { weights } = this;
		const convDim = weights.linearKeyHeads * weights.linearKeyDim * 2 + weights.linearValueHeads * weights.linearValueDim;
		const mixedQKV = linear( input, delta.qkvWeight, null, weights.hiddenSize, convDim );
		const conv = causalConv1dStep( mixedQKV, cache.conv, delta.convWeight, weights.linearConvKernel, 'silu' );
		const keySize = weights.linearKeyHeads * weights.linearKeyDim;
		const valueSize = weights.linearValueHeads * weights.linearValueDim;
		const query = conv.slice( 0, keySize );
		const key = conv.slice( keySize, keySize * 2 );
		const value = conv.slice( keySize * 2 );
		const z = linear( input, delta.zWeight, null, weights.hiddenSize, valueSize );
		const b = linear( input, delta.bWeight, null, weights.hiddenSize, weights.linearValueHeads );
		const a = linear( input, delta.aWeight, null, weights.hiddenSize, weights.linearValueHeads );
		const beta = new Float32Array( weights.linearValueHeads );
		const decay = new Float32Array( weights.linearValueHeads );

		for ( let head = 0; head < weights.linearValueHeads; head ++ ) {

			beta[ head ] = sigmoid( b[ head ] );
			decay[ head ] = Math.exp( - Math.exp( delta.aLog[ head ] ) * softplus( a[ head ] + delta.dtBias[ head ] ) );

		}

		const repeat = weights.linearValueHeads / weights.linearKeyHeads;
		const qUsed = repeat === 1 ? query : repeatHeads( query, weights.linearKeyHeads, weights.linearKeyDim, repeat );
		const kUsed = repeat === 1 ? key : repeatHeads( key, weights.linearKeyHeads, weights.linearKeyDim, repeat );

		for ( let head = 0; head < weights.linearValueHeads; head ++ ) {

			l2norm( qUsed, head * weights.linearKeyDim, weights.linearKeyDim );
			l2norm( kUsed, head * weights.linearKeyDim, weights.linearKeyDim );

			const qScale = weights.linearKeyDim ** - 0.5;
			const qOff = head * weights.linearKeyDim;

			for ( let i = 0; i < weights.linearKeyDim; i ++ ) qUsed[ qOff + i ] *= qScale;

		}

		const mixed = gatedDeltaRuleStep( qUsed, kUsed, value, decay, beta, cache.state, {
			numVHeads: weights.linearValueHeads,
			keyDim: weights.linearKeyDim,
			valueDim: weights.linearValueDim
		} );
		const gated = rmsNormGated( mixed, z, delta.normWeight, weights.linearValueHeads, weights.linearValueDim, weights.rmsNormEps );
		return linear( gated, delta.outWeight, null, valueSize, weights.hiddenSize );

	}

	resetCache(): void {

		this._cacheTokens = [];
		this._cacheLogits = null;

		for ( const cache of this.caches ) {

			if ( cache.key ) {

				cache.key.fill( 0 );
				cache.value.fill( 0 );

			} else {

				cache.conv.fill( 0 );
				cache.state.fill( 0 );

			}

		}

	}

	generate( prompt: string, options: GenerateOptions = {} ): GenerationResult {

		return generateSync( this, prompt, options, {
			rewindable: false,
			resetCache: () => this.resetCache(),
			forwardToken: ( tokenId, position ) => this.forwardToken( tokenId, position )
		} );

	}

}

function repeatHeads( vector: Float32Array, headCount: number, headDim: number, repeat: number ): Float32Array {

	const target = new Float32Array( headCount * repeat * headDim );

	for ( let head = 0; head < headCount; head ++ ) {

		const source = vector.subarray( head * headDim, ( head + 1 ) * headDim );

		for ( let copy = 0; copy < repeat; copy ++ ) {

			target.set( source, ( head * repeat + copy ) * headDim );

		}

	}

	return target;

}

export { QwenCPURunner };
