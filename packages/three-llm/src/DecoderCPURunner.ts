import { causalAttention, geluNew, layerNorm, linear, logitSoftcap, rmsNorm, silu } from './LLMMath.js';
import { generateSync } from './LLMGenerate.js';
import type { DecoderWeights } from './DecoderWeights.js';
import type { DecoderBlock, DecoderRecipe, GenerateOptions, GenerationResult, RunnerOptions } from './types.js';

interface KvCache {
	key: Float32Array;
	value: Float32Array;
}

/**
 * CPU reference for the parameterized decoder graph.
 *
 */
class DecoderCPURunner {

	weights: DecoderWeights;
	recipe: DecoderRecipe;
	maxTokens: number;
	hiddenSize: number;
	caches: KvCache[];
	_cacheTokens?: number[];
	_cacheLogits?: Float32Array | null;

	constructor( weights: DecoderWeights, options: RunnerOptions = {} ) {

		this.weights = weights;
		this.recipe = weights.recipe;
		this.maxTokens = Math.min( options.maxTokens || weights.contextLimit(), weights.contextLimit() );
		this.hiddenSize = weights.hiddenSize;
		this.caches = [];

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			this.caches.push( {
				key: new Float32Array( weights.kvSize * this.maxTokens ),
				value: new Float32Array( weights.kvSize * this.maxTokens )
			} );

		}

	}

	norm( x: Float32Array, weight: Float32Array | undefined, bias?: Float32Array | null ): Float32Array {

		if ( this.recipe.norm === 'layer_norm' ) {

			return layerNorm( x, weight!, bias!, this.recipe.normEps );

		}

		return rmsNorm( x, weight ?? null, this.recipe.normEps, this.recipe.norm === 'rms_offset' );

	}

	forwardToken( tokenId: number, position: number ): Float32Array {

		const { weights, recipe, hiddenSize } = this;
		const x = weights.embedding( tokenId, position );

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			const block = weights.block( i );
			const cache = this.caches[ i ];

			if ( recipe.residual === 'parallel' ) {

				const normed = this.norm( x, block.lnWeight, block.lnBias );
				const qkv = linear( normed, block.attnQKVWeight, block.attnQKVBias ?? null, hiddenSize, weights.qSize + 2 * weights.kvSize );
				const attn = this.attention( qkv, block, cache, position );
				const attnOut = linear( attn, block.attnProjWeight, block.attnProjBias ?? null, weights.qSize, hiddenSize );
				const inner = linear( normed, block.mlpFCWeight!, block.mlpFCBias ?? null, hiddenSize, weights.innerSize );

				for ( let dim = 0; dim < inner.length; dim ++ ) inner[ dim ] = geluNew( inner[ dim ] );

				const mlpOut = linear( inner, block.mlpProjWeight!, block.mlpProjBias ?? null, weights.innerSize, hiddenSize );

				for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] += attnOut[ dim ] + mlpOut[ dim ];
				continue;

			}

			if ( recipe.postNorms ) {

				const residualAttn = x.slice();
				const norm1 = this.norm( x, block.ln1Weight );
				const qkv = linear( norm1, block.attnQKVWeight, null, hiddenSize, weights.qSize + 2 * weights.kvSize );
				const attn = this.attention( qkv, block, cache, position );
				let attnOut = linear( attn, block.attnProjWeight, null, weights.qSize, hiddenSize );
				attnOut = this.norm( attnOut, block.postAttnNormWeight );

				for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] = residualAttn[ dim ] + attnOut[ dim ];

				const residualMlp = x.slice();
				const preMlp = this.norm( x, block.preMlpNormWeight );
				const gate = linear( preMlp, block.mlpGateWeight!, null, hiddenSize, weights.innerSize );
				const up = linear( preMlp, block.mlpUpWeight!, null, hiddenSize, weights.innerSize );
				const hidden = new Float32Array( weights.innerSize );
				const activate = recipe.mlpActivation === 'silu' ? silu : geluNew;

				for ( let dim = 0; dim < hidden.length; dim ++ ) hidden[ dim ] = activate( gate[ dim ] ) * up[ dim ];

				let mlpOut = linear( hidden, block.mlpDownWeight!, null, weights.innerSize, hiddenSize );
				mlpOut = this.norm( mlpOut, block.postMlpNormWeight );

				for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] = residualMlp[ dim ] + mlpOut[ dim ];
				continue;

			}

			const qkvOut = recipe.architecture === 'gpt2' ? hiddenSize * 3 : weights.qSize + 2 * weights.kvSize;
			const attnIn = recipe.architecture === 'gpt2' ? hiddenSize : weights.qSize;
			const norm1 = this.norm( x, block.ln1Weight, block.ln1Bias );
			const qkv = linear( norm1, block.attnQKVWeight, block.attnQKVBias ?? null, hiddenSize, qkvOut );
			const attn = this.attention( qkv, block, cache, position );
			const attnOut = linear( attn, block.attnProjWeight, block.attnProjBias ?? null, attnIn, hiddenSize );

			for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] += attnOut[ dim ];

			const norm2 = this.norm( x, block.ln2Weight, block.ln2Bias );

			if ( recipe.mlp === 'dense_gelu' ) {

				const inner = linear( norm2, block.mlpFCWeight!, block.mlpFCBias ?? null, hiddenSize, weights.innerSize );

				for ( let dim = 0; dim < inner.length; dim ++ ) inner[ dim ] = geluNew( inner[ dim ] );

				const mlpOut = linear( inner, block.mlpProjWeight!, block.mlpProjBias ?? null, weights.innerSize, hiddenSize );

				for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] += mlpOut[ dim ];

			} else {

				const gate = linear( norm2, block.mlpGateWeight!, null, hiddenSize, weights.innerSize );
				const up = linear( norm2, block.mlpUpWeight!, null, hiddenSize, weights.innerSize );
				const hidden = new Float32Array( weights.innerSize );
				const activate = recipe.mlpActivation === 'silu' ? silu : geluNew;

				for ( let dim = 0; dim < hidden.length; dim ++ ) hidden[ dim ] = activate( gate[ dim ] ) * up[ dim ];

				const mlpOut = linear( hidden, block.mlpDownWeight!, null, weights.innerSize, hiddenSize );

				for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] += mlpOut[ dim ];

			}

		}

		const normed = this.norm( x, weights.outputNormWeight ?? undefined, weights.outputNormBias );
		const logits = linear( normed, weights.logitWeight!, null, hiddenSize, weights.vocabSize );
		return logitSoftcap( logits, recipe.finalLogitSoftcap );

	}

	attention( qkv: Float32Array, block: DecoderBlock, cache: KvCache, position: number ): Float32Array {

		const { weights, recipe } = this;

		return causalAttention( qkv, {
			headCount: weights.headCount,
			kvHeadCount: weights.kvHeadCount,
			headDim: weights.headDim,
			position,
			keyCache: cache.key,
			valueCache: cache.value,
			ropeTheta: block.ropeTheta !== undefined ? block.ropeTheta : recipe.ropeTheta,
			rotaryDim: recipe.rotaryDim || weights.headDim,
			slidingWindow: block.slidingWindow || 0,
			attnScale: recipe.attnScale,
			qNormWeight: block.qNormWeight,
			kNormWeight: block.kNormWeight,
			rmsEpsilon: recipe.normEps,
			offsetRMSNorm: recipe.norm === 'rms_offset'
		} );

	}

	resetCache(): void {

		this._cacheTokens = [];
		this._cacheLogits = null;

		for ( const cache of this.caches ) {

			cache.key.fill( 0 );
			cache.value.fill( 0 );

		}

	}

	generate( prompt: string, options: GenerateOptions = {} ): GenerationResult {

		return generateSync( this, prompt, options, {
			rewindable: true,
			resetCache: () => this.resetCache(),
			forwardToken: ( tokenId, position ) => this.forwardToken( tokenId, position )
		} );

	}

}

export { DecoderCPURunner };
