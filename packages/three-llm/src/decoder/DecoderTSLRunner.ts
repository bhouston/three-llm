import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, If, instanceIndex, storage, uint } from 'three/tsl';

import { DecoderWeights } from './DecoderWeights.js';
import { generateAsync } from '../runtime/generate.js';
import { logitSoftcap } from '../runtime/math.js';
import { TSLAdd } from '../tsl/TSLAdd.js';
import { TSLAttention } from '../tsl/TSLAttention.js';
import { orderedComputeNodes } from '../tsl/TSLCompute.js';
import { TSLGatedMLP } from '../tsl/TSLGatedMLP.js';
import { TSLLinear } from '../tsl/TSLLinear.js';
import { createChunkedLogitLayers, createLogitSampler, readChunkedLogits, TSLLogitSampler } from '../tsl/TSLLogits.js';
import type { LogitChunk } from '../tsl/TSLLogits.js';
import { TSLMLP } from '../tsl/TSLMLP.js';
import { TSLNormalize } from '../tsl/TSLNormalize.js';
import { TSLRMSNorm } from '../tsl/TSLRMSNorm.js';
import {
	collectAttentionWeights, collectLinearWeights, collectLogitWeights, collectMlpWeights, collectNormWeights,
	uploadAndReleaseStaticWeights
} from '../tsl/releaseCPU.js';
import type {
	ComputeNode, DecoderBlock, DecoderRecipe, GenerateOptions, GenerationResult, LoaderOptions, Renderer,
	RunnerOptions, SampleOptions, TslNode
} from '../types.js';

type NormKernel = TSLNormalize | TSLRMSNorm;
type MlpKernel = TSLMLP | TSLGatedMLP;

interface DecoderLayer {
	kind: 'parallel' | 'gemma' | 'sequential';
	outputNode: TslNode;
	attention: TSLAttention;
	ln?: NormKernel;
	ln1?: NormKernel;
	ln2?: NormKernel;
	qkv: TSLLinear;
	attnProj: TSLLinear;
	mlp: MlpKernel;
	addAttention: TSLAdd;
	addMLP: TSLAdd;
	postAttnNorm?: NormKernel;
	preMlp?: NormKernel;
	postMlpNorm?: NormKernel;
}

/**
 * Parameterized TSL decoder for GPT-2, Llama-family, Phi, and Gemma 3.
 *
 */
class DecoderTSLRunner {

	weights: DecoderWeights;
	recipe: DecoderRecipe;
	maxTokens: number;
	workgroupSize: number;
	logitChunkSize: number;
	prefillChunkSize: number;
	hiddenSize: number;
	embeddingBuffer: Float32Array;
	embeddingAttribute: StorageBufferAttribute;
	embeddingNode: TslNode;
	positionBuffer: Uint32Array;
	positionAttribute: StorageBufferAttribute;
	positionNode: TslNode;
	prefillCursorBuffer: Uint32Array;
	prefillCursorAttribute: StorageBufferAttribute;
	prefillCursorNode: TslNode;
	prefillEmbeddingBuffer: Float32Array;
	prefillEmbeddingAttribute: StorageBufferAttribute;
	prefillEmbeddingNode: TslNode;
	prefillCopyNode: ComputeNode;
	prefillAdvanceNode: ComputeNode;
	layers: DecoderLayer[];
	finalNorm: NormKernel;
	logits: LogitChunk[];
	logitSampler: TSLLogitSampler;
	prefillComputeNodes: ComputeNode[];
	computeNodes: ComputeNode[];
	_cacheTokens?: number[];
	_cacheLogits?: Float32Array | null;
	_releasedCPUWeights: boolean;

	constructor( weights: DecoderWeights, options: RunnerOptions = {} ) {

		this.weights = weights;
		this.recipe = weights.recipe;
		this.maxTokens = Math.min( options.maxTokens || weights.contextLimit(), weights.contextLimit() );
		this.workgroupSize = options.workgroupSize || 64;
		this.logitChunkSize = options.logitChunkSize || 8192;
		this.prefillChunkSize = options.prefillChunkSize || 32;
		this.hiddenSize = weights.hiddenSize;
		this.embeddingBuffer = new Float32Array( this.hiddenSize );
		this.embeddingAttribute = new StorageBufferAttribute( this.embeddingBuffer, 1 );
		this.embeddingNode = storage( this.embeddingAttribute, 'float', this.hiddenSize ).setName( `${ weights.architecture }Embedding` );
		this.positionBuffer = new Uint32Array( 1 );
		this.positionAttribute = new StorageBufferAttribute( this.positionBuffer, 1 );
		this.positionNode = storage( this.positionAttribute, 'uint', 1 ).setName( `${ weights.architecture }Position` );
		this.prefillCursorBuffer = new Uint32Array( 1 );
		this.prefillCursorAttribute = new StorageBufferAttribute( this.prefillCursorBuffer, 1 );
		this.prefillCursorNode = storage( this.prefillCursorAttribute, 'uint', 1 ).setName( `${ weights.architecture }PrefillCursor` );
		this.prefillEmbeddingBuffer = new Float32Array( this.prefillChunkSize * this.hiddenSize );
		this.prefillEmbeddingAttribute = new StorageBufferAttribute( this.prefillEmbeddingBuffer, 1 );
		this.prefillEmbeddingNode = storage( this.prefillEmbeddingAttribute, 'float', this.prefillEmbeddingBuffer.length ).setName( `${ weights.architecture }PrefillEmbeddings` );
		this.prefillCopyNode = this.createPrefillCopyNode( `${ weights.architecture }PrefillCopy` );
		this.prefillAdvanceNode = this.createPrefillAdvanceNode( `${ weights.architecture }PrefillAdvance` );
		this.layers = [];

		let currentNode = this.embeddingNode;

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			const built = this.buildLayer( weights.block( i ), i, currentNode );
			this.layers.push( built );
			currentNode = built.outputNode;

		}

		this.finalNorm = this.buildFinalNorm( currentNode );
		this.logits = createChunkedLogitLayers( this.finalNorm.outputNode, weights, this.logitChunkSize, `${ weights.architecture }Logits` );
		weights.logitWeight = null;
		this.logitSampler = createLogitSampler( this.logits, {
			candidateCount: options.logitCandidateCount || 8,
			logitSoftcap: this.recipe.finalLogitSoftcap,
			name: `${ weights.architecture }Logits`
		} );
		this.prefillComputeNodes = this.createComputeNodes( false );
		this.computeNodes = this.createComputeNodes();
		this.weights.releaseCheckpointTensors();
		this._releasedCPUWeights = false;

	}

	static async fromURL( baseURL: string, options?: LoaderOptions & RunnerOptions ) {

		return new this( await DecoderWeights.fromURL( baseURL, options ), options );

	}

	buildNorm( node: TslNode, weight: Float32Array | null | undefined, bias: Float32Array | null | undefined, name: string ) {

		if ( this.recipe.norm === 'layer_norm' ) {

			return new TSLNormalize( node, weight!, bias ?? null, this.hiddenSize, {
				epsilon: this.recipe.normEps,
				name,
				workgroupSize: this.workgroupSize
			} );

		}

		return new TSLRMSNorm( node, weight!, this.hiddenSize, {
			epsilon: this.recipe.normEps,
			offsetWeight: this.recipe.norm === 'rms_offset',
			name,
			workgroupSize: this.workgroupSize
		} );

	}

	buildAttention( qkvNode: TslNode, block: DecoderBlock, name: string ) {

		const { weights, recipe } = this;

		return new TSLAttention( qkvNode, this.hiddenSize, weights.headCount, this.maxTokens, {
			name: `${ name }Attention`,
			workgroupSize: this.workgroupSize,
			headDim: weights.headDim,
			kvHeadCount: weights.kvHeadCount,
			ropeTheta: block.ropeTheta !== undefined ? block.ropeTheta : recipe.ropeTheta,
			rotaryDim: recipe.rotaryDim || weights.headDim,
			slidingWindow: block.slidingWindow || 0,
			attnScale: recipe.attnScale,
			qNormWeight: block.qNormWeight,
			kNormWeight: block.kNormWeight,
			rmsEpsilon: recipe.normEps,
			offsetRMSNorm: recipe.norm === 'rms_offset',
			positionNode: this.positionNode.element( uint( 0 ) )
		} );

	}

	createPrefillCopyNode( name: string ) {

		const { hiddenSize, workgroupSize, embeddingNode, prefillCursorNode, prefillEmbeddingNode } = this;

		return Fn( () => {

			const dim = instanceIndex.toVar( 'dim' );

			If( dim.lessThan( uint( hiddenSize ) ), () => {

				const offset = prefillCursorNode.element( uint( 0 ) ).mul( uint( hiddenSize ) ).add( dim );
				embeddingNode.element( dim ).assign( prefillEmbeddingNode.element( offset ) );

			} );

		} )().compute( hiddenSize, [ workgroupSize ] ).setName( name );

	}

	createPrefillAdvanceNode( name: string ) {

		const { prefillCursorNode, positionNode } = this;

		return Fn( () => {

			prefillCursorNode.element( uint( 0 ) ).assign( prefillCursorNode.element( uint( 0 ) ).add( uint( 1 ) ) );
			positionNode.element( uint( 0 ) ).assign( positionNode.element( uint( 0 ) ).add( uint( 1 ) ) );

		} )().compute( 1, [ 1 ] ).setName( name );

	}

	buildLayer( block: DecoderBlock, index: number, residualNode: TslNode ): DecoderLayer {

		const { weights, recipe } = this;
		const name = `${ weights.architecture }Layer${ index }`;

		if ( recipe.residual === 'parallel' ) {

			const ln = this.buildNorm( residualNode, block.lnWeight, block.lnBias, `${ name }LN` );
			const qkv = new TSLLinear( ln.outputNode, block.attnQKVWeight, block.attnQKVBias ?? null, this.hiddenSize, weights.qSize + 2 * weights.kvSize, {
				name: `${ name }QKV`,
				workgroupSize: this.workgroupSize
			} );
			const attention = this.buildAttention( qkv.outputNode, block, name );
			const attnProj = new TSLLinear( attention.outputNode, block.attnProjWeight, block.attnProjBias ?? null, weights.qSize, this.hiddenSize, {
				name: `${ name }AttnProj`,
				workgroupSize: this.workgroupSize
			} );
			const mlp = new TSLMLP( ln.outputNode, block.mlpFCWeight!, block.mlpFCBias, block.mlpProjWeight!, block.mlpProjBias, this.hiddenSize, weights.innerSize, {
				name: `${ name }MLP`,
				workgroupSize: this.workgroupSize
			} );
			const addAttention = new TSLAdd( residualNode, attnProj.outputNode, this.hiddenSize, {
				name: `${ name }AddAttention`,
				workgroupSize: this.workgroupSize
			} );
			const addMLP = new TSLAdd( addAttention.outputNode, mlp.outputNode, this.hiddenSize, {
				name: `${ name }AddMLP`,
				workgroupSize: this.workgroupSize
			} );

			return {
				kind: 'parallel',
				ln, qkv, attention, attnProj, mlp, addAttention, addMLP,
				outputNode: addMLP.outputNode
			};

		}

		const ln1 = this.buildNorm( residualNode, block.ln1Weight, block.ln1Bias || null, `${ name }LN1` );
		const qkvOut = recipe.architecture === 'gpt2' ? this.hiddenSize * 3 : weights.qSize + 2 * weights.kvSize;
		const qkv = new TSLLinear( ln1.outputNode, block.attnQKVWeight, block.attnQKVBias || null, this.hiddenSize, qkvOut, {
			name: `${ name }QKV`,
			workgroupSize: this.workgroupSize
		} );
		const attention = this.buildAttention( qkv.outputNode, block, name );
		const attnIn = recipe.architecture === 'gpt2' ? this.hiddenSize : weights.qSize;
		const attnProj = new TSLLinear( attention.outputNode, block.attnProjWeight, block.attnProjBias || null, attnIn, this.hiddenSize, {
			name: `${ name }AttnProj`,
			workgroupSize: this.workgroupSize
		} );

		if ( recipe.postNorms ) {

			const postAttnNorm = this.buildNorm( attnProj.outputNode, block.postAttnNormWeight, null, `${ name }PostAttn` );
			const addAttention = new TSLAdd( residualNode, postAttnNorm.outputNode, this.hiddenSize, {
				name: `${ name }AddAttention`,
				workgroupSize: this.workgroupSize
			} );
			const preMlp = this.buildNorm( addAttention.outputNode, block.preMlpNormWeight, null, `${ name }PreMLP` );
			const mlp = new TSLGatedMLP( preMlp.outputNode, block.mlpGateWeight!, block.mlpUpWeight!, block.mlpDownWeight!, this.hiddenSize, weights.innerSize, {
				name: `${ name }MLP`,
				workgroupSize: this.workgroupSize,
				activation: recipe.mlpActivation
			} );
			const postMlpNorm = this.buildNorm( mlp.outputNode, block.postMlpNormWeight, null, `${ name }PostMLP` );
			const addMLP = new TSLAdd( addAttention.outputNode, postMlpNorm.outputNode, this.hiddenSize, {
				name: `${ name }AddMLP`,
				workgroupSize: this.workgroupSize
			} );

			return {
				kind: 'gemma',
				ln1, qkv, attention, attnProj, postAttnNorm, addAttention, preMlp, mlp, postMlpNorm, addMLP,
				outputNode: addMLP.outputNode
			};

		}

		const addAttention = new TSLAdd( residualNode, attnProj.outputNode, this.hiddenSize, {
			name: `${ name }AddAttention`,
			workgroupSize: this.workgroupSize
		} );
		const ln2 = this.buildNorm( addAttention.outputNode, block.ln2Weight, block.ln2Bias || null, `${ name }LN2` );
		const mlp = recipe.mlp === 'dense_gelu'
			? new TSLMLP( ln2.outputNode, block.mlpFCWeight!, block.mlpFCBias, block.mlpProjWeight!, block.mlpProjBias, this.hiddenSize, weights.innerSize, {
				name: `${ name }MLP`,
				workgroupSize: this.workgroupSize
			} )
			: new TSLGatedMLP( ln2.outputNode, block.mlpGateWeight!, block.mlpUpWeight!, block.mlpDownWeight!, this.hiddenSize, weights.innerSize, {
				name: `${ name }MLP`,
				workgroupSize: this.workgroupSize,
				activation: recipe.mlpActivation
			} );
		const addMLP = new TSLAdd( addAttention.outputNode, mlp.outputNode, this.hiddenSize, {
			name: `${ name }AddMLP`,
			workgroupSize: this.workgroupSize
		} );

		return {
			kind: 'sequential',
			ln1, qkv, attention, attnProj, addAttention, ln2, mlp, addMLP,
			outputNode: addMLP.outputNode
		};

	}

	buildFinalNorm( node: TslNode ) {

		return this.buildNorm( node, this.weights.outputNormWeight, this.weights.outputNormBias, `${ this.weights.architecture }FinalNorm` );

	}

	createComputeNodes( includeLogits = true ): ComputeNode[] {

		const nodes: ComputeNode[] = [];

		for ( const layer of this.layers ) {

			if ( layer.kind === 'parallel' ) {

				nodes.push( ...orderedComputeNodes(
					layer.ln, layer.qkv, layer.attention, layer.attnProj, layer.mlp, layer.addAttention, layer.addMLP
				) );

			} else if ( layer.kind === 'gemma' ) {

				nodes.push( ...orderedComputeNodes(
					layer.ln1, layer.qkv, layer.attention, layer.attnProj, layer.postAttnNorm,
					layer.addAttention, layer.preMlp, layer.mlp, layer.postMlpNorm, layer.addMLP
				) );

			} else {

				nodes.push( ...orderedComputeNodes(
					layer.ln1, layer.qkv, layer.attention, layer.attnProj,
					layer.addAttention, layer.ln2, layer.mlp, layer.addMLP
				) );

			}

		}

		if ( includeLogits ) {

			nodes.push( ...orderedComputeNodes( this.finalNorm, ...this.logits.map( ( logit ) => logit.layer ) ) );

		}

		return nodes;

	}

	sampleComputeNodes( candidateCount: number ) {

		return this.computeNodes.concat( this.logitSampler.computeNodesFor( candidateCount ) );

	}

	prefillChunkComputeNodes( count: number ) {

		const nodes: ComputeNode[] = [];

		for ( let i = 0; i < count; i ++ ) {

			nodes.push( this.prefillCopyNode, ...this.prefillComputeNodes, this.prefillAdvanceNode );

		}

		return nodes;

	}

	setPosition( position: number ) {

		this.positionBuffer[ 0 ] = position;
		this.positionAttribute.needsUpdate = true;

	}

	computeToken( renderer: Renderer, tokenId: number, position: number, computeLogits = true, sampleCandidateCount = 0 ) {

		this.weights.embedding( tokenId, position, this.embeddingBuffer );
		this.embeddingAttribute.needsUpdate = true;
		this.setPosition( position );

		for ( const layer of this.layers ) layer.attention.setPosition( position );

		renderer.compute( computeLogits
			? ( sampleCandidateCount > 0 ? this.sampleComputeNodes( sampleCandidateCount ) : this.computeNodes )
			: this.prefillComputeNodes );
		this.prepare( renderer );

	}

	async prefillTokens( renderer: Renderer, inputTokens: number[], start: number, end: number, onProgress?: ( n: number ) => void | Promise<void> ) {

		for ( let offset = start; offset < end; offset += this.prefillChunkSize ) {

			const count = Math.min( this.prefillChunkSize, end - offset );

			for ( let i = 0; i < count; i ++ ) {

				this.weights.embedding(
					inputTokens[ offset + i ],
					offset + i,
					this.prefillEmbeddingBuffer.subarray( i * this.hiddenSize, ( i + 1 ) * this.hiddenSize )
				);

			}

			this.prefillEmbeddingAttribute.needsUpdate = true;
			this.prefillCursorBuffer[ 0 ] = 0;
			this.prefillCursorAttribute.needsUpdate = true;
			this.setPosition( offset );
			renderer.compute( this.prefillChunkComputeNodes( count ) );
			this.prepare( renderer );
			if ( onProgress ) await onProgress( offset + count );

		}

	}

	async readLogits( renderer: Renderer ) {

		const logits = await readChunkedLogits( renderer, this.logits, this.weights.vocabSize );
		return logitSoftcap( logits, this.recipe.finalLogitSoftcap );

	}

	async sampleToken( renderer: Renderer, candidateCount: number, options: SampleOptions ) {

		return this.logitSampler.sampleToken( renderer, candidateCount, options );

	}

	resetCache() {

		this._cacheTokens = [];
		this._cacheLogits = null;

		for ( const layer of this.layers ) layer.attention.reset();

	}

	collectStaticWeightAttributes() {

		const attributes: StorageBufferAttribute[] = [];

		for ( const layer of this.layers ) {

			if ( layer.ln ) collectNormWeights( layer.ln, attributes );
			if ( layer.ln1 ) collectNormWeights( layer.ln1, attributes );
			if ( layer.ln2 ) collectNormWeights( layer.ln2, attributes );
			if ( layer.postAttnNorm ) collectNormWeights( layer.postAttnNorm, attributes );
			if ( layer.preMlp ) collectNormWeights( layer.preMlp, attributes );
			if ( layer.postMlpNorm ) collectNormWeights( layer.postMlpNorm, attributes );
			collectLinearWeights( layer.qkv, attributes );
			collectAttentionWeights( layer.attention, attributes );
			collectLinearWeights( layer.attnProj, attributes );
			collectMlpWeights( layer.mlp, attributes );

		}

		collectNormWeights( this.finalNorm, attributes );
		collectLogitWeights( this.logits, attributes );
		return attributes;

	}

	prepare( renderer: Renderer ) {

		if ( this._releasedCPUWeights ) return;

		if ( uploadAndReleaseStaticWeights( renderer, this.computeNodes, this.collectStaticWeightAttributes() ) === false ) return;

		this.weights.releaseCheckpointTensors();
		this.weights.releaseUnpackedWeightArrays();
		this._releasedCPUWeights = true;

	}

	async generate( renderer: Renderer, prompt: string, options: GenerateOptions = {} ): Promise<GenerationResult> {

		this.prepare( renderer );
		return generateAsync( this, prompt, options, {
			rewindable: true,
			resetCache: () => this.resetCache(),
			computeToken: ( tokenId, position, computeLogits, sampleCandidateCount ) => this.computeToken( renderer, tokenId, position, computeLogits, sampleCandidateCount ),
			prefillTokens: ( inputTokens, start, end, onProgress ) => this.prefillTokens( renderer, inputTokens, start, end, onProgress ),
			readLogits: () => this.readLogits( renderer ),
			sampleToken: ( candidateCount, sampleOptions ) => this.sampleToken( renderer, candidateCount, sampleOptions ),
			maxGpuCandidateCount: this.logitSampler.candidateCount
		} );

	}

}

export { DecoderTSLRunner };
