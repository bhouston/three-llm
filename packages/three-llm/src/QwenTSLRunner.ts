import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, If, instanceIndex, storage, uint } from 'three/tsl';

import { generateAsync } from './LLMGenerate.js';
import { TSLAdd } from './TSLAdd.js';
import { TSLAttention } from './TSLAttention.js';
import { TSLConcat } from './TSLConcat.js';
import { orderedComputeNodes } from './TSLCompute.js';
import { TSLGatedDeltaNet } from './TSLGatedDeltaNet.js';
import { TSLGatedMLP } from './TSLGatedMLP.js';
import { TSLLinear } from './TSLLinear.js';
import { createChunkedLogitLayers, createLogitSampler, readChunkedLogits, TSLLogitSampler } from './TSLLogits.js';
import type { LogitChunk } from './TSLLogits.js';
import { TSLRMSNorm } from './TSLRMSNorm.js';
import { TSLSplitHeadGate } from './TSLSplitHeadGate.js';
import { QwenWeights } from './QwenWeights.js';
import type {
	ComputeNode, GenerateOptions, GenerationResult, LoaderOptions, Renderer, RunnerOptions, SampleOptions, TslNode
} from './types.js';

interface FullAttentionMixer {
	qGate: TSLLinear;
	split: TSLSplitHeadGate;
	kv: TSLLinear;
	packed: TSLConcat;
	attention: TSLAttention;
	attnProj: TSLLinear;
	outputNode: TslNode;
	computeNodes: ComputeNode[];
	compute: ( renderer: Renderer, position: number ) => void;
}

type QwenMixer = TSLGatedDeltaNet | FullAttentionMixer;

interface QwenLayer {
	ln1: TSLRMSNorm;
	mixer: QwenMixer;
	addAttention: TSLAdd;
	ln2: TSLRMSNorm;
	mlp: TSLGatedMLP;
	addMLP: TSLAdd;
	layerType?: string;
}

/**
 * Qwen3.5 text generation runner backed by TSL compute kernels.
 *
 */
class QwenTSLRunner {

	weights: QwenWeights;
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
	layers: QwenLayer[];
	finalNorm: TSLRMSNorm;
	logits: LogitChunk[];
	logitSampler: TSLLogitSampler;
	computeNodes: ComputeNode[];
	prefillComputeNodes: ComputeNode[];
	_cacheTokens?: number[];
	_cacheLogits?: Float32Array | null;

	constructor( weights: QwenWeights, options: RunnerOptions = {} ) {

		this.weights = weights;
		this.maxTokens = Math.min( options.maxTokens || weights.contextLimit(), weights.contextLimit() );
		this.workgroupSize = options.workgroupSize || 64;
		this.logitChunkSize = options.logitChunkSize || 8192;
		this.prefillChunkSize = options.prefillChunkSize || 32;
		this.hiddenSize = weights.hiddenSize;
		this.embeddingBuffer = new Float32Array( this.hiddenSize );
		this.embeddingAttribute = new StorageBufferAttribute( this.embeddingBuffer, 1 );
		this.embeddingNode = storage( this.embeddingAttribute, 'float', this.hiddenSize ).setName( 'QwenEmbedding' );
		this.positionBuffer = new Uint32Array( 1 );
		this.positionAttribute = new StorageBufferAttribute( this.positionBuffer, 1 );
		this.positionNode = storage( this.positionAttribute, 'uint', 1 ).setName( 'QwenPosition' );
		this.prefillCursorBuffer = new Uint32Array( 1 );
		this.prefillCursorAttribute = new StorageBufferAttribute( this.prefillCursorBuffer, 1 );
		this.prefillCursorNode = storage( this.prefillCursorAttribute, 'uint', 1 ).setName( 'QwenPrefillCursor' );
		this.prefillEmbeddingBuffer = new Float32Array( this.prefillChunkSize * this.hiddenSize );
		this.prefillEmbeddingAttribute = new StorageBufferAttribute( this.prefillEmbeddingBuffer, 1 );
		this.prefillEmbeddingNode = storage( this.prefillEmbeddingAttribute, 'float', this.prefillEmbeddingBuffer.length ).setName( 'QwenPrefillEmbeddings' );
		this.prefillCopyNode = this.createPrefillCopyNode( 'QwenPrefillCopy' );
		this.prefillAdvanceNode = this.createPrefillAdvanceNode( 'QwenPrefillAdvance' );
		this.layers = [];

		let currentNode = this.embeddingNode;

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			const block = weights.block( i );
			const name = `QwenLayer${ i }`;
			const ln1 = new TSLRMSNorm( currentNode, block.ln1Weight!, this.hiddenSize, {
				epsilon: weights.rmsNormEps,
				offsetWeight: true,
				name: `${ name }LN1`,
				workgroupSize: this.workgroupSize
			} );

			let mixer: QwenMixer;

			if ( block.layerType === 'linear_attention' ) {

				mixer = new TSLGatedDeltaNet( ln1.outputNode, block.delta!, {
					name: `${ name }Delta`,
					hiddenSize: this.hiddenSize,
					numKHeads: weights.linearKeyHeads,
					numVHeads: weights.linearValueHeads,
					keyDim: weights.linearKeyDim,
					valueDim: weights.linearValueDim,
					kernelSize: weights.linearConvKernel,
					epsilon: weights.rmsNormEps,
					workgroupSize: this.workgroupSize
				} );

			} else {

				const qGate = new TSLLinear( ln1.outputNode, block.qGateWeight!, null, this.hiddenSize, weights.qSize * 2, {
					name: `${ name }QGate`,
					workgroupSize: this.workgroupSize
				} );
				const split = new TSLSplitHeadGate( qGate.outputNode, weights.headCount, weights.headDim, {
					name: `${ name }Split`,
					workgroupSize: this.workgroupSize
				} );
				const kv = new TSLLinear( ln1.outputNode, block.attnQKVWeight, null, this.hiddenSize, 2 * weights.kvSize, {
					name: `${ name }KV`,
					workgroupSize: this.workgroupSize
				} );
				const packed = new TSLConcat( [
					{ node: split.queryNode, size: weights.qSize },
					{ node: kv.outputNode, size: 2 * weights.kvSize }
				], {
					name: `${ name }Pack`,
					workgroupSize: this.workgroupSize
				} );
				const attention = new TSLAttention( packed.outputNode, this.hiddenSize, weights.headCount, this.maxTokens, {
					name: `${ name }Attention`,
					workgroupSize: this.workgroupSize,
					headDim: weights.headDim,
					kvHeadCount: weights.kvHeadCount,
					ropeTheta: weights.ropeTheta,
					rotaryDim: weights.rotaryDim,
					attnScale: weights.attnScale,
					qNormWeight: block.qNormWeight,
					kNormWeight: block.kNormWeight,
					rmsEpsilon: weights.rmsNormEps,
					offsetRMSNorm: true,
					gateNode: split.gateNode,
					positionNode: this.positionNode.element( uint( 0 ) )
				} );
				const attnProj = new TSLLinear( attention.outputNode, block.attnProjWeight, null, weights.qSize, this.hiddenSize, {
					name: `${ name }AttnProj`,
					workgroupSize: this.workgroupSize
				} );
				mixer = {
					qGate,
					split,
					kv,
					packed,
					attention,
					attnProj,
					outputNode: attnProj.outputNode,
					computeNodes: orderedComputeNodes( qGate, split, kv, packed, attention, attnProj ),
					compute: ( renderer: Renderer, position: number ) => {

						qGate.compute( renderer );
						split.compute( renderer );
						kv.compute( renderer );
						packed.compute( renderer );
						attention.compute( renderer, position );
						attnProj.compute( renderer );

					}
				};

			}

			const addAttention = new TSLAdd( currentNode, mixer.outputNode, this.hiddenSize, {
				name: `${ name }AddAttention`,
				workgroupSize: this.workgroupSize
			} );
			const ln2 = new TSLRMSNorm( addAttention.outputNode, block.ln2Weight!, this.hiddenSize, {
				epsilon: weights.rmsNormEps,
				offsetWeight: true,
				name: `${ name }LN2`,
				workgroupSize: this.workgroupSize
			} );
			const mlp = new TSLGatedMLP( ln2.outputNode, block.mlpGateWeight!, block.mlpUpWeight!, block.mlpDownWeight!, this.hiddenSize, weights.innerSize, {
				name: `${ name }MLP`,
				workgroupSize: this.workgroupSize,
				activation: weights.mlpActivation
			} );
			const addMLP = new TSLAdd( addAttention.outputNode, mlp.outputNode, this.hiddenSize, {
				name: `${ name }AddMLP`,
				workgroupSize: this.workgroupSize
			} );

			this.layers.push( { ln1, mixer, addAttention, ln2, mlp, addMLP, layerType: block.layerType } );
			currentNode = addMLP.outputNode;

		}

		this.finalNorm = new TSLRMSNorm( currentNode, weights.outputNormWeight!, this.hiddenSize, {
			epsilon: weights.rmsNormEps,
			offsetWeight: true,
			name: 'QwenFinalNorm',
			workgroupSize: this.workgroupSize
		} );
		this.logits = createChunkedLogitLayers( this.finalNorm.outputNode, weights, this.logitChunkSize, 'QwenLogits' );
		this.logitSampler = createLogitSampler( this.logits, {
			candidateCount: options.logitCandidateCount || 8,
			name: 'QwenLogits'
		} );
		this.computeNodes = [];

		for ( const layer of this.layers ) {

			this.computeNodes.push( ...orderedComputeNodes(
				layer.ln1, layer.mixer, layer.addAttention, layer.ln2, layer.mlp, layer.addMLP
			) );

		}

		this.prefillComputeNodes = this.computeNodes.slice();
		this.computeNodes.push( ...orderedComputeNodes( this.finalNorm, ...this.logits.map( ( logit ) => logit.layer ) ) );

	}

	static async fromURL( baseURL: string, options: LoaderOptions & RunnerOptions = {} ) {

		return new this( await QwenWeights.fromURL( baseURL, options ), options );

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

		for ( const layer of this.layers ) {

			if ( layer.layerType !== 'linear_attention' ) ( layer.mixer as FullAttentionMixer ).attention.setPosition( position );

		}

		renderer.compute( computeLogits
			? ( sampleCandidateCount > 0 ? this.sampleComputeNodes( sampleCandidateCount ) : this.computeNodes )
			: this.prefillComputeNodes );

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
			if ( onProgress ) await onProgress( offset + count );

		}

	}

	async readLogits( renderer: Renderer ) {

		return readChunkedLogits( renderer, this.logits, this.weights.vocabSize );

	}

	async sampleToken( renderer: Renderer, candidateCount: number, options: SampleOptions ) {

		return this.logitSampler.sampleToken( renderer, candidateCount, options );

	}

	resetCaches() {

		for ( const layer of this.layers ) {

			if ( layer.layerType === 'linear_attention' ) ( layer.mixer as TSLGatedDeltaNet ).reset();
			else ( layer.mixer as FullAttentionMixer ).attention.reset();

		}

	}

	resetCache() {

		this._cacheTokens = [];
		this._cacheLogits = null;
		this.resetCaches();

	}

	async generate( renderer: Renderer, prompt: string, options: GenerateOptions = {} ): Promise<GenerationResult> {

		return generateAsync( this, prompt, options, {
			rewindable: false,
			resetCache: () => this.resetCache(),
			computeToken: ( tokenId, position, computeLogits, sampleCandidateCount ) => this.computeToken( renderer, tokenId, position, computeLogits, sampleCandidateCount ),
			prefillTokens: ( inputTokens, start, end, onProgress ) => this.prefillTokens( renderer, inputTokens, start, end, onProgress ),
			readLogits: () => this.readLogits( renderer ),
			sampleToken: ( candidateCount, sampleOptions ) => this.sampleToken( renderer, candidateCount, sampleOptions ),
			maxGpuCandidateCount: this.logitSampler.candidateCount
		} );

	}

}

export { QwenTSLRunner };
