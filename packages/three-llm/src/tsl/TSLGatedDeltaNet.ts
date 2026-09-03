import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, If, Loop, exp, float, instanceIndex, inversesqrt, log, storage, uint } from 'three/tsl';

import { TSLLinear } from './TSLLinear.js';
import type { KernelOptions, QwenDeltaWeights, Renderer, TslNode } from '../types.js';

interface GatedDeltaNetOptions extends KernelOptions {
	hiddenSize?: number;
	numKHeads?: number;
	numVHeads?: number;
	keyDim?: number;
	valueDim?: number;
	kernelSize?: number;
	epsilon?: number;
}

/**
 * One-token Gated DeltaNet decode: causal conv, recurrent delta rule, gated RMSNorm.
 *
 */
class TSLGatedDeltaNet {

	hiddenSize: number;
	numKHeads: number;
	numVHeads: number;
	keyDim: number;
	valueDim: number;
	kernelSize: number;
	workgroupSize: number;
	epsilon: number;
	keySize: number;
	valueSize: number;
	convDim: number;
	stateSize: number;
	repeat: number;
	qkv: TSLLinear;
	zProj: TSLLinear;
	bProj: TSLLinear;
	aProj: TSLLinear;
	convStateAttribute: StorageBufferAttribute;
	convOutAttribute: StorageBufferAttribute;
	queryAttribute: StorageBufferAttribute;
	keyAttribute: StorageBufferAttribute;
	valueAttribute: StorageBufferAttribute;
	recurrentAttribute: StorageBufferAttribute;
	mixedAttribute: StorageBufferAttribute;
	outputAttribute: StorageBufferAttribute;
	convStateNode: TslNode;
	convOutNode: TslNode;
	queryNode: TslNode;
	keyNode: TslNode;
	valueNode: TslNode;
	recurrentNode: TslNode;
	mixedNode: TslNode;
	normWeightAttribute: StorageBufferAttribute;
	normWeightNode: TslNode;
	aLogAttribute: StorageBufferAttribute;
	dtBiasAttribute: StorageBufferAttribute;
	convWeightAttribute: StorageBufferAttribute;
	aLogNode: TslNode;
	dtBiasNode: TslNode;
	convWeightNode: TslNode;
	decayAttribute: StorageBufferAttribute;
	betaAttribute: StorageBufferAttribute;
	decayNode: TslNode;
	betaNode: TslNode;
	outProj: TSLLinear;
	outputNode: TslNode;
	convComputeNode: TslNode;
	prepareComputeNode: TslNode;
	normQKComputeNode: TslNode;
	decayComputeNode: TslNode;
	deltaComputeNode: TslNode;
	normComputeNode: TslNode;
	computeNodes: TslNode[];

	constructor( inputNode: TslNode, weights: QwenDeltaWeights, options: GatedDeltaNetOptions = {} ) {

		this.hiddenSize = options.hiddenSize as number;
		this.numKHeads = options.numKHeads as number;
		this.numVHeads = options.numVHeads as number;
		this.keyDim = options.keyDim as number;
		this.valueDim = options.valueDim as number;
		this.kernelSize = options.kernelSize || 4;
		this.workgroupSize = options.workgroupSize || 64;
		this.epsilon = options.epsilon || 1e-6;
		this.keySize = this.numKHeads * this.keyDim;
		this.valueSize = this.numVHeads * this.valueDim;
		this.convDim = this.keySize * 2 + this.valueSize;
		this.stateSize = this.numVHeads * this.keyDim * this.valueDim;
		this.repeat = this.numVHeads / this.numKHeads;

		this.qkv = new TSLLinear( inputNode, weights.qkvWeight, null, this.hiddenSize, this.convDim, {
			name: options.name ? `${ options.name }QKV` : 'LLMDeltaQKV',
			workgroupSize: this.workgroupSize
		} );
		this.zProj = new TSLLinear( inputNode, weights.zWeight, null, this.hiddenSize, this.valueSize, {
			name: options.name ? `${ options.name }Z` : 'LLMDeltaZ',
			workgroupSize: this.workgroupSize
		} );
		this.bProj = new TSLLinear( inputNode, weights.bWeight, null, this.hiddenSize, this.numVHeads, {
			name: options.name ? `${ options.name }B` : 'LLMDeltaB',
			workgroupSize: this.workgroupSize
		} );
		this.aProj = new TSLLinear( inputNode, weights.aWeight, null, this.hiddenSize, this.numVHeads, {
			name: options.name ? `${ options.name }A` : 'LLMDeltaA',
			workgroupSize: this.workgroupSize
		} );

		this.convStateAttribute = new StorageBufferAttribute( new Float32Array( this.convDim * this.kernelSize ), 1 );
		this.convOutAttribute = new StorageBufferAttribute( new Float32Array( this.convDim ), 1 );
		this.queryAttribute = new StorageBufferAttribute( new Float32Array( this.numVHeads * this.keyDim ), 1 );
		this.keyAttribute = new StorageBufferAttribute( new Float32Array( this.numVHeads * this.keyDim ), 1 );
		this.valueAttribute = new StorageBufferAttribute( new Float32Array( this.valueSize ), 1 );
		this.recurrentAttribute = new StorageBufferAttribute( new Float32Array( this.stateSize ), 1 );
		this.mixedAttribute = new StorageBufferAttribute( new Float32Array( this.valueSize ), 1 );
		this.outputAttribute = new StorageBufferAttribute( new Float32Array( this.hiddenSize ), 1 );

		this.convStateNode = storage( this.convStateAttribute, 'float', this.convDim * this.kernelSize ).setName( options.name ? `${ options.name }ConvState` : 'LLMDeltaConvState' );
		this.convOutNode = storage( this.convOutAttribute, 'float', this.convDim ).setName( options.name ? `${ options.name }ConvOut` : 'LLMDeltaConvOut' );
		this.queryNode = storage( this.queryAttribute, 'float', this.numVHeads * this.keyDim ).setName( options.name ? `${ options.name }Query` : 'LLMDeltaQuery' );
		this.keyNode = storage( this.keyAttribute, 'float', this.numVHeads * this.keyDim ).setName( options.name ? `${ options.name }Key` : 'LLMDeltaKey' );
		this.valueNode = storage( this.valueAttribute, 'float', this.valueSize ).setName( options.name ? `${ options.name }Value` : 'LLMDeltaValue' );
		this.recurrentNode = storage( this.recurrentAttribute, 'float', this.stateSize ).setName( options.name ? `${ options.name }State` : 'LLMDeltaState' );
		this.mixedNode = storage( this.mixedAttribute, 'float', this.valueSize ).setName( options.name ? `${ options.name }Mixed` : 'LLMDeltaMixed' );
		this.normWeightAttribute = new StorageBufferAttribute( weights.normWeight, 1 );
		this.normWeightNode = storage( this.normWeightAttribute, 'float', this.valueDim ).toReadOnly().setName( options.name ? `${ options.name }Norm` : 'LLMDeltaNorm' );
		this.aLogAttribute = new StorageBufferAttribute( weights.aLog, 1 );
		this.dtBiasAttribute = new StorageBufferAttribute( weights.dtBias, 1 );
		this.convWeightAttribute = new StorageBufferAttribute( weights.convWeight, 1 );
		this.aLogNode = storage( this.aLogAttribute, 'float', this.numVHeads ).toReadOnly().setName( options.name ? `${ options.name }ALog` : 'LLMDeltaALog' );
		this.dtBiasNode = storage( this.dtBiasAttribute, 'float', this.numVHeads ).toReadOnly().setName( options.name ? `${ options.name }DtBias` : 'LLMDeltaDtBias' );
		this.convWeightNode = storage( this.convWeightAttribute, 'float', this.convDim * this.kernelSize ).toReadOnly().setName( options.name ? `${ options.name }ConvW` : 'LLMDeltaConvW' );

		this.decayAttribute = new StorageBufferAttribute( new Float32Array( this.numVHeads ), 1 );
		this.betaAttribute = new StorageBufferAttribute( new Float32Array( this.numVHeads ), 1 );
		this.decayNode = storage( this.decayAttribute, 'float', this.numVHeads ).setName( options.name ? `${ options.name }Decay` : 'LLMDeltaDecay' );
		this.betaNode = storage( this.betaAttribute, 'float', this.numVHeads ).setName( options.name ? `${ options.name }Beta` : 'LLMDeltaBeta' );

		this.outProj = new TSLLinear( this.mixedNode, weights.outWeight, null, this.valueSize, this.hiddenSize, {
			name: options.name ? `${ options.name }Out` : 'LLMDeltaOut',
			workgroupSize: this.workgroupSize
		} );
		this.outputNode = this.outProj.outputNode;

		this.convComputeNode = this.createConvNode( options.name );
		this.prepareComputeNode = this.createPrepareNode( options.name );
		this.normQKComputeNode = this.createNormQKNode( options.name );
		this.decayComputeNode = this.createDecayNode( options.name );
		this.deltaComputeNode = this.createDeltaNode( options.name );
		this.normComputeNode = this.createNormNode( options.name );
		this.computeNodes = [
			this.qkv.computeNode,
			this.zProj.computeNode,
			this.bProj.computeNode,
			this.aProj.computeNode,
			this.convComputeNode,
			this.prepareComputeNode,
			this.normQKComputeNode,
			this.decayComputeNode,
			this.deltaComputeNode,
			this.normComputeNode,
			this.outProj.computeNode
		];

	}

	createConvNode( name?: string ) {

		const { convDim, kernelSize, workgroupSize, qkv, convStateNode, convOutNode, convWeightNode } = this;

		return Fn( () => {

			const channel = instanceIndex.toVar( 'channel' );

			If( channel.lessThan( uint( convDim ) ), () => {

				const stateOffset = channel.mul( uint( kernelSize ) );
				const weightOffset = channel.mul( uint( kernelSize ) );
				const input = qkv.outputNode.element( channel );
				const sum = float( 0 ).toVar( 'convSum' );

				Loop( { start: uint( 1 ), end: uint( kernelSize ), type: 'uint', condition: '<' }, ( { i }: { i: TslNode } ) => {

					sum.addAssign( convWeightNode.element( weightOffset.add( i.sub( uint( 1 ) ) ) ).mul( convStateNode.element( stateOffset.add( i ) ) ) );

				} );

				sum.addAssign( convWeightNode.element( weightOffset.add( uint( kernelSize - 1 ) ) ).mul( input ) );
				const silu = sum.div( float( 1 ).add( exp( sum.negate() ) ) );
				convOutNode.element( channel ).assign( silu );

				Loop( { start: uint( 0 ), end: uint( kernelSize - 1 ), type: 'uint', condition: '<' }, ( { i }: { i: TslNode } ) => {

					convStateNode.element( stateOffset.add( i ) ).assign( convStateNode.element( stateOffset.add( i.add( uint( 1 ) ) ) ) );

				} );

				convStateNode.element( stateOffset.add( uint( kernelSize - 1 ) ) ).assign( input );

			} );

		} )().compute( convDim, [ workgroupSize ] ).setName( name ? `${ name }Conv` : 'LLMDeltaConv' );

	}

	createPrepareNode( name?: string ) {

		const {
			keySize, valueSize, keyDim, numVHeads, repeat, convOutNode, queryNode, keyNode, valueNode, workgroupSize
		} = this;
		const prepared = numVHeads * keyDim;

		return Fn( () => {

			const index = instanceIndex.toVar( 'index' );

			If( index.lessThan( uint( prepared ) ), () => {

				const vHead = index.div( uint( keyDim ) );
				const local = index.mod( uint( keyDim ) );
				const kHead = vHead.div( uint( repeat ) );
				const source = kHead.mul( uint( keyDim ) ).add( local );
				const q = convOutNode.element( source );
				const k = convOutNode.element( uint( keySize ).add( source ) );
				queryNode.element( index ).assign( q );
				keyNode.element( index ).assign( k );

			} );

			If( index.lessThan( uint( valueSize ) ), () => {

				valueNode.element( index ).assign( convOutNode.element( uint( keySize * 2 ).add( index ) ) );

			} );

		} )().compute( Math.max( prepared, valueSize ), [ workgroupSize ] ).setName( name ? `${ name }Prepare` : 'LLMDeltaPrepare' );

	}

	createNormQKNode( name?: string ) {

		const { keyDim, numVHeads, queryNode, keyNode, workgroupSize } = this;

		return Fn( () => {

			const head = instanceIndex.toVar( 'head' );

			If( head.lessThan( uint( numVHeads ) ), () => {

				const offset = head.mul( uint( keyDim ) );
				const qSum = float( 0 ).toVar( 'qSum' );
				const kSum = float( 0 ).toVar( 'kSum' );

				Loop( { start: uint( 0 ), end: uint( keyDim ), type: 'uint', condition: '<' }, ( { i }: { i: TslNode } ) => {

					const qv = queryNode.element( offset.add( i ) );
					const kv = keyNode.element( offset.add( i ) );
					qSum.addAssign( qv.mul( qv ) );
					kSum.addAssign( kv.mul( kv ) );

				} );

				const qInv = inversesqrt( qSum.add( 1e-6 ) ).div( float( Math.sqrt( keyDim ) ) );
				const kInv = inversesqrt( kSum.add( 1e-6 ) );

				Loop( { start: uint( 0 ), end: uint( keyDim ), type: 'uint', condition: '<' }, ( { i }: { i: TslNode } ) => {

					queryNode.element( offset.add( i ) ).assign( queryNode.element( offset.add( i ) ).mul( qInv ) );
					keyNode.element( offset.add( i ) ).assign( keyNode.element( offset.add( i ) ).mul( kInv ) );

				} );

			} );

		} )().compute( numVHeads, [ workgroupSize ] ).setName( name ? `${ name }NormQK` : 'LLMDeltaNormQK' );

	}

	createDecayNode( name?: string ) {

		const { numVHeads, workgroupSize, aProj, bProj, aLogNode, dtBiasNode, decayNode, betaNode } = this;

		return Fn( () => {

			const head = instanceIndex.toVar( 'head' );

			If( head.lessThan( uint( numVHeads ) ), () => {

				const a = aProj.outputNode.element( head ).add( dtBiasNode.element( head ) );
				const softplus = a.greaterThan( float( 20 ) ).select( a, log( float( 1 ).add( exp( a ) ) ) );
				decayNode.element( head ).assign( exp( exp( aLogNode.element( head ) ).negate().mul( softplus ) ) );
				betaNode.element( head ).assign( float( 1 ).div( float( 1 ).add( exp( bProj.outputNode.element( head ).negate() ) ) ) );

			} );

		} )().compute( numVHeads, [ workgroupSize ] ).setName( name ? `${ name }Decay` : 'LLMDeltaDecay' );

	}

	createDeltaNode( name?: string ) {

		const {
			numVHeads, keyDim, valueDim, workgroupSize, queryNode, keyNode, valueNode, recurrentNode, mixedNode,
			decayNode, betaNode
		} = this;
		const count = numVHeads * valueDim;
		const chunk = 16;

		return Fn( () => {

			const index = instanceIndex.toVar( 'index' );

			If( index.lessThan( uint( count ) ), () => {

				const head = index.div( uint( valueDim ) );
				const v = index.mod( uint( valueDim ) );
				const decay = decayNode.element( head );
				const beta = betaNode.element( head );
				const qOff = head.mul( uint( keyDim ) );
				const stateOff = head.mul( uint( keyDim ) ).mul( uint( valueDim ) );
				const kvMem = float( 0 ).toVar( 'kvMem' );

				for ( let start = 0; start < keyDim; start += chunk ) {

					const end = Math.min( start + chunk, keyDim );

					( Loop as TslNode )( { start: uint( start ), end: uint( end ), type: 'uint', condition: '<', name: `kA${ start }` }, ( vars: Record<string, TslNode> ) => {

						const k = vars[ `kA${ start }` ];
						const sIndex = stateOff.add( k.mul( uint( valueDim ) ) ).add( v );
						recurrentNode.element( sIndex ).assign( recurrentNode.element( sIndex ).mul( decay ) );
						kvMem.addAssign( recurrentNode.element( sIndex ).mul( keyNode.element( qOff.add( k ) ) ) );

					} );

				}

				const delta = valueNode.element( head.mul( uint( valueDim ) ).add( v ) ).sub( kvMem ).mul( beta );
				const mixed = float( 0 ).toVar( 'mixed' );

				for ( let start = 0; start < keyDim; start += chunk ) {

					const end = Math.min( start + chunk, keyDim );

					( Loop as TslNode )( { start: uint( start ), end: uint( end ), type: 'uint', condition: '<', name: `kB${ start }` }, ( vars: Record<string, TslNode> ) => {

						const k = vars[ `kB${ start }` ];
						const sIndex = stateOff.add( k.mul( uint( valueDim ) ) ).add( v );
						recurrentNode.element( sIndex ).assign( recurrentNode.element( sIndex ).add( keyNode.element( qOff.add( k ) ).mul( delta ) ) );
						mixed.addAssign( recurrentNode.element( sIndex ).mul( queryNode.element( qOff.add( k ) ) ) );

					} );

				}

				mixedNode.element( index ).assign( mixed );

			} );

		} )().compute( count, [ workgroupSize ] ).setName( name ? `${ name }Delta` : 'LLMDeltaRule' );

	}

	createNormNode( name?: string ) {

		const { numVHeads, valueDim, workgroupSize, mixedNode, zProj, normWeightNode, epsilon } = this;

		return Fn( () => {

			const head = instanceIndex.toVar( 'head' );

			If( head.lessThan( uint( numVHeads ) ), () => {

				const offset = head.mul( uint( valueDim ) );
				const sumSquares = float( 0 ).toVar( 'sumSquares' );

				Loop( { start: uint( 0 ), end: uint( valueDim ), type: 'uint', condition: '<' }, ( { i }: { i: TslNode } ) => {

					const value = mixedNode.element( offset.add( i ) );
					sumSquares.addAssign( value.mul( value ) );

				} );

				const invRms = inversesqrt( sumSquares.div( float( valueDim ) ).add( epsilon ) );

				Loop( { start: uint( 0 ), end: uint( valueDim ), type: 'uint', condition: '<' }, ( { i }: { i: TslNode } ) => {

					const index = offset.add( i );
					const z = zProj.outputNode.element( index );
					const silu = z.div( float( 1 ).add( exp( z.negate() ) ) );
					mixedNode.element( index ).assign(
						mixedNode.element( index ).mul( invRms ).mul( normWeightNode.element( i ) ).mul( silu )
					);

				} );

			} );

		} )().compute( numVHeads, [ workgroupSize ] ).setName( name ? `${ name }Norm` : 'LLMDeltaGatedNorm' );

	}

	reset() {

		this.convStateAttribute.array.fill( 0 );
		this.convStateAttribute.needsUpdate = true;
		this.recurrentAttribute.array.fill( 0 );
		this.recurrentAttribute.needsUpdate = true;

	}

	compute( renderer: Renderer ) {

		this.qkv.compute( renderer );
		this.zProj.compute( renderer );
		this.bProj.compute( renderer );
		this.aProj.compute( renderer );
		renderer.compute( this.convComputeNode );
		renderer.compute( this.prepareComputeNode );
		renderer.compute( this.normQKComputeNode );
		renderer.compute( this.decayComputeNode );
		renderer.compute( this.deltaComputeNode );
		renderer.compute( this.normComputeNode );
		this.outProj.compute( renderer );
		return this.outputNode;

	}

}

export { TSLGatedDeltaNet };
