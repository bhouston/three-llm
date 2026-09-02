import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, If, instanceIndex, storage, uint } from 'three/tsl';

import type { KernelOptions, Renderer, TslNode } from './types.js';

/**
 * Split packed `[q_h, gate_h]` heads into separate query and gate vectors.
 *
 */
class TSLSplitHeadGate {

	qSize: number;
	queryAttribute: StorageBufferAttribute;
	gateAttribute: StorageBufferAttribute;
	queryNode: TslNode;
	gateNode: TslNode;
	computeNode: TslNode;

	constructor( packedNode: TslNode, headCount: number, headDim: number, options: KernelOptions = {} ) {

		this.qSize = headCount * headDim;
		this.queryAttribute = new StorageBufferAttribute( new Float32Array( this.qSize ), 1 );
		this.gateAttribute = new StorageBufferAttribute( new Float32Array( this.qSize ), 1 );
		this.queryNode = storage( this.queryAttribute, 'float', this.qSize ).setName( options.name ? `${ options.name }Query` : 'LLMSplitQuery' );
		this.gateNode = storage( this.gateAttribute, 'float', this.qSize ).setName( options.name ? `${ options.name }Gate` : 'LLMSplitGate' );

		const packedWidth = headDim * 2;

		this.computeNode = Fn( () => {

			const index = instanceIndex.toVar( 'index' );

			If( index.lessThan( uint( this.qSize ) ), () => {

				const head = index.div( uint( headDim ) );
				const local = index.mod( uint( headDim ) );
				const packedOffset = head.mul( uint( packedWidth ) );

				this.queryNode.element( index ).assign( packedNode.element( packedOffset.add( local ) ) );
				this.gateNode.element( index ).assign( packedNode.element( packedOffset.add( uint( headDim ) ).add( local ) ) );

			} );

		} )().compute( this.qSize, [ options.workgroupSize || 64 ] ).setName( options.name || 'LLMSplitHeadGate' );

	}

	compute( renderer: Renderer ) {

		renderer.compute( this.computeNode );
		return this.queryNode;

	}

}

export { TSLSplitHeadGate };
