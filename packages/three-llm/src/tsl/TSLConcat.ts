import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, If, instanceIndex, storage, uint } from 'three/tsl';

import type { KernelOptions, Renderer, TslNode } from '../types.js';

export interface ConcatPart {
	node: TslNode;
	size: number;
}

/**
 * Concatenate several 1D storage vectors into one buffer.
 *
 */
class TSLConcat {

	parts: ConcatPart[];
	size: number;
	outputAttribute: StorageBufferAttribute;
	outputNode: TslNode;
	computeNode: TslNode;

	constructor( parts: ConcatPart[], options: KernelOptions = {} ) {

		this.parts = parts;
		this.size = parts.reduce( ( sum, part ) => sum + part.size, 0 );
		this.outputAttribute = new StorageBufferAttribute( new Float32Array( this.size ), 1 );
		this.outputNode = storage( this.outputAttribute, 'float', this.size ).setName( options.name ? `${ options.name }Output` : 'LLMConcatOutput' );
		this.computeNode = this.createComputeNode( options.name || 'LLMConcat', options.workgroupSize || 64 );

	}

	createComputeNode( name: string, workgroupSize: number ) {

		const { parts, outputNode, size } = this;
		const ranges: Array<{ node: TslNode; start: number; end: number }> = [];
		let offset = 0;

		for ( let i = 0; i < parts.length; i ++ ) {

			ranges.push( { node: parts[ i ].node, start: offset, end: offset + parts[ i ].size } );
			offset += parts[ i ].size;

		}

		return Fn( () => {

			const index = instanceIndex.toVar( 'index' );

			If( index.lessThan( uint( size ) ), () => {

				for ( let i = 0; i < ranges.length; i ++ ) {

					const range = ranges[ i ];
					If( index.greaterThanEqual( uint( range.start ) ).and( index.lessThan( uint( range.end ) ) ), () => {

						outputNode.element( index ).assign( range.node.element( index.sub( uint( range.start ) ) ) );

					} );

				}

			} );

		} )().compute( size, [ workgroupSize ] ).setName( name );

	}

	compute( renderer: Renderer ) {

		renderer.compute( this.computeNode );
		return this.outputNode;

	}

}

export { TSLConcat };
