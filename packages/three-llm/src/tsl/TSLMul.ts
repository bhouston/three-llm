import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, instanceIndex, storage } from 'three/tsl';

import type { KernelOptions, Renderer, TslNode } from '../types.js';

/**
 * Element-wise multiply for gated MLPs.
 *
 */
class TSLMul {

	outputAttribute: StorageBufferAttribute;
	outputNode: TslNode;
	computeNode: TslNode;

	constructor( aNode: TslNode, bNode: TslNode, size: number, options: KernelOptions = {} ) {

		this.outputAttribute = new StorageBufferAttribute( new Float32Array( size ), 1 );
		this.outputNode = storage( this.outputAttribute, 'float', size ).setName( options.name ? `${ options.name }Output` : 'LLMMulOutput' );

		this.computeNode = Fn( () => {

			this.outputNode.element( instanceIndex ).assign( aNode.element( instanceIndex ).mul( bNode.element( instanceIndex ) ) );

		} )().compute( size, [ options.workgroupSize || 64 ] ).setName( options.name || 'LLMMul' );

	}

	compute( renderer: Renderer ) {

		renderer.compute( this.computeNode );

		return this.outputNode;

	}

}

export { TSLMul };
