import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, instanceIndex, storage } from 'three/tsl';

import type { KernelOptions, Renderer, TslNode } from '../types.js';

/**
 * Element-wise residual add for hidden vectors.
 *
 */
class TSLAdd {

	outputAttribute: StorageBufferAttribute;
	outputNode: TslNode;
	computeNode: TslNode;

	constructor( aNode: TslNode, bNode: TslNode, size: number, options: KernelOptions = {} ) {

		this.outputAttribute = new StorageBufferAttribute( new Float32Array( size ), 1 );
		this.outputNode = storage( this.outputAttribute, 'float', size ).setName( options.name ? `${ options.name }Output` : 'LLMAddOutput' );

		this.computeNode = Fn( () => {

			this.outputNode.element( instanceIndex ).assign( aNode.element( instanceIndex ).add( bNode.element( instanceIndex ) ) );

		} )().compute( size, [ options.workgroupSize || 64 ] ).setName( options.name || 'LLMAdd' );

	}

	compute( renderer: Renderer ) {

		renderer.compute( this.computeNode );

		return this.outputNode;

	}

}

export { TSLAdd };
