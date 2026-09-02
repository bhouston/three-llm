import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, float, instanceIndex, storage, tanh } from 'three/tsl';

import type { KernelOptions, Renderer, TslNode } from './types.js';

/**
 * GPT-2 `gelu_new` activation.
 *
 */
class TSLGELU {

	outputAttribute: StorageBufferAttribute;
	outputNode: TslNode;
	computeNode: TslNode;

	constructor( inputNode: TslNode, size: number, options: KernelOptions = {} ) {

		this.outputAttribute = new StorageBufferAttribute( new Float32Array( size ), 1 );
		this.outputNode = storage( this.outputAttribute, 'float', size ).setName( options.name ? `${ options.name }Output` : 'LLMGELUOutput' );

		this.computeNode = Fn( () => {

			const x = inputNode.element( instanceIndex );
			const cubic = x.mul( x ).mul( x ).mul( 0.044715 ).add( x );
			const inner = cubic.mul( Math.sqrt( 2 / Math.PI ) ).clamp( - 10, 10 );
			const value = x.mul( 0.5 ).mul( tanh( inner ).add( float( 1 ) ) );

			this.outputNode.element( instanceIndex ).assign( value );

		} )().compute( size, [ options.workgroupSize || 64 ] ).setName( options.name || 'LLMGELU' );

	}

	compute( renderer: Renderer ) {

		renderer.compute( this.computeNode );

		return this.outputNode;

	}

}

export { TSLGELU };
