import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, exp, float, instanceIndex, storage } from 'three/tsl';

import type { KernelOptions, Renderer, TslNode } from '../types.js';

/**
 * Element-wise `silu(gate) * up` used by SwiGLU MLPs.
 *
 */
class TSLSiLUMul {

	outputAttribute: StorageBufferAttribute;
	outputNode: TslNode;
	computeNode: TslNode;

	constructor( gateNode: TslNode, upNode: TslNode, size: number, options: KernelOptions = {} ) {

		this.outputAttribute = new StorageBufferAttribute( new Float32Array( size ), 1 );
		this.outputNode = storage( this.outputAttribute, 'float', size ).setName( options.name ? `${ options.name }Output` : 'LLMSiLUMulOutput' );

		this.computeNode = Fn( () => {

			const x = gateNode.element( instanceIndex );
			const silu = x.div( float( 1 ).add( exp( x.negate() ) ) );

			this.outputNode.element( instanceIndex ).assign( silu.mul( upNode.element( instanceIndex ) ) );

		} )().compute( size, [ options.workgroupSize || 64 ] ).setName( options.name || 'LLMSiLUMul' );

	}

	compute( renderer: Renderer ) {

		renderer.compute( this.computeNode );

		return this.outputNode;

	}

}

export { TSLSiLUMul };
