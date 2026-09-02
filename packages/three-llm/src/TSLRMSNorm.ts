import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, Loop, float, instanceIndex, inversesqrt, storage, uint } from 'three/tsl';

import type { KernelOptions, Renderer, TslNode } from './types.js';

interface RMSNormOptions extends KernelOptions {
	epsilon?: number;
	offsetWeight?: boolean;
}

/**
 * RMS normalization for a single hidden vector.
 *
 * Llama scales by `weight`. Gemma scales by `1 + weight`.
 *
 */
class TSLRMSNorm {

	inputNode: TslNode;
	hiddenSize: number;
	epsilon: number;
	offsetWeight: boolean;
	workgroupSize: number;
	weightAttribute: StorageBufferAttribute;
	outputAttribute: StorageBufferAttribute;
	weightNode: TslNode;
	outputNode: TslNode;
	computeNode: TslNode;

	constructor( inputNode: TslNode, weightArray: Float32Array, hiddenSize: number, options: RMSNormOptions = {} ) {

		this.inputNode = inputNode;
		this.hiddenSize = hiddenSize;
		this.epsilon = options.epsilon || 1e-5;
		this.offsetWeight = options.offsetWeight === true;
		this.workgroupSize = options.workgroupSize || 64;

		this.weightAttribute = new StorageBufferAttribute( weightArray, 1 );
		this.outputAttribute = new StorageBufferAttribute( new Float32Array( hiddenSize ), 1 );

		this.weightNode = storage( this.weightAttribute, 'float', hiddenSize ).toReadOnly().setName( options.name ? `${ options.name }Weight` : 'LLMRMSNormWeight' );
		this.outputNode = storage( this.outputAttribute, 'float', hiddenSize ).setName( options.name ? `${ options.name }Output` : 'LLMRMSNormOutput' );

		this.computeNode = this.createComputeNode( options.name || 'LLMRMSNorm' );

	}

	createComputeNode( name: string ) {

		const { inputNode, weightNode, outputNode, hiddenSize, epsilon, offsetWeight, workgroupSize } = this;

		return Fn( () => {

			const index = instanceIndex.toVar( 'index' );
			const sumSquares = float( 0 ).toVar( 'sumSquares' );

			Loop( { start: uint( 0 ), end: uint( hiddenSize ), type: 'uint', condition: '<' }, ( { i }: { i: TslNode } ) => {

				const value = inputNode.element( i );
				sumSquares.addAssign( value.mul( value ) );

			} );

			const invRms = inversesqrt( sumSquares.div( float( hiddenSize ) ).add( epsilon ) );
			const scale = offsetWeight ? weightNode.element( index ).add( 1 ) : weightNode.element( index );

			outputNode.element( index ).assign( inputNode.element( index ).mul( invRms ).mul( scale ) );

		} )().compute( hiddenSize, [ workgroupSize ] ).setName( name );

	}

	compute( renderer: Renderer ) {

		renderer.compute( this.computeNode );

		return this.outputNode;

	}

}

export { TSLRMSNorm };
