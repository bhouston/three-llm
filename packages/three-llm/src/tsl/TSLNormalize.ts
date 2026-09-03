import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, Loop, float, instanceIndex, inversesqrt, storage, uint } from 'three/tsl';

import type { KernelOptions, Renderer, TslNode } from '../types.js';

interface NormalizeOptions extends KernelOptions {
	epsilon?: number;
}

/**
 * Layer normalization for a single hidden vector.
 *
 */
class TSLNormalize {

	inputNode: TslNode;
	hiddenSize: number;
	epsilon: number;
	workgroupSize: number;
	weightAttribute: StorageBufferAttribute;
	biasAttribute: StorageBufferAttribute;
	outputAttribute: StorageBufferAttribute;
	weightNode: TslNode;
	biasNode: TslNode;
	outputNode: TslNode;
	computeNode: TslNode;

	constructor( inputNode: TslNode, weightArray: Float32Array, biasArray: Float32Array | null, hiddenSize: number, options: NormalizeOptions = {} ) {

		this.inputNode = inputNode;
		this.hiddenSize = hiddenSize;
		this.epsilon = options.epsilon || 1e-5;
		this.workgroupSize = options.workgroupSize || 64;

		this.weightAttribute = new StorageBufferAttribute( weightArray, 1 );
		this.biasAttribute = new StorageBufferAttribute( biasArray as Float32Array, 1 );
		this.outputAttribute = new StorageBufferAttribute( new Float32Array( hiddenSize ), 1 );

		this.weightNode = storage( this.weightAttribute, 'float', hiddenSize ).toReadOnly().setName( options.name ? `${ options.name }Weight` : 'LLMLayerNormWeight' );
		this.biasNode = storage( this.biasAttribute, 'float', hiddenSize ).toReadOnly().setName( options.name ? `${ options.name }Bias` : 'LLMLayerNormBias' );
		this.outputNode = storage( this.outputAttribute, 'float', hiddenSize ).setName( options.name ? `${ options.name }Output` : 'LLMLayerNormOutput' );

		this.computeNode = this.createComputeNode( options.name || 'LLMLayerNorm' );

	}

	createComputeNode( name: string ) {

		const { inputNode, weightNode, biasNode, outputNode, hiddenSize, epsilon, workgroupSize } = this;

		return Fn( () => {

			const index = instanceIndex.toVar( 'index' );
			const mean = float( 0 ).toVar( 'mean' );

			Loop( { start: uint( 0 ), end: uint( hiddenSize ), type: 'uint', condition: '<' }, ( { i }: { i: TslNode } ) => {

				mean.addAssign( inputNode.element( i ) );

			} );

			mean.divAssign( float( hiddenSize ) );

			const variance = float( 0 ).toVar( 'variance' );

			Loop( { start: uint( 0 ), end: uint( hiddenSize ), type: 'uint', condition: '<' }, ( { i }: { i: TslNode } ) => {

				const delta = inputNode.element( i ).sub( mean );
				variance.addAssign( delta.mul( delta ) );

			} );

			variance.divAssign( float( hiddenSize ) );

			const value = inputNode.element( index ).sub( mean )
				.mul( inversesqrt( variance.add( epsilon ) ) )
				.mul( weightNode.element( index ) )
				.add( biasNode.element( index ) );

			outputNode.element( index ).assign( value );

		} )().compute( hiddenSize, [ workgroupSize ] ).setName( name );

	}

	compute( renderer: Renderer ) {

		renderer.compute( this.computeNode );

		return this.outputNode;

	}

}

export { TSLNormalize };
