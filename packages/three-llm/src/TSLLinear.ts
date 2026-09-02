import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, If, Loop, instanceIndex, storage, uint } from 'three/tsl';

import type { KernelOptions, Renderer, TslNode } from './types.js';

/**
 * One-token dense layer implemented as a TSL compute pass.
 *
 * Weight layout is `[inputSize, outputSize]`, matching GPT-2 Conv1D tensors
 * stored by Hugging Face.
 *
 */
class TSLLinear {

	inputNode: TslNode;
	inputSize: number;
	outputSize: number;
	workgroupSize: number;
	weightAttribute: StorageBufferAttribute;
	biasAttribute: StorageBufferAttribute;
	outputAttribute: StorageBufferAttribute;
	weightNode: TslNode;
	biasNode: TslNode;
	outputNode: TslNode;
	computeNode: TslNode;

	constructor( inputNode: TslNode, weightArray: Float32Array, biasArray: Float32Array | null, inputSize: number, outputSize: number, options: KernelOptions = {} ) {

		this.inputNode = inputNode;
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.workgroupSize = options.workgroupSize || 64;

		this.weightAttribute = new StorageBufferAttribute( weightArray, 1 );
		this.biasAttribute = new StorageBufferAttribute( biasArray || new Float32Array( outputSize ), 1 );
		this.outputAttribute = new StorageBufferAttribute( new Float32Array( outputSize ), 1 );

		this.weightNode = storage( this.weightAttribute, 'float', weightArray.length ).toReadOnly().setName( options.name ? `${ options.name }Weight` : 'LLMLinearWeight' );
		this.biasNode = storage( this.biasAttribute, 'float', outputSize ).toReadOnly().setName( options.name ? `${ options.name }Bias` : 'LLMLinearBias' );
		this.outputNode = storage( this.outputAttribute, 'float', outputSize ).setName( options.name ? `${ options.name }Output` : 'LLMLinearOutput' );

		this.computeNode = this.createComputeNode( options.name || 'LLMLinear' );

	}

	createComputeNode( name: string ) {

		const { inputNode, weightNode, biasNode, outputNode, inputSize, outputSize, workgroupSize } = this;

		return Fn( () => {

			const outputIndex = instanceIndex.toVar( 'outputIndex' );

			If( outputIndex.lessThan( uint( outputSize ) ), () => {

				const sum = biasNode.element( outputIndex ).toVar( 'sum' );

				Loop( { start: uint( 0 ), end: uint( inputSize ), type: 'uint', condition: '<' }, ( { i }: { i: TslNode } ) => {

					const weightIndex = i.mul( uint( outputSize ) ).add( outputIndex );
					sum.addAssign( inputNode.element( i ).mul( weightNode.element( weightIndex ) ) );

				} );

				outputNode.element( outputIndex ).assign( sum );

			} );

		} )().compute( outputSize, [ workgroupSize ] ).setName( name );

	}

	compute( renderer: Renderer ) {

		renderer.compute( this.computeNode );

		return this.outputNode;

	}

}

export { TSLLinear };
