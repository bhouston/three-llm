import { TSLGELU } from './TSLGELU.js';
import { TSLLinear } from './TSLLinear.js';
import type { KernelOptions, Renderer, TslNode } from './types.js';

/**
 * GPT-2 MLP block: dense -> gelu_new -> dense.
 *
 */
class TSLMLP {

	fc: TSLLinear;
	gelu: TSLGELU;
	proj: TSLLinear;
	outputNode: TslNode;
	computeNodes: TslNode[];

	constructor( inputNode: TslNode, fcWeight: Float32Array, fcBias: Float32Array | null | undefined, projWeight: Float32Array, projBias: Float32Array | null | undefined, hiddenSize: number, innerSize: number, options: KernelOptions = {} ) {

		this.fc = new TSLLinear( inputNode, fcWeight, fcBias ?? null, hiddenSize, innerSize, {
			name: options.name ? `${ options.name }FC` : 'LLMMLPFC',
			workgroupSize: options.workgroupSize
		} );

		this.gelu = new TSLGELU( this.fc.outputNode, innerSize, {
			name: options.name ? `${ options.name }GELU` : 'LLMMLPGELU',
			workgroupSize: options.workgroupSize
		} );

		this.proj = new TSLLinear( this.gelu.outputNode, projWeight, projBias ?? null, innerSize, hiddenSize, {
			name: options.name ? `${ options.name }Proj` : 'LLMMLPProj',
			workgroupSize: options.workgroupSize
		} );

		this.outputNode = this.proj.outputNode;
		this.computeNodes = [ this.fc.computeNode, this.gelu.computeNode, this.proj.computeNode ];

	}

	compute( renderer: Renderer ) {

		this.fc.compute( renderer );
		this.gelu.compute( renderer );
		this.proj.compute( renderer );

		return this.outputNode;

	}

}

export { TSLMLP };
