import type { ComputeNode, ComputeOp } from '../types.js';

function orderedComputeNodes( ...operations: Array<ComputeOp | null | undefined> ): ComputeNode[] {

	const nodes: ComputeNode[] = [];

	for ( const operation of operations ) {

		if ( operation === null || operation === undefined ) continue;

		if ( Array.isArray( operation.computeNodes ) ) {

			nodes.push( ...operation.computeNodes );

		} else if ( operation.computeNode ) {

			nodes.push( operation.computeNode );

		} else {

			throw new Error( 'TSLCompute: Operation does not expose compute nodes.' );

		}

	}

	return nodes;

}

export { orderedComputeNodes };
