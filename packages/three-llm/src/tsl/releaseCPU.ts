import type { StorageBufferAttribute } from 'three/webgpu';

import type { ComputeNode, Renderer } from '../types.js';
import type { TSLAttention } from './TSLAttention.js';
import type { TSLGatedDeltaNet } from './TSLGatedDeltaNet.js';
import type { TSLGatedMLP } from './TSLGatedMLP.js';
import type { TSLLinear } from './TSLLinear.js';
import type { LogitChunk } from './TSLLogits.js';
import type { TSLMLP } from './TSLMLP.js';
import type { TSLNormalize } from './TSLNormalize.js';
import type { TSLRMSNorm } from './TSLRMSNorm.js';

type BindGroup = {
	bindings: Array<{
		isStorageBuffer?: boolean;
		attribute?: StorageBufferAttribute;
		_buffer?: unknown;
	}>;
};

type BindingsHost = {
	getForCompute: ( node: ComputeNode ) => BindGroup[];
};

function pushAttribute( list: StorageBufferAttribute[], attribute?: StorageBufferAttribute ): void {

	if ( attribute ) list.push( attribute );

}

function collectLinearWeights( linear: TSLLinear, list: StorageBufferAttribute[] ): void {

	pushAttribute( list, linear.weightAttribute );
	pushAttribute( list, linear.biasAttribute );

}

function collectNormWeights( norm: TSLRMSNorm | TSLNormalize, list: StorageBufferAttribute[] ): void {

	pushAttribute( list, norm.weightAttribute );
	if ( 'biasAttribute' in norm ) pushAttribute( list, ( norm as TSLNormalize ).biasAttribute );

}

function collectAttentionWeights( attention: TSLAttention, list: StorageBufferAttribute[] ): void {

	pushAttribute( list, attention.qNormAttribute );
	pushAttribute( list, attention.kNormAttribute );

}

function collectGatedDeltaWeights( delta: TSLGatedDeltaNet, list: StorageBufferAttribute[] ): void {

	collectLinearWeights( delta.qkv, list );
	collectLinearWeights( delta.zProj, list );
	collectLinearWeights( delta.bProj, list );
	collectLinearWeights( delta.aProj, list );
	collectLinearWeights( delta.outProj, list );
	pushAttribute( list, delta.normWeightAttribute );
	pushAttribute( list, delta.aLogAttribute );
	pushAttribute( list, delta.dtBiasAttribute );
	pushAttribute( list, delta.convWeightAttribute );

}

function collectMlpWeights( mlp: TSLMLP | TSLGatedMLP, list: StorageBufferAttribute[] ): void {

	if ( 'gate' in mlp ) {

		collectLinearWeights( mlp.gate, list );
		collectLinearWeights( mlp.up, list );
		collectLinearWeights( mlp.down, list );
		return;

	}

	collectLinearWeights( mlp.fc, list );
	collectLinearWeights( mlp.proj, list );

}

function collectLogitWeights( chunks: LogitChunk[], list: StorageBufferAttribute[] ): void {

	for ( const chunk of chunks ) collectLinearWeights( chunk.layer, list );

}

function emptyAttributeArray( attribute: StorageBufferAttribute ): void {

	const array = attribute.array;
	if ( array === undefined || array.length === 0 ) return;
	const ArrayType = array.constructor as new ( n: number ) => typeof array;
	attribute.array = new ArrayType( 0 );

}

function uploadAndReleaseStaticWeights( renderer: Renderer, computeNodes: ComputeNode[], attributes: StorageBufferAttribute[] ): boolean {

	const uniqueAttributes = [ ...new Set( attributes ) ];
	const uniqueNodes = [ ...new Set( computeNodes ) ];
	const bindings = ( renderer as { _bindings?: BindingsHost } )._bindings;

	if ( ! bindings || uniqueNodes.length === 0 ) return false;

	try {

		for ( const node of uniqueNodes ) bindings.getForCompute( node );

	} catch {

		return false;

	}

	const attributeSet = new Set( uniqueAttributes );

	for ( const node of uniqueNodes ) {

		for ( const group of bindings.getForCompute( node ) ) {

			for ( const binding of group.bindings ) {

				if ( binding.isStorageBuffer && binding.attribute && attributeSet.has( binding.attribute ) ) {

					binding._buffer = null;

				}

			}

		}

	}

	for ( const attribute of uniqueAttributes ) emptyAttributeArray( attribute );
	return true;

}

export {
	collectAttentionWeights,
	collectGatedDeltaWeights,
	collectLinearWeights,
	collectLogitWeights,
	collectMlpWeights,
	collectNormWeights,
	uploadAndReleaseStaticWeights
};
