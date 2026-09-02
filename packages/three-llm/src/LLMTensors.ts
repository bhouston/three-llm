/**
 * Shared tensor helpers for Hugging Face causal LM loaders.
 *
 */

import type { HuggingFaceConfig, PreparedGeneration, ProgressCallback, Tensor, TensorMap, Tokenizer } from './types.js';

function transpose2D( data: Float32Array, rows: number, columns: number ): Float32Array {

	const target = new Float32Array( data.length );

	for ( let row = 0; row < rows; row ++ ) {

		for ( let column = 0; column < columns; column ++ ) {

			target[ column * rows + row ] = data[ row * columns + column ];

		}

	}

	return target;

}

function float16ToFloat32( value: number ): number {

	const sign = ( value >> 15 ) & 1;
	const exponent = ( value >> 10 ) & 0x1f;
	const fraction = value & 0x3ff;

	if ( exponent === 0 ) {

		if ( fraction === 0 ) return sign ? - 0 : 0;

		return ( sign ? - 1 : 1 ) * Math.pow( 2, - 14 ) * ( fraction / 1024 );

	}

	if ( exponent === 31 ) {

		return fraction ? NaN : ( sign ? - Infinity : Infinity );

	}

	return ( sign ? - 1 : 1 ) * Math.pow( 2, exponent - 15 ) * ( 1 + fraction / 1024 );

}

const _bf16Bits = new Uint32Array( 1 );
const _bf16Float = new Float32Array( _bf16Bits.buffer );
const CONVERT_CHUNK_ELEMENTS = 1 << 20; // 1,048,576 values ≈ 2 MB of BF16
const STREAM_BUFFER_LIMIT = 256 * 1024 * 1024;

function bfloat16ToFloat32( value: number ): number {

	_bf16Bits[ 0 ] = value << 16;
	return _bf16Float[ 0 ];

}

function convertBF16Range( source: Uint16Array, target: Float32Array, start: number, end: number ): void {

	const bits = _bf16Bits;
	const float = _bf16Float;

	for ( let i = start; i < end; i ++ ) {

		bits[ 0 ] = source[ i ] << 16;
		target[ i ] = float[ 0 ];

	}

}

function convertF16Range( source: Uint16Array, target: Float32Array, start: number, end: number ): void {

	for ( let i = start; i < end; i ++ ) target[ i ] = float16ToFloat32( source[ i ] );

}

function tensorToFloat32( tensor: Tensor ): Float32Array {

	if ( tensor.dtype === 'F32' ) return tensor.data as Float32Array;

	const source = tensor.data as Uint16Array;
	const target = new Float32Array( source.length );

	if ( tensor.dtype === 'F16' ) {

		convertF16Range( source, target, 0, source.length );

	} else if ( tensor.dtype === 'BF16' ) {

		convertBF16Range( source, target, 0, source.length );

	} else {

		throw new Error( `LLMTensors: Tensor "${ tensor.name }" uses dtype "${ tensor.dtype }"; only F32, F16, and BF16 are supported.` );

	}

	return target;

}

async function convertAllTensors( tensors: TensorMap, onProgress?: ProgressCallback, label = 'LLMTensors', skipTensor?: ( name: string ) => boolean ): Promise<number> {

	const names = Object.keys( tensors ).filter( ( name ) => {

		if ( skipTensor && skipTensor( name ) ) return false;

		const dtype = tensors[ name ].dtype;
		return dtype === 'BF16' || dtype === 'F16';

	} );

	if ( names.length === 0 ) return 0;

	let total = 0;

	for ( let i = 0; i < names.length; i ++ ) total += tensors[ names[ i ] ].data.length;

	const dtype = tensors[ names[ 0 ] ].dtype;
	const report = createProgress( label, onProgress );
	let done = 0;

	await report( `Converting ${ names.length } ${ dtype } tensors (${ formatBytes( total * 2 ) })...` );

	for ( let n = 0; n < names.length; n ++ ) {

		const name = names[ n ];
		const tensor = tensors[ name ];
		const source = tensor.data as Uint16Array;
		const target = new Float32Array( source.length );
		const convertRange = tensor.dtype === 'BF16' ? convertBF16Range : convertF16Range;

		for ( let start = 0; start < source.length; start += CONVERT_CHUNK_ELEMENTS ) {

			const end = Math.min( start + CONVERT_CHUNK_ELEMENTS, source.length );
			convertRange( source, target, start, end );
			done += end - start;

			const pct = Math.min( 100, Math.round( ( 100 * done ) / total ) );
			await report( `Converting ${ tensor.dtype } ${ pct }% (${ formatBytes( done * 2 ) } / ${ formatBytes( total * 2 ) }) — ${ n + 1 }/${ names.length } ${ name }` );

		}

		tensor.data = target;
		tensor.dtype = 'F32';

	}

	return names.length;

}

function packProjections( projections: Float32Array[], inputSize: number ): Float32Array {

	const outputSizes = projections.map( ( projection ) => projection.length / inputSize );
	const outputSize = outputSizes.reduce( ( sum, size ) => sum + size, 0 );
	const packed = new Float32Array( inputSize * outputSize );

	for ( let row = 0; row < inputSize; row ++ ) {

		let offset = row * outputSize;

		for ( let i = 0; i < projections.length; i ++ ) {

			const size = outputSizes[ i ];
			packed.set( projections[ i ].subarray( row * size, row * size + size ), offset );
			offset += size;

		}

	}

	return packed;

}

function packBiases( biases: Array<Float32Array | null> ): Float32Array | null {

	if ( biases.every( ( bias ) => bias === null ) ) return null;

	const parts = biases.map( ( bias ) => bias || new Float32Array( 0 ) );
	const packed = new Float32Array( parts.reduce( ( sum, part ) => sum + part.length, 0 ) );
	let offset = 0;

	for ( const part of parts ) {

		packed.set( part, offset );
		offset += part.length;

	}

	return packed;

}

function prepareGeneration( tokenizer: Tokenizer, prompt: string, maxTokens: number, maxNewTokens: number, endOfTextTokenId: number ): PreparedGeneration {

	const encoded = tokenizer.encode( prompt );
	const promptBudget = Math.max( 1, maxTokens - 1 );
	const inputTokens = encoded.length === 0 ? [ endOfTextTokenId ] : encoded.slice( - promptBudget );
	const newTokenBudget = Math.max( 0, Math.min( maxNewTokens, maxTokens - inputTokens.length ) );

	return { inputTokens, newTokenBudget };

}

function unwrapTextConfig( config: HuggingFaceConfig ): HuggingFaceConfig {

	if ( config && config.text_config && typeof config.text_config === 'object' ) {

		return {
			...config.text_config,
			model_type: config.text_config.model_type || config.model_type,
			_parent_model_type: config.model_type
		};

	}

	return config;

}

function detectLanguagePrefix( tensors: TensorMap ): string {

	if ( tensors[ 'model.language_model.embed_tokens.weight' ] !== undefined ) return 'model.language_model.';
	if ( tensors[ 'language_model.embed_tokens.weight' ] !== undefined ) return 'language_model.';
	if ( tensors[ 'model.embed_tokens.weight' ] !== undefined ) return 'model.';
	if ( tensors[ 'embed_tokens.weight' ] !== undefined ) return '';

	return 'model.';

}

async function fetchJSON<T = any>( url: string, label = 'LLM' ): Promise<T> {

	const response = await fetch( url );

	if ( response.ok === false ) {

		throw new Error( `${ label }: Failed to load "${ url }" (${ response.status } ${ response.statusText })` );

	}

	return response.json() as Promise<T>;

}

function formatBytes( bytes: number ): string {

	if ( bytes < 1024 ) return `${ bytes } B`;
	if ( bytes < 1024 * 1024 ) return `${ ( bytes / 1024 ).toFixed( 1 ) } KB`;
	return `${ ( bytes / ( 1024 * 1024 ) ).toFixed( 1 ) } MB`;

}

function yieldToBrowser(): Promise<void> {

	return new Promise( ( resolve ) => setTimeout( resolve, 0 ) );

}

function createProgress( label: string, onProgress?: ProgressCallback ) {

	return async function report( message: string ): Promise<void> {

		if ( onProgress ) onProgress( `${ label }: ${ message }` );
		await yieldToBrowser();

	};

}

async function fetchArrayBuffer( url: string, label = 'LLM', onProgress?: ProgressCallback ): Promise<ArrayBuffer> {

	const response = await fetch( url );

	if ( response.ok === false ) {

		throw new Error( `${ label }: Failed to load "${ url }" (${ response.status } ${ response.statusText })` );

	}

	const total = Number( response.headers.get( 'Content-Length' ) ) || 0;
	const report = createProgress( label, onProgress );

	if ( response.body === null || typeof response.body.getReader !== 'function' || total > STREAM_BUFFER_LIMIT ) {

		await report( `Downloading ${ url }${ total ? ` (${ formatBytes( total ) })` : '' }...` );
		const buffer = await response.arrayBuffer();
		await report( `Downloaded ${ formatBytes( buffer.byteLength ) }` );
		return buffer;

	}

	const reader = response.body.getReader();
	const chunks: Uint8Array[] = [];
	let received = 0;
	let lastReport = 0;

	await report( `Downloading ${ url }${ total ? ` (${ formatBytes( total ) })` : '' }...` );

	while ( true ) {

		const { done, value } = await reader.read();
		if ( done ) break;

		chunks.push( value );
		received += value.byteLength;

		if ( received - lastReport >= 8 * 1024 * 1024 || ( total > 0 && received === total ) ) {

			lastReport = received;
			const totalText = total > 0 ? ` / ${ formatBytes( total ) }` : '';
			await report( `Downloading weights ${ formatBytes( received ) }${ totalText }` );

		}

	}

	const bytes = new Uint8Array( received );
	let offset = 0;

	for ( let i = 0; i < chunks.length; i ++ ) {

		bytes.set( chunks[ i ], offset );
		offset += chunks[ i ].byteLength;

	}

	await report( `Downloaded ${ formatBytes( received ) }` );
	return bytes.buffer;

}

export {
	bfloat16ToFloat32,
	convertAllTensors,
	createProgress,
	detectLanguagePrefix,
	fetchArrayBuffer,
	fetchJSON,
	float16ToFloat32,
	formatBytes,
	packBiases,
	packProjections,
	prepareGeneration,
	tensorToFloat32,
	transpose2D,
	unwrapTextConfig,
	yieldToBrowser
};
