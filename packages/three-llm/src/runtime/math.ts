import type { CausalAttentionOptions } from '../types.js';

type FloatVec = Float32Array;

function linear( input: FloatVec, weight: FloatVec, bias: FloatVec | null, inputSize: number, outputSize: number, target = new Float32Array( outputSize ) ): FloatVec {

	for ( let output = 0; output < outputSize; output ++ ) {

		let sum = bias !== null ? bias[ output ] : 0;

		for ( let i = 0; i < inputSize; i ++ ) {

			sum += input[ i ] * weight[ i * outputSize + output ];

		}

		target[ output ] = sum;

	}

	return target;

}

function layerNorm( input: FloatVec, weight: FloatVec, bias: FloatVec, epsilon = 1e-5, target = new Float32Array( input.length ) ): FloatVec {

	let mean = 0;

	for ( let i = 0; i < input.length; i ++ ) mean += input[ i ];

	mean /= input.length;

	let variance = 0;

	for ( let i = 0; i < input.length; i ++ ) {

		const d = input[ i ] - mean;
		variance += d * d;

	}

	const invStd = 1 / Math.sqrt( variance / input.length + epsilon );

	for ( let i = 0; i < input.length; i ++ ) {

		target[ i ] = ( input[ i ] - mean ) * invStd * weight[ i ] + bias[ i ];

	}

	return target;

}

function geluNew( x: number ): number {

	return 0.5 * x * ( 1 + Math.tanh( Math.sqrt( 2 / Math.PI ) * ( x + 0.044715 * x * x * x ) ) );

}

function softmax( values: ArrayLike<number>, target = new Float32Array( values.length ) ): FloatVec {

	let maxValue = - Infinity;

	for ( let i = 0; i < values.length; i ++ ) maxValue = Math.max( maxValue, values[ i ] );

	let sum = 0;

	for ( let i = 0; i < values.length; i ++ ) {

		const value = Math.exp( values[ i ] - maxValue );
		target[ i ] = value;
		sum += value;

	}

	for ( let i = 0; i < target.length; i ++ ) target[ i ] /= sum;

	return target;

}

function rmsNorm( input: FloatVec, weight: FloatVec | null | undefined, epsilon = 1e-5, offsetWeight = false, target = new Float32Array( input.length ) ): FloatVec {

	let sumSquares = 0;

	for ( let i = 0; i < input.length; i ++ ) sumSquares += input[ i ] * input[ i ];

	const invRms = 1 / Math.sqrt( sumSquares / input.length + epsilon );

	for ( let i = 0; i < input.length; i ++ ) {

		const scale = weight == null ? 1 : ( offsetWeight ? 1 + weight[ i ] : weight[ i ] );
		target[ i ] = input[ i ] * invRms * scale;

	}

	return target;

}

function silu( x: number ): number {

	return x / ( 1 + Math.exp( - x ) );

}

function geluPytorchTanh( x: number ): number {

	return geluNew( x );

}

function rotaryAngle( position: number, freqIndex: number, rotaryDim: number, theta: number ): number {

	return position * Math.pow( theta, - 2 * freqIndex / rotaryDim );

}

interface RopeOptions {
	ropeFreqDim?: number;
	ropePairCount?: number;
}

function applyRoPE( vector: FloatVec, headOffset: number, rotaryDim: number, position: number, theta: number, options: RopeOptions = {} ): FloatVec {

	if ( rotaryDim <= 0 ) return vector;

	const half = rotaryDim / 2;
	const freqDim = options.ropeFreqDim || rotaryDim;
	const pairCount = options.ropePairCount !== undefined ? options.ropePairCount : half;
	const rotated = new Float32Array( rotaryDim );

	for ( let i = 0; i < rotaryDim; i ++ ) {

		const x = vector[ headOffset + i ];
		const freqIndex = i < half ? i : i - half;

		if ( freqIndex >= pairCount ) {

			rotated[ i ] = x;
			continue;

		}

		const partner = i < half ? - vector[ headOffset + i + half ] : vector[ headOffset + i - half ];
		const angle = rotaryAngle( position, freqIndex, freqDim, theta );

		rotated[ i ] = x * Math.cos( angle ) + partner * Math.sin( angle );

	}

	for ( let i = 0; i < rotaryDim; i ++ ) vector[ headOffset + i ] = rotated[ i ];

	return vector;

}

function rmsNormPackedHeads( vector: FloatVec, headCount: number, headDim: number, weight: FloatVec | null, epsilon: number, offsetWeight: boolean ): void {

	for ( let head = 0; head < headCount; head ++ ) {

		const offset = head * headDim;
		const slice = vector.subarray( offset, offset + headDim );
		slice.set( rmsNorm( slice, weight, epsilon, offsetWeight ) );

	}

}

interface AttentionOptions extends CausalAttentionOptions {
	hiddenSize?: number;
	ropeFreqDim?: number;
	ropePairCount?: number;
	queryOnly?: boolean;
	writeCache?: boolean;
	vNorm?: boolean;
	outputGate?: FloatVec | null;
}

function causalAttention( qkv: FloatVec, options: AttentionOptions ): FloatVec {

	const {
		headCount,
		position,
		keyCache,
		valueCache,
		headDim = ( options.hiddenSize as number ) / headCount,
		kvHeadCount = headCount,
		ropeTheta = 0,
		rotaryDim = headDim,
		slidingWindow = 0,
		attnScale = 1 / Math.sqrt( headDim ),
		qNormWeight = null,
		kNormWeight = null,
		rmsEpsilon = 1e-6,
		offsetRMSNorm = false,
		ropeFreqDim = rotaryDim,
		ropePairCount,
		queryOnly = false,
		writeCache = true,
		vNorm = false,
		outputGate = null
	} = options;
	const qSize = headCount * headDim;
	const kvSize = kvHeadCount * headDim;
	const query = qkv.slice( 0, qSize );
	const key = queryOnly ? null : qkv.slice( qSize, qSize + kvSize );
	const firstToken = slidingWindow > 0 ? Math.max( 0, position - slidingWindow + 1 ) : 0;
	const ropeOptions = { ropeFreqDim, ropePairCount };

	if ( qNormWeight !== null ) rmsNormPackedHeads( query, headCount, headDim, qNormWeight, rmsEpsilon, offsetRMSNorm );

	if ( queryOnly === false && kNormWeight !== null ) {

		rmsNormPackedHeads( key!, kvHeadCount, headDim, kNormWeight, rmsEpsilon, offsetRMSNorm );

	}

	if ( ropeTheta > 0 ) {

		for ( let head = 0; head < headCount; head ++ ) {

			applyRoPE( query, head * headDim, rotaryDim, position, ropeTheta, ropeOptions );

		}

		if ( queryOnly === false ) {

			for ( let head = 0; head < kvHeadCount; head ++ ) {

				applyRoPE( key!, head * headDim, rotaryDim, position, ropeTheta, ropeOptions );

			}

		}

	}

	if ( queryOnly === false && writeCache ) {

		const value = qkv.slice( qSize + kvSize, qSize + 2 * kvSize );

		if ( vNorm ) rmsNormPackedHeads( value, kvHeadCount, headDim, null, rmsEpsilon, false );

		for ( let dim = 0; dim < kvSize; dim ++ ) {

			keyCache[ position * kvSize + dim ] = key![ dim ];
			valueCache[ position * kvSize + dim ] = value[ dim ];

		}

	}

	const output = new Float32Array( qSize );

	for ( let head = 0; head < headCount; head ++ ) {

		const qOffset = head * headDim;
		const kvHead = Math.floor( head * kvHeadCount / headCount );
		const kvOffset = kvHead * headDim;
		const scores = new Float32Array( position - firstToken + 1 );

		for ( let token = firstToken; token <= position; token ++ ) {

			let dot = 0;

			for ( let i = 0; i < headDim; i ++ ) {

				dot += query[ qOffset + i ] * keyCache[ token * kvSize + kvOffset + i ];

			}

			scores[ token - firstToken ] = dot * attnScale;

		}

		const weights = softmax( scores );

		for ( let i = 0; i < headDim; i ++ ) {

			let sum = 0;

			for ( let token = firstToken; token <= position; token ++ ) {

				sum += weights[ token - firstToken ] * valueCache[ token * kvSize + kvOffset + i ];

			}

			output[ qOffset + i ] = sum;

		}

	}

	if ( outputGate !== null ) {

		for ( let i = 0; i < output.length; i ++ ) output[ i ] *= sigmoid( outputGate[ i ] );

	}

	return output;

}

function sigmoid( x: number ): number {

	return 1 / ( 1 + Math.exp( - x ) );

}

function softplus( x: number ): number {

	if ( x > 20 ) return x;
	if ( x < - 20 ) return Math.exp( x );
	return Math.log( 1 + Math.exp( x ) );

}

function logitSoftcap( logits: FloatVec, cap?: number | null ): FloatVec {

	if ( cap === null || cap === undefined ) return logits;

	for ( let i = 0; i < logits.length; i ++ ) logits[ i ] = cap * Math.tanh( logits[ i ] / cap );

	return logits;

}

function l2norm( vector: FloatVec, offset: number, size: number, epsilon = 1e-6 ): FloatVec {

	let sumSquares = 0;

	for ( let i = 0; i < size; i ++ ) sumSquares += vector[ offset + i ] * vector[ offset + i ];

	const inv = 1 / Math.sqrt( sumSquares + epsilon );

	for ( let i = 0; i < size; i ++ ) vector[ offset + i ] *= inv;

	return vector;

}

function splitHeadGate( packed: FloatVec, headCount: number, headDim: number ): { query: FloatVec; gate: FloatVec } {

	const qSize = headCount * headDim;
	const query = new Float32Array( qSize );
	const gate = new Float32Array( qSize );

	for ( let head = 0; head < headCount; head ++ ) {

		const packedOffset = head * headDim * 2;
		const headOffset = head * headDim;
		query.set( packed.subarray( packedOffset, packedOffset + headDim ), headOffset );
		gate.set( packed.subarray( packedOffset + headDim, packedOffset + headDim * 2 ), headOffset );

	}

	return { query, gate };

}

function causalConv1dStep( input: FloatVec, state: FloatVec, weight: FloatVec, kernelSize: number, activation = 'silu' ): FloatVec {

	const convDim = input.length;
	const output = new Float32Array( convDim );
	const window = new Float32Array( kernelSize );

	for ( let channel = 0; channel < convDim; channel ++ ) {

		const stateOffset = channel * kernelSize;
		const weightOffset = channel * kernelSize;

		for ( let k = 0; k < kernelSize; k ++ ) window[ k ] = state[ stateOffset + k ];

		for ( let k = 0; k < kernelSize - 1; k ++ ) state[ stateOffset + k ] = window[ k + 1 ];

		state[ stateOffset + kernelSize - 1 ] = input[ channel ];

		let sum = 0;

		for ( let k = 0; k < kernelSize - 1; k ++ ) sum += weight[ weightOffset + k ] * window[ k + 1 ];

		sum += weight[ weightOffset + kernelSize - 1 ] * input[ channel ];
		output[ channel ] = activation === 'silu' ? silu( sum ) : sum;

	}

	return output;

}

function gatedDeltaRuleStep( query: FloatVec, key: FloatVec, value: FloatVec, decay: FloatVec, beta: FloatVec, state: FloatVec, options: { numVHeads: number; keyDim: number; valueDim: number } ): FloatVec {

	const { numVHeads, keyDim, valueDim } = options;
	const output = new Float32Array( numVHeads * valueDim );

	for ( let head = 0; head < numVHeads; head ++ ) {

		const decayH = decay[ head ];
		const betaH = beta[ head ];
		const qOffset = head * keyDim;
		const vOffset = head * valueDim;
		const stateOffset = head * keyDim * valueDim;

		for ( let v = 0; v < valueDim; v ++ ) {

			let kvMem = 0;

			for ( let k = 0; k < keyDim; k ++ ) {

				const index = stateOffset + k * valueDim + v;
				state[ index ] *= decayH;
				kvMem += state[ index ] * key[ qOffset + k ];

			}

			const delta = ( value[ vOffset + v ] - kvMem ) * betaH;
			let mixed = 0;

			for ( let k = 0; k < keyDim; k ++ ) {

				const index = stateOffset + k * valueDim + v;
				state[ index ] += key[ qOffset + k ] * delta;
				mixed += state[ index ] * query[ qOffset + k ];

			}

			output[ vOffset + v ] = mixed;

		}

	}

	return output;

}

function rmsNormGated( input: FloatVec, gate: FloatVec, weight: FloatVec, headCount: number, headDim: number, epsilon = 1e-6 ): FloatVec {

	const output = new Float32Array( input.length );

	for ( let head = 0; head < headCount; head ++ ) {

		const offset = head * headDim;
		const slice = input.subarray( offset, offset + headDim );
		const normed = rmsNorm( slice, weight, epsilon, false );

		for ( let i = 0; i < headDim; i ++ ) output[ offset + i ] = normed[ i ] * silu( gate[ offset + i ] );

	}

	return output;

}

interface RepeatPenaltyOptions {
	repetitionPenalty?: number;
	presencePenalty?: number;
	frequencyPenalty?: number;
	repeatLastN?: number;
	noRepeatNgramSize?: number;
}

function applyRepeatPenalties( logits: FloatVec, tokens: number[] | undefined, {
	repetitionPenalty = 1,
	presencePenalty = 0,
	frequencyPenalty = 0,
	repeatLastN = 64,
	noRepeatNgramSize = 0
}: RepeatPenaltyOptions = {} ): FloatVec {

	if ( tokens === undefined || tokens.length === 0 ) return logits;

	const useRepetition = repetitionPenalty !== 1 && repetitionPenalty > 0;
	const usePresence = presencePenalty !== 0;
	const useFrequency = frequencyPenalty !== 0;
	const ngramSize = Math.max( 0, Math.floor( noRepeatNgramSize ) );
	const useNgrams = ngramSize >= 2 && tokens.length >= ngramSize - 1;

	if ( ! useRepetition && ! usePresence && ! useFrequency && ! useNgrams ) return logits;

	const penalized = logits.slice();

	if ( useRepetition || usePresence || useFrequency ) {

		const windowStart = repeatLastN > 0 ? Math.max( 0, tokens.length - repeatLastN ) : 0;
		const counts = new Map<number, number>();

		for ( let i = windowStart; i < tokens.length; i ++ ) {

			const tokenId = tokens[ i ];
			counts.set( tokenId, ( counts.get( tokenId ) || 0 ) + 1 );

		}

		for ( const [ tokenId, count ] of counts ) {

			let score = penalized[ tokenId ];

			if ( useRepetition ) {

				score = score < 0 ? score * repetitionPenalty : score / repetitionPenalty;

			}

			if ( usePresence ) score -= presencePenalty;
			if ( useFrequency ) score -= frequencyPenalty * count;

			penalized[ tokenId ] = score;

		}

	}

	if ( useNgrams ) {

		const prefixLength = ngramSize - 1;
		const prefixStart = tokens.length - prefixLength;

		for ( let i = 0; i <= tokens.length - ngramSize; i ++ ) {

			let match = true;

			for ( let j = 0; j < prefixLength; j ++ ) {

				if ( tokens[ i + j ] !== tokens[ prefixStart + j ] ) {

					match = false;
					break;

				}

			}

			if ( match ) penalized[ tokens[ i + prefixLength ] ] = - Infinity;

		}

	}

	return penalized;

}

interface SamplingNeedOptions extends RepeatPenaltyOptions {
	forceFullLogits?: boolean;
}

function needsFullLogitsForSampling( {
	repetitionPenalty = 1,
	presencePenalty = 0,
	frequencyPenalty = 0,
	noRepeatNgramSize = 0,
	forceFullLogits = false
}: SamplingNeedOptions = {} ): boolean {

	return forceFullLogits === true ||
		( repetitionPenalty !== 1 && repetitionPenalty > 0 ) ||
		presencePenalty !== 0 ||
		frequencyPenalty !== 0 ||
		Math.max( 0, Math.floor( noRepeatNgramSize ) ) >= 2;

}

type LogitCandidate = [ number, number ];

function sampleTopKCandidates( candidates: LogitCandidate[], {
	temperature = 0.8,
	random = Math.random
}: { temperature?: number; random?: () => number } = {} ): number {

	if ( candidates.length === 0 ) return 0;
	if ( temperature <= 0 || candidates.length === 1 ) return candidates[ 0 ][ 0 ];

	const values = new Float32Array( candidates.length );

	for ( let i = 0; i < candidates.length; i ++ ) values[ i ] = candidates[ i ][ 1 ] / Math.max( temperature, 1e-6 );

	const probabilities = softmax( values );
	let r = random();

	for ( let i = 0; i < probabilities.length; i ++ ) {

		r -= probabilities[ i ];
		if ( r <= 0 ) return candidates[ i ][ 0 ];

	}

	return candidates[ candidates.length - 1 ][ 0 ];

}

interface SampleTopKOptions extends RepeatPenaltyOptions {
	temperature?: number;
	topK?: number;
	random?: () => number;
	tokens?: number[];
}

function sampleTopK( logits: FloatVec, {
	temperature = 0.8,
	topK = 40,
	random = Math.random,
	tokens,
	repetitionPenalty = 1,
	presencePenalty = 0,
	frequencyPenalty = 0,
	repeatLastN = 64,
	noRepeatNgramSize = 0
}: SampleTopKOptions = {} ): number {

	logits = applyRepeatPenalties( logits, tokens, {
		repetitionPenalty,
		presencePenalty,
		frequencyPenalty,
		repeatLastN,
		noRepeatNgramSize
	} );

	const k = Math.min( topK, logits.length );
	const candidates: LogitCandidate[] = [];

	if ( temperature <= 0 || k === 1 ) {

		let bestIndex = 0;
		let bestValue = logits[ 0 ];

		for ( let i = 1; i < logits.length; i ++ ) {

			if ( logits[ i ] > bestValue ) {

				bestIndex = i;
				bestValue = logits[ i ];

			}

		}

		return bestIndex;

	}

	for ( let i = 0; i < logits.length; i ++ ) {

		const value = logits[ i ] / Math.max( temperature, 1e-6 );

		if ( candidates.length < k ) {

			candidates.push( [ i, value ] );
			candidates.sort( ( a, b ) => b[ 1 ] - a[ 1 ] );

		} else if ( value > candidates[ k - 1 ][ 1 ] ) {

			candidates[ k - 1 ] = [ i, value ];
			candidates.sort( ( a, b ) => b[ 1 ] - a[ 1 ] );

		}

	}

	const values = new Float32Array( candidates.length );

	for ( let i = 0; i < candidates.length; i ++ ) values[ i ] = candidates[ i ][ 1 ];

	const probabilities = softmax( values );
	let r = random();

	for ( let i = 0; i < probabilities.length; i ++ ) {

		r -= probabilities[ i ];
		if ( r <= 0 ) return candidates[ i ][ 0 ];

	}

	return candidates[ candidates.length - 1 ][ 0 ];

}

export {
	applyRoPE,
	causalAttention,
	causalConv1dStep,
	gatedDeltaRuleStep,
	geluNew,
	geluPytorchTanh,
	l2norm,
	layerNorm,
	linear,
	logitSoftcap,
	needsFullLogitsForSampling,
	rmsNorm,
	rmsNormGated,
	rmsNormPackedHeads,
	rotaryAngle,
	sampleTopKCandidates,
	sampleTopK,
	sigmoid,
	silu,
	softplus,
	softmax,
	splitHeadGate
};
