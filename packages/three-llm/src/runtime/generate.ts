import { needsFullLogitsForSampling, sampleTopK } from './math.js';
import type { GenerateOptions, GenerationResult, PreparedGeneration, PromptCachePlan, SampleOptions, Tokenizer } from '../types.js';

interface GenerateWeights {
	stopTokenIds?: number[];
	endOfTextTokenId: number;
	tokenizer: Tokenizer;
	prepareGeneration: ( prompt: string, maxTokens: number, maxNewTokens: number ) => PreparedGeneration;
}

interface GenerateRunner {
	weights: GenerateWeights;
	maxTokens: number;
	_cacheTokens?: number[];
	_cacheLogits?: Float32Array | null;
}

interface GenerateSyncControls {
	rewindable: boolean;
	resetCache: () => void;
	forwardToken: ( tokenId: number, position: number ) => Float32Array;
}

interface GenerateAsyncControls {
	rewindable: boolean;
	resetCache: () => void;
	computeToken: ( tokenId: number, position: number, computeLogits: boolean, candidateCount: number ) => void | Promise<void>;
	prefillTokens?: ( inputTokens: number[], start: number, end: number, onProgress: ( n: number ) => Promise<void> ) => Promise<void>;
	readLogits: () => Promise<Float32Array | null>;
	sampleToken?: ( candidateCount: number, options: SampleOptions ) => Promise<number>;
	maxGpuCandidateCount?: number;
}

function isStopToken( runner: GenerateRunner, tokenId: number ): boolean {

	const stopTokenIds = runner.weights.stopTokenIds;

	if ( Array.isArray( stopTokenIds ) && stopTokenIds.length > 0 ) {

		return stopTokenIds.includes( tokenId );

	}

	return tokenId === runner.weights.endOfTextTokenId;

}

function sharedPrefixLength( a: number[] = [], b: number[] = [] ): number {

	const n = Math.min( a.length, b.length );
	let i = 0;

	while ( i < n && a[ i ] === b[ i ] ) i ++;

	return i;

}

function prepareGenerationFromTokens( tokens: number[], maxTokens: number, maxNewTokens: number, endOfTextTokenId: number ): PreparedGeneration {

	const promptBudget = Math.max( 1, maxTokens - 1 );
	const inputTokens = tokens.length === 0 ? [ endOfTextTokenId ] : tokens.slice( - promptBudget );
	const newTokenBudget = Math.max( 0, Math.min( maxNewTokens, maxTokens - inputTokens.length ) );

	return { inputTokens, newTokenBudget };

}

function resolvePromptTokens( runner: GenerateRunner, prompt: string, options: GenerateOptions ): PreparedGeneration {

	const maxNewTokens = options.maxNewTokens ?? 32;

	if ( options.inputTokens ) {

		return prepareGenerationFromTokens(
			options.inputTokens,
			runner.maxTokens,
			maxNewTokens,
			runner.weights.endOfTextTokenId
		);

	}

	return runner.weights.prepareGeneration( prompt, runner.maxTokens, maxNewTokens );

}

function planPromptCache( cachedTokens: number[] | undefined, cachedLogits: Float32Array | null | undefined, inputTokens: number[], rewindable: boolean ): PromptCachePlan {

	const prefix = sharedPrefixLength( cachedTokens, inputTokens );
	const hasLogits = cachedLogits !== null && cachedLogits !== undefined;
	const appendOnly = prefix > 0 && prefix === ( cachedTokens || [] ).length && hasLogits;

	if ( appendOnly ) {

		return { start: prefix, logits: cachedLogits as Float32Array, reset: false, reused: prefix };

	}

	if ( rewindable && prefix > 0 ) {

		const start = prefix >= inputTokens.length ? inputTokens.length - 1 : prefix;
		return { start, logits: null, reset: false, reused: start };

	}

	return { start: 0, logits: null, reset: true, reused: 0 };

}

function gpuCandidateCount( options: GenerateOptions, maxCandidateCount: number ): number {

	if ( maxCandidateCount === 0 || options.gpuSampling === false || needsFullLogitsForSampling( options ) ) return 0;

	const temperature = options.temperature ?? 0.8;
	const topK = Math.max( 1, Math.floor( options.topK ?? 40 ) );

	if ( temperature <= 0 || topK === 1 ) return 1;
	if ( topK <= maxCandidateCount ) return topK;

	return 0;

}

function yieldToBrowser(): Promise<void> {

	if ( typeof requestAnimationFrame === 'function' ) {

		return new Promise( ( resolve ) => requestAnimationFrame( () => resolve() ) );

	}

	if ( typeof setTimeout === 'function' ) {

		return new Promise( ( resolve ) => setTimeout( resolve, 0 ) );

	}

	return Promise.resolve();

}

function finishGeneration( runner: GenerateRunner, weights: GenerateWeights, allTokens: number[], generatedTokens: number[], logits: Float32Array | null, extra: { reused: number; promptTokens: number; rest: object } ): GenerationResult {

	runner._cacheTokens = allTokens.slice();
	runner._cacheLogits = logits;

	return {
		tokens: allTokens,
		generatedTokens,
		text: weights.tokenizer.decode( allTokens ),
		generatedText: weights.tokenizer.decode( generatedTokens ),
		cachedPromptTokens: extra.reused,
		promptTokens: extra.promptTokens,
		... extra.rest
	};

}

function generateSync( runner: GenerateRunner, prompt: string, options: GenerateOptions = {}, controls: GenerateSyncControls ): GenerationResult {

	const { rewindable, resetCache, forwardToken } = controls;
	const { inputTokens, newTokenBudget } = resolvePromptTokens( runner, prompt, options );
	const plan = planPromptCache( runner._cacheTokens || [], runner._cacheLogits, inputTokens, rewindable );

	if ( plan.reset ) resetCache();

	if ( options.onPrefill ) {

		options.onPrefill( {
			cachedPromptTokens: plan.reused,
			promptTokens: inputTokens.length
		} );

	}

	const allTokens = inputTokens.slice();
	const generatedTokens: number[] = [];
	let logits = plan.logits;

	for ( let i = plan.start; i < inputTokens.length; i ++ ) {

		logits = forwardToken( inputTokens[ i ], i );

	}

	for ( let i = 0; i < newTokenBudget; i ++ ) {

		const nextToken = sampleTopK( logits!, { ...options, tokens: allTokens } );

		if ( isStopToken( runner, nextToken ) ) break;

		allTokens.push( nextToken );
		generatedTokens.push( nextToken );
		logits = forwardToken( nextToken, allTokens.length - 1 );

	}

	return finishGeneration( runner, runner.weights, allTokens, generatedTokens, logits, {
		reused: plan.reused,
		promptTokens: inputTokens.length,
		rest: {}
	} );

}

async function generateAsync( runner: GenerateRunner, prompt: string, options: GenerateOptions = {}, controls: GenerateAsyncControls ): Promise<GenerationResult> {

	const { rewindable, resetCache, computeToken, prefillTokens, readLogits, sampleToken, maxGpuCandidateCount = 0 } = controls;
	const { inputTokens, newTokenBudget } = resolvePromptTokens( runner, prompt, options );
	const plan = planPromptCache( runner._cacheTokens || [], runner._cacheLogits, inputTokens, rewindable );
	const signal = options.signal;
	const candidateCount = gpuCandidateCount( options, maxGpuCandidateCount );
	const useGpuSampling = candidateCount > 0 && typeof sampleToken === 'function';

	if ( plan.reset ) resetCache();

	if ( options.onPrefill ) {

		options.onPrefill( {
			cachedPromptTokens: plan.reused,
			promptTokens: inputTokens.length
		} );

	}

	const allTokens = inputTokens.slice();
	const generatedTokens: number[] = [];
	let logits = plan.logits;
	const needsPromptLogits = newTokenBudget > 0;
	const prefillEnd = needsPromptLogits ? inputTokens.length - 1 : inputTokens.length;
	let promptLoopStart = plan.start;
	const reportPrefillProgress = async ( completedPromptTokens: number ) => {

		if ( options.onPrefillProgress ) {

			options.onPrefillProgress( {
				cachedPromptTokens: plan.reused,
				completedPromptTokens,
				freshPromptTokens: inputTokens.length - plan.start,
				promptTokens: inputTokens.length
			} );
			await yieldToBrowser();

		}

	};

	if ( options.prefillMode !== false && typeof prefillTokens === 'function' && plan.start < prefillEnd ) {

		await prefillTokens( inputTokens, plan.start, prefillEnd, reportPrefillProgress );
		promptLoopStart = prefillEnd;

	}

	for ( let i = promptLoopStart; i < inputTokens.length; i ++ ) {

		if ( signal !== undefined && signal.aborted ) break;

		const computeLogits = needsPromptLogits && i === inputTokens.length - 1;
		await computeToken( inputTokens[ i ], i, computeLogits, useGpuSampling ? candidateCount : 0 );
		if ( computeLogits ) logits = useGpuSampling ? null : await readLogits();
		await reportPrefillProgress( i + 1 );

	}

	if ( options.onPrefillComplete ) {

		options.onPrefillComplete( {
			cachedPromptTokens: plan.reused,
			promptTokens: inputTokens.length
		} );

	}

	for ( let i = 0; i < newTokenBudget; i ++ ) {

		if ( signal !== undefined && signal.aborted ) break;
		if ( useGpuSampling === false && logits === null ) break;

		const nextToken = useGpuSampling
			? await sampleToken!( candidateCount, { ...options, tokens: allTokens } )
			: sampleTopK( logits!, { ...options, tokens: allTokens } );

		if ( isStopToken( runner, nextToken ) ) break;

		allTokens.push( nextToken );
		generatedTokens.push( nextToken );

		if ( options.onToken ) {

			options.onToken( runner.weights.tokenizer.decode( allTokens ), nextToken );

		}

		await computeToken( nextToken, allTokens.length - 1, true, useGpuSampling ? candidateCount : 0 );
		logits = useGpuSampling ? null : await readLogits();

	}

	return finishGeneration( runner, runner.weights, allTokens, generatedTokens, logits, {
		reused: plan.reused,
		promptTokens: inputTokens.length,
		rest: { aborted: signal !== undefined && signal.aborted }
	} );

}

export {
	generateAsync,
	generateSync,
	gpuCandidateCount,
	planPromptCache,
	prepareGenerationFromTokens,
	sharedPrefixLength
};
