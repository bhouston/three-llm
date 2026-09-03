import type { AddedToken, GPT2TokenizerOptions, Tokenizer } from '../types.js';

/**
 * Minimal GPT-2 byte-level BPE tokenizer.
 *
 */
class GPT2Tokenizer implements Tokenizer {

	encoder: Record<string, number>;
	decoder: string[];
	cache: Map<string, string>;
	byteEncoder: Record<number, string>;
	byteDecoder: Record<string, number>;
	unknownToken: string;
	endOfTextToken: string;
	tokenPattern: RegExp;
	addedTokenIds: Map<string, number>;
	addedTokenPattern: RegExp | null;
	endOfTextTokenId?: number;
	bpeRanks: Map<string, number>;

	constructor( vocab: Record<string, number>, merges: string[], options: GPT2TokenizerOptions = {} ) {

		this.encoder = Object.assign( {}, vocab );
		this.decoder = [];
		this.cache = new Map();
		this.byteEncoder = bytesToUnicode();
		this.byteDecoder = {};
		this.unknownToken = options.unknownToken || '<|endoftext|>';
		this.endOfTextToken = options.endOfTextToken || '<|endoftext|>';
		this.tokenPattern = options.tokenPattern || GPT2_TOKEN_PATTERN;
		this.addedTokenIds = new Map();
		this.addedTokenPattern = null;

		const addedTokens: AddedToken[] = options.addedTokens || [];

		for ( const added of addedTokens ) {

			if ( added === undefined || added.content === undefined || added.id === undefined ) continue;

			this.encoder[ added.content ] = added.id;
			this.addedTokenIds.set( added.content, added.id );

		}

		if ( this.addedTokenIds.size > 0 ) {

			const escaped = Array.from( this.addedTokenIds.keys() )
				.sort( ( a, b ) => b.length - a.length )
				.map( escapeRegExp )
				.join( '|' );
			this.addedTokenPattern = new RegExp( `(${ escaped })` );

		}

		this.endOfTextTokenId = this.encoder[ this.endOfTextToken ];

		for ( const token in this.encoder ) {

			this.decoder[ this.encoder[ token ] ] = token;

		}

		for ( const key in this.byteEncoder ) {

			this.byteDecoder[ this.byteEncoder[ Number( key ) ] ] = Number( key );

		}

		this.bpeRanks = new Map();

		for ( let i = 0; i < merges.length; i ++ ) {

			const merge = merges[ i ].trim();

			if ( merge === '' || merge.startsWith( '#version' ) ) continue;

			const pair = merge.split( /\s+/ );
			this.bpeRanks.set( pair.join( '\u0000' ), this.bpeRanks.size );

		}

	}

	static async fromURLs( vocabURL: string, mergesURL: string, options?: GPT2TokenizerOptions ): Promise<GPT2Tokenizer> {

		const [ vocabResponse, mergesResponse ] = await Promise.all( [
			fetch( vocabURL ),
			fetch( mergesURL )
		] );

		if ( vocabResponse.ok === false ) {

			throw new Error( `GPT2Tokenizer: Failed to load "${ vocabURL }" (${ vocabResponse.status } ${ vocabResponse.statusText })` );

		}

		if ( mergesResponse.ok === false ) {

			throw new Error( `GPT2Tokenizer: Failed to load "${ mergesURL }" (${ mergesResponse.status } ${ mergesResponse.statusText })` );

		}

		return new GPT2Tokenizer( await vocabResponse.json() as Record<string, number>, ( await mergesResponse.text() ).split( /\r?\n/ ), options );

	}

	encode( text: string ): number[] {

		const tokens: number[] = [];
		const chunks = this.addedTokenPattern === null
			? [ text ]
			: String( text ).split( this.addedTokenPattern );

		for ( const chunk of chunks ) {

			if ( chunk === '' ) continue;

			if ( this.addedTokenIds.has( chunk ) ) {

				tokens.push( this.addedTokenIds.get( chunk )! );
				continue;

			}

			const matches = chunk.match( this.tokenPattern ) || [];

			for ( const match of matches ) {

				const encoded = byteEncode( match, this.byteEncoder );
				const bpeTokens = this.bpe( encoded ).split( ' ' );

				for ( const token of bpeTokens ) {

					const id = this.encoder[ token ];
					tokens.push( id === undefined ? this.encoder[ this.unknownToken ] : id );

				}

			}

		}

		return tokens;

	}

	decode( tokenIds: number[] ): string {

		let text = '';

		for ( const id of tokenIds ) {

			text += this.decoder[ id ] || '';

		}

		const bytes: number[] = [];

		for ( const char of text ) {

			const byte = this.byteDecoder[ char ];
			if ( byte !== undefined ) bytes.push( byte );

		}

		return new TextDecoder( 'utf-8', { fatal: false } ).decode( new Uint8Array( bytes ) );

	}

	bpe( token: string ): string {

		if ( this.cache.has( token ) ) return this.cache.get( token )!;

		let word = Array.from( token );
		let pairs = getPairs( word );

		if ( pairs.size === 0 ) return token;

		while ( true ) {

			let bestPair: string | null = null;
			let bestRank = Infinity;

			for ( const pair of pairs ) {

				const rank = this.bpeRanks.get( pair );

				if ( rank !== undefined && rank < bestRank ) {

					bestPair = pair;
					bestRank = rank;

				}

			}

			if ( bestPair === null ) break;

			const [ first, second ] = bestPair.split( '\u0000' );
			const nextWord: string[] = [];
			let i = 0;

			while ( i < word.length ) {

				if ( word[ i ] === first && i < word.length - 1 && word[ i + 1 ] === second ) {

					nextWord.push( first + second );
					i += 2;

				} else {

					nextWord.push( word[ i ] );
					i ++;

				}

			}

			word = nextWord;

			if ( word.length === 1 ) break;

			pairs = getPairs( word );

		}

		const value = word.join( ' ' );
		this.cache.set( token, value );

		return value;

	}

}

const GPT2_TOKEN_PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const QWEN_TOKEN_PATTERN = /(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

function bytesToUnicode(): Record<number, string> {

	const bs: number[] = [];
	const cs: number[] = [];

	for ( let i = 33; i <= 126; i ++ ) bs.push( i );
	for ( let i = 161; i <= 172; i ++ ) bs.push( i );
	for ( let i = 174; i <= 255; i ++ ) bs.push( i );

	for ( const b of bs ) cs.push( b );

	let n = 0;

	for ( let b = 0; b < 256; b ++ ) {

		if ( bs.includes( b ) === false ) {

			bs.push( b );
			cs.push( 256 + n );
			n ++;

		}

	}

	const table: Record<number, string> = {};

	for ( let i = 0; i < bs.length; i ++ ) {

		table[ bs[ i ] ] = String.fromCodePoint( cs[ i ] );

	}

	return table;

}

function byteEncode( text: string, byteEncoder: Record<number, string> ): string {

	const bytes = new TextEncoder().encode( text );
	let result = '';

	for ( const byte of bytes ) {

		result += byteEncoder[ byte ];

	}

	return result;

}

function escapeRegExp( string: string ): string {

	return string.replace( /[.*+?^${}()|[\]\\]/g, '\\$&' );

}

function getPairs( word: string[] ): Set<string> {

	const pairs = new Set<string>();

	for ( let i = 0; i < word.length - 1; i ++ ) {

		pairs.add( `${ word[ i ] }\u0000${ word[ i + 1 ] }` );

	}

	return pairs;

}

export { GPT2Tokenizer, GPT2_TOKEN_PATTERN, QWEN_TOKEN_PATTERN };
