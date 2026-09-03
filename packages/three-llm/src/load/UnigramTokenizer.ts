import { createProgress } from './tensors.js';
import type { LoaderOptions, Tokenizer } from '../types.js';

interface TrieNode {
	children: Map<string, TrieNode>;
	id: number;
}

interface TokenizerModel {
	type?: string;
	vocab?: Record<string, number> | Array<[string, number]>;
	merges?: Array<string | [string, string]>;
	byte_fallback?: boolean;
	unk_id?: number;
	unk_token?: string;
}

interface TokenizerJSON {
	model?: TokenizerModel;
	post_processor?: { bos_token_id?: number };
	added_tokens?: Array<{ id: number; content: string; special?: boolean }>;
}

interface TokenizerConfig {
	bos_token_id?: number;
	eos_token_id?: number | number[];
	add_bos_token?: boolean;
	[key: string]: unknown;
}

interface UnigramLoadOptions extends LoaderOptions {
	tokenizerConfig?: TokenizerConfig;
}

/**
 * Hugging Face `tokenizer.json` loader for Unigram and SentencePiece-style BPE.
 *
 * Gemma 3 uses BPE with a space→▁ normalizer, byte fallback, and a BOS
 * post-processor. Older Unigram vocabs (array of `[token, score]`) still work.
 *
 */
class UnigramTokenizer implements Tokenizer {

	tokens: string[];
	scores: number[];
	tokenToId: Map<string, number>;
	bpeRanks: Map<string, number>;
	bpeCache: Map<string, string>;
	useBpe: boolean;
	byteFallback: boolean;
	replacement: string;
	bosTokenId?: number;
	eosTokenId?: number | number[];
	addBos: boolean;
	endOfTextTokenId?: number;
	trie: TrieNode;
	unkId: number;

	constructor( tokenizerJSON: TokenizerJSON, tokenizerConfig: TokenizerConfig = {} ) {

		const model = tokenizerJSON.model || {};
		const vocab = model.vocab || [];

		this.tokens = [];
		this.scores = [];
		this.tokenToId = new Map();
		this.bpeRanks = new Map();
		this.bpeCache = new Map();
		this.useBpe = model.type === 'BPE' || ( vocab !== null && Array.isArray( vocab ) === false );
		this.byteFallback = model.byte_fallback === true;
		this.replacement = '▁';
		this.bosTokenId = tokenizerConfig.bos_token_id ?? tokenizerJSON.post_processor?.bos_token_id;
		this.eosTokenId = tokenizerConfig.eos_token_id;
		this.addBos = tokenizerConfig.add_bos_token !== false;
		this.endOfTextTokenId = ( this.eosTokenId ?? 1 ) as number;
		this.trie = { children: new Map(), id: - 1 };

		if ( this.useBpe ) {

			const bpeVocab = vocab as Record<string, number>;

			for ( const token in bpeVocab ) {

				const id = bpeVocab[ token ];
				this.tokens[ id ] = token;
				this.tokenToId.set( token, id );

			}

			const merges = model.merges || [];

			for ( let i = 0; i < merges.length; i ++ ) {

				const pair = Array.isArray( merges[ i ] ) ? merges[ i ] : String( merges[ i ] ).split( ' ' );
				this.bpeRanks.set( `${ pair[ 0 ] }\u0000${ pair[ 1 ] }`, i );

			}

		} else {

			const unigramVocab = vocab as Array<[string, number]>;

			for ( let i = 0; i < unigramVocab.length; i ++ ) {

				const [ token, score ] = unigramVocab[ i ];
				this.tokens[ i ] = token;
				this.scores[ i ] = score;
				this.tokenToId.set( token, i );
				insertTrie( this.trie, token, i );

			}

		}

		this.unkId = model.unk_id ?? this.tokenToId.get( model.unk_token || '<unk>' ) ?? 0;

		const added = tokenizerJSON.added_tokens || [];

		for ( const addedToken of added ) {

			if ( addedToken.content === undefined || addedToken.id === undefined ) continue;

			this.tokens[ addedToken.id ] = addedToken.content;
			if ( this.scores[ addedToken.id ] === undefined ) this.scores[ addedToken.id ] = 0;
			this.tokenToId.set( addedToken.content, addedToken.id );
			if ( this.useBpe === false ) insertTrie( this.trie, addedToken.content, addedToken.id );

			if ( addedToken.special && addedToken.content.includes( 'bos' ) ) this.bosTokenId = addedToken.id;
			if ( addedToken.special && addedToken.content.includes( 'eos' ) ) this.eosTokenId = addedToken.id;

		}

		if ( this.bosTokenId === undefined ) this.bosTokenId = this.tokenToId.get( '<bos>' ) ?? 2;
		if ( this.eosTokenId === undefined ) this.eosTokenId = this.tokenToId.get( '<eos>' ) ?? 1;
		this.endOfTextTokenId = this.eosTokenId as number;

	}

	static async fromURL( baseURL: string, options: UnigramLoadOptions = {} ): Promise<UnigramTokenizer> {

		const root = baseURL.endsWith( '/' ) ? baseURL : `${ baseURL }/`;
		const report = createProgress( 'UnigramTokenizer', options.onProgress );

		await report( 'Fetching tokenizer.json' );
		const tokenizerResponse = await fetch( `${ root }tokenizer.json` );

		if ( tokenizerResponse.ok === false ) {

			throw new Error( `UnigramTokenizer: Failed to load "${ root }tokenizer.json" (${ tokenizerResponse.status } ${ tokenizerResponse.statusText })` );

		}

		await report( `Parsing tokenizer.json (${ tokenizerResponse.headers.get( 'Content-Length' ) || 'unknown size' })` );
		const tokenizerJSON = await tokenizerResponse.json() as TokenizerJSON;

		let tokenizerConfig: TokenizerConfig = options.tokenizerConfig || {};

		try {

			const configResponse = await fetch( `${ root }tokenizer_config.json` );
			if ( configResponse.ok ) {

				await report( 'Parsing tokenizer_config.json' );
				tokenizerConfig = { ...await configResponse.json() as TokenizerConfig, ...tokenizerConfig };

			}

		} catch ( _error ) {

			// tokenizer_config.json is optional.

		}

		await report( `Building ${ tokenizerJSON.model?.type || 'tokenizer' } tables` );
		return new UnigramTokenizer( tokenizerJSON, tokenizerConfig );

	}

	encode( text: string ): number[] {

		const source = String( text );
		const ids = this.useBpe
			? this.encodeBpe( source.replaceAll( ' ', this.replacement ) )
			: this.unigram( `${ this.replacement }${ source.replaceAll( ' ', this.replacement ) }` );

		if ( this.addBos ) ids.unshift( this.bosTokenId! );

		return ids;

	}

	decode( tokenIds: number[] ): string {

		let text = '';

		for ( const id of tokenIds ) {

			if ( id === this.bosTokenId || id === this.eosTokenId ) continue;

			const token = this.tokens[ id ];
			if ( token === undefined ) continue;

			const hex = token.match( /^<0x([0-9A-Fa-f]{2})>$/ );
			text += hex ? String.fromCharCode( parseInt( hex[ 1 ], 16 ) ) : token;

		}

		return text.split( this.replacement ).join( ' ' ).replace( /^\s/, '' );

	}

	encodeBpe( text: string ): number[] {

		const pieces = this.bpe( text ).split( ' ' );
		const ids: number[] = [];

		for ( let i = 0; i < pieces.length; i ++ ) {

			if ( pieces[ i ] === '' ) continue;
			ids.push( this.tokenToId.has( pieces[ i ] ) ? this.tokenToId.get( pieces[ i ] )! : this.unkId );

		}

		return ids;

	}

	bpe( token: string ): string {

		if ( this.bpeCache.has( token ) ) return this.bpeCache.get( token )!;

		let word = symbolsFor( token, this.tokenToId, this.byteFallback );
		let pairs = getPairs( word );

		if ( pairs.size === 0 ) return word.join( ' ' );

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
		this.bpeCache.set( token, value );
		return value;

	}

	unigram( text: string ): number[] {

		const chars = Array.from( text );
		const length = chars.length;
		const best = new Float64Array( length + 1 );
		const back = new Int32Array( length + 1 );
		const backId = new Int32Array( length + 1 );

		best.fill( - Infinity );
		best[ 0 ] = 0;
		back.fill( - 1 );

		for ( let start = 0; start < length; start ++ ) {

			if ( best[ start ] === - Infinity ) continue;

			let node: TrieNode | undefined = this.trie;
			let matched = false;

			for ( let end = start; end < length; end ++ ) {

				node = node.children.get( chars[ end ] );
				if ( node === undefined ) break;

				if ( node.id >= 0 ) {

					const score = best[ start ] + this.scores[ node.id ];

					if ( score > best[ end + 1 ] ) {

						best[ end + 1 ] = score;
						back[ end + 1 ] = start;
						backId[ end + 1 ] = node.id;

					}

					matched = true;

				}

			}

			if ( matched === false && best[ start + 1 ] === - Infinity ) {

				const unkTokens = byteFallback( chars[ start ], this.tokenToId, this.unkId );
				best[ start + 1 ] = best[ start ] + ( this.scores[ unkTokens[ 0 ] ] || - 10 );
				back[ start + 1 ] = start;
				backId[ start + 1 ] = unkTokens[ 0 ];

			}

		}

		const ids: number[] = [];
		let index = length;

		if ( best[ length ] === - Infinity ) {

			for ( let i = 0; i < length; i ++ ) ids.push( this.unkId );
			return ids;

		}

		while ( index > 0 ) {

			ids.push( backId[ index ] );
			index = back[ index ];

		}

		ids.reverse();
		return ids;

	}

}

function insertTrie( root: TrieNode, token: string, id: number ): void {

	const chars = Array.from( token );
	let node = root;

	for ( let i = 0; i < chars.length; i ++ ) {

		let next = node.children.get( chars[ i ] );

		if ( next === undefined ) {

			next = { children: new Map(), id: - 1 };
			node.children.set( chars[ i ], next );

		}

		node = next;

	}

	node.id = id;

}

function byteFallback( char: string, tokenToId: Map<string, number>, unkId: number ): number[] {

	const bytes = new TextEncoder().encode( char );
	const ids: number[] = [];

	for ( let i = 0; i < bytes.length; i ++ ) {

		const token = `<0x${ bytes[ i ].toString( 16 ).toUpperCase().padStart( 2, '0' ) }>`;
		ids.push( tokenToId.has( token ) ? tokenToId.get( token )! : unkId );

	}

	return ids;

}

function symbolsFor( text: string, tokenToId: Map<string, number>, useByteFallback: boolean ): string[] {

	const chars = Array.from( text );
	const symbols: string[] = [];

	for ( let i = 0; i < chars.length; i ++ ) {

		if ( tokenToId.has( chars[ i ] ) || useByteFallback === false ) {

			symbols.push( chars[ i ] );

		} else {

			const bytes = new TextEncoder().encode( chars[ i ] );

			for ( let j = 0; j < bytes.length; j ++ ) {

				symbols.push( `<0x${ bytes[ j ].toString( 16 ).toUpperCase().padStart( 2, '0' ) }>` );

			}

		}

	}

	return symbols;

}

function getPairs( word: string[] ): Set<string> {

	const pairs = new Set<string>();

	for ( let i = 0; i < word.length - 1; i ++ ) {

		pairs.add( `${ word[ i ] }\u0000${ word[ i + 1 ] }` );

	}

	return pairs;

}

export { UnigramTokenizer };
