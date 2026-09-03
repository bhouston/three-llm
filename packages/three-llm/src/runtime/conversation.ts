import type { ChatMessage, FormatChatOptions } from '../types.js';

interface PromptFormatter {
	formatChat?: ( messages: ChatMessage[], options?: FormatChatOptions ) => string;
}

/**
 * Base GPT-2 / Llama checkpoints are completion models. Feed the user text as a
 * prefix instead of wrapping it in a chat template they were never trained on.
 *
 */
function formatCompletionPrompt( messages: ChatMessage[] ): string {

	let prompt = '';

	for ( const message of messages ) {

		const text = message.text ?? message.content ?? '';

		if ( prompt === '' ) {

			prompt = text;
			continue;

		}

		if ( message.role === 'assistant' ) {

			prompt += text;
			continue;

		}

		prompt += `\n\n${ text }`;

	}

	return prompt;

}

function completionFollowUpText( userText: string ): string {

	return `\n\n${ userText }`;

}

function formatPrompt( weights: PromptFormatter, messages: ChatMessage[], options: FormatChatOptions = {} ): string {

	if ( typeof weights.formatChat === 'function' ) {

		return weights.formatChat( messages, options );

	}

	return formatCompletionPrompt( messages );

}

export { completionFollowUpText, formatCompletionPrompt, formatPrompt };
