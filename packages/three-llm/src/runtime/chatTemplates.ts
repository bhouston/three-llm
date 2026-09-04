import type { ChatMessage, ChatTemplateKind, FormatChatOptions, Tokenizer } from '../types.js';

function textFor(message: ChatMessage): string {
	return message.text ?? message.content ?? '';
}

function chatML(messages: ChatMessage[], assistantSuffix = '', options: FormatChatOptions = {}): string {
	const addGenerationPrompt = options.addGenerationPrompt !== false;
	let prompt = '';

	for (const message of messages) {
		if (message.role !== 'system' && message.role !== 'user' && message.role !== 'assistant') continue;
		prompt += `<|im_start|>${message.role}\n${textFor(message)}<|im_end|>\n`;
	}

	if (addGenerationPrompt) prompt += `<|im_start|>assistant\n${assistantSuffix}`;
	return prompt;
}

function qwenThinkingSuffix(options: FormatChatOptions): string {
	return options.enableThinking === true ? '<think>\n' : '<think>\n\n</think>\n\n';
}

function gemmaChat(messages: ChatMessage[], options: FormatChatOptions = {}): string {
	let prompt = '';

	for (const message of messages) {
		if (message.role === 'system') {
			prompt += textFor(message).trim() === '' ? '' : `${textFor(message)}\n\n`;
			continue;
		}

		if (message.role === 'user') prompt += `<start_of_turn>user\n${textFor(message)}<end_of_turn>\n`;
		else if (message.role === 'assistant') prompt += `<start_of_turn>model\n${textFor(message)}<end_of_turn>\n`;
	}

	if (options.addGenerationPrompt !== false) prompt += '<start_of_turn>model\n';
	return prompt;
}

function kananaChat(messages: ChatMessage[], options: FormatChatOptions = {}): string {
	let prompt = '';

	for (const message of messages) {
		if (message.role !== 'system' && message.role !== 'user' && message.role !== 'assistant') continue;
		prompt += `<|start_header_id|>${message.role}<|end_header_id|>\n\n${textFor(message)}<|eot_id|>`;
	}

	if (options.addGenerationPrompt !== false) prompt += '<|start_header_id|>assistant<|end_header_id|>\n\n';
	return prompt;
}

function formatChatTemplate(kind: ChatTemplateKind, messages: ChatMessage[], options: FormatChatOptions = {}): string {
	if (kind === 'gemma3') return gemmaChat(messages, options);
	if (kind === 'kanana') return kananaChat(messages, options);
	if (kind === 'qwen3' || kind === 'qwen3_5') return chatML(messages, qwenThinkingSuffix(options), options);
	if (kind === 'smollm2') return chatML(messages, '', options);
	if (kind === 'deepseek-r1') return chatML(messages, '<think>\n', options);
	return chatML(messages, '', options);
}

function tokenizerId(tokenizer: Tokenizer, token: string): number | undefined {
	return tokenizer.encoder?.[token];
}

function stopTokenIdsFor(kind: ChatTemplateKind | undefined, tokenizer: Tokenizer, configured: number[] = []): number[] {
	const ids = new Set<number>(configured);

	for (const id of tokenizer.stopTokenIds || []) ids.add(id);
	if (tokenizer.endOfTextTokenId !== undefined) ids.add(tokenizer.endOfTextTokenId);
	if (Array.isArray(tokenizer.eosTokenId)) for (const id of tokenizer.eosTokenId) ids.add(id);
	else if (tokenizer.eosTokenId !== undefined) ids.add(tokenizer.eosTokenId);

	const templateStops: Record<ChatTemplateKind, string[]> = {
		qwen2: ['<|im_end|>'],
		qwen3: ['<|im_end|>', '<|endoftext|>'],
		qwen3_5: ['<|im_end|>', '<|endoftext|>'],
		'deepseek-r1': ['<|im_end|>', '<｜end▁of▁sentence｜>'],
		gemma3: ['<end_of_turn>', '<eos>'],
		smollm2: ['<|im_end|>', '<|endoftext|>'],
		kanana: ['<|eot_id|>', '<|end_of_text|>'],
	};

	for (const token of kind === undefined ? [] : templateStops[kind]) {
		const id = tokenizerId(tokenizer, token);
		if (id !== undefined) ids.add(id);
	}

	return Array.from(ids);
}

export { formatChatTemplate, stopTokenIdsFor };
