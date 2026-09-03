# three-llm

[![npm version](https://img.shields.io/npm/v/three-llm.svg)](https://www.npmjs.com/package/three-llm)
[![live demo](https://img.shields.io/badge/demo-three--llm.ben3d.ca-blue)](https://three-llm.ben3d.ca)

Run large language models in the browser with WebGPU. `three-llm` implements transformer inference with [Three.js](https://threejs.org/) and its TSL compute shader system, so model execution stays on the user's GPU without a server-side inference runtime.

**[Try the live demo](https://three-llm.ben3d.ca)** · **[Read the technical write-up](https://ben3d.ca/blog/running-llms-in-the-browser-with-threejs)**

The library loads Hugging Face model configs, tokenizers, and SafeTensors weights directly in the browser. This repository also includes a chat demo that exercises the full stack.

## Try it

The live demo at [three-llm.ben3d.ca](https://three-llm.ben3d.ca) runs five checkpoints in a multi-turn chat UI with streaming replies, model switching, and generation settings. For how the models are implemented and why the architecture choices matter, see [Running LLMs in the Browser with Three.js](https://ben3d.ca/blog/running-llms-in-the-browser-with-threejs).

| Model | Size | Notes |
| --- | --- | --- |
| TinyStories GPT-2 3M | 15 MB | Short children's stories |
| GPT-2 124M | 548 MB | Classic dense GPT-2 |
| SmolLM2 135M | 269 MB | Llama-style chat model (default) |
| Qwen3.5 0.8B | 1.7 GB | Hybrid linear + full attention |
| Phi-1.5 1.3B | 2.8 GB | Microsoft Phi architecture |

On phones and other memory-constrained devices, stick with TinyStories or SmolLM2. Qwen and Phi need enough GPU memory that many mobile browsers will fail to load them.

## Features

- WebGPU inference through Three.js TSL compute shaders
- CPU reference runners for testing and validation
- Prompt caching, chunked prefill, streaming token callbacks, and GPU sampling
- GPT-2, Llama-style, Gemma 3, Phi, and Qwen 3.5 decoder architectures
- GPT-2 BPE, Qwen BPE, and unigram tokenizers
- Chat prompt formatting via `formatPrompt`
- Direct loading of Hugging Face SafeTensors checkpoints

## Requirements

- A browser with WebGPU support, such as a recent Chrome, Edge, or Safari release
- Enough device memory for the selected model and its intermediate buffers

Model files can range from a few megabytes to several gigabytes. Remote Hugging Face repositories must allow browser CORS requests.

## Install

```sh
pnpm add three-llm three
```

## Usage

Create a Three.js WebGPU renderer, load a compatible Hugging Face checkpoint, and generate text:

```ts
import { createTSLRunner } from 'three-llm';
import { WebGPURenderer } from 'three/webgpu';

const renderer = new WebGPURenderer();
await renderer.init();

const runner = await createTSLRunner('https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/', {
  onProgress: console.log,
  prefillChunkSize: 4,
});

const result = await runner.generate(renderer, 'Once upon a time,', {
  maxNewTokens: 64,
  temperature: 0.7,
  topK: 10,
  onToken: (text) => {
    // Append each decoded token to your UI.
    console.log(text);
  },
});

console.log(result.generatedText);
```

For multi-turn chat, pass formatted messages with `formatPrompt` from `three-llm`. For catalog entries and URL resolution, import `MODEL_CATALOG` and `resolveModelURL` from `three-llm/catalog`.

## Run the demo locally

To work on the chat app or test against the hosted model bucket:

```sh
corepack enable
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). The demo loads checkpoints through the website's `/api/models/` proxy backed by the public [`gs://three-llm`](https://storage.googleapis.com/three-llm/) bucket, and falls back to Hugging Face if a file is missing.

## Development

This monorepo uses pnpm workspaces:

- `packages/three-llm`: the inference library, model loaders, tokenizers, and TSL kernels
- `packages/website`: the React chat demo

Requirements: Node.js 20 or newer, pnpm 11.

```sh
pnpm dev            # watch the library and run the demo
pnpm build          # build every workspace package
pnpm test           # run type checks and unit tests
pnpm test:e2e       # run Playwright tests
pnpm lint           # check source with Oxlint
pnpm format         # format the repository with Oxfmt
```

## License

[MIT](LICENSE) © 2026 Ben Houston
