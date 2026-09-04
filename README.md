# vgpu-llm

[![npm version](https://img.shields.io/npm/v/vgpu-llm.svg)](https://www.npmjs.com/package/vgpu-llm)
[![live demo](https://img.shields.io/badge/demo-three--llm.ben3d.ca-blue)](https://three-llm.ben3d.ca)

Run large language models in the browser with WebGPU. `vgpu-llm` implements transformer inference as
[`vgpu`](https://github.com/vercel-labs/vgpu) WGSL compute kernels, so model execution stays on the user's GPU
without a server-side inference runtime.

**[Try the live demo: three-llm.ben3d.ca](https://three-llm.ben3d.ca)** · **[Background: the original
Three.js-based write-up](https://ben3d.ca/blog/running-llms-in-the-browser-with-threejs)** (the engine has since
moved from Three.js/TSL to `vgpu`)

<img src="three-llm-chat.webp" alt="Demo App Interface" width="500" />

## Features

- WebGPU inference through `vgpu` WGSL compute kernels
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
pnpm add vgpu-llm vgpu
```

## Usage

Initialize a `vgpu` context, load a compatible Hugging Face checkpoint, and generate text:

```ts
import { createGpuRunner } from 'vgpu-llm';
import { init } from 'vgpu';

const gpu = await init();

const runner = await createGpuRunner(gpu, 'https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/', {
  onProgress: console.log,
  prefillChunkSize: 4,
});

const result = await runner.generate('Once upon a time,', {
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

For multi-turn chat, pass formatted messages with `formatPrompt` from `vgpu-llm`. For catalog entries and URL resolution, import `MODEL_CATALOG` and `resolveModelURL` from `vgpu-llm/catalog`.

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

- `packages/vgpu-llm`: the inference library, model loaders, tokenizers, and `vgpu` WGSL kernels
- `packages/website`: the React chat demo

Requirements: Node.js 20 or newer, pnpm 11.

```sh
pnpm dev            # watch the library and run the demo
pnpm build          # build every workspace package
pnpm test           # run type checks and unit tests
pnpm test:gpu       # run real-GPU kernel/runner tests via vgpu/node (Dawn), no browser needed
pnpm test:checkpoints # run CPU + real-GPU tests against downloaded checkpoints
pnpm test:e2e       # run Playwright tests
pnpm lint           # check source with Oxlint
pnpm format         # format the repository with Oxfmt
```

## License

[MIT](LICENSE) © 2026 Ben Houston
