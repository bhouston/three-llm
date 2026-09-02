# three-llm

Run large language models in the browser with WebGPU. `three-llm` implements transformer inference with [Three.js](https://threejs.org/) and its TSL compute shader system, so model execution stays on the user's GPU without a server-side inference runtime.

The project includes a TypeScript library and a browser chat demo. It loads Hugging Face model configs, tokenizers, and SafeTensors weights directly.

## Features

- WebGPU inference through Three.js TSL compute shaders
- CPU reference runners for testing and validation
- Prompt caching, chunked prefill, streaming token callbacks, and GPU sampling
- GPT-2, Llama-style, Gemma 3, Phi, and Qwen 3.5 decoder architectures
- GPT-2 BPE, Qwen BPE, and unigram tokenizers
- Direct loading of Hugging Face SafeTensors checkpoints

The included model catalog covers TinyStories GPT-2 3M, GPT-2 124M, SmolLM2 135M, Qwen3.5 0.8B, and Phi-1.5.

## Requirements

- A browser with WebGPU support, such as a recent Chrome, Edge, or Safari release
- Node.js 20 or newer
- pnpm 11
- Enough device memory for the selected model and its intermediate buffers

Model files can range from a few megabytes to several gigabytes. Remote Hugging Face repositories must allow browser CORS requests.

## Run the demo

```sh
corepack enable
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). The demo loads checkpoints from `/api/models/`, which proxies the public `gs://three-llm` bucket, and falls back to Hugging Face if a file is missing.

## Library usage

Install the library with Three.js:

```sh
pnpm add three-llm three
```

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

For catalog entries and URL resolution, import `MODEL_CATALOG` and `resolveModelURL` from `three-llm/catalog`.

## Development

This repository uses a pnpm workspace:

- `packages/three-llm`: the inference library, model loaders, tokenizers, and TSL kernels
- `packages/website`: the React chat demo

Useful commands:

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
