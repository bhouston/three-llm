# three-llm

[![CI](https://github.com/bhouston/three-llm/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bhouston/three-llm/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/bhouston/three-llm/branch/main/graph/badge.svg)](https://codecov.io/gh/bhouston/three-llm)

[![npm version](https://img.shields.io/npm/v/three-llm.svg)](https://www.npmjs.com/package/three-llm)
[![live demo](https://img.shields.io/badge/demo-three--llm.ben3d.ca-blue)](https://three-llm.ben3d.ca)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/fwupDN493R)

Run large language models in the browser with WebGPU. `three-llm` implements transformer inference with [Three.js](https://threejs.org/) and its TSL compute shader system, so model execution stays on the user's GPU without a server-side inference runtime.

**[Try the live demo: threekit-llm.ben3d.ca](https://three-llm.ben3d.ca)** · **[Read the technical write-up](https://ben3d.ca/blog/running-llms-in-the-browser-with-threejs)**

<img src="three-llm-chat.webp" alt="Demo App Interface" width="500" />

See [packages/three-llm/README.md](packages/three-llm/README.md) for full documentation, including features, requirements, install, and usage.

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

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for the issue, branch, PR, and release workflow and [SECURITY.md](SECURITY.md) for private vulnerability reporting.
