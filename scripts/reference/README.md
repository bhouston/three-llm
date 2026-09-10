# Independent inference fixtures

`generate.py` imports **Transformers 4.51.3**, source commit
`5f4ecf2d9f867a1255131d2461d75793c0cf1db2`, and **PyTorch 2.7.0**. It never
imports the TypeScript inference engine or its math. No model download is needed.

Generate in an isolated Python 3.12 environment:

```sh
uv venv --python 3.12 /tmp/three-llm-reference
uv pip install --python /tmp/three-llm-reference/bin/python -r scripts/reference/requirements.txt
/tmp/three-llm-reference/bin/python scripts/reference/generate.py
pnpm test:unit
pnpm test:gpu
pnpm test:browser
```

The checked-in fixture contains the exact deterministic synthetic model weights,
configuration and weight hashes, generator/source hashes, direct input token IDs,
all 19 positions' logits/final hidden states, and rotated Q/K at positions
0, 1, 15, 16, 17 and 63. It exercises default RoPE, linear, YaRN (including the
original-context ratio), explicit YaRN attention scaling, and Llama 3 scaling.
The generator also compares upstream cached and full-sequence logits.

Separate FP16-storage expectations round non-embedding weights with PyTorch
before running FP32 inference. Embeddings remain FP32, matching the library's
storage contract. This avoids treating quantization error as kernel error.
The SHA256 companion file and generator hash are checked in Vitest. Updating a
fixture requires deliberately regenerating both; do not regenerate fixtures to
accommodate unexplained implementation errors.
The generated JSON is excluded from Oxfmt so formatting hooks preserve its
byte-level hash.

These fixtures validate inference math and tensor mapping, not tokenization,
real-checkpoint quality, or every architecture. They contain no pretrained model
weights. The reference uses Transformers' eager attention backend on CPU. All
normal tests consume checked-in JSON and do not require Python.
