"""Generate independent, offline test vectors. Run with requirements.txt's pinned versions."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy
import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers import modeling_rope_utils

assert torch.__version__.split('+')[0] == '2.7.0'
assert transformers.__version__ == '4.51.3'
assert numpy.__version__ == '2.2.5'
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
root = Path(__file__).resolve().parents[2]

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

base = dict(hidden_size=32, intermediate_size=48, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=17,
            max_position_embeddings=64, rope_theta=10000., rms_norm_eps=1e-5,
            tie_word_embeddings=False, attention_bias=False, mlp_bias=False,
            hidden_act='silu', bos_token_id=1, eos_token_id=0, attention_dropout=0.)
variants = [
    ('default', None),
    ('linear', dict(rope_type='linear', factor=4.)),
    ('yarn', dict(rope_type='yarn', factor=8., original_max_position_embeddings=16,
                  beta_fast=2., beta_slow=0.25)),
    ('yarn_custom', dict(rope_type='yarn', factor=4., beta_fast=4., beta_slow=0.5, attention_factor=1.3)),
    ('llama3', dict(rope_type='llama3', factor=4., original_max_position_embeddings=16,
                    low_freq_factor=1., high_freq_factor=4.)),
]
ids = [1, 5, 2, 9, 3, 12, 4, 7, 6, 11, 8, 15, 10, 13, 2, 16, 3, 7, 9]
fixture = {'metadata': {'transformers': transformers.__version__, 'torch': torch.__version__,
                       'numpy': numpy.__version__,
                       'transformers_commit': '5f4ecf2d9f867a1255131d2461d75793c0cf1db2',
                       'rope_source_sha256': hashlib.sha256(Path(inspect.getfile(modeling_rope_utils)).read_bytes()).hexdigest(),
                       'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       'tokenizer': 'none: direct token IDs isolate inference from tokenization',
                       'attention': 'eager', 'dtype': 'float32', 'device': 'cpu'},
           'input_ids': ids, 'cases': []}
for name, scaling in variants:
    config = LlamaConfig(**base, rope_scaling=scaling)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        for i, (key, parameter) in enumerate(model.named_parameters()):
            values = torch.sin(torch.arange(parameter.numel(), dtype=torch.float32) * 0.17 + i * 0.53)
            parameter.copy_((1 + values * .05 if 'norm' in key else values * .15).reshape(parameter.shape))
        tensors = {key: {'shape': list(value.shape), 'values': value.flatten().tolist()} for key, value in model.state_dict().items()}
        output = model(torch.tensor([ids]), output_hidden_states=True, use_cache=False)
        # Verify the upstream full and cached implementations themselves agree.
        past = None
        cached = []
        for token in ids:
            step = model(torch.tensor([[token]]), past_key_values=past, use_cache=True)
            past = step.past_key_values
            cached.append(step.logits[0, 0])
        torch.testing.assert_close(torch.stack(cached), output.logits[0], atol=2e-6, rtol=2e-5)
        positions = [0, 1, 15, 16, 17, 63]
        q = torch.sin(torch.arange(len(positions)*8, dtype=torch.float32)*.23).reshape(1, 1, len(positions), 8)
        k = torch.cos(torch.arange(len(positions)*8, dtype=torch.float32)*.31).reshape_as(q)
        cos, sin = model.model.rotary_emb(q, torch.tensor([positions]))
        rq, rk = apply_rotary_pos_emb(q, k, cos, sin)
        case_config = {**base, 'model_type': 'llama', 'rope_scaling': scaling}
        fixture['cases'].append({'name': name, 'config': case_config, 'config_sha256': digest(case_config),
            'logits': output.logits[0].tolist(), 'final_hidden': output.hidden_states[-1][0].tolist(),
            'rope': {'positions': positions, 'inv_freq': model.model.rotary_emb.inv_freq.tolist(),
                     'attention_factor': model.model.rotary_emb.attention_scaling,
                     'query': q[0, 0].tolist(), 'key': k[0, 0].tolist(),
                     'rotated_query': rq[0, 0].tolist(), 'rotated_key': rk[0, 0].tolist()}})
        # Simulate this backend's FP16 storage contract: embeddings stay FP32,
        # other weights are rounded once, then all runtime arithmetic is FP32.
        for key, parameter in model.named_parameters():
            if key != 'model.embed_tokens.weight':
                parameter.copy_(parameter.half().float())
        narrowed = model(torch.tensor([ids]), output_hidden_states=True, use_cache=False)
        fixture['cases'][-1]['fp16_storage_logits'] = narrowed.logits[0].tolist()
        fixture['cases'][-1]['fp16_storage_final_hidden'] = narrowed.hidden_states[-1][0].tolist()
        if 'tensors' not in fixture:
            fixture['tensors'] = tensors
            fixture['metadata']['weights_sha256'] = digest(tensors)
        else:
            assert digest(tensors) == fixture['metadata']['weights_sha256']
path = root / 'packages/three-llm/src/test/fixtures/transformers-llama.json'
path.write_text(json.dumps(fixture, separators=(',', ':')) + '\n')
path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest() + '\n')
print(f'{path}: {path.stat().st_size} bytes; sha256={hashlib.sha256(path.read_bytes()).hexdigest()}')
