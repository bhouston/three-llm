"""Summarize paired browser benchmarks; timings remain machine-specific."""
import json
import random
import statistics
from pathlib import Path

rng = random.Random(1729)

def summarize(samples):
    ratios = [before / after for before, after in samples]
    boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(5000))
    return {'before_ms': statistics.median(x[0] for x in samples),
            'after_ms': statistics.median(x[1] for x in samples),
            'paired_speedup': statistics.median(ratios),
            'bootstrap_95_percent': [boot[125], boot[4875]]}

root = Path(__file__).resolve().parents[1] / 'profile-output'
norm = json.loads((root / 'astra-normalization.json').read_text())
submission = json.loads((root / 'astra-submission.json').read_text())
result = {'adapter': norm['adapter'], 'user_agent': norm['userAgent'],
          'normalization': [{'kind': row['kind'], 'size': row['size'], **summarize(row['samples'])}
                            for row in norm['results']],
          'attention_submission': summarize(submission['samples']),
          'delta_submission': summarize(submission['delta']['samples']),
          'submissions': submission['submissions']}
(root / 'astra-summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
