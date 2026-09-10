"""Summarize raw Vitest benchmark samples; standard library only."""
import json
import random
import statistics
from pathlib import Path

random.seed(1729)
for name in ['normalization', 'submission']:
    path = Path(f'profile-output/astra-{name}.json')
    report = json.loads(path.read_text())
    for row in report['rows']:
        before, after = row['baseline'], row['candidate']
        n = len(before)
        estimates = []
        for _ in range(5000):
            indices = random.choices(range(n), k=n)
            if name == 'normalization':
                estimates.append(statistics.median(before[i] / after[i] for i in indices))
            else:
                estimates.append(statistics.median(before[i] for i in indices) /
                                 statistics.median(after[i] for i in indices))
        estimates.sort()
        row['bootstrap95MedianSpeedup'] = [estimates[125], estimates[4874]]
        row['baselineMedianMs'] = statistics.median(before)
        row['candidateMedianMs'] = statistics.median(after)
        row['baselineP95Ms'] = sorted(before)[min(n - 1, int(n * .95))]
        row['candidateP95Ms'] = sorted(after)[min(n - 1, int(n * .95))]
        print(name, row.get('kind', row.get('workload')), row.get('size', ''), row.get('precision', ''),
              f"{row['medianSpeedup']:.2f}x", row['bootstrap95MedianSpeedup'])
    report['statistics'] = '5000 paired bootstrap resamples, seed 1729; local-run uncertainty only, not cross-device confidence'
    path.with_name(f'astra-{name}-summary.json').write_text(json.dumps(report, indent=2) + '\n')
