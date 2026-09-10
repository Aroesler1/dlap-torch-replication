#!/usr/bin/env python
"""Collect results.json from several logdirs into one comparison table (markdown + csv)."""
import sys, os, json
rows = []
for d in sys.argv[1:]:
    p = os.path.join(d, 'results.json')
    if not os.path.exists(p): continue
    r = json.load(open(p)); t = r['test']; v = r['valid']; tr = r['train']
    rows.append((os.path.basename(d.rstrip('/')), r['n_seeds'],
                 tr['SR_monthly'], v['SR_monthly'], t['SR_monthly'], t['SR_annual'],
                 t.get('EV_beta', float('nan')), t.get('XS_R2_weighted_beta', float('nan')), t.get('XS_R2_beta', float('nan')),
                 t['turnover_mean'], list(t['SR_annual_net_of_cost'].values())[-1], t['max_drawdown']))
hdr = ['model', 'seeds', 'SR train', 'SR valid', 'SR test (monthly)', 'SR test (annual)', 'EV test', 'XS-R2 test (T_i-weighted, paper def.)', 'XS-R2 test (unweighted)', 'turnover', 'net SR (worst cost)', 'max DD']
print('| ' + ' | '.join(hdr) + ' |'); print('|' + '---|' * len(hdr))
for r in rows:
    print('| ' + ' | '.join([r[0], str(r[1])] + [f'{x:.2f}' for x in r[2:]]) + ' |')
print('| paper: GAN (hidden states) | 9 | 2.68 | 1.43 | 0.75 | 2.60 | 0.08 | 0.23 | – | – | – | – |')
print('| paper: UNC | 9 | 1.93 | 1.33 | 0.53 | 1.84 | 0.07 | 0.19 | – | – | – | – |')
print('| paper: GAN (no macro) | 9 | 1.90 | 1.35 | 0.69 | 2.39 | – | – | – | – | – | – |')
print('| paper: FFN forecast (no macro) | 9 | 0.45 | 0.42 | 0.44 | 1.52 | 0.04 | 0.15 | – | – | – | – |')
with open('summary.csv', 'w') as f:
    f.write(','.join(hdr) + '\n')
    for r in rows: f.write(','.join(map(str, r)) + '\n')
