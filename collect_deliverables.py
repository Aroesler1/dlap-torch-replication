"""Gather everything for the presentation into deliverables/ and zip it."""
import os, shutil, glob, subprocess, sys
PY = sys.executable
D = 'deliverables'; shutil.rmtree(D, ignore_errors=True); os.makedirs(D)
runs = ['output/gan', 'output/unc', 'output/gan_nomacro', 'output/gan_allmacro_raw', 'output/ext_turnover', 'output/ext_positive']
md = subprocess.run([PY, 'summarize.py'] + runs, capture_output=True, text=True).stdout
open(f'{D}/summary.md', 'w').write(md); shutil.copy('summary.csv', f'{D}/summary.csv')
for r in glob.glob('output/*/results.json'):
    shutil.copy(r, f'{D}/results_{os.path.basename(os.path.dirname(r))}.json')
for s in ('sim1', 'sim2'):
    if os.path.exists(f'output/{s}/table2.json'): shutil.copy(f'output/{s}/table2.json', f'{D}/{s}_table2.json')
if os.path.exists('output/sim2/hidden_state_recovery.png'): shutil.copy('output/sim2/hidden_state_recovery.png', D)
for r in ('gan', 'ext_turnover', 'ext_positive'):
    os.makedirs(f'{D}/plots_{r}', exist_ok=True)
    for p in glob.glob(f'output/{r}/plots/*.png'): shutil.copy(p, f'{D}/plots_{r}/')
if os.path.exists('notes.md'): shutil.copy('notes.md', D)
print(md); print(sorted(os.listdir(D)))
shutil.make_archive('deliverables', 'zip', D); print('deliverables.zip', os.path.getsize('deliverables.zip'))
