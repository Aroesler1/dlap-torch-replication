"""Run this first on the VM: prints the shapes of the downloaded npz files and flags anything the code won't like."""
import numpy as np, os, sys
ok = True
for f in ['char/Char_train', 'char/Char_valid', 'char/Char_test', 'macro/macro_train', 'macro/macro_valid', 'macro/macro_test']:
    p = f'datasets/{f}.npz'
    if not os.path.exists(p):
        print('MISSING', p); ok = False; continue
    d = np.load(p, allow_pickle=True)
    print(f'{p:32s} keys={d.files}  data.shape={d["data"].shape}  dtype={d["data"].dtype}')
    if 'macro' in f and d['data'].shape[1] not in (178,):
        print(f'   -> macro file has {d["data"].shape[1]} columns, not 178. If it is 338, add "macro_idx": "178" to every config in configs/.')
    if 'char' in f:
        r = d['data'][:, :, 0]; m = r != -99.99
        print(f'   months={d["data"].shape[0]}  permnos={d["data"].shape[1]}  chars={d["data"].shape[2]-1}  avg stocks/month={m.sum(1).mean():.0f}  dates {d["date"][0]}..{d["date"][-1]}')
print('\nAll files present.' if ok else '\nFix the missing files before running.')
