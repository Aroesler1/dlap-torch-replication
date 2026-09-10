# Full run plan for the Windows H100 VM. Run from PowerShell inside the dlap_torch folder:
#   .\run_all.ps1
# Each baseline seed takes ~5-10 min on an H100. Ctrl-C at any point keeps everything finished so far.
$ErrorActionPreference = "Continue"
python run.py --config configs/gan.json               --logdir output/gan             --seeds 9   # headline replication
python run.py --config configs/unc.json               --logdir output/unc             --seeds 9   # ablation: no adversary
python run.py --config configs/gan_nomacro.json       --logdir output/gan_nomacro     --seeds 9   # ablation: no macro
python run.py --config configs/gan_allmacro_raw.json  --logdir output/gan_allmacro_raw --seeds 4  # ablation: raw macro, no LSTM
python run.py --config configs/ext_turnover.json      --logdir output/ext_turnover    --seeds 9   # extension 1: cost-aware
python run.py --config configs/ext_positive_sdf.json  --logdir output/ext_positive    --seeds 9   # extension 2: positive SDF
python simulate.py --setup 1 --seeds 3                                                             # Table II
python simulate.py --setup 2 --seeds 3
python summarize.py output/gan output/unc output/gan_nomacro output/gan_allmacro_raw output/ext_turnover output/ext_positive
