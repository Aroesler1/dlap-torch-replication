#!/bin/bash
# Full run plan for the H100 box. Each baseline seed ~5-10 min; adjust --seeds if short on time.
set -e
python run.py --config configs/gan.json           --logdir output/gan            --seeds 9        # headline replication
python run.py --config configs/unc.json           --logdir output/unc            --seeds 9        # ablation: no adversary
python run.py --config configs/gan_nomacro.json   --logdir output/gan_nomacro    --seeds 9        # ablation: no macro
python run.py --config configs/gan_allmacro_raw.json --logdir output/gan_allmacro_raw --seeds 4  # ablation: raw macro, no LSTM
python run.py --config configs/ext_turnover.json  --logdir output/ext_turnover   --seeds 9        # extension 1
python run.py --config configs/ext_positive_sdf.json --logdir output/ext_positive --seeds 9      # extension 2
python summarize.py output/*                                                                     # one table for the slides
