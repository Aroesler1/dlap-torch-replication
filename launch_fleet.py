"""Launch detached per-seed training processes on the GPU.  Usage:
   python launch_fleet.py gan:0-8 unc:0-8 ...        (config name : seed range, one process per seed)
   python launch_fleet.py sim1 sim2                   (simulations)
Writes <logdir>/proc_seed<k>.log and fleet_pids.json; processes survive this shell closing."""
import os, sys, json, subprocess, time
PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'Scripts', 'python.exe')
CFG = {'gan': 'configs/gan.json', 'unc': 'configs/unc.json', 'gan_nomacro': 'configs/gan_nomacro.json',
       'gan_allmacro_raw': 'configs/gan_allmacro_raw.json', 'ext_turnover': 'configs/ext_turnover.json',
       'ext_positive': 'configs/ext_positive_sdf.json', 'quick': 'configs/quick.json'}
FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS
pids = {}
for job in sys.argv[1:]:
    if job.startswith('sim'):
        setup = job[3]; logdir = f'output/sim{setup}'; os.makedirs(logdir, exist_ok=True)
        cmd = [PY, 'simulate.py', '--setup', setup, '--seeds', '3']
        out = open(f'{logdir}/proc.log', 'w')
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, creationflags=FLAGS)
        pids[job] = p.pid; print(job, p.pid); continue
    if '=' in job:   # name=0-8 -> ONE process running seeds a..b serially (clean log, ensemble eval at the end)
        name, rng = job.split('='); a, b = (rng.split('-') + [rng])[:2]
        logdir = f'output/{name}'; os.makedirs(logdir, exist_ok=True)
        cmd = [PY, 'run.py', '--config', CFG[name], '--logdir', logdir, '--seed_start', a, '--seeds', str(int(b) - int(a) + 1)]
        out = open(f'{logdir}/proc_seeds{a}-{b}.log', 'w')
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, creationflags=FLAGS)
        pids[f'{name}={a}-{b}'] = p.pid; print(name, a, b, p.pid); time.sleep(2); continue
    name, rng = job.split(':'); a, b = (rng.split('-') + [rng])[:2]
    logdir = f'output/{name}'; os.makedirs(logdir, exist_ok=True)
    for k in range(int(a), int(b) + 1):
        cmd = [PY, 'run.py', '--config', CFG[name], '--logdir', logdir, '--seed_start', str(k), '--seeds', '1']
        out = open(f'{logdir}/proc_seed{k}.log', 'w')
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, creationflags=FLAGS)
        pids[f'{name}:{k}'] = p.pid; print(name, k, p.pid)
        time.sleep(2)
old = json.load(open('fleet_pids.json')) if os.path.exists('fleet_pids.json') else {}
old.update(pids); json.dump(old, open('fleet_pids.json', 'w'), indent=1)
