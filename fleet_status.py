"""Cheap status: which seeds have checkpoints, which procs are alive, last log line per proc, GPU util."""
import os, json, glob, subprocess, ctypes
pids = json.load(open('fleet_pids.json')) if os.path.exists('fleet_pids.json') else {}
k32 = ctypes.windll.kernel32
def alive(pid):
    h = k32.OpenProcess(0x1000, False, pid)
    if not h: return False
    code = ctypes.c_ulong(); k32.GetExitCodeProcess(h, ctypes.byref(code)); k32.CloseHandle(h)
    return code.value == 259
for job, pid in pids.items():
    if job.startswith('sim'):
        log = f'output/{job}/proc.log'
    elif '=' in job:
        name, rng = job.split('='); log = f'output/{name}/proc_seeds{rng}.log'
    else:
        name, k = job.split(':'); log = f'output/{name}/proc_seed{k}.log'
    last = ''
    if os.path.exists(log):
        lines = [l.rstrip() for l in open(log, errors='replace') if l.strip()]
        last = lines[-1][:110] if lines else ''
    if job.startswith('sim'): done = os.path.exists(f'output/{job}/table2.json')
    elif '=' in job: done = os.path.exists(f'output/{name}/results.json')
    else: done = os.path.exists(f'output/{name}/seed_{k}/checkpoint.pt')
    print(f"{job:22s} pid={pid:6d} {'ALIVE' if alive(pid) else 'exit '} {'ckpt' if done else '    '}  {last}")
try:
    print(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip())
except Exception as e: print(e)
