"""Wait for the existing downloader; then resume only unfinished chunks."""
from common import *
import argparse, psutil, subprocess, time, traceback
p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int,required=True);args=p.parse_args()
statefile=R/'logs/download_recovery_state.json';rec=R/'recovery/download_retry_20261013T0402'
meta=dict(status='WAITING_EXISTING_DOWNLOADER',pid=os.getpid(),original_pid=args.wait_pid,started_utc=utc(),network_workers_started=0);save(statefile,meta)
try:
    try:
        original=psutil.Process(args.wait_pid)
        assert 'acquire_parallel.py' in ' '.join(original.cmdline()),'PID no longer belongs to original downloader'
        while original.is_running():
            try:original.wait(timeout=10);break
            except psutil.TimeoutExpired:pass
    except psutil.NoSuchProcess:pass
    for cycle in range(1,4):
        current=json.loads((R/'logs/download_state.json').read_text())
        if current['status']=='COMPLETED':
            meta.update(status='COMPLETED',ended_utc=utc());save(statefile,meta);sys.exit(0)
        assert current['status']=='FAILED', 'Original downloader did not finish with a recorded failure'
        # Exclude ourselves and verify no other downloader process remains alive.
        for proc in psutil.process_iter(['pid','name','cmdline']):
            line=' '.join(proc.info['cmdline'] or [])
            if (proc.info['name'] or '').lower() in ['python.exe','pythonw.exe'] and proc.pid!=os.getpid() and str(R).lower() in line.lower() and ('acquire_parallel.py' in line or 'acquire_parallel_retry.py' in line):
                raise RuntimeError('Another downloader is alive; do not duplicate')
        save(rec/f'cycle_{cycle}_prior_download_state.json',current)
        current.update(pid=os.getpid(),status='RECOVERING',updated_utc=utc());save(R/'logs/download_state.json',current)
        meta.update(status='RECOVERING',cycle=cycle,updated_utc=utc());save(statefile,meta)
        env=os.environ.copy();env['DOWNLOAD_SUPERVISOR_PID']=str(os.getpid())
        with (R/'logs'/f'download_recovery_cycle{cycle}.log').open('w',encoding='utf8') as log:
            child=subprocess.Popen([sys.executable,'-X','utf8','-u',str(R/'scripts/acquire_parallel_retry.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            meta.update(child_pid=child.pid,network_workers_started=cycle);save(statefile,meta)
            code=child.wait()
        current=json.loads((R/'logs/download_state.json').read_text());meta.update(last_exit_code=code,updated_utc=utc());save(statefile,meta)
        if code==0 and current['status']=='COMPLETED':
            meta.update(status='COMPLETED',ended_utc=utc());save(statefile,meta);sys.exit(0)
        if 'Archive size/checksum mismatch' in current.get('error',''):raise RuntimeError('Archive checksum mismatch requires investigation')
        if cycle<3:time.sleep(60)
    raise RuntimeError('Bounded download recovery exhausted; partial files retained')
except BaseException as error:
    if isinstance(error,SystemExit) and error.code==0:raise
    meta.update(status='FAILED',error=repr(error),traceback=traceback.format_exc(),ended_utc=utc());save(statefile,meta);raise
