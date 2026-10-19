"""Plot relaxation activity, complete-window criterion, density and tensor strain."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='output_relax')
    args=parser.parse_args()
    base=Path(__file__).resolve().parent
    folder=base/args.input
    with (folder/'relax.csv').open(encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    # A live writer may leave one incomplete trailing record.
    rows=[r for r in rows if r.get('converged') not in (None,'')]
    if not rows: raise ValueError('No complete relaxation records')
    def values(key): return np.array([float(r[key]) for r in rows])
    t=values('time_s')*1e9
    summary_file=folder/'summary.json'
    summary=json.loads(summary_file.read_text(encoding='utf-8')) if summary_file.exists() else {}
    threshold=summary.get('tolerance',1e-7)
    old_activity_criterion = summary.get('criterion', '').startswith('sum of per-step')
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    axs[0,0].plot(t,values('density_m-2'))
    axs[0,0].set_ylabel('Driver density (m$^{-2}$)')
    complete=[r for r in rows if r['window_activity']]
    if complete:
        tw=[float(r['time_s'])*1e9 for r in complete]
        axs[0,1].plot(tw,[float(r['window_net_norm']) for r in complete],'o-',label='Net tensor norm' + (' (diagnostic)' if old_activity_criterion else ' (criterion)'))
        axs[0,1].plot(tw,[float(r['window_activity']) for r in complete],'s-',label='Sum of step norms' + (' (old criterion)' if old_activity_criterion else ' (diagnostic)'))
    axs[0,1].axhline(threshold,color='red',ls='--',label='1e-7 threshold')
    axs[0,1].set_yscale('symlog',linthresh=threshold*.01)
    axs[0,1].set_ylabel('Full-window plastic strain increment')
    axs[0,1].legend(fontsize=8)
    for key in ('ep_xx','ep_yy','ep_zz','ep_yz','ep_xz','ep_xy'):
        axs[1,0].plot(t,values(key),label=key)
    axs[1,0].set_ylabel('Cumulative plastic strain (tensor components)')
    axs[1,0].legend(fontsize=8,ncol=3)
    axs[1,1].plot(t,values('cumulative_activity'))
    axs[1,1].set_ylabel('Cumulative sum of tensor increment norms')
    for ax in axs.flat: ax.set_xlabel('Relaxation time (ns)');ax.grid(alpha=.25)
    status='CONVERGED' if summary.get('converged') else ('NOT CONVERGED' if summary else 'RUNNING / incomplete')
    fig.suptitle(f'Zero-stress relaxation | {status} | windows >= 10 ns')
    out=base/'plots';out.mkdir(exist_ok=True)
    fig.savefig(out/'relaxation.png',dpi=250);plt.close(fig)
    print(out/'relaxation.png')
if __name__ == '__main__': main()
