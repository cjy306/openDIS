"""Compare both groups at loading total strain; run after both jobs finish."""
import io
import argparse
from pathlib import Path
import numpy as np

def parse_response(text):
    if text and not text.endswith('\n'):
        text = text.rsplit('\n',1)[0]+'\n' if '\n' in text else ''
    data = np.loadtxt(io.StringIO(text), comments='#', ndmin=2)
    if data.shape[0] == 0 or data.shape[1] < 4 or not np.isfinite(data[:,1:4]).all():
        raise ValueError('Empty or invalid loading data')
    if np.any(np.diff(data[:,1]) < 0):
        raise ValueError('Nonmonotone strain; inspect restart overlap')
    return data[:,1]*100, data[:,2]/1e6, data[:,3]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='output')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base = Path(__file__).resolve().parent
    missing = [str(base/case/args.input/'stress_strain_dens.dat')
               for case in ('no_loops','with_loops')
               if not (base/case/args.input/'stress_strain_dens.dat').is_file()]
    if missing:
        raise FileNotFoundError('Both groups are required; missing: '+', '.join(missing))
    fig,axs = plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    for case,label,color in [('no_loops','FR only','black'),('with_loops','FR + loops','tab:red')]:
        path = base/case/args.input/'stress_strain_dens.dat'
        e,s,r = parse_response(path.read_text(encoding='utf-8'))
        if e[-1] < 0.5:
            print(f'WARNING: {case} incomplete: final total strain {e[-1]:.6g}%')
        for ax,y in zip(axs,[s,r,r-r[0]]):
            ax.plot(e,y,label=label,color=color,lw=1)
        print(f'{case}: first recorded rho={r[0]:.6e}, last strain={e[-1]:.6g}%')
    for ax,title in zip(axs,['Axial stress (MPa)','Total line density (m$^{-2}$)',
                             'Density change from first loading record (m$^{-2}$)']):
        ax.set(xlabel='Loading total strain (%)',ylabel=title)
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('BCC Fe mobility approximation | 1 um | 300 K | seed 12345')
    out = base/'plots'
    out.mkdir(exist_ok=True)
    fig.savefig(out/'loops_compare.png',dpi=250)
    plt.close(fig)
    print(out/'loops_compare.png')

if __name__ == '__main__':
    main()
