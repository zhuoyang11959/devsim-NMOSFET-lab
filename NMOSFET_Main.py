"""Unified entry point. Spyder: open this file and press F5.
Notebook: import NMOSFET_Main as demo, then demo.run().
"""
from pathlib import Path
import json,time,csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import nmos_lab as lab

# %% Settings: edit here in Spyder, or set these variables from the notebook.
RUN_CURVES = True
RUN_MAPS = True
REL_ERROR = 1e-5
BIAS_STEP = 0.25
VG_VALUES = np.linspace(0,2,21)
VD_VALUES = np.linspace(0,2,21)
OUTPUT_GATES = [1.0,1.5,2.0]
TRANSFER_DRAIN = 0.05
STATES = [('Cutoff',0.,.05),('Linear',1.,.05),
          ('Near pinch-off',1.,.7),('Saturation',1.,1.5)]
OUT = Path(__file__).resolve().parent / 'simulation_results'

def run():
    """Run selected tasks. Every invocation starts from a fresh device."""
    if not RUN_CURVES and not RUN_MAPS:
        raise ValueError('Enable RUN_CURVES or RUN_MAPS')
    OUT.mkdir(exist_ok=True)
    lab.REL_ERROR=REL_ERROR
    start=time.perf_counter();maps={};result={}
    def bias(c,v):lab.bias(c,float(v),step=BIAS_STEP)
    def capture(vg,vd):
        if not RUN_MAPS:return
        for name,g,d in STATES:
            if abs(vg-g)>1e-9 or abs(vd-d)>1e-9:continue
            ds=lab.ds
            a={k:np.asarray(ds.get_node_model_values(device='mos2d',region='bulk',name=n))
               for k,n in [('x','x'),('y','y'),('n','Electrons'),('p','Holes'),('psi','Potential')]}
            a['elements']=np.asarray(ds.get_element_node_list(device='mos2d',region='bulk'),dtype=int)
            maps[name]=a
            np.savez(OUT/('carriers_'+name.replace(' ','_').replace('-','_')+'.npz'),**a)
    def sweep(c,values,fixed):
        rows=[]
        for v in values:
            bias(c,v);rows.append({'bias':float(v),**lab.currents()})
            capture(float(v) if c=='gate' else fixed, fixed if c=='gate' else float(v))
        return rows
    if RUN_CURVES:
        lab.build();bias('drain',TRANSFER_DRAIN)
        result['transfer']=sweep('gate',VG_VALUES,TRANSFER_DRAIN)
        result['output']={}
        for vg in OUTPUT_GATES:
            lab.build();bias('gate',vg)
            result['output'][str(vg)]=sweep('drain',VD_VALUES,vg)
        for name,rows in [('transfer',result['transfer'])]+[(f'output_VG_{g}',rr) for g,rr in result['output'].items()]:
            with (OUT/(name+'.csv')).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        v=[a['bias'] for a in result['transfer']];i=100*np.abs([a['drain'] for a in result['transfer']])
        axs[0].plot(v,i,'o-');axs[1].semilogy(v,np.maximum(i,1e-30),'o-')
        for ax in axs:ax.set(xlabel='VG (V)',ylabel='|ID| (µA for W=1 µm)');ax.grid(True,alpha=.3)
        fig.suptitle(f'Transfer curve: VD={TRANSFER_DRAIN:g} V');fig.savefig(OUT/'Id_Vg.png',dpi=180)
        fig,ax=plt.subplots(layout='constrained')
        for g,rows in result['output'].items():ax.plot([a['bias'] for a in rows],100*np.abs([a['drain'] for a in rows]),'o-',label=f'VG={g} V')
        ax.set(xlabel='VD (V)',ylabel='|ID| (µA for W=1 µm)');ax.grid(True,alpha=.3);ax.legend();fig.savefig(OUT/'Id_Vd.png',dpi=180)
    if RUN_MAPS:
        # Reuse states solved in the current sweeps. Solve only missing states.
        missing=[s for s in STATES if s[0] not in maps]
        if missing:
            lab.build()
            for name,vg,vd in missing:bias('gate',vg);bias('drain',vd);capture(vg,vd)
        fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for ax,(name,vg,vd) in zip(axs.flat,STATES):
            a=maps[name];tr=mtri.Triangulation(a['x']*1e4,a['y']*1e7,a['elements']);z=np.log10(np.maximum(a['n'],1))
            im=ax.tripcolor(tr,z,shading='gouraud',vmin=8,vmax=19,cmap='viridis')
            ax.tricontour(tr,z,levels=[17],colors='white',linewidths=.8);ax.plot([1,2],[0,0],color='#df7c21',lw=4)
            ax.set(xlim=(.85,2.15),ylim=(60,0),xlabel='x (µm)',ylabel='Depth (nm)',title=f'{name}: VG={vg:g}, VD={vd:g} V')
        fig.colorbar(im,ax=axs,label='log10 n (cm⁻³)');fig.suptitle('Source left, drain right. White: n=NA. Vertical scale enlarged.')
        fig.savefig(OUT/'Channel_carrier_maps.png',dpi=180)
        fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
        for name,vg,vd in STATES:
            a=maps[name];mask=(abs(a['y'])<1e-12)&(a['x']>=1e-4)&(a['x']<=2e-4);idx=np.argsort(a['x'][mask])
            ax.semilogy(a['x'][mask][idx]*1e4,a['n'][mask][idx],label=name)
        ax.axhline(1e17,color='gray',ls='--',label='NA');ax.set(xlabel='x (µm)',ylabel='Surface n (cm⁻³)');ax.legend();ax.grid(True,alpha=.3)
        fig.savefig(OUT/'Channel_surface_profiles.png',dpi=180)
    result['settings']={'relative_tolerance':REL_ERROR,'bias_step':BIAS_STEP,'maps':RUN_MAPS,'curves':RUN_CURVES}
    result['elapsed_seconds']=time.perf_counter()-start
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print(f"Finished in {result['elapsed_seconds']:.1f} s. Results: {OUT}")
    plt.show()
    return result

# %% Main entry point
if __name__=='__main__':
    results=run()
