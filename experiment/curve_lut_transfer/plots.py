"""Scientific figures with exported numerical arrays and explicit units."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from utils.normal_mapping import lookup_slopes_raw
from .metrics import support
from .storage import npz_write


def make_plots(root: Path, prepared: dict, luts: dict, summaries: dict,
               rows: dict, details: dict, datasets: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = root / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    names = sorted(prepared['conditions'], key=lambda n: 0 if n=='flat' else 1/prepared['conditions'][n])
    curvature = np.array([0 if n=='flat' else 1/prepared['conditions'][n] for n in names])
    def value(key):
        v = summaries[key]['rmse']
        return np.nan if v is None else v
    direct = np.array([value(f'flat/{n}/common') for n in names])
    own = np.array([value(f'{n}/{n}/common') for n in names])
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(curvature, direct, 'o-', label='Frozen planar LUT')
    ax.plot(curvature, own, 's--', label='Same-curvature LUT (see validation status)')
    for x, y, n in zip(curvature, own, names):
        if n != 'flat' and summaries[f'{n}/{n}/common'].get('lut_status') != 'passed':
            ax.annotate('diagnostic / failed', (x, y), xytext=(-5,8),
                        textcoords='offset points', ha='right', fontsize=7)
    ax.axhline(direct[0], color='gray', lw=.8, label='Planar independent baseline')
    ax.set(xlabel='Curvature (1/mm)', ylabel='Slope vector RMSE (dimensionless; lower better)',
           title='Common material domain; equal poses, reference area weights')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(folder/'slope_rmse.png', dpi=160); plt.close(fig)
    npz_write(folder/'slope_rmse.npz', curvature=curvature, direct=direct, recalibrated=own)

    matrix = np.array([[value(f'{ln}/{dn}/common') for dn in names] for ln in names])
    fig, ax = plt.subplots(figsize=(6,5))
    im=ax.imshow(np.ma.masked_invalid(matrix), vmin=0, cmap='magma')
    labels=[n + (' [diagnostic]' if n!='flat' and summaries[f'{n}/{n}/common'].get('lut_status')!='passed' else '') for n in names]
    ax.set(xticks=range(len(names)), xticklabels=names, yticks=range(len(names)), yticklabels=labels,
           xlabel='Test condition', ylabel='Calibration condition', title='Slope vector RMSE (lower better)')
    for i in range(len(names)):
        for j in range(len(names)):
            color='black' if np.isfinite(matrix[i,j]) and im.norm(matrix[i,j])>.6 else 'white'
            ax.text(j,i, f'{matrix[i,j]:.3f}' if np.isfinite(matrix[i,j]) else 'N/A', ha='center',color=color)
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(folder/'cross_curvature.png',dpi=160); plt.close(fig)
    npz_write(folder/'cross_curvature.npz', values=matrix, conditions=np.asarray(names))

    us=sorted({p['center_uv'][0] for p in prepared['test']}); vs=sorted({p['center_uv'][1] for p in prepared['test']})
    heat=[]
    for n in names:
        rr=rows[f'flat/{n}/common']
        heat.append(np.array([[np.sqrt(np.mean([r['mse'] for r in rr if r['u']==u and r['v']==v]))
                               if all(r['mse'] is not None for r in rr if r['u']==u and r['v']==v) else np.nan for u in us] for v in vs]))
    vmax=max((float(np.nanmax(h)) for h in heat if np.isfinite(h).any()),default=1.)
    fig, axes=plt.subplots(1,len(names),figsize=(4*len(names),3.7),squeeze=False,layout='constrained')
    for ax,n,h in zip(axes[0],names,heat):
        # Grid coordinates may be nonuniform: show actual normalized arclength cell edges.
        def edges(v):
            return np.r_[v[0]-(v[1]-v[0])/2,(np.array(v[1:])+v[:-1])/2,v[-1]+(v[-1]-v[-2])/2] if len(v)>1 else [v[0]-.05,v[0]+.05]
        im=ax.pcolormesh(edges(us),edges(vs),h,vmin=0,vmax=vmax,cmap='magma')
        ax.set(xlabel='u (normalized arclength)',ylabel='v',title=n+'; frozen planar LUT')
    fig.colorbar(im,ax=axes[0].tolist(),label='Slope RMSE (depths weighted equally)')
    fig.savefig(folder/'position_heatmaps.png',dpi=160); plt.close(fig)
    npz_write(folder/'position_heatmaps.npz',u=np.array(us),v=np.array(vs),values=np.array(heat),conditions=np.asarray(names))

    # Preselected centre/left/right plus worst pose; numbers remain full resolution.
    for n in names:
        key=f'flat/{n}/common'
        poses=prepared['test']
        selected={min(range(len(poses)),key=lambda j: abs(poses[j]['center_uv'][0]-u)+abs(poses[j]['center_uv'][1])) for u in (min(us),0,max(us))}
        good=[j for j,r in enumerate(rows[key]) if r['status']=='ok']
        if good:
            selected.add(max(good,key=lambda j: rows[key][j]['rmse']))
        for j in selected:
            detail=details[key][j]
            if not detail:
                continue
            s=datasets[n][j]; take=np.searchsorted(s['grid_index'],detail['grid_index'])
            xy=s['material_coordinates'][take]; truth=s['slope_samples'][take]; pred=detail['prediction']
            fig,axes=plt.subplots(1,2,figsize=(9,3.6))
            stride=max(1,len(xy)//150)  # Display-only arrow density; all numeric samples are retained.
            peak=max(float(np.linalg.norm(truth,axis=1).max()),float(np.linalg.norm(pred,axis=1).max()),1e-6)
            scale=peak/(.12*max(float(np.ptp(xy,axis=0).max()),1e-3))
            arrow=axes[0].quiver(xy[::stride,0],xy[::stride,1],truth[::stride,0],truth[::stride,1],
                                angles='xy',scale_units='xy',scale=scale,color='black',label='Truth')
            axes[0].quiver(xy[::stride,0],xy[::stride,1],pred[::stride,0],pred[::stride,1],
                            angles='xy',scale_units='xy',scale=scale,color='tab:red',alpha=.6,label='Frozen LUT')
            axes[0].quiverkey(arrow,.15,.95,.5,'0.5 slope',labelpos='S')
            axes[0].legend(fontsize=8)
            im=axes[1].scatter(xy[:,0],xy[:,1],c=detail['error'],s=3,vmin=0,cmap='magma')
            fig.colorbar(im,ax=axes[1],label='Slope vector error')
            for ax in axes:
                ax.set(xlabel='s (mm)',ylabel='t (mm)',aspect='equal')
            fig.suptitle(n+' / '+poses[j]['pose_id']); fig.tight_layout()
            tag=n+'_'+poses[j]['pose_id']
            fig.savefig(folder/(tag+'.png'),dpi=140); plt.close(fig)
            npz_write(folder/(tag+'.npz'),material_coordinates=xy,truth=truth,prediction=pred,error=detail['error'])

    for n,lut in luts.items():
        if n=='flat' or lut is None:
            continue
        flat=luts['flat']; lower=np.maximum(lut.color_min,flat.color_min); upper=np.minimum(lut.color_max,flat.color_max)
        if np.any(upper<=lower):
            continue
        r,g=np.meshgrid(np.linspace(lower[0],upper[0],121),np.linspace(lower[1],upper[1],121))
        blue=lower[2]+np.array([.25,.5,.75])*(upper[2]-lower[2])
        c=np.stack([np.stack((r,g,np.full_like(r,b)),-1) for b in blue])
        delta=(lookup_slopes_raw(c,lut)[0]-lookup_slopes_raw(c,flat)[0])
        strict=(support(c.reshape(-1,3),lut)[1]&support(c.reshape(-1,3),flat)[1]).reshape(c.shape[:-1])
        fig,axes=plt.subplots(3,3,figsize=(10,9),layout='constrained')
        for i,b in enumerate(blue):
            for j,values in enumerate((np.linalg.norm(delta[i],axis=-1),delta[i,:,:,0],delta[i,:,:,1])):
                vmax=float(np.max(np.abs(delta))) if j else float(np.max(np.linalg.norm(delta,axis=-1)))
                im=axes[i,j].pcolormesh(r,g,np.ma.masked_where(~strict[i],values),cmap='magma' if j==0 else 'coolwarm',
                                       vmin=0 if j==0 else -vmax,vmax=vmax or 1,shading='auto')
                axes[i,j].set(xlabel='Delta R (decoded)',ylabel='Delta G (decoded)',title=f'Delta B={b:.3f}; '+['norm difference','p difference','q difference'][j])
                axes[i,j].set_facecolor('lightgray'); fig.colorbar(im,ax=axes[i,j])
        fig.suptitle(n+': function difference (not accuracy); gray = no original joint support')
        fig.savefig(folder/('mapping_slices_'+n+'.png'),dpi=140); plt.close(fig)
        npz_write(folder/('mapping_slices_'+n+'.npz'),colors=c,delta=delta,strict_support=strict)
