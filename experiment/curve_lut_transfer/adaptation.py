"""Optional preregistered global corrections; independent tests never enter fits."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from utils.normal_mapping import lookup_slopes_raw, NormalCalibration
from .storage import load_arrays, json_write, Checkpoints
from .metrics import evaluate_pose, aggregate


def budget_order(poses: list[dict], seed: int) -> list[int]:
    """Seeded round-robin depth strata; complete poses and nested budgets."""
    rng=np.random.default_rng(seed)
    groups=[]
    for d in sorted({p['depth_mm'] for p in poses}):
        g=[i for i,p in enumerate(poses) if p['depth_mm']==d]
        rng.shuffle(g); groups.append(g)
    return [g[j] for j in range(max(map(len,groups))) for g in groups if j<len(g)]


def fit_corrections(samples: list[dict], lut: NormalCalibration, *, ridge: float, iterations: int) -> dict:
    if any(len(s['grid_index'])==0 for s in samples):
        return dict(status='failed', reason='Selected budget contains an empty pose')
    colors=np.concatenate([s['color_residual_samples'] for s in samples])
    target=np.concatenate([s['slope_samples'] for s in samples]).astype(float)
    weights=np.concatenate([s['area_mm2']/s['area_mm2'].sum()/len(samples) for s in samples])
    pred=lookup_slopes_raw(colors,lut)[0].astype(float)
    # Row-vector convention: pred @ B.T, so stored B matches the mathematical model.
    B=np.linalg.solve(pred.T@(weights[:,None]*pred)+ridge*np.eye(2),pred.T@(weights[:,None]*target)+ridge*np.eye(2)).T
    def objective(values):
        A=values.reshape(3,3)
        estimates=lookup_slopes_raw(colors@A.T,lut)[0].astype(float)
        return float(weights@np.sum((estimates-target)**2,axis=1)+ridge*np.sum((A-np.eye(3))**2))
    opt=minimize(objective,np.eye(3).ravel(),method='L-BFGS-B',options={'maxiter':iterations,'eps':1e-3,'ftol':1e-10})
    return dict(status='ok', B=B.tolist(), A=opt.x.reshape(3,3).tolist(),
                color_fit_converged=bool(opt.success), color_fit_message=str(opt.message),
                color_fit_iterations=int(opt.nit), objective=float(opt.fun), ridge=ridge,
                sample_count=len(colors), validation_labels_used=0,
                hyperparameters='fixed before independent evaluation; no candidate selection on tests')


def run_adaptation(p: dict, ck: Checkpoints, datasets: dict,
                   common: list[np.ndarray], settings: dict) -> dict:
    from .runner import sample_path
    a=settings['adaptation']; lut=p['lut']; m=settings['metrics']; results={}
    flat_ref=load_arrays(ck.root/'samples/flat/reference.npz')
    order=budget_order(p['train'],m['seed'])
    for name,radius in p['conditions'].items():
        if radius is None:
            continue
        ref=load_arrays(ck.root/'samples'/name/'reference.npz')
        valid=(flat_ref['valid']&ref['valid']).reshape(-1)
        bright=(flat_ref['color']>=a['reference_floor'])&(ref['color']>=a['reference_floor'])
        valid &= bright.all(-1).reshape(-1)
        gain=np.divide(flat_ref['color'],np.maximum(ref['color'],a['reference_floor'])).reshape(-1,3)
        train=[load_arrays(sample_path(ck.root,name,'training',pose)) for pose in p['train']]
        fits={}
        for budget in a['pose_budgets']:
            chosen=order[:budget]
            fit=fit_corrections([train[i] for i in chosen],lut,ridge=a['ridge'],iterations=a['maximum_iterations'])
            fit['pose_ids']=[p['train'][i]['pose_id'] for i in chosen]
            fits[str(budget)]=fit
        result=dict(fits=fits, evaluations={}, reference_floor=a['reference_floor'])
        methods=['reference_gain']+[f'{kind}_{budget}' for budget in a['pose_budgets'] for kind in ('slope','color')]
        for method in methods:
            rows=[]; detail=[]; cover=[]; rawrows=[]; rawdetails=[]
            if method!='reference_gain':
                kind,budget=method.split('_'); fit=fits[budget]
                if fit['status']!='ok':
                    result['evaluations'][method]=fit; continue
            for pose,sample,indices in zip(p['test'],datasets[name],common):
                colors=sample['color_residual_samples']
                use=indices
                if method=='reference_gain':
                    use=indices[valid[indices]]
                    colors=colors*gain[sample['grid_index']]
                    # Drop only reference-dark points, report coverage and compare uncorrected on exactly that subset.
                    positions=np.searchsorted(sample['grid_index'],indices)
                    used=np.searchsorted(sample['grid_index'],use)
                    denom=sample['area_mm2'][positions].sum()
                    cover.append(float(sample['area_mm2'][used].sum()/denom) if denom else 0.)
                elif kind=='color':
                    colors=colors@np.array(fit['A']).T
                pred=lookup_slopes_raw(colors,lut)[0]
                if method!='reference_gain' and kind=='slope':
                    pred=pred@np.array(fit['B']).T
                transformed=dict(sample,color_residual_samples=colors)
                row,dd=evaluate_pose(transformed,lut,use,prediction=pred)
                row.update(pose_id=pose['pose_id'],anchor_id=pose['anchor_id']); rows.append(row); detail.append(dd)
                raw,rd=evaluate_pose(sample,lut,use); raw.update(pose_id=pose['pose_id'],anchor_id=pose['anchor_id'])
                rawrows.append(raw); rawdetails.append(rd)
            result['evaluations'][method]=dict(corrected=aggregate(rows,detail,m['slope_tolerances']),
                uncorrected_same_domain=aggregate(rawrows,rawdetails,m['slope_tolerances']),
                reference_coverage=float(np.mean(cover)) if cover else 1.,per_pose=rows)
        results[name]=result
    json_write(ck.root/'adaptation.json',results)
    return results
