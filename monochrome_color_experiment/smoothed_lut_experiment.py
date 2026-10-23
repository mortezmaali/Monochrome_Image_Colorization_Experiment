"""PCHIP versus smoothed PCHIP and kernel LUT. Train/CV Macbeth only."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from skimage.color import deltaE_ciede2000
import compare_models as b
O=b.ROOT/'smoothed_lut';O.mkdir(exist_ok=True)
x=b.x.copy();y=b.y.copy(); order=np.argsort(x); xs=x[order];ys=y[order]
span=np.ptp(xs)
# kernel-smoothed calibration anchors; lambda controls smoothing of the 24 RGB anchor values
# leave-one-out prediction selects bandwidth without inspecting test datasets.
def smooth_anchors(x0,y0,bandwidth):
    d=(x0[:,None]-x0[None,:])/bandwidth
    w=np.exp(-.5*d*d);np.fill_diagonal(w,0)
    denom=w.sum(axis=1)
    return (w@y0)/np.maximum(denom[:,None],1e-100)

def fit_smooth_pchip(x0,y0,bandwidth,blend):
    ii=np.argsort(x0);xx=x0[ii];yy=y0[ii]
    uniq,inv=np.unique(xx,return_inverse=True)
    means=np.array([yy[inv==i].mean(axis=0) for i in range(len(uniq))])
    # smooth with local kernel average INCLUDING self for fitted LUT, blend original with smoothed targets
    d=(uniq[:,None]-uniq[None,:])/bandwidth
    w=np.exp(-.5*d*d);w/=w.sum(axis=1,keepdims=True)
    smoothed=(1-blend)*means+blend*(w@means)
    return PchipInterpolator(uniq,smoothed,axis=0,extrapolate=False),uniq

def predict_pchip(model,limits,z):return model(np.clip(z,limits[0],limits[-1]))

# fixed, predeclared bandwidth/strength grid; LOOCV selects based on training-only linear RGB error
bands=span*np.array([.025,.05,.1,.2,.35,.5])
blends=[.25,.5,.75,1.0]
candidates=[]
for band in bands:
 for blend in blends:
  preds=[]
  for i in range(len(xs)):
   mask=np.arange(len(xs))!=i
   mod,limits=fit_smooth_pchip(xs[mask],ys[mask],band,blend)
   preds.append(predict_pchip(mod,limits,np.array([xs[i]]))[0])
  score=float(np.mean((np.array(preds)-ys)**2))
  candidates.append((score,band,blend))
candidates.sort()
_,best_band,best_blend=candidates[0]
smoothed,limits=fit_smooth_pchip(xs,ys,best_band,best_blend)
# Extra baseline: strong smoothing, predeclared rather than tuned to test
fixed,fixlim=fit_smooth_pchip(xs,ys,.2*span,1.0)
models=['Affine','PCHIP LUT','Smoothed PCHIP (CV)','Smoothed PCHIP (strong)']
rows=[]
for name,r in b.sets.items():
 signal=b.mono(r);truth=b.reference(r);truthxyz=b.xyz(r)
 predictions={'Affine':b.predict('Affine',signal),'PCHIP LUT':b.predict('PCHIP LUT',signal),
  'Smoothed PCHIP (CV)':predict_pchip(smoothed,limits,signal),
  'Smoothed PCHIP (strong)':predict_pchip(fixed,fixlim,signal)}
 fig,axes=plt.subplots(2+len(models),1,figsize=(13,15 if len(r)==24 else 23))
 colors=[b.srgb(truth),np.repeat(signal[:,None],3,axis=1)]+[b.srgb(predictions[m]) for m in models]
 for ax,col,label in zip(axes,colors,['Reference sRGB','Monochrome']+models):
  ax.imshow(b.base.chart(col,6 if len(r)==24 else 10,patch=30,gap=3));ax.set_title(label);ax.axis('off')
 fig.suptitle(name+' | trained and tuned ONLY on Macbeth 24 | modeled sensitivity')
 fig.tight_layout();fig.savefig(O/(name+'.png'),dpi=130);plt.close(fig)
 for model,pred in predictions.items():
  err=b.de00(truthxyz,pred)
  rows.append([name,model,len(r),np.sqrt(np.mean((pred-truth)**2)),err.mean(),np.median(err),np.percentile(err,90)])
  print(name,model,'RMSE',round(rows[-1][3],4),'DE00',round(rows[-1][4],2))
 np.savetxt(O/(name+'_predictions.csv'),np.column_stack([signal,truth]+[predictions[m] for m in models]),delimiter=',',comments='',header='mono,ref_R,ref_G,ref_B,'+','.join(m.replace(' ','_')+'_'+c for m in models for c in ['R','G','B']))
with (O/'summary.csv').open('w') as f:
 f.write('dataset,model,n,linear_rgb_rmse,mean_DE00,median_DE00,p90_DE00\n')
 for row in rows:f.write(','.join(map(str,row))+'\n')
with (O/'README.txt').open('w') as f:
 f.write(f'Modeled sensor, NOT measured OV9281. Reflectance 380:10:730, rows 3:33 -> 400:10:700.\nD65 and CIE 1931 CMFs from lightSource.m. Macbeth 24 only for training and hyperparameter selection.\nSmoothed PCHIP CV selects bandwidth and blending via leave-one-out linear RGB MSE on Macbeth only.\nChosen bandwidth={best_band:.6g}, fraction of mono range={best_band/span:.3f}, blend={best_blend}.\nSecond strongly smoothed PCHIP is fixed bandwidth=20% of range, blend=1.\nClamps input to Macbeth range; clipped sRGB for display only. DeltaE00 calculated from unclipped linear RGB after converting to XYZ; out-of-gamut values can make comparisons less physically meaningful.\nCaution: LOOCV on 24 points can be unstable; test sets were not used for tuning.\n')
fig,axes=plt.subplots(3,1,figsize=(9,10),sharex=True)
grid=np.linspace(xs.min(),xs.max(),400)
for c,ax in enumerate(axes):
 ax.scatter(xs,ys[:,c],label='Macbeth calibration samples',s=26)
 ax.plot(grid,b.predict('PCHIP LUT',grid)[:,c],label='PCHIP')
 ax.plot(grid,smoothed(grid)[:,c],label='Smoothed PCHIP (CV)')
 ax.plot(grid,fixed(grid)[:,c],label='Smoothed PCHIP (strong)')
 ax.set_ylabel(['Linear R','Linear G','Linear B'][c]);ax.grid(alpha=.2)
axes[0].legend(fontsize=8);axes[-1].set_xlabel('Simulated monochrome response')
fig.tight_layout();fig.savefig(O/'calibration_curves.png',dpi=160);plt.close(fig)
print('selected:',best_band/span,best_blend,'output',O)
