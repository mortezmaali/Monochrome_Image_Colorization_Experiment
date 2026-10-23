"""Compare grayscale->RGB models. Train only on Macbeth24; never tune on test sets.
Run: python compare_models.py. Illustrative sensor, NOT measured OV9281 QE.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import PchipInterpolator
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from skimage.color import deltaE_ciede2000
import monochrome_color_simulation as base
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'model_comparison';OUT.mkdir(exist_ok=True)
WL=np.arange(400.,701.,10.); E=base.D65[::2]; CMF=base.CMF[::2]
assert len(WL)==len(E)==len(CMF)==31
white=(E[:,None]*CMF).sum(axis=0)/(E*CMF[:,1]).sum()
M=np.array([[3.2406,-1.5372,-.4986],[-.9689,1.8758,.0415],[.0557,-.2040,1.0570]])
M_inv=np.linalg.inv(M)
def read(fn,key):
    r=np.asarray(loadmat(ROOT/fn)[key],float)
    assert r.shape[0]==36 and np.isfinite(r).all()
    return r[2:33].T
def xyz(r):return (r*E)@CMF/(E*CMF[:,1]).sum()
def mono(r):
    w=E*base.modeled_sensor_sensitivity(WL)
    return r@w/w.sum()
def reference(r):return xyz(r)@M.T
train=read('Ref_Macbeth24.mat','RefT')
sets={'Macbeth24_training':train,'Ref_new2_testing':read('Ref_new2.mat','Ref'),'Ref_training_testing':read('Ref_training.mat','RefTrain')}
x=mono(train);y=reference(train)
# Pseudoinverse with bias
W=np.linalg.pinv(np.c_[x,np.ones_like(x)])@y
# Degree 3 polynomial, ridge stabilization; coefficient shrinkage only on non-intercept
poly=PolynomialFeatures(degree=3,include_bias=False)
Xp=poly.fit_transform(x[:,None]); scaler=StandardScaler();Xs=scaler.fit_transform(Xp)
poly_model=Ridge(alpha=1.0).fit(Xs,y)
# PCHIP per channel; for duplicate x values, average target colors
order=np.argsort(x);xs=x[order];ys=y[order]
unique, inverse=np.unique(xs,return_inverse=True)
means=np.array([ys[inverse==i].mean(axis=0) for i in range(len(unique))])
lut=PchipInterpolator(unique,means,axis=0,extrapolate=False)
# Small NN; standardize inputs and targets. Fixed seed, mild regularization.
xscale=StandardScaler().fit(x[:,None]);yscale=StandardScaler().fit(y)
net=MLPRegressor(hidden_layer_sizes=(8,),activation='tanh',solver='lbfgs',alpha=1.0,max_iter=2500,random_state=42)
net.fit(xscale.transform(x[:,None]),yscale.transform(y))
def predict(name,signal):
    if name=='Affine':return np.c_[signal,np.ones_like(signal)]@W
    if name=='Polynomial':return poly_model.predict(scaler.transform(poly.transform(signal[:,None])))
    if name=='PCHIP LUT':return lut(np.clip(signal,unique[0],unique[-1]))
    if name=='Neural network':return yscale.inverse_transform(net.predict(xscale.transform(signal[:,None])))
    raise ValueError(name)
models=['Affine','Polynomial','PCHIP LUT','Neural network']
def srgb(rgb):return np.clip(base.srgb_encode(rgb),0,1)
def de00(refxyz,estimate):
    a=base.lab_from_xyz(refxyz,white)
    b=base.lab_from_xyz(estimate@M_inv.T,white)
    return deltaE_ciede2000(a[:,None,:],b[:,None,:]).ravel()
rows=[]
for setname,r in sets.items():
    signal=mono(r);truth=reference(r);truth_xyz=xyz(r)
    predictions={name:predict(name,signal) for name in models}
    ncols=6 if len(r)==24 else 10
    fig,axes=plt.subplots(2+len(models),1,figsize=(12,15 if len(r)==24 else 23))
    all_colors=[srgb(truth),np.repeat(signal[:,None],3,axis=1)]+[srgb(predictions[n]) for n in models]
    labels=['Reference sRGB','Simulated monochrome']+models
    for ax,colors,label in zip(axes,all_colors,labels):
        ax.imshow(base.chart(colors,ncols,patch=32,gap=3));ax.set_title(label);ax.axis('off')
    fig.suptitle(setname+' | 24 Macbeth training samples ONLY | modeled sensor',fontsize=14)
    fig.tight_layout();fig.savefig(OUT/(setname+'.png'),dpi=130);plt.close(fig)
    csv=[signal,truth[:,0],truth[:,1],truth[:,2]]
    headers=['mono','reference_linear_R','reference_linear_G','reference_linear_B']
    for name in models:
        p=predictions[name]; err=de00(truth_xyz,p)
        rmse=np.sqrt(np.mean((p-truth)**2))
        rows.append((setname,name,len(r),rmse,float(err.mean()),float(np.median(err)),float(np.percentile(err,90)),int(np.sum((signal<unique[0])|(signal>unique[-1])))))
        csv.extend([p[:,0],p[:,1],p[:,2],err]);headers.extend([name.replace(' ','_')+'_'+z for z in ['R','G','B','DE00']])
    np.savetxt(OUT/(setname+'.csv'),np.column_stack(csv),delimiter=',',header=','.join(headers),comments='')
with (OUT/'summary.csv').open('w') as f:
    f.write('dataset,model,n,linear_rgb_rmse,mean_DE00,median_DE00,p90_DE00,signals_outside_training_range\n')
    for row in rows:f.write(','.join(map(str,row))+'\n')
for row in rows:print('%-23s %-15s N=%3d RMSE=%.4f meanDE00=%.2f median=%.2f out-of-range=%d'%(row[0],row[1],row[2],row[3],row[4],row[5],row[7]))
print('Outputs:',OUT)
