"""Monochrome camera -> RGB simulation. See README for spectral assumptions."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
# IMPORTANT: confirm wavelength coordinates with original dataset author.
REFLECTANCE_WAVELENGTHS_NM = np.arange(380., 731., 10.)  # 36 samples, ASSUMPTION
# Use only overlap with available MATLAB D65 and CIE 1931 2-degree data (400-700 nm).
# Values below are transcribed from the supplied lightSource.m file.
D65 = np.array([82.75,87.12,91.49,92.46,93.43,90.06,86.68,95.77,104.87,110.94,117.01,117.41,117.81,116.34,114.86,115.39,115.92,112.37,108.81,109.08,109.35,108.58,107.8,106.3,104.79,106.24,107.69,106.05,104.41,104.23,104.05,102.02,100,98.17,96.33,96.06,95.79,92.24,88.69,89.35,90.01,89.8,89.6,88.65,87.7,85.49,83.29,83.49,83.7,81.86,80.03,80.12,80.21,81.25,82.28,80.28,78.28,74,69.72,70.67,71.61])
CMF = np.array([
[.01431,.00040,.06785],[.02319,.00064,.11020],[.04351,.00121,.20740],[.07763,.00218,.37130],[.13438,.00400,.64560],[.21477,.00730,1.03905],[.28390,.01160,1.38560],[.32850,.01684,1.62296],[.34828,.02300,1.74706],[.34806,.02980,1.78260],[.33620,.03800,1.77211],[.31870,.04800,1.74410],[.29080,.06000,1.66920],[.25110,.07390,1.52810],[.19536,.09098,1.28764],[.14210,.11260,1.04190],[.09564,.13902,.81295],[.05795,.16930,.61620],[.03201,.20802,.46518],[.01470,.25860,.35330],[.00490,.32300,.27200],[.00240,.40730,.21230],[.00930,.50300,.15820],[.02910,.60820,.11170],[.06327,.71000,.07825],[.10960,.79320,.05725],[.16550,.86200,.04216],[.22575,.91485,.02984],[.29040,.95400,.02030],[.35970,.98030,.01340],[.43345,.99495,.00875],[.51205,1.,.00575],[.59450,.99500,.00390],[.67840,.97860,.00275],[.76210,.95200,.00210],[.84250,.91540,.00180],[.91630,.87000,.00165],[.97860,.81630,.00140],[1.02630,.75700,.00110],[1.05670,.69490,.00100],[1.06220,.63100,.00080],[1.04560,.56680,.00060],[1.00260,.50300,.00034],[.93840,.44120,.00024],[.85445,.38100,.00019],[.75140,.32100,.00010],[.64240,.26500,.00005],[.54190,.21700,.00003],[.44790,.17500,.00002],[.36080,.13820,.00001],[.28350,.10700,.00000],[.21870,.08160,.00000],[.16490,.06100,.00000],[.12120,.04458,.00000],[.08740,.03200,.00000],[.06360,.02320,.00000],[.04677,.01700,.00000],[.03290,.01192,.00000],[.02270,.00821,.00000],[.01584,.00572,.00000],[.01136,.00410,.00000]])
WL = np.arange(400.,701.,5.)
# Modeled broad silicon-like response, NOT a measured OV9281 spectral response.
# Replace with actual measured sensitivity samples for quantitative sensor predictions.
def modeled_sensor_sensitivity(wavelength):
    s = (0.42*np.exp(-0.5*((wavelength-510)/115)**2)
         + 0.58*np.exp(-0.5*((wavelength-670)/165)**2))
    return s / s.max()

def load_ref(name, key):
    d=loadmat(ROOT/name)[key].astype(float)
    if d.shape[0]!=len(REFLECTANCE_WAVELENGTHS_NM):
        raise ValueError(f'{name}: expected {len(REFLECTANCE_WAVELENGTHS_NM)} wavelength samples, got {d.shape}')
    if np.any((d<0)|(d>1)):
        raise ValueError(f'{name}: reflectances outside 0..1')
    # PCHIP minimizes interpolation overshoot; clipping handles tiny boundary artifacts.
    return np.clip(PchipInterpolator(REFLECTANCE_WAVELENGTHS_NM,d,axis=0)(WL),0,1).T

def spectral_to_xyz(R):
    # D65 and CIE 1931 2-degree observer from lightSource.m
    E=D65
    k=1/np.trapezoid(E*CMF[:,1],WL)
    return k*np.trapezoid(R[:,:,None]*E[None,:,None]*CMF[None,:,:],WL,axis=1)

def xyz_to_linear_rgb(XYZ):
    # Standard D65 sRGB matrix. No white-point adaptation necessary with D65 reference.
    M=np.array([[3.2406,-1.5372,-.4986],[-.9689,1.8758,.0415],[.0557,-.2040,1.0570]])
    return XYZ@M.T

def srgb_encode(rgb):
    return np.where(rgb<=.0031308,12.92*rgb,1.055*np.maximum(rgb,0)**(1/2.4)-.055)

def srgb_decode(rgb):
    return np.where(rgb<=.04045,rgb/12.92,((np.maximum(rgb,0)+.055)/1.055)**2.4)

def camera_response(R):
    # Exposure normalization: a perfect diffuser under the same D65 has response 1.
    weights=D65*modeled_sensor_sensitivity(WL)
    return np.trapezoid(R*weights[None,:],WL,axis=1)/np.trapezoid(weights,WL)

def chart(colors, columns, patch=46, gap=4):
    n=len(colors); rows=int(np.ceil(n/columns))
    out=np.full((rows*(patch+gap)+gap,columns*(patch+gap)+gap,3),.17)
    for i,c in enumerate(np.clip(colors,0,1)):
        r,col=divmod(i,columns)
        y=gap+r*(patch+gap);x=gap+col*(patch+gap)
        out[y:y+patch,x:x+patch]=c
    return out

def lab_from_xyz(xyz,white):
    d=6/29
    q=xyz/white
    f=np.where(q>d**3,np.cbrt(np.maximum(q,0)),q/(3*d*d)+4/29)
    return np.stack((116*f[:,1]-16,500*(f[:,0]-f[:,1]),200*(f[:,1]-f[:,2])),axis=1)

def run():
    print('WARNING: reflectance wavelengths 380:10:730 nm are an UNVERIFIED assumption.')
    print('WARNING: monochrome spectral sensitivity is illustrative, not measured OV9281 data.')
    train=load_ref('Ref_Macbeth24.mat','RefT')
    test1=load_ref('Ref_new2.mat','Ref')
    test2=load_ref('Ref_training.mat','RefTrain')
    sets={'Macbeth24_train':train,'Ref_new2_test':test1,'Ref_training_test':test2}
    out=ROOT/'results';out.mkdir(exist_ok=True)
    white=spectral_to_xyz(np.ones((1,len(WL))))[0]
    # Fit in LINEAR RGB, then sRGB encode for display. Fit an affine 2x3 transform.
    # A 1x3 fit with no intercept is also provided for direct comparison.
    c_train=camera_response(train)
    target_train=xyz_to_linear_rgb(spectral_to_xyz(train))
    X=np.column_stack((c_train,np.ones_like(c_train)))
    W=np.linalg.pinv(X)@target_train
    W_no_bias=np.linalg.pinv(c_train[:,None])@target_train
    print('Affine coefficients [slope; intercept] for linear RGB:\n',W)
    print('No-intercept coefficients for linear RGB:\n',W_no_bias)
    np.savetxt(out/'affine_coefficients.csv',W,delimiter=',')
    np.savetxt(out/'no_intercept_coefficients.csv',W_no_bias,delimiter=',')
    for name,R in sets.items():
        c=camera_response(R)
        xyz=spectral_to_xyz(R)
        ref_lin=xyz_to_linear_rgb(xyz)
        pred_lin=np.column_stack((c,np.ones_like(c)))@W
        pred0_lin=c[:,None]@W_no_bias
        ref=srgb_encode(ref_lin)
        pred=srgb_encode(pred_lin)
        pred0=srgb_encode(pred0_lin)
        # CIE76 computed from un-clipped linear predictions. Negative predictions
        # are mathematically allowed in the fit but not valid display RGB.
        M_inv=np.linalg.inv(np.array([[3.2406,-1.5372,-.4986],[-.9689,1.8758,.0415],[.0557,-.2040,1.0570]]))
        pred_xyz=pred_lin@M_inv.T
        de=np.linalg.norm(lab_from_xyz(xyz,white)-lab_from_xyz(pred_xyz,white),axis=1)
        rmse=np.sqrt(np.mean((pred_lin-ref_lin)**2))
        print(f'{name}: n={len(c)}, linear-RGB RMSE={rmse:.5f}, mean ΔE76={de.mean():.2f}, median ΔE76={np.median(de):.2f}')
        clipped_ref=np.clip(ref,0,1);clipped_pred=np.clip(pred,0,1)
        clipped0=np.clip(pred0,0,1)
        mono=np.repeat(c[:,None],3,axis=1)
        cols=6 if len(c)==24 else 10
        fig,axes=plt.subplots(4,1,figsize=(12, 4*(int(np.ceil(len(c)/cols))*.45+1)))
        for ax,title,values in zip(axes,['Reference sRGB','Monochrome sensor','Affine estimated sRGB','No-intercept estimated sRGB'],[clipped_ref,mono,clipped_pred,clipped0]):
            ax.imshow(chart(values,cols));ax.set_title(title);ax.axis('off')
        fig.suptitle(name+' | patch order = source column order',fontsize=13)
        fig.tight_layout();fig.savefig(out/(name+'_charts.png'),dpi=150);plt.close(fig)
        np.savetxt(out/(name+'_patch_results.csv'),np.column_stack([c,ref,pred,pred0,de]),delimiter=',',header='mono,ref_R,ref_G,ref_B,affine_R,affine_G,affine_B,no_bias_R,no_bias_G,no_bias_B,DeltaE76',comments='')
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(WL,modeled_sensor_sensitivity(WL),label='Illustrative monochrome sensitivity')
    ax.plot(WL,CMF[:,0],label='CIE xbar');ax.plot(WL,CMF[:,1],label='CIE ybar');ax.plot(WL,CMF[:,2],label='CIE zbar')
    ax.set(xlabel='Wavelength (nm)',ylabel='Relative sensitivity',title='Spectral curves (camera curve is NOT OV9281 measurement)')
    ax.legend();fig.tight_layout();fig.savefig(out/'spectral_curves.png',dpi=150);plt.close(fig)
    print('Outputs:',out)

if __name__=='__main__':run()
