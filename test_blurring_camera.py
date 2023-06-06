import numpy as np
import myUtils.myOperators as mo
import myUtils.myFunctions as mf
import mySolvers.mySolvers as ms
from myUtils.visualTools import createVideo, createPlot
from skimage import data
from skimage.color import rgb2gray
from skimage.util import img_as_float64 
from time import time

MAXITER = 600
SEED = 2357
NOISE_LEVEL = .001
LAMBDA = 0.01
EPSILON = 1e-2

L1 = mo.Blur(9,5)
L2 = mo.DG()

original = data.camera()
X = img_as_float64(original,force_copy=False)

np.random.seed(SEED)
noise = NOISE_LEVEL * np.random.randn(*X.shape)
B_noise = np.maximum(0, np.minimum(1, L1(X) + noise))

f = mf.IndicatorR1()
g1 = mf.Norm_1(3., B_noise)
g2 = mf.Normx(LAMBDA)


x1 = B_noise
v1 = L1(X)
v2 = L2(X)


mysolFRB = ms.FRB(x1,v1,v2,f_primal=f,Gs_dual=[g1,g2],LinOp=[L1,L2], maxIT = 800)
mysolFRBD = ms.FRBD(x1,v1,v2,f_primal=f,Gs_dual=[g1,g2],LinOp=[L1,L2], maxIT = 800)
mysolFBF = ms.FBF(x1,v1,v2,f_primal=f,Gs_dual=[g1,g2],LinOp=[L1,L2],maxIT = 800)

tic = time()
mysolFBF.solve()
print(f"Elapsed time: {(time() - tic):.2f} sec")


tic = time()
mysolFRB.solve()
print(f"Elapsed time: {(time() - tic):.2f} sec")

tic = time()
mysolFRBD.solve()
print(f"Elapsed time: {(time() - tic):.2f} sec")

createVideo(X,
            B_noise, 
            mysolFBF, 
            mysolFRBD,
            './Visualization/cameradeblurring.mp4',
           'Blurred')


createPlot(mysolFRB,
           mysolFRBD,
           mysolFBF,
           name='./Visualization/plotCamera.pdf')




