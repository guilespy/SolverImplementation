import numpy as np
import myUtils.myOperators as mo
import myUtils.myFunctions as mf
import mySolvers.mySolvers as ms
import scipy
from myUtils.visualTools import createVideo, createPlot
from time import time

SEED = 2357
LOST_LEVEL = 0.8
NOISE_LEVEL = 0.01
LAMBDA = 0.1
CMAP = 'gray'

original = scipy.datasets.face()[:, :, 0]
X = np.asarray(original)/255.

mask = np.random.seed(SEED)
noise = NOISE_LEVEL * np.random.randn(*X.shape)
mask = np.random.rand(*X.shape)
mask = mask > LOST_LEVEL


L1 = mo.Masking(mask)
L2 = mo.DG()

np.random.seed(SEED)
noise = NOISE_LEVEL * np.random.randn(*X.shape)
B_noise = np.maximum(np.minimum(X + noise, 1), 0)
B_noise = L1(B_noise)

f = mf.IndicatorR1()
g1 = mf.Norm_1(1, B_noise)
g2 = mf.Normx(LAMBDA)

x1 = B_noise
v1 = L1(X)
v2 = L2(X)



mysolFBF = ms.FBF(x1, v1, v2, f_primal=f,
                  Gs_dual=[g1, g2],
                  LinOp=[L1, L2],
                  maxIT=350)



mysolFRB = ms.FRB(x1, v1, v2, f_primal=f,
                  Gs_dual=[g1, g2],
                  LinOp=[L1, L2],
                  maxIT=350)




mysolFRBD = ms.FRBD(x1, v1, v2,
                  f_primal=f,
                  Gs_dual=[g1, g2],
                  LinOp=[L1, L2],
                  numberSteps = 4,
                  maxIT=350)

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
            './Visualization/faceReconstruction.mp4',
           'Noisy')


createPlot(mysolFRB,
           mysolFRBD,
           mysolFBF,
           name='./Visualization/plotFace.pdf')

