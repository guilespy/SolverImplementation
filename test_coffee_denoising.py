import numpy as np
from skimage import data
from skimage.util import img_as_float
from skimage.color import rgb2gray
import myUtils.myOperators as mo
import myUtils.myFunctions as mf
import mySolvers.mySolvers as ms
from myUtils.visualTools import createVideo, createPlot


SEED = 2357
LOST_LEVEL = 0.8
NOISE_LEVEL = 0.001
LAMBDA = 0.1
CMAP = 'gray'


original = data.coffee()
grayscale = rgb2gray(original)
X = img_as_float(grayscale, force_copy=False)


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
g1 = mf.Norm_1(1., B_noise)
g2 = mf.Normx(LAMBDA)

x1 = B_noise
v1 = L1(X)
v2 = L2(X)


mysolFRB = ms.FRB(x1, v1, v2,
                  f_primal=f,
                  Gs_dual=[g1, g2],
                  LinOp=[L1, L2],
                  maxIT=300)

mysolFBF = ms.FBF(x1, v1, v2,
                  f_primal=f,
                  Gs_dual=[g1, g2],
                  LinOp=[L1, L2],
                  maxIT=300)

mysolFRBD = ms.FRBD(x1, v1, v2,
                  f_primal=f,
                  Gs_dual=[g1, g2],
                  LinOp=[L1, L2],
                  maxIT=300)
mysolFRBD.solve()
mysolFRB.solve()
mysolFBF.solve()


createPlot(mysolFBF,
           mysolFRB,
           mysolFRBD,
           name='./Visualization/plotCoffee.pdf')


createVideo(X, B_noise, mysolFRBD, mysolFBF,
            './Visualization/coffeeReconstruction.mp4')

