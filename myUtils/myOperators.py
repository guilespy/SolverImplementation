import numpy as np
from copy import copy 
import cv2
from cv2 import GaussianBlur as blurring
import numba


class Blur():
    def __init__(self, p1 = 9, p2 = 4, bt = cv2.BORDER_REFLECT101):
        self.p1 = p1
        self.p2 = p2
        self.bt = bt
        self.LC = 1. #Lipschitz Constant

    def __call__(self, x):
        return blurring(x, (self.p1,self.p1), sigmaX = self.p2,
                        borderType = self.bt)
    def T(self,x):
        return self.__call__(x)
    

def DiscreteGrandient(xx):
    """ Input: X \in R^m_1 x ... x R^m_n
        Output: [G_1, ..., G_n]
                where G_k = X[:,..,:1,...,:] - X[:,...,:-1,...,:]
    \Cup zeros(m_1,...,m_{k-1},1,m_{k+1},...m_n)"""

    x = copy(np.asarray(xx))
    dim = np.asarray(x.shape)
    Gradient = []

    for i,_ in enumerate(x.shape):
        ndim_aux = copy(dim)
        ndim_aux[i] = 1
        Diff_i = np.diff(x,axis=i)
        Zeros_i = np.zeros(ndim_aux)
        Gradient.append(np.concatenate((Diff_i,Zeros_i),axis=i))

    return np.asarray(Gradient)

def DiscreteGradientTranspose(xy):
    r"""xy : ndarray"""
    xy = np.asarray(xy)
    x = copy(xy[0])
    y = copy(xy[1])

    dim_x = x.shape
    dim_y = y.shape

    Dx = -np.diff(x[:-1],axis=0)
    Dx = np.concatenate((-x[0].reshape(1,-1),Dx,x[-2].reshape(1,-1)),axis=0)

    Dy = -np.diff(y[:,:-1],axis=1)
    Dy = np.concatenate((-y[:,0].reshape(-1,1),Dy,y[:,-2].reshape(-1,1)),axis=1)

    return Dx + Dy

    

class DG():
    def __init__(self):
        self.LC = np.sqrt(8)

    def __call__(self,x):
        return DiscreteGrandient(x)

    def T(self,x=None):
        return DiscreteGradientTranspose(x)
    
        
class Masking():
   def __init__(self, M):
       self.LC = 1.
       self.mask = M
   def __call__(self, X):
       return self.mask * X
   def T(self,X):
       return self.mask * X






