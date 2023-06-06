import numpy as np
import myUtils.myFunctions as mf


def prox_conjugate(C,x,l=1.):
    return x - l*C.prox(x/l,1/l)


x = np.random.randn(3,3)* 10
b = np.random.rand(3,3) * 2




n1 = mf.Norm_1(.444,b)
print("A ver que nos da:")
print(n1.prox(x))
print("*******")
pcx1 = n1.prox_conj(x,2.3)
print(pcx1)
pcx2 = prox_conjugate(n1,x,2.3)
print("*******")
print(pcx2)
print(f"The difference is: {np.linalg.norm(pcx1 - pcx2)}")



