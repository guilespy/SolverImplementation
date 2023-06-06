import numpy as np
from copy import copy 


class EukNorm():
    r"""
    Euklidean norm \lambda \|x\|
    """
    def __init__(self,lambda_=1.0,c=0):
        self.lambda_ = lambda_
        self.c = np.asarray(c)

    def __call__(self,x):
        return self.lambda_ * np.linalg.norm(x - self.c)

    def prox(self,x,gamma=1.0):
        if np.linalg.norm(x - self.c) <= gamma*self.lambda_:
            return self.c
        else:
            return x - self.lambda_*gamma/np.linalg.norm(x-self.c)*(x-self.c)

    def prox_conj(self,x,gamma = 1.0):
        if np.linalg.norm(x-gamma*self.c) <= self.lambda_:
            return x - gamma * self.c
        else:
            return (self.lambda_/np.linalg.norm(x-gamma*self.c))*(x-gamma*self.c)
                    
        

class Norm_1():
    r"""
    return lamda_ * ||x - y||_1
    """
    def __init__(self,lambda_ = 1.0, y = 0.):
        self.lambda_ = lambda_
        self.y = np.asarray(y)

    def __call__(self,x):
        return self.lambda_ * np.sum(np.abs(x - self.y))

    def prox(self,x,mu=1.):
        return self.y + np.sign(x - self.y)*np.maximum(np.abs(x-self.y) - self.lambda_ * mu,0)

    def prox_conj(self, x, mu=1):
        temp = x - mu * self.y
        return np.maximum(-self.lambda_,np.minimum(temp,self.lambda_))
    
class Norm_2():
    def __init__(self,lambda_ = 1.0, y = 0):
        self.lambda_ = lambda_
        self.LC = 2.0
        self.y = np.asarray(y)

    def __call__(self,x):
        return self.lambda_ * np.sum((x-self.y)**2)

    def grad(self, x):
        return 2 * self.lambda_ * (x - self.y)

class Normx():
    """
    It works only with R^{2x2}
    """

    def __init__(self,lambda_ = 1.):
        self.lambda_ = lambda_

    def __call__(self,x):
        return np.sum(np.sqrt(x[0]**2 + x[1]**2))

    def prox_conj(self, y, mu=1.):
        temp = np.sqrt(y[0]**2 + y[1]**2)
        yy = y[0] / np.maximum(1., temp / self.lambda_)
        zz = y[1] / np.maximum(1., temp / self.lambda_)

        return np.asarray((yy,zz))

class IndicatorB():
    def __init__(self, b, lambda_ = 1.0):
        assert b != None
        self.b = np.asarray(b)
        self.lambda_ = lambda_

    def __call__(self,x):
        return 0
#Check this part!!!!!!
    def prox_conj(self, x , mu = 1.0):
        assert x.shape == self.b.shape
        return x - self.mu * self.b 
        

class IndicatorR1():
    def __init__(self, lambda_ = 1.0):
        self.lambda_ = lambda_

    def __call__(self,x):
        return 0

    def prox(self, x, mu=1.):
        return np.maximum(0,np.minimum(x,1))


class IndicatorR1N1():
    def __init__(self, lambda_ = 1):
        self.lambda_ = lambda_
    
    def __call__(self,x):
        return self.lambda_ * np.sum(np.abs(x))

    def prox(self,x,mu=1):
        return np.maximum(0,np.minimum(x - self.lambda_ * mu,1))


class IndicatorR1N2():
    def __init__(self, lambda_ = 1.):
        self.lambda_ = lambda_
    
    def __call__(self, x):
        return self.lambda_ * np.sum(x**2)

    def prox(self,x,mu = 1.):
        aux = (1./(1. + 2 * self.lambda_ * mu))*x
        return np.maximum(0,np.minimum(aux,1))


