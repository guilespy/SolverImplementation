import numpy as np
from copy import copy

class mySolver():
    r"""This is the base class for solvers
    each child needs an update method
    Stop Criteria: f(x_{n+i}) - f(x_n) < tol for i=1:_REP for _REP > 0 (i.e. we are on a plain)"""
    def __init__(self,*args,**kwargs):
        self.iteration = 0
        self.maxIT = 600
        self.tol = 1e-5
        self.learning_rate = 1.1
        self.verbose = True
        self.solutionsPath = []
        self.history = []
        self.f_primal = None #must be define! it needs a .prox method to be define
        self.h_primal = None #it needs a .grad method to be define

        self.Gs_dual = [] #List of g_s functions! they need a .prox_conj method

        self.LinOp = [] # List of linear operators, each of them needs a .T (transpose) method defined

        self.x1 = args[0] #primal variable. It must be nd.array

        self._bestIT = 1
        self._fval = np.Inf
        self._REP = 10
        self._StoppingCriteria = np.Inf * np.ones(self._REP)
        self._Method = "None"
        #dual Variables. They must be ndarray or tuple of ndarrays
        self.dualVars = []
        for i in range(1,len(args)):
            self.dualVars.append(args[i])
            
        self.__dict__.update(kwargs)

    def eval(self,x):
        fval = self.f_primal(x)
        if self.h_primal:
            fval += self.h_primal(x)

        fval += sum([self.Gs_dual[i](self.LinOp[i](x)) for i in range(len(self.LinOp))])
        fval += sum([self.Gs_dual[i](x) for i in range(len(self.Gs_dual) - len(self.LinOp))])
        return fval
 
  
    def update(self):
        pass
        
    def solve(self):

        print("******************************")
        print("Solving using method " + self._Method + "\n")
        
        for i in range(self.maxIT):
            self.update()
            fval = self.eval(self.x1)
            #store the best value
            if fval < self._fval:
                self._fval = fval
                self._bestIT = i
            #update StoppingCriteria
            if any(fval < self._StoppingCriteria):
                self._StoppingCriteria[np.argmax(self._StoppingCriteria)] = fval
            #check if you are in the plane
            if all(np.abs(fval - self._StoppingCriteria) < self.tol):
                print(f"Stopcriteria: {self._REP} minimum points on a plane satisfied")
                break
            if self.verbose:
                self.solutionsPath.append(self.x1)
                self.history.append(fval)
        print(f"\nThe algorithm run {i} iterations.")
        print(f"The minimum was found at the iteration {self._bestIT}.\n\n")

       

class FBF(mySolver):
    r"""This algorithm solves problems in the form
    \minimize f(x) + \sum_{i=1}^n(g_i(L_i(x))) + h(x)

    where f,g_i are convex functions (we need to know prox_f and prox_{g^*},
    L_i is a linear operator (we need to knwo \|L_i\|)
    and h is convex differentiable with Lipschitz constinuous gradient (we need to know the Lipschitz constant)
    Ref https://doi.org/10.48550/arXiv.1107.0081
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._Method = "FBF"
        self.p_1 = 0
        self.SumLTp_2 = 0
        self.y_1 = 0
        self.solutionsPath.append(self.x1)
        #now we determine the learning rate
        beta = 0
        for i in range(0,len(self.Gs_dual) - len(self.LinOp)):
            beta +=1
        for L in self.LinOp:
            beta += L.LC ** 2
        beta = np.sqrt(beta)
        if self.h_primal:
            beta += self.h_primal.LC
        eta = (1-1e-12)/(beta+1)
        self.learning_rate = (1.- eta)/beta

        # new set the update x method
    def forward_1(self):
        self.y_1 = 0
        
        for i, L in enumerate(self.LinOp):
            self.y_1 += L.T(self.dualVars[i])  
        for i in range(len(self.Gs_dual) - len(self.LinOp)):
            self.y_1 += self.dualVars[-1-i]
        if self.h_primal:
            self.y_1 += self.h_primal.grad(self.x1) 
        self.p_1 = self.f_primal.prox(self.x1 - self.learning_rate * self.y_1,
                                      self.learning_rate)
               
    def updateV(self):
        self.SumLTp_2 = 0
        def recFBF_v(v1,g,L=None):
            if not L:
                L = lambda x : x
                L.T = lambda x: x
            p_2 = g.prox_conj(v1 + self.learning_rate * L(self.x1), self.learning_rate)
            self.SumLTp_2 += L.T(p_2)
            q_2 = p_2 + self.learning_rate * L(self.p_1)
            return q_2 - self.learning_rate * L(self.x1)
            
        for i,L in enumerate(self.LinOp):
            self.dualVars[i] = recFBF_v(self.dualVars[i],
                                        self.Gs_dual[i],
                                        L)
        for i in range(len(self.Gs_dual) - len(self.LinOp)):
            self.dualVars[-1-i] = recFBF_v(self.dualVars[-1-i],self.Gs_dual[-1-i])

    def forward_2(self):
        q_1 = self.p_1 - self.learning_rate * self.SumLTp_2
        if self.h_primal:
            q_1 -= self.learning_rate * self.h_primal.grad(self.p_1)

        self.x1 = self.learning_rate * self.y_1 + q_1

    def update(self):
        self.forward_1()
        self.updateV()
        self.forward_2()
        
    
                 
class FRB(mySolver):
    r"""This algorithm solves problems in the form
    \minimize f(x) + \sum_{i=1}^n(g_i(L_i(x))) + h(x)

    where f,g_i are convex functions (we need to know prox_f and prox_{g^*},
    L_i is a linear operator (we need to knwo \|L_i\|)
    and h is convex differentiable with Lipschitz constinuous gradient (we need to know the Lipschitz constant)
    Ref https://doi.org/10.48550/arXiv.1808.04162 """
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._Method = "FRB"
        self.solutionsPath.append(self.x1)
        self.x0 = self.x1.copy()
        self.dualVars0 = self.dualVars.copy()
        beta = 0

        # Determine learning rate
        for i in range(0,len(self.Gs_dual) - len(self.LinOp)):
            beta +=1
        for L in self.LinOp:
            beta += L.LC ** 2
        beta = np.sqrt(beta)
        if self.h_primal:
            beta += self.h_primal.LC

        self.learning_rate = ((1.- 1e-10)/(2*beta), (1. - 1e-10)/(2*beta))
        # new set the update x method
    def updateX(self):
        p = 0
        partial_sum_dual = 0
        partial_sum_dual_noL = 0
        new_alpha = sum(self.learning_rate)
        old_alpha = self.learning_rate[0]

        for i,L  in enumerate(self.LinOp):
            partial_sum_dual += L.T(new_alpha * self.dualVars[i] - old_alpha * self.dualVars0[i])  

        for i in range(len(self.Gs_dual) - len(self.LinOp)):
            partial_sum_dual += new_alpha * self.dualVars[-1-i] - old_alpha * self.dualVars0[-1-i]

        if self.h_primal:
            partial_sum_dual += new_alpha * self.h_primal.grad(self.x1) - old_alpha * self.h_primal.grad(self.x0)

        p = self.x1 -  partial_sum_dual 
        p = self.f_primal.prox(p,old_alpha)
        self.x1, self.x0 =  p,self.x1
               
    def updateV(self):
        def recFRB_v(v1,x,g,gamma,L=None):
            if not L:
                L = lambda x : x
            new_gamma = sum(gamma)
            old_gamma = gamma[0]
            y = L(new_gamma *  x[1] - old_gamma * x[0])
            p = g.prox_conj(v1[1] + y,old_gamma)        
            return p.copy(),v1[1].copy()


        for i,L in enumerate(self.LinOp):
            self.dualVars[i],self.dualVars0[i] = recFRB_v((self.dualVars0[i],self.dualVars[i]),
                                                          (self.x0,self.x1),
                                                          self.Gs_dual[i],
                                                          self.learning_rate,
                                                          L)

        for i in range(len(self.Gs_dual) - len(self.LinOp)):
            self.dualVars[-1-i],self.dualVars0[-1-i] = recFRB_v((self.dualVars0[-1-i],self.dualVars[-1-i]),
                                                          (self.x0,self.x1),
                                                          self.Gs_dual[-1-i],
                                                          self.learning_rate)


    def update(self):
        self.updateX()
        self.updateV()





class FRBD(mySolver):
    r"""This algorithm solves problems in the form
    \minimize f(x) + \sum_{i=1}^n(g_i(L_i(x))) + h(x)
    in the same way as FRB but including an adaptative line search
    where f,g_i are convex functions (we need to know prox_f and prox_{g^*},
    L_i is a linear operator (we need to knwo \|L_i\|)
    and h is convex differentiable with Lipschitz constinuous gradient (we need to know the Lipschitz constant)
    Ref https://doi.org/10.48550/arXiv.1808.04162 """
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._Method = "FRBD"
        self.solutionsPath.append(self.x1)
        self.x0 = self.x1.copy()
        self.dualVars0 = self.dualVars.copy()
        self.numberSteps = 3 #how many steps do you want to ckeck?
        beta = 0

        # Determine learning rate
        for i in range(0,len(self.Gs_dual) - len(self.LinOp)):
            beta +=1
        for L in self.LinOp:
            beta += L.LC ** 2
        beta = np.sqrt(beta)
        if self.h_primal:
            beta += self.h_primal.LC

        maxEpsilon = np.linspace(0,1./(2*(1+beta)),self.numberSteps)
        self.steps =  (1 - maxEpsilon)/(2*beta)   #Broadcasting
        self.steps = np.append(1./(1+beta),self.steps) #add a big stepsize
        self.learning_rate = np.array((self.steps[0] ,self.steps[0]))
            # new set the update x method
    def updateX(self):


        actual_fval = self.eval(self.x1)

        for step in self.steps:
            self.learning_rate[1] = step
            new_alpha = sum(self.learning_rate)
            old_alpha = self.learning_rate[0]


            p = 0
            partial_sum_dual = 0
            partial_sum_dual_noL = 0



            for i,L  in enumerate(self.LinOp):
                partial_sum_dual += L.T(new_alpha * self.dualVars[i] - old_alpha * self.dualVars0[i])  

            for i in range(len(self.Gs_dual) - len(self.LinOp)):
                partial_sum_dual += new_alpha * self.dualVars[-1-i] - old_alpha * self.dualVars0[-1-i]

            if self.h_primal:
                partial_sum_dual += new_alpha * self.h_primal.grad(self.x1) - old_alpha * self.h_primal.grad(self.x0)

            

            p = self.x1 -  partial_sum_dual 
            p = self.f_primal.prox(p,old_alpha)

            if self.eval(p) < actual_fval:
                self.learning_rate[0] = step
                break
            else:
                self.learning_rate[0] = step


        self.x1, self.x0 =  p,self.x1

        
               
    def updateV(self):
        def recFRB_v(v1,x,g,gamma,L=None):
            if not L:
                L = lambda x : x
            new_gamma = sum(gamma)
            old_gamma = gamma[0]
            y = L(new_gamma *  x[1] - old_gamma * x[0])
            p = g.prox_conj(v1[1] + y,old_gamma)        
            return p.copy(),v1[1].copy()


        for i,L in enumerate(self.LinOp):
            self.dualVars[i],self.dualVars0[i] = recFRB_v((self.dualVars0[i],self.dualVars[i]),
                                                          (self.x0,self.x1),
                                                          self.Gs_dual[i],
                                                          self.learning_rate,
                                                          L)

        for i in range(len(self.Gs_dual) - len(self.LinOp)):
            self.dualVars[-1-i],self.dualVars0[-1-i] = recFRB_v((self.dualVars0[-1-i],self.dualVars[-1-i]),
                                                          (self.x0,self.x1),
                                                          self.Gs_dual[-1-i],
                                                          self.learning_rate)


    def update(self):
            self.updateX()
            self.updateV()
                
       
   
    
