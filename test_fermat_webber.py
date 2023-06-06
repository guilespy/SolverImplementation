import numpy as np
import matplotlib.pyplot as plt
import myUtils.myOperators as mo
import myUtils.myFunctions as mf
import mySolvers.mySolvers as ms


MAXITER = 10000
x_sol = np.array([0, 0])

# Starting Points and functions.

x1 = np.array([144,-150])

v1 = np.array([10,10])
v2 = np.array([-1,-1])
v3 = np.array([1,-1])

c1 = np.array([59, 0])
c2 = np.array([20, 0])
c3 = np.array([-20, 48])
c4 = np.array([-20,-48])


lambda_1 = 5
lambda_2 = 5
lambda_3 = 13
lambda_4= 13

f = mf.EukNorm(lambda_1, c1)
g1 = mf.EukNorm(lambda_2, c2)
g2 = mf.EukNorm(lambda_3, c3)
g3 = mf.EukNorm(lambda_4, c4)

mysolFRB = ms.FRB(x1,v1,v2,v3,f_primal=f,Gs_dual=[g1,g2,g3], tol = 1e-10,maxIT = 1500)
mysolFBF = ms.FBF(x1,v1,v2,v3,f_primal=f,Gs_dual=[g1,g2,g3], tol = 1e-10, maxIT = 1500)
mysolFRBD = ms.FRBD(x1,v1,v2,v3,f_primal=f,Gs_dual=[g1,g2,g3], tol = 1e-10, maxIT = 1500)


mysolFRB.solve()
mysolFBF.solve()
mysolFRBD.solve()


print(f"Solution FRB: {mysolFRB.solutionsPath[mysolFRB._bestIT]}")
print(f"Solution FBF: {mysolFBF.solutionsPath[mysolFBF._bestIT]}")
print(f"Solution FRBD: {mysolFBF.solutionsPath[mysolFBF._bestIT]}")


def plottingResults(*methods,
                    c=[c1,c2,c3,c4],
                    name='pathFW.pdf'):
    

    style = ['b-d','g-o','r-^']
    c = np.asarray(c)
    
    
    fig,axes = plt.subplots(1,1,figsize=(20,10))
    
    for index,m in enumerate(methods):
        ax = axes
        data = np.asarray(m.solutionsPath)
        #ax.scatter(data[:,0],data[:,1],color='red')
        ax.plot(data[:,0],data[:,1],style[index],label=m._Method)
    
    ax.set_title("Fermat Weber")
    ax.scatter(c[:,0],c[:,1],s=50,color='black')
    ax.legend()
    plt.show()

def plottingResults3d(sFRB,sFBF,
                    c=[c1,c2,c3,c4],
                    name='pathFW.pdf'):
    style = ['b-d','g-o']
    c = np.asarray(c)
    
    
    fig,axes = plt.subplots(1,2,figsize=(20,40))
    axes[0] = plt.axes(projection = "3d")
    ax = axes[0]
    ax.scatter3D(c[:,0],c[:,1],c[:,2],s=50,color='black')
    #axes[1] = plt.axes(projection="3d")
    #ax = axes[1]
    #ax.scatter3D(c[:,0],c[:,1],c[:,2],s=50,color='black')
    #ax.scatter3D(sFBF[:,0],sFBF[:,1],sFBF[:,2],color='green')

    ax.scatter3D(sFRB[:,0],sFRB[:,1],sFRB[:,2],color='red')


    plt.show()


plottingResults(mysolFRB,mysolFBF,mysolFRBD)


    
