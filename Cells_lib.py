import numpy as np
import scipy.misc
#import scipy as scp
#import pylab as pyl
#import pandas as pd
#import holoviews as hv
#import param
#import panel as pn
#import requests
import matplotlib.pyplot as plt

#from panel.pane import LaTeX
#hv.extension('bokeh')
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
#from io import BytesIO

#import time
options = dict(cmap='gray',xaxis=None,yaxis=None,width=400,height=400,toolbar=None)

def discreteP(amin,amax,bmin,bmax,Na,Nb,Ntheta):
    A=np.linspace(amin,amax,Na) #discrétisation des valeurs possibles pour le premier demi-axe 
    B=np.linspace(bmin,bmax,Nb) #discrétisation des valeurs possibles pour le second demi-axe
    Theta=np.linspace(0,90,Ntheta+1)[:-1] #discrétisation des valeurs possibles pour l'inclinaison
    [X,Y,Z]=np.meshgrid(A,B,Theta)
    P=np.zeros((Nb,Na,Ntheta,3))
    P[:,:,:,0]=X
    P[:,:,:,1]=Y
    P[:,:,:,2]=Z
    return P.reshape(Na*Nb*Ntheta,3) 

def create_Image(a,b,theta,m,n):
    #crée une image m*n contenant une ellipse de demis axes a et b et d'inclinaison theta
    I=np.zeros((m,n))
    [X,Y]=np.meshgrid(np.linspace(-(m-1)/2,(m-1)/2,m),np.linspace(-(n-1)/2,(n-1)/2,n))
    I=(((X*np.cos(theta*np.pi/180)+Y*np.sin(theta*np.pi/180))/a)**2+((Y*np.cos(theta*np.pi/180)-X*np.sin(theta*np.pi/180))/b)**2)<1
    return I/np.sum(I)

def spline(m,n,a,b):
    I=np.zeros((m,n))
    [X,Y]=np.meshgrid(np.linspace(-a,m-a,m),np.linspace(-b,n-b,n))
    I=(X**2+Y**2)*np.log(np.sqrt(X**2+Y**2))
    return I/np.sum(I)


def dictionnaire(Param,k,l):
    d=Param.shape[0]
    D=np.zeros((k,l,d),dtype=complex)
    for i in range(d):
        a,b,theta=Param[i]
        D[:,:,i]=np.fft.fft2(np.fft.fftshift(create_Image(a,b,theta,k,l)))
    return D

def Df(x,D):
    #Calcul de Dx
    u=np.zeros((D.shape[0],D.shape[1]),dtype='complex')
    for i in range(D.shape[2]):
        xx=x[:,:,i]
        u+=np.fft.ifft2(D[:,:,i]*np.fft.fft2(xx))
    return np.real(u) 


def Dtf(x,D):
    #Calcul de D*x
    u=np.zeros((D.shape[0],D.shape[1],D.shape[2]),dtype='complex')
    for i in range(D.shape[2]):
        di=D[:,:,i]
        u[:,:,i]=np.fft.ifft2(np.fft.fft2(x)*(di.conjugate()))
    return np.real(u)


def Gradient(x,D,u):
    delta=Df(x,D)-u
    return Dtf(delta,D)

def stepFrank(x,D,u,tau,lbd):
    k,l,d=x.shape
    g=Gradient(x,D,u)
    arg=np.argmin(g)
    s=np.zeros(k*l*d)
    s[arg]=1
    s=s.reshape(k,l,d)
    #recherche linaire d'un pas "optimal" entre 0 et tau
    beta=tau*Df(s,D)
    gamma=(1-tau)*Df(x,D)-u
    err=np.linalg.norm(gamma)
    n=100000
    pas=lbd
    for alpha in np.linspace(0,lbd,n):
        if np.linalg.norm(alpha*beta+gamma)<err:
            pas=alpha
            err=np.linalg.norm(alpha*beta+gamma)
    return x+tau*(pas*s-x)

def frankWolfe(x0,D,u,niter,lbd):
    #Compute the optimization of1/2||Dx-u||_2^2
    k=0
    x=x0*1.
    F=[]
    while k<niter:
        tau=2/(2+k)
        x=stepFrank(x,D,u,tau,lbd)
        F.append(1/2*np.linalg.norm(Df(x,D)-u))
        k+=1
    return x,F

def fondPol(m,n):
    P=np.ones((m,n,5))
    X,Y=np.meshgrid(range(m),range(n))
    P[:,:,1]=X
    P[:,:,2]=Y
    P[:,:,3]=X**2
    P[:,:,4]=Y**2
    P=P.reshape((m*n,5))
    return P

def frankWolfe2(x0,D,u,niter,lbd):
    #Compute the optimization of 1/2||Dx+Py-u||_2^2
    #At each iteration we find optimal y with least square-method
    umin=np.min(u)
    k=0
    x=x0*1.
    m,n=u.shape
    F=[]
    P=FondSplines(m,n,3)
    while k<niter:
        tau=2/(2+k)
        Dx=Df(x,D)
        y=np.linalg.lstsq(P,(u-Dx).reshape(m*n))[0]
        Py=np.clip((P@y).reshape(m,n),umin,None)
        x=stepFrank(x,D,u-Py,tau,lbd)
        F.append(1/2*np.linalg.norm(Dx+Py-u))
        k+=1
    return x,F

def stepFrank2(x,D,u):
    k,l,d=x.shape
    g=Gradient(x,D,u)
    arg=np.argmin(g)
    s=np.zeros(k*l*d)
    s[arg]=1
    s=s.reshape(k,l,d)
    x=x+s
    indices=np.where(x>0)
    N=len(indices[0])
    Di=np.zeros((k*l,N))
    for i in range(N):
        Di[:,i]=(np.roll(np.roll(np.fft.ifft2(D[:,:,indices[2][i]]),indices[0][i],axis=0),indices[1][i],axis=1)).reshape(k*l)
    xopt=np.linalg.lstsq(Di,u.reshape(k*l))
    x[indices]=xopt[0]
    return x

def FondSplines(m,n,k):
    P=np.ones((m,n,k**2+5))
    X,Y=np.meshgrid(range(m),range(n))
    P[:,:,1]=X/np.sum(X)
    P[:,:,2]=Y/np.sum(Y)
    P[:,:,3]=X**2/np.sum(X**2)
    P[:,:,4]=Y**2/np.sum(Y**2)
    for i in range(k):
        for j in range(k):
            P[:,:,4+i*k+j]=spline(m,n,int((i+1/2)*m/k),int((j+1/2)*n/k))
    P=P.reshape((m*n,k**2+5))
    return P


def frankWolfe3(x0,D,u,niter,lbd):
    #Compute the optimization of 1/2||Dx+Py-u||_2^2
    #At each iteration we find optimal y with least square-method, and then the cell to add with Frank-Wolfe and then we find optimal x with least square-method 
    umin=np.min(u)
    k=0
    x=x0*1.
    m,n=u.shape
    F1=[]
    P=FondSplines(m,n,3)
    while k<niter:
        y=np.linalg.lstsq(P,(u-Df(x,D)).reshape(m*n))[0]
        if False:
            Py=np.clip((P@y).reshape(m,n),umin,None)
        else :
            Py=(P@y).reshape(m,n)
        x=stepFrank2(x,D,u-Py)
        F1.append(1/2*np.linalg.norm(Df(x,D)+Py-u))
        k+=1
    return x,F1


def ForwardBackward(x,b,A,step,lam,Niter):
    #Forward-Backward algorithm to minimize F(x)=f(x)+g(x) avec f(x)=1/2||Ax-b||_2^2 et g(x)=lam||x||_1
    for n in range(Niter):
        Grad=np.conjugate(A).transpose()@(A@x-b)
        x=euclidean_proj_simplex((x-step*Grad).reshape(np.product((x-step*Grad).shape)),lam).reshape(x.shape)
        if False:    
            plt.imshow((A@x).reshape((100,100)))
            plt.show()
    return x

def frankWolfe4(x0,D,u,niter,lbd,step=1,Niter=50):
    #Wompute the optimization of 1/2||Dx+Py-u||_2^2
    #At each iteration we find optimal y with least square-method, and then the cell to add with Frank-Wolfe
    k=0
    x=x0*1.
    m,n=u.shape
    F1=[]
    P=FondSplines(m,n,3)
    while k<niter:
        y=np.linalg.lstsq(P,(u-Df(x,D)).reshape(m*n))[0]
        Py=(P@y).reshape(m,n)
        x=stepFrank(x,D,u-Py,1/2,100)
#        y=np.linalg.lstsq(P,(u-Df(x,D)).reshape(m*n))[0]
#        Py=(P@y).reshape(m,n)
        indices=np.where(x>0)
        N=len(indices[0])
        Dk=np.zeros((m*n,N))
        xx=x[np.where(x>0)]
        for i in range(N):
            Dk[:,i]=(np.roll(np.roll(np.fft.ifft2(D[:,:,indices[2][i]]),indices[0][i],axis=0),indices[1][i],axis=1)).reshape(m*n)
        xx=ForwardBackward(xx.reshape(N,1),(u-Py).reshape(m*n,1),Dk,100/(k+1),lbd,Niter)
        x[indices]=xx.reshape(N)
        F1.append(1/2*np.linalg.norm(Df(x,D)+Py-u))
        k+=1
        #if true : plot the image every per_aff iteration
        per_aff=20
        if k%per_aff==1:
            plt.imshow(Df(x,D))
            plt.show()
    return x,F1


""" Module to compute projections on the positive simplex or the L1-ball

A positive simplex is a set X = { \mathbf{x} | \sum_i x_i = s, x_i \geq 0 }

The (unit) L1-ball is the set X = { \mathbf{x} | || x ||_1 \leq 1 }

Adrien Gaidon - INRIA - 2011
"""


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.

    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() <= s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / (rho+1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

