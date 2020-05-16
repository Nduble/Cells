import numpy as np
import scipy as scp
import pylab as pyl
import pywt
import pandas as pd
import holoviews as hv
import param
import panel as pn
import requests
import matplotlib.pyplot as plt

from panel.pane import LaTeX
hv.extension('bokeh')
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from io import BytesIO

import time
from PIL import Image
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

def spline(m,n):
    I=np.zeros((m,n))
    [X,Y]=np.meshgrid(np.linspace(-(m-1)/2,(m-1)/2,m),np.linspace(-(n-1)/2,(n-1)/2,n))
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
    n=100
    pas=lbd
    for alpha in np.linspace(0,lbd,n):
        if np.linalg.norm(alpha*beta+gamma)<err:
            pas=alpha
            err=np.linalg.norm(alpha*beta+gamma)
    return x+tau*(pas*s-x)

def frankWolfe(x0,D,u,niter,lbd):
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
    #On optimise 1/2||Dx+Py-u||_2^2
    #A chaque itération on trouve y optimal par MC
    umin=np.min(u)
    k=0
    x=x0*1.
    m,n=u.shape
    F=[]
    P=fondPol(m,n)
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

def frankWolfe3(x0,D,u,niter,lbd):
    #On optimise 1/2||Dx+Py-u||_2^2
    #A chaque itération on trouve y optimal par MC puis on optimise les coefficients de x par MC
    umin=np.min(u)
    k=0
    x=x0*1.
    m,n=u.shape
    F1=[]
    P=np.ones((m,n,5))
    X,Y=np.meshgrid(range(m),range(n))
    P[:,:,1]=X
    P[:,:,2]=Y
    P[:,:,3]=X**2
    P[:,:,4]=Y**2
    P=P.reshape((m*n,5))
    while k<niter:
        y=np.linalg.lstsq(P,(u-Df(x,D)).reshape(m*n))[0]
        Py=np.clip((P@y).reshape(m,n),umin,None)
        x=stepFrank2(x,D,u-Py)
        F1.append(1/2*np.linalg.norm(Df(x,D)+Py-u))
        k+=1
    return x,F1