import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

amin,amax,bmin,bmax,Na,Nb,Ntheta=5,8,5,8,3,3,10

def discreteP(amin,amax,bmin,bmax,Na,Nb,Ntheta):
    A=np.linspace(amin,amax,Na)
    B=np.linspace(bmin,bmax,Nb)
    Theta=np.linspace(0,90,Ntheta+1)[:-1]
    [X,Y,Z]=np.meshgrid(A,B,Theta)
#    print(X)
    P=np.zeros((Nb,Na,Ntheta,3))
#    print(P,P[:,:,:,0])
    P[:,:,:,0]=X
    P[:,:,:,1]=Y
    P[:,:,:,2]=Z
    return P.reshape(Na*Nb*Ntheta,3)
t0=time.time()

P=discreteP(amin,amax,bmin,bmax,Na,Nb,Ntheta)
#print(P)

a,b,theta,m,n=10,30,30,100,100

def Image(a,b,theta,m,n):
    I=np.zeros((m,n))
    [X,Y]=np.meshgrid(np.linspace(-(m-1)/2,(m-1)/2,m),np.linspace(-(n-1)/2,(n-1)/2,n))
    I=(((X*np.cos(theta*np.pi/180)+Y*np.sin(theta*np.pi/180))/a)**2+((Y*np.cos(theta*np.pi/180)-X*np.sin(theta*np.pi/180))/b)**2)<1
    return I

I=Image(a,b,theta,m,n)
#print(I)

k,l=100,100
    
def dicotisation(Param,k,l):
    d=Param.shape[0]
    D=np.zeros((k,l,d))
    for i in range(d):
        a,b,theta=Param[i]
        D[:,:,i]=np.fft.fft2(np.fft.fftshift(Image(a,b,theta,k,l)))
    return D

def Df(x,D):
    u=np.zeros((D.shape[0],D.shape[1]),dtype='complex')
    for i in range(D.shape[2]):
        u+=np.fft.ifft2(np.fft.fft2(x[:,:,i])*D[:,:,i])
    return np.real(u)
        
d=P.shape[0]
x=np.random.rand(k,l,d)
x[x<=0.99999]=0

D=dicotisation(P,k,l)
uu=Df(x,D)

plt.figure(1)
plt.imshow(uu)
#symétrie hermitienne
#\mathop{\mathrm{argmin}}