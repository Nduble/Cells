import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
from PIL import Image

IMAGES=["Optimum_Grid_18.tif","Optimum_Grid_19.tif","Optimum_Grid_20.tif","Optimum_Grid_25.tif","Optimum_Grid_26.tif"]
image=np.array(Image.open("Optimum_Grid_19.tif"))
for file in IMAGES :
    image=np.array(Image.open(file))
    plt.hist(np.sort(image.reshape(10000)))
    plt.show()
def traitement(image,seuil=0.3):
    N=image.shape[0]*image.shape[1]
    S=np.sort(image,axis=None)[int(N*(1-seuil))]
    print(S)
    I=image*1.
    I[I<S]=0
    return I

Amin,Amax,Bmin,Bmax,Na,Nb,Ntheta=5,8,5,8,3,3,5

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

P=discreteP(Amin,Amax,Bmin,Bmax,Na,Nb,Ntheta) 
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
        xx=x[:,:,i]
        u+=np.fft.ifft2(D[:,:,i]*np.fft.fft2(xx))
#        if np.max(xx)>0:
#            plt.imshow(np.real(u))
#        plt.show()
    return np.real(u)
        
d=P.shape[0]
x=np.random.rand(k,l,d)
x[x<=0.99999]=0
x=x*np.random.rand(k,l,d)*5

x0=x*1.
#x0[:,:,:50]=0

D=dicotisation(P,k,l)
u0=Df(x0,D)

uu=Df(x,D)

#plt.figure(1)
#plt.imshow(uu)
#plt.show()
#plt.imshow(u0)
#plt.show()
#symétrie hermitienne
#\mathop{\mathrm{argmin}}

def Dtf(x,D):
    u=np.zeros((D.shape[0],D.shape[1],D.shape[2]),dtype='complex')
    for i in range(D.shape[2]):
        di=D[:,:,i]
        u[:,:,i]=np.fft.ifft2(np.fft.fft2(x)*(di.conjugate()))
    return np.real(u)
        

def Gradient(x,D,u):
    delta=Df(x,D)-u
    return Dtf(delta,D)

x=np.zeros((k,l,d))
g=Gradient(x,D,uu)

def stepFrank(x,D,u,tau,lbd):
    k,l,d=x.shape
    g=Gradient(x,D,u)
    arg=np.argmin(g)
    s=np.zeros(k*l*d)
    s[arg]=1
    s=s.reshape(k,l,d)
    n=10
    beta=tau*Df(s,D)
    gamma=(1-tau)*Df(x,D)-u
    err=np.linalg.norm(gamma)
    pas=lbd
    for alpha in np.linspace(0,lbd,n):
        if np.linalg.norm(alpha*beta+gamma)<err:
            pas=alpha
            err=np.linalg.norm(alpha*beta+gamma)
    return x+tau*(pas*s-x)

def frankWolfe(x0,D,u,niter,lbd):
    k=0
    x=x0*1.
    Lbd=[]
    while k<niter:
        tau=2/(2+k)
        x=stepFrank(x,D,u,tau,lbd)
        Lbd.append(np.sum(np.abs(x)))
        if k%100==0:
            plt.imshow(Df(x,D))
            plt.show()
            #plt.imshow(uu)
            #plt.show()
        k+=1
    return x,Lbd





plt.imshow(image)

plt.show()
for seuil in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    plt.imshow(traitement(image,seuil))
    plt.show()
"""
plt.show()
time.sleep(10)
"""
"""
for i in np.linspace(1,7,5):
    xx=frankWolfe(x,D,uu,100,i)
    plt.imshow(Df(xx,D))
    plt.show()
"""
"""
xx,ll=frankWolfe(x,D,uu,1000,500000)
plt.imshow(Df(xx,D))
plt.show()

"""
"""
xx,ll=frankWolfe(x,D,image,500,100000)
plt.imshow(Df(xx,D))
plt.show()
"""