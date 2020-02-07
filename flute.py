import numpy as np
import matplotlib.pyplot as plt
import scipy

amin,amax,bmin,bmax,Na,Nb,Ntheta=3,15,3,15,30,30,10

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

P=discreteP(amin,amax,bmin,bmax,Na,Nb,Ntheta)
#print(P)

def rotation2(u,theta):
    return np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])@u

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
    n=Param.shape[0]
    m=k*l
    D=np.zeros((m,n))
    for i in range(n):
        a,b,theta=Param[i]
        D[:,i]=Image(a,b,theta,k,l).reshape(1,m)
    return D    

def Translation(Dk,dx,dy):
    N=int(np.sqrt(Dk.shape[0]))
    T=np.eye(Dk.shape[0],k=dx+N*dy)
    return np.abs(np.fft.ifft2(np.fft.fft2(T)@np.fft.fft2(Dk))/N**2)

D1=dicotisation(P,k,l)
D2=Translation(D1,20,-30)
D3=Translation(D1,30,40)

Cell1=D1[:,int(D1.shape[1]*np.random.rand())].reshape(k,l)
Cell2=D2[:,int(D1.shape[1]*np.random.rand())].reshape(k,l)
Cell3=D3[:,int(D1.shape[1]*np.random.rand())].reshape(k,l)

u=2*Cell1+Cell2+0.4*Cell3
plt.imshow(u)

#def Descente(D,x,u):
#    Grad=D.transpose()@(D@x-u)
    