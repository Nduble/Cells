import Cells_lib 
#scipy.misc.imsave('outfile.jpg', image_array)
##----------------- Initialisations Données -------------------##

#paramètres du dictionnaire 
Amin,Amax,Bmin,Bmax,Na,Nb,Ntheta=5,8,5,8,3,3,5

#Taille de l'image standard
k,l=100,100 

##-------- Initialisations des variables du calcul----------------##


P=discreteP(Amin,Amax,Bmin,Bmax,Na,Nb,Ntheta)
d=P.shape[0] 
D=dictionnaire(P,k,l)

#Image : 1 pour image synthétique, 2 pour image réelle
IMAGE=3

if IMAGE==0:
    #Création d'une image de synthèse aléatoire
#    x=np.random.rand(k,l,d)
#    x[x<=0.99999]=0
#    x=x*np.random.rand(k,l,d)*5
    x=np.load("xtest.npy")
    utest=Df(x,D)
    lbd=30
else :
    file="Cell"+str(IMAGE)+".tif"
    utest=np.array(Image.open(file))
    lbd=1000000
    
plt.imshow(utest)
plt.show()
TEST=2

fonction_test= frankWolfe3
Niter=30


uref=np.linalg.norm(utest)        
seuil=0.8

if TEST==1:
    x,F=fonction_test(np.zeros((k,l,d)),D,utest,Niter,lbd)
    plt.imshow(Df(x,D))
    plt.show()
    plt.loglog(F/uref)
elif TEST==2:
    x1,F1=frankWolfe(np.zeros((k,l,d)),D,utest,Niter,lbd)
    x2,F2=frankWolfe2(np.zeros((k,l,d)),D,utest,Niter,lbd)
    x3,F3=frankWolfe3(np.zeros((k,l,d)),D,utest,Niter,lbd)
    x4,F4=frankWolfe4(np.zeros((k,l,d)),D,utest,Niter,lbd)
    if False :
#        pn.Column(pn.Row(hv.Image(Df(x1,D)).opts(**options),hv.Image(Df(x2,D)).opts(**options)),
#              pn.Row(hv.Image(Df(x3,D)).opts(**options),hv.Image(image).opts(**options)))
        plt.imshow(Df(x1,D))
        plt.show()
        plt.imshow(Df(x2,D))
        plt.show()
#        plt.imshow(Df(x3,D))
#        plt.show()
        plt.imshow(Df(x4,D))
        plt.show()
        plt.imshow(utest)
        plt.show()
    if True :
        N0=5
        plt.loglog(F1[N0:]/uref,label="algorithme naif")
        plt.loglog(F2[N0:]/uref,label="ajout du fond")
        plt.loglog(F3[N0:]/uref,label="optimisation de x")
        plt.loglog(F4[N0:]/uref,label="ajout de splines")
        plt.legend
        plt.show()
    if False :
        xp1=x1[np.where(x1!=0)]
        plt.hist(xp1)
        plt.show()
        print("norme l0 de x:",np.sum(x1!=0),"norme l1 de x:",np.sum(np.abs(x1)))
        xp2=x2[np.where(x2!=0)]
        plt.hist(xp2)
        plt.show()
        print("norme l0 de x:",np.sum(x2!=0),"norme l1 de x:",np.sum(np.abs(x2)))
        xp3=x3[np.where(x3!=0)]
        plt.hist(xp3)
        plt.show()
        print("norme l0 de x:",np.sum(x3!=0),"norme l1 de x:",np.sum(np.abs(x3)))
        xp4=x4[np.where(x4!=0)]
        plt.hist(xp4)
        plt.show()
        print("norme l0 de x:",np.sum(x4!=0),"norme l1 de x:",np.sum(np.abs(x4)))
elif TEST==3:
    utest2=utest*(np.random.rand(k,l)<seuil)
    xx1,FF1=frankWolfe(np.zeros((k,l,d)),D,utest,Niter,lbd)
    xx2,FF2=frankWolfe2(np.zeros((k,l,d)),D,utest,Niter,lbd)
    xx4,FF4=frankWolfe4(np.zeros((k,l,d)),D,utest,Niter,lbd,Niter=30)
    if True :
#        pn.Column(pn.Row(hv.Image(Df(x1,D)).opts(**options),hv.Image(Df(x2,D)).opts(**options)),
#              pn.Row(hv.Image(Df(x3,D)).opts(**options),hv.Image(image).opts(**options)))
        plt.imshow(Df(xx1,D))
        plt.show()
        plt.imshow(Df(xx2,D))
        plt.show()
        plt.imshow(Df(xx3,D))
        plt.show()
        plt.imshow(Df(xx4,D))
        plt.show()
        plt.imshow(utest)
        plt.show()
        plt.imshow(utest2)
        plt.show()
    if True :
        N0=5
        plt.loglog(F1[N0:]/uref,label="algorithme naif")
        plt.loglog(F2[N0:]/uref,label="ajout du fond")
        plt.loglog(F4[N0:]/uref,label="ajout de splines")
        plt.loglog(FF1[N0:]/uref,label="algorithme naif")
        plt.loglog(FF2[N0:]/uref,label="ajout du fond")
        plt.loglog(FF4[N0:]/uref,label="ajout de splines")
        plt.legend
        plt.show()
    