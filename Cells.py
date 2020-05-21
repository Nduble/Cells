import Cells_lib 

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
    x=np.random.rand(k,l,d)
    x[x<=0.99999]=0
    x=x*np.random.rand(k,l,d)*5
    utest=Df(x,D)
    lbd=30
else :
    file="Cell"+str(IMAGE)+".tif"
    utest=np.array(Image.open(file))
    lbd=150000000
    
plt.imshow(utest)
plt.show()
TEST=1
fonction_test= frankWolfe4
Niter=100

if TEST==1:
    x,F=fonction_test(np.zeros((k,l,d)),D,utest,50,lbd)
    plt.imshow(Df(x,D))
    plt.show()
    plt.loglog(F)
elif TEST==2:
    x1,F1=frankWolfe(np.zeros((k,l,d)),D,utest,Niter,lbd)
    x2,F2=frankWolfe2(np.zeros((k,l,d)),D,utest,Niter,lbd)
    x3,F3=frankWolfe3(np.zeros((k,l,d)),D,utest,Niter,lbd)
    x4,F4=frankWolfe4(np.zeros((k,l,d)),D,utest,Niter,lbd)
    if True :
#        pn.Column(pn.Row(hv.Image(Df(x1,D)).opts(**options),hv.Image(Df(x2,D)).opts(**options)),
#              pn.Row(hv.Image(Df(x3,D)).opts(**options),hv.Image(image).opts(**options)))
        plt.imshow(Df(x1,D))
        plt.show()
        plt.imshow(Df(x2,D))
        plt.show()
        plt.imshow(Df(x3,D))
        plt.show()
        plt.imshow(Df(x4,D))
        plt.show()
        plt.imshow(utest)
        plt.show()
    if True :
        uref=np.linalg.norm(utest)        
        N0=5
        plt.loglog(F1[N0:]/uref,label="algorithme naif")
        plt.loglog(F2[N0:]/uref,label="ajout du fond")
        plt.loglog(F3[N0:]/uref,label="optimisation de x")
        plt.loglog(F4[N0:]/uref,label="ajout de splines")
        plt.legend
        plt.show()
    if True :
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