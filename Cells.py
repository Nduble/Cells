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
IMAGE=0

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
    
TEST=2

if TEST==1:
    x,F=frankWolfe(np.zeros((k,l,d)),D,utest,100,lbd)
elif TEST==2:
    x1,F1=frankWolfe(np.zeros((k,l,d)),D,utest,500,lbd)
    x2,F2=frankWolfe(np.zeros((k,l,d)),D,utest,500,lbd)
    x3,F3=frankWolfe(np.zeros((k,l,d)),D,utest,500,lbd)
    if True :
        pn.Column(pn.Row(hv.Image(Df(xx1,D)).opts(**options),hv.Image(Df(xx2,D)).opts(**options)),
              pn.Row(hv.Image(Df(xx3,D)).opts(**options),hv.Image(image).opts(**options)))
    if True :
        uref=np.linalg.norm(utest)        
        N0=5
        plt.loglog(F1[N0:]/uref,lbl="algorithme naif")
        plt.loglog(F2[N0:]/uref,lbl="ajout du fond")
        plt.loglog(F3[N0:]/uref,lbl="optimisation de x")
        plt.legend
        plt.show()
    if True :
        xp1=xx1[np.where(xx1!=0)]
        plt.hist(xp1)
        plt.show()
        print("norme l0 de x:",np.sum(xp1),"norme l1 de x:",np.sum(np.abs(xx1)))
        xp2=xx2[np.where(xx2!=0)]
        plt.hist(xp2)
        plt.show()
        print("norme l0 de x:",np.sum(xp2),"norme l1 de x:",np.sum(np.abs(xx2)))
        xp3=xx3[np.where(xx3!=0)]
        plt.hist(xp3)
        plt.show()
        print("norme l0 de x:",np.sum(xp3),"norme l1 de x:",np.sum(np.abs(xx3)))