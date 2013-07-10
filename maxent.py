from math import *
import numpy as np
from scipy import linalg
from scipy import weave
import maxent_def
import matplotlib.pyplot as plt
from copy import *
import string


# Construct list of omega points
wmin=-10
wmax=10
wnum=401
dw=float(wmax-wmin)/(wnum-1)
wlist=np.zeros(wnum)
for i in range(wnum):
    wlist[i]=wmin+dw*i
    

# Read data and set parameters for Gtau
beta=10 
fname='../dnGtau_U2J023B10.dat'
itr=0
for i in open(fname,"r"):
    itr+=1
print("itr",itr)
taunum=deepcopy(itr)
taulist=np.zeros(taunum)
Gin=np.zeros(taunum)
Err=np.zeros(taunum)
itr=0
for i in open(fname,"r"):
    #x,y,z=i.split()
    #taulist[itr],Gin[itr],Err[itr]=string.atof(x),-string.atof(y),string.atof(z)
    x,y=i.split()
    taulist[itr],Gin[itr]=string.atof(x),-string.atof(y)
    itr+=1
#Err=Err*100
#sigma=0.005
#Err=sigma*np.ones(taunum)
#Err[0]=0.0001
#Err[taunum-1]=0.001
itr=0
for i in open("../delG.dat","r"):
    #x,y,z=i.split()
    #taulist[itr],Gin[itr],Err[itr]=string.atof(x),-string.atof(y),string.atof(z)
    Err[itr]=string.atof(i)
    itr+=1



# Generate kernel
Ker=maxent_def.kernel(taulist,wlist,beta) 
# Hessian Matrix
hessian=np.dot(Ker.transpose(),(Ker.transpose()/(Err**2)).transpose())

# Model for prior knowlegde
model=np.ones(wnum)
norm=dw*model.sum()
model=model/norm

# Trial spectral function
Aguess1=np.ones(wnum)
norm=dw*Aguess1.sum()
Aguess1=Aguess1/norm
Gguess1=dw*np.dot(Ker,Aguess1)
Gguess2=Gguess1.copy()
Chi1=maxent_def.chisquared(Gin,Gguess1,Err)


# Parameters for MaxEnt
alpha=2000.0
tempstart=10
refrac=1.0
LHS=1.0
RHS=1.0
tol=0.01
MCcycle=4000

# MaxEnt Loop
itr=0
sam=0
np.random.seed(seed=423945)
for alphacycle in range(50):
    acc=0
    temp=tempstart
    for i in range(100*MCcycle):
        Aguess2,w1,w2=maxent_def.propose_move(Aguess1,refrac)
        Gguess2=Gguess1+dw*Ker[:,w1]*(Aguess2[w1]-Aguess1[w1])+dw*Ker[:,w2]*(Aguess2[w2]-Aguess1[w2])
        Chi2=maxent_def.chisquared(Gin,Gguess2,Err)
        dS=dw*(- Aguess2[w1]*log(Aguess2[w1]/model[w1]) \
                     - Aguess2[w2]*log(Aguess2[w2]/model[w2]) \
                     + Aguess1[w1]*log(Aguess1[w1]/model[w1]) \
                     + Aguess1[w2]*log(Aguess1[w2]/model[w2]))
        weight=((Chi2-Chi1)/2.0 - alpha*dS)/temp
        if weight < 0:
            prob=1
        else:
            prob=exp(-weight)
        if prob < 0: print "Error"
        if (np.random.rand() < prob):
            Aguess1=Aguess2.copy()
            Chi1=Chi2.copy()
            Gguess1=Gguess2.copy()
            acc+=1.0
        #if abs(dw*Aguess1.sum()-1)>tol: print "normalization error"
        if i%(MCcycle)==0 and i!=0:
            print "cycles,acc,temp,chi",i,acc/MCcycle,temp,Chi1
            if float(acc)/MCcycle > 0.1:
                if refrac < 0.01: refrac *= 1.5
                if refrac > 0.001: refrac /= 1.5    
            if acc/(MCcycle)<0.001: break
            acc=0
            temp/=1.5
    #Check convergence
    tempstart=1
    refrac=0.05        
    lmda=dw*(np.sqrt(Aguess1)*(hessian*np.sqrt(Aguess1)).transpose()).transpose()
    ev=linalg.eig(lmda,right=False,left=False).real
    RHS=(ev/(alpha+ev)).sum()
    S=maxent_def.entropy(Aguess1,model,dw)
    LHS=-2.0*alpha*S
    if abs((RHS/LHS).real-1.)<tol: 
        print "Converged" 
        break                
    #alpha=RHS/LHS
    print alphacycle,alpha,LHS/RHS,Chi1,S
    if (RHS/LHS).real < 0.05:
        alpha*=0.05
    else:
        alpha*=(RHS/LHS)*(1.0-0.001*np.random.uniform(-0.5,0.5))
    print "New alpha",alpha

#for i in range(Aguess1.size-2):
#    Aguess1[i+1]=sum(Aguess1[i:i+2])/3.0
#Aguess1=Aguess1/(dw*Aguess1.sum()) 
