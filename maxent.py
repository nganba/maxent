# This script does maxent calculation to get G of omega from G of iomega from TRIQS.
# Before running the script, start ipytriqs in interactive more. Load the necessary HDF file
# and set ImOmegaG to the required G of iomega. 

from math import *
import numpy as np
from scipy import linalg
from scipy import weave
import maxent_def
import matplotlib.pyplot as plt
from copy import *
import string
from pytriqs.Base.GF_Local import *


# Construct list of omega points
wmin=-8
wmax=8
wnum=201
dw=float(wmax-wmin)/(wnum-1)
wlist=np.zeros(wnum)
for i in range(wnum):
    wlist[i]=wmin+dw*i
    


# Do maxent for all orbitals in G of iomega
Beta=ImOmegaG.Beta 
taunum=200
for orb,g in ImOmegaG:
    # Read G of imag omega from HDF file 
    # Note:  Make sure the HDF archive is imported using pytriqs before running
    #        this maxent script
    taulist,Gin=maxent_def.GtaufromGomega(ImOmegaG,orb,Beta,taunum)
    taulist=np.append(taulist,Beta)
    print taulist
    Gin=np.append(Gin,-ImOmegaG[orb].density().real)
    Gin=-Gin
    Err=maxent_def.std_error(Gin)

    # Generate kernel
    Ker=maxent_def.kernel(taulist,wlist,Beta) 
    # Hessian Matrix
    hessian=np.dot(Ker.transpose(),(Ker.transpose()/(Err**2)).transpose())

    # Model for prior knowlegde
    model=np.ones(wnum)
    norm=dw*model.sum()
    model=model/norm

    # MaxEnt Loop
    avgA=np.zeros(wnum)
    # Average over random seeds 
    rndseeds=[423945,102342,945323,453123,32481,76231]
    for rndnum in rndseeds:
        # Initialize parameters for MaxEnt
        alpha=2000.0
        tempstart=10
        refrac=1.0
        LHS=1.0
        RHS=1.0
        tol=0.01
        MCcycle=5000
        # Trial spectral function
        Aguess1=np.ones(wnum)
        norm=dw*Aguess1.sum()
        Aguess1=Aguess1/norm
        Gguess1=dw*np.dot(Ker,Aguess1)
        Gguess2=Gguess1.copy()
        Chi1=maxent_def.chisquared(Gin,Gguess1,Err)
        np.random.seed(seed=rndnum)
        for alphacycle in range(50):
            acc=0
            temp=tempstart
            for i in range(60*MCcycle):
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
                if i%(10*MCcycle)==0 and i!=0:
                    print "cycles,acc,temp,chi",i,acc/MCcycle,temp,Chi1
                if i%(MCcycle)==0 and i!=0:
                    if refrac < 0.01: refrac *= 1.5
                    if refrac > 0.001: refrac /= 1.5 
                    acc=0
                    temp/=1.5
            print "cycles,acc,temp,chi",i,acc/MCcycle,temp,Chi1
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
        avgA=avgA+Aguess1

    avgA=avgA/np.size(rndseeds)
    # save spectral function
    A=[[wlist[i],avgA[i]] for i in range(wnum)]
    np.savetxt("../DOSU2J023B"+int(Beta).__str__()+orb+".dat",A)
