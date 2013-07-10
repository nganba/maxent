import numpy as np
from math import *
from copy import *
from scipy import weave

def chisquared(data,calc,err):
    """ chisquared calcuates the deviation of "calc" from "data" normalized by 
       "err"
        
        chisquared = sum_{i} (data[i]-calc[i])**2/err[i]**2
        
        data = given data 
        calc = calucated values
        err  = error in given data
    """
    if data.shape != calc.shape != err.shape:
        print ("Error: data, calc and err needs to be of same shape for calculating chisquared")
    diff=((data-calc)/err)**2
    #diff=np.zeros(data.size)
    #expr="diff=((data-calc)/err)*((data-calc)/err)"
    #weave.blitz(expr,check_size=0)
    chi=diff.sum()
    return chi


def entropy(probden,model,dw):
    """ Calculates the generalized Shannon-Jaynes entropy of "probden" with
        zero entropy set at "probden"="model".
        
        probden = probability distribution
        model   = probability distribution that corresponds to zero entropy
        dw      = step size in numerical integration
    """
    if probden.shape != model.shape:
        print("Error: probden and model needs to be of same shape for calculating entropy")
    tmp=np.zeros(probden.size)
    #expr="tmp = probden - model - probden*np.log(probden/model)"
    #weave.blitz(expr,check_size=0)
    tmp = -probden*np.log(probden/model)
    ent = dw*tmp.sum()
    return ent


def kernel(tau,omega,beta):
    """ Generates kernel that relates G(tau) with A(omega)
        
        tau =  list of imaginary time points
        omega = list of omega points
        beta = inverse temperature
    """
    K = np.zeros((tau.size,omega.size))
    for i in range(tau.size):
        for j in range(omega.size):
            if omega[j]>0 :
            	K[i,j] = e**(-tau[i]*omega[j])/(1 + e**(-beta*omega[j]))
            else:
                K[i,j] = e**((beta-tau[i])*omega[j])/(1 + e**(beta*omega[j]))  
    return K


def transform(data,ker,dw):
    """ Transform "data" with kernel "ker" 
        dw = step size in numerical integration    
    """
    if ker.shape[1]!=data.shape:
        print("Error: shape mismatch for matrix multiplication")
    out = dw*dot(ker,data)
    return out

def propose_move(Aguess,refrac):
    """ Propose a new guess spectral function by transferring weight between
        two different omega points
    """
    while True:
        w1=np.random.randint(0,Aguess.size)
        #while Aguess[w1]<0.00001: w1=np.random.randint(0,Aguess.size)
        w2=np.random.randint(0,Aguess.size)
        while w1==w2: w2=np.random.randint(0,Aguess.size)
        d1=Aguess[w1]*np.random.uniform(-1.0,1.0)*refrac
        d2=-d1
        if Aguess[w1]+d1>0 and Aguess[w2]+d2>0: break
    Anew=Aguess.copy()
    Anew[w1]=Anew[w1]+d1
    Anew[w2]=Anew[w2]+d2
    #print d1, d2
    return (Anew,w1,w2)

    
    
        
    
        
    


    
    
        
