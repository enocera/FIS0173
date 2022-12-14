import numpy as np
from tqdm import tqdm
import pandas as pd

def pair_check(p1,p2,delta,s_com):
    '''Check proximity of pair of momenta
    :param p1, p2: 4-momenta
    :param delta: proximity measure according to the JADE algorithm - e.g. 0.01
    
    returns: boolean - True if too close, False otherwise
    '''
    
    distance = (dot(p1,p2))/s_com
        
    close = False
    if distance <= delta:
        close = True
        
    return close

def check_all(p_array,delta,s_com):
    'Given an array of 4-momenta, check proximity of all pairs'
    too_close = False
    
    p_array = p_array[2:]
    
    for idx, p in enumerate(p_array):
        to_check = p_array[idx+1:]
        for j in to_check:
            proximity = pair_check(p,j,delta,s_com=s_com)
            if proximity == True:
                too_close = True
                
    return too_close 

def random_moms(num_points):
    'Generate 4-mom components from a uniform distribution'
    moms = []
    for i in range(num_points):
        moms.append(np.random.uniform(0,1,4))
    return np.array(moms)

def isotropic_moms(mom):
    'Create massles 4-mom with isotropic angular distribution'
    c = 2*mom[0] - 1
    phi = 2*np.pi*mom[1]
    q_0 = -np.log(mom[2]*mom[3])
    q_1 = q_0*np.sqrt(1-c**2)*np.cos(phi)
    q_2 = q_0*np.sqrt(1-c**2)*np.sin(phi)
    q_3 = q_0*c
    return np.array([q_0,q_1,q_2,q_3])

def dot(p1,p2):
    'Minkowski metric dot product'
    prod = p1[0]*p2[0]-(p1[1]*p2[1]+p1[2]*p2[2]+p1[3]*p2[3])
    return prod
    
def boost(mom, moms, rts):
    '''
    Boost and scale isotropic 4-mom for momentum conservation
    :param mom: one isotropic 4-mom
    :param moms: array of all isotropic momenta
    :param rts: centre of mass energy
    
    returns: np array of boosted an scaled 4-mom corresponding to mom
    '''
    q = mom
    Q = np.zeros(4)
    for i in moms:
        Q+=i
    M = np.sqrt(dot(Q,Q))
    b = -Q[1:]/M
    x = rts/M
    gamma = Q[0]/M
    a = 1/(1+gamma)
    
    p_0 = x*(gamma*q[0]+np.dot(b,q[1:]))
    p_space = x*(q[1:]+b*q[0]+a*(np.dot(b,q[1:]))*b)
    return np.array([p_0,p_space[0],p_space[1],p_space[2]])

# rts = sqrt(s) = sqrt(Q^2) = sqrt( 2*p1.p2 ) = sqrt( (p1+p2)^2 )
def generate(num_finalstates, num_points, rts, delta=0.01):
    
    p_1 = np.array([rts/2.,0.,0.,rts/2.])
    p_2 = np.array([rts/2.,0.,0.,-rts/2.])
    
    # store the momenta only if they pass the cut conditions
    cut_momenta = []
    
    # for the progress bar
    pbar = tqdm(total=num_points)
    # for storing number of trials
    n_trials = 0;
    while len(cut_momenta) < num_points:
        n_trials += 1
        moms = random_moms(num_finalstates)
        
        iso_moms = []
        for i in moms:
            iso_moms.append(isotropic_moms(i))
            
        iso_moms = np.array(iso_moms)
        
        boost_moms = []
        boost_moms.append(p_1)
        boost_moms.append(p_2)
        for i in iso_moms:
            boost_moms.append(boost(i,iso_moms,rts))
         
        # NB s_com = rts^2
        close = check_all(boost_moms, delta=delta, s_com=dot(p_1,p_2))
        
        if close == False:
            cut_momenta.append(boost_moms)
            pbar.update(1)

    pbar.close()


    cut_mom = pd.DataFrame({'momenta':list(cut_momenta)})

    return cut_mom['momenta'], n_trials
