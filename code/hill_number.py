import numpy as np 
from simulation import load_empirical_data
from skbio.diversity.alpha import hill   # nombres de Hill (diversité générale, paramétrés par q)

from pathlib import Path

freqs = [40, 30, 20, 10]

eq_freqs=[25, 25, 25, 25] 

def hill_number(x,q):
    #raise error au cas où
    if type(q) != int and q != np.inf : 
        print('q must be either int or np.inf')
        raise ValueError

    # on force un array pour être sûr 
    x = np.array(x)
    x_prop = x/x.sum()

    if q == 1 : 
        return np.exp(-((np.log(x_prop)*x_prop).sum()))
    elif q == np.inf:
        return 1/x_prop.max()
    else :
        return (x_prop**q).sum()**(1/(1-q))




if __name__ == '__main__' : 
    ROOT = Path(__file__).parent
    DATA_PATH = ROOT.parent / "data" / "fandom_data.csv"   # chemin relatif au script (portable, remplace le chemin en dur)

    df,counter,freq = load_empirical_data(DATA_PATH)
    
    for q in [0,1,2,np.inf] : 
        print(f'(fonction maison )Test pour q = {q} : {hill_number(freq,q)}')
        print(f'(fonction skbio)Test pour q = {q} : {hill(freq,order = q)}')

