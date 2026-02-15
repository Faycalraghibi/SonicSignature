""" First, some useful functions to get maxima of the spectrogram for a Shazam-like algorithm.
    Make sure to have numba installed so that it work fast.

    The second part of the file includes a class to compute conformal prediction intervals from a KNN classifier.

Author: J. Miramont -  Jan 30 2026
"""

from numba import jit
import numpy as np


@jit(nopython=True)
def filter_by_distance(order,maxima,idx,idx2):
    for k,i in enumerate(order):
        for q in range(k,len(order)):
            j = order[q]
            d = (maxima[idx2[i],0]-maxima[idx2[j],0])**2 + (maxima[idx2[i],1]-maxima[idx2[j],1])**2 
            d = d**0.5
            if d>0 and d<10:
                if idx[idx2[i]]:
                    idx[idx2[j]]=False
    return idx

def filter_maxima(maxima,S):
    idx = np.zeros((len(maxima),), dtype=bool)
    thr = np.quantile(S,0.9)
    for k in range(len(maxima)):
        if S[maxima[k,0],maxima[k,1]]>thr:
            idx[k] = True

    idx2 = np.where(idx)[0]
    sel_maxima = maxima[idx]
    amps = np.zeros((len(sel_maxima),),)
    for i,k in enumerate(sel_maxima):
        amps[i] = S[sel_maxima[i,0],sel_maxima[i,1]] 

    order = np.argsort(amps)[::-1]
    idx = filter_by_distance(order,maxima,idx,idx2)
    maxima = maxima[idx,:]
    return maxima

def get_maxima(S):
    """ Find local maxima of the spectrogram S as maxima in 3x3 grids.
    Includes first/last rows/columns.
    """
    aux_S = np.zeros((S.shape[0]+2,S.shape[1]+2))-np.inf
    aux_S[1:-1,1:-1] = S
    S = aux_S
    aux_ceros = ((S >= np.roll(S,  1, 0)) &
            (S >= np.roll(S, -1, 0)) &
            (S >= np.roll(S,  1, 1)) &
            (S >= np.roll(S, -1, 1)) &
            (S >= np.roll(S, [-1, -1], [0,1])) &
            (S >= np.roll(S, [1, 1], [0,1])) &
            (S >= np.roll(S, [-1, 1], [0,1])) &
            (S >= np.roll(S, [1, -1], [0,1])) 
            )
    [y, x] = np.where(aux_ceros==True)
    pos = np.zeros((len(x), 2)) # Position of zeros in norm. coords.
    pos[:, 0] = y-1
    pos[:, 1] = x-1

    pos = filter_maxima(pos.astype(int),S) # For shazam application, we need to filter some points...

    return pos



class ConformalPrediction:
    """ A class implementing conformal prediction for a KNN classifier. 
    Make sure to pass a KNN classifier from sklearn.
    """
    def __init__(self, KNNmodel,X,y):
        self.model = KNNmodel
        self.y = y
        self.X = X
        self.pz = None


    def conformal_measure(self,y,x_new,y_new):
        # Search the distance to an example of the same class
        numerator = None
        k=1
        while numerator is None:
            dist, ind1 = self.model.kneighbors(x_new,k)
            
            if dist[0][-1] == 0:
                k+=1
                continue

            if y[ind1[0][-1]] == y_new:
                numerator = dist[0][-1]
            k+=1    
        
        # Search the distance to a different class.
        denominator = None
        while denominator is None:
            dist, ind1 = self.model.kneighbors(x_new,k)
            if y[ind1[0][-1]] != y_new:
                denominator = dist[0][-1]
            k+=1
        return numerator/denominator     

    def predict(self,x_new):
        pz = []

        for label in np.unique(self.y):
            print(label)
            # x_new = X_test[10]
            y_new = label
            alpha = []

            Xbag = np.vstack((self.X,x_new))
            ybag = np.append(self.y,y_new)

            for i in range(len(ybag)):
                alpha.append(self.conformal_measure(ybag,
                                            Xbag[i].reshape(1,-1),
                                            ybag[i]))

            alpha = np.array(alpha)
            pz.append(np.sum(alpha>=alpha[-1])/len(alpha))
        
        self.pz = pz
   
    def compute_interval(self,eps=0.05):
        assert self.pz is not None, 'You must run predict(x) first.'

        interval = []
        for i,p in enumerate(self.pz):
            if p >= eps:
                interval.append(np.unique(self.y)[i])

        return interval



def search_song(db_hashes,song_hashes):
    scores =  []
    for hashes in db_hashes:
        tokens_present = [token for token in song_hashes.keys() if token in hashes.keys()]
        offsets = [hashes[token]-song_hashes[token] for token in tokens_present]
        count, bins  = np.histogram(offsets,)
        scores.append(np.max(count))

    return np.argsort(scores)[::-1][:3] # Returns the index of the five more similar songs by similarity.
