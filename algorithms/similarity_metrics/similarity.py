'''
SIMILARITY

Object used to represent a similarity metric.
'''
from scipy.signal import find_peaks
import numpy as np


def getSimilarityClass(name):
    if name == 'cosine':
        return CosineSimilarity()
    if name == 'peak':
        return PeakSimilarity()

class CosineSimilarity:
    def similarity(self,V1,V2):
        return np.dot(V1,V2)/np.linalg.norm(V1)/np.linalg.norm(V2)

class PeakSimilarity:
    def similarity(self,A,B):
        pA, _ = find_peaks(A)
        pB, _ = find_peaks(B)
        p = np.append(pA,pB,axis=0)
        return np.dot(A[p],B[p])/np.linalg.norm(A[p])/np.linalg.norm(B[p])
