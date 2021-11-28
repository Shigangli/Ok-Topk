# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import time
import logging
import utils
import sys
import math


def intersect1d_searchsorted(A, B_ar):
    idx = np.searchsorted(B_ar, A)
    idx[idx==len(B_ar)] = 0
    return A[B_ar[idx] == A]


a1 = np.zeros(1332134, dtype='int32')
a2 = np.zeros(1412321, dtype='int32')
for i in range(a1.size):
    a1[i] = i
for i in range(a2.size):
    a2[i] = i
    
stimein = time.time()
#involved_indexes = np.intersect1d(a1, a2, return_indices=False, assume_unique=True)
involved_indexes = intersect1d_searchsorted(a2, a1)
#involved_indexes = intersect1d_searchsorted(a1, a2)
elapsein = time.time() - stimein
print("intersec time sorted: ", elapsein)

print(involved_indexes)

stimein = time.time()
involved_indexes = np.intersect1d(a1, a2, return_indices=False, assume_unique=True)
elapsein = time.time() - stimein
print("intersec time np: ", elapsein)

print(involved_indexes)
