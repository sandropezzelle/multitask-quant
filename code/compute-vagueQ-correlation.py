# -*- coding: utf-8 -*-
"""
#sandro pezzelle, nov 2017
"""
import numpy as np
from scipy import spatial
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_absolute_error

"""
choose the model for which you need Pearson correlations

"""
model = 'multi-task-prop'
#model = 'vagueq-frozen'
#model = 'vagueq-end2end'

with open(model+'/vagueq_predictions-bestmodel.txt', 'r') as pred:
    reader = [line.split() for line in pred]
    lines = (len(reader)/2)

predicted = []
gt = []

for i,val in enumerate(reader):
   if i==0:
     predicted.append(val)

   elif i==1:
       gt.append(val)

   else:
       if i%2==0:
         predicted.append(val)
       else:
            gt.append(val)

print(len(predicted),len(gt))

gtm = []
for j,w in enumerate(gt):
   gtm.append(np.array(w).astype(np.float))

gtmajority = np.mean(gtm, axis=0)


corr,dot,mae = [],[],[]
majcorr,majdot,majmae = [],[],[]

for i,v in enumerate(predicted):
   l1 = np.array(v).astype(np.float)
   l2 = np.array(gt[i]).astype(np.float)

   corr.append(pearsonr(l1,l2)[0]) #Pearson correlation for each predicted/gt pair
   majcorr.append(pearsonr(l2,gtmajority)[0])
   dot.append(np.dot(l1,l2)) #dot-product
   majdot.append(np.dot(l2,gtmajority))
   mae.append(mean_absolute_error(l1,l2)) #mean absolute error
   majmae.append(mean_absolute_error(l2,gtmajority))

avgcorr = np.mean(corr, axis=0)
dotcorr = np.mean(dot, axis=0)
maecorr = np.mean(mae, axis=0)

print("pearson-correlation:",avgcorr,"dot-product:",dotcorr,"meanabserror:",maecorr)

"""
these lines below compute the chance level for each measure

"""
mavgcorr = np.mean(majcorr, axis=0)
mdotcorr = np.mean(majdot, axis=0)
mmaecorr = np.mean(majmae, axis=0)

print("maj-pearson-correlation:",mavgcorr,"maj-dot-product:",mdotcorr,"maj-meanabserror:",mmaecorr)
