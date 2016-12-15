from __future__ import division
import numpy as np
from scipy.spatial.distance import pdist, jaccard, squareform, cosine
from itertools import combinations
import random
import sklearn.metrics.pairwise as pair
from sklearn.metrics import log_loss
import pandas as pd
from math import radians, cos, sin, asin, sqrt
#size=2000
print 'size,accuracy'
for m in range(5,3000):
    xids = pd.read_csv('xdevrecon.csv',nrows=m)
    xids_sym = pd.read_csv('xdevrecon2.csv',nrows=m)
    xids_sym = xids_sym[['xid', 'match', 'score', 'sde', 'sos', 'sbr', 'lat', 'long', 'lci',
           'lco', '_col10']]

    xids = xids.append(xids_sym)

    #xids = xids.drop_duplicates(['xid','match'])
    size = len(xids.drop_duplicates('xid'))
    #print 'The size is ', size
    #print len(xids)
    #start with our lat/long data. We're going to run haversine dist on these dudes.
    #define the haversine function
    def haversine(latlon1, latlon2):
        lon1 = latlon1[1]
        lat1 = latlon1[0]
        lon2 = latlon2[1]
        lat2 = latlon2[0]
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 1.0
        return c * r

    xids_la = xids.drop_duplicates('xid')
    xids_la = xids_la.drop_duplicates('match')
    xids_la=xids_la.reset_index()
    xids=xids_la
    size = len(xids)

    #print 'length is ',  len(xids_la)
    X = xids_la[['lat','long']]
    X
    dm=pdist(X.as_matrix(), lambda u, v: haversine(u,v))
    dm
    q = lambda i,j,n: n*j - j*(j+1)/2 + i - 1 - j
    ds = squareform( dm )
    for i in xrange( 1, size ):
        for j in xrange( i ):
          assert ds[ i, j ] == dm[ q( i, j, size ) ]

    ds


    #now let's do our cosine matrix for the categorical data
    #cats = xids[:5000]
    cats = xids.drop_duplicates('xid')
    cats = cats.drop('xid',1)
    cats = cats.drop('match',1)
    cats = cats.drop('lco',1)
    cats = cats.drop('lci',1)
    cats = cats.drop('lat',1)
    cats = cats.drop('long',1)
    cats = pd.get_dummies(cats)
    catcosdf = 1- pair.cosine_similarity(cats)


    #now multiply these two matrices to get our total similarity matrix
    haversine_dist = ds
    cosine_dist=catcosdf
    alpha = 0.9
    y_pred = alpha*haversine_dist + (1-alpha)*cosine_dist
    y_pred


    #now make our score set
    #this function creates a dict with the xid and its index position
    def createMapping(listuniqueid):
        mapping = {}
        for n,xid in enumerate(listuniqueid):
            mapping[xid] = n
        return mapping

    #this creates a matrix whose l/w are the lengths of the mapping dict
    def createGTMatrix(mapping, matches):
        m = np.zeros([len(mapping), len(mapping)])
        for xid1, xid2 in matches:
            try:
                m[mapping[xid1],mapping[xid2]] = 1.0
                m[mapping[xid2],mapping[xid1]] = 1.0
            except:
                pass
                #continue
        return m

    xids1 = xids
    #here are my matched pairs
    matches = xids1[['xid', 'match']]
    matches = [tuple(x) for x in matches.values]
    listuniqueid = xids1['xid']

    mapping = createMapping(listuniqueid.unique())
    y_true = createGTMatrix(mapping,matches)
    #print 'y_pred', y_pred
    X_i_pred = np.argsort(y_pred)[1:]

    ###SHORTLIST SECTION -- FOR FULLSIES PLZ SKIP TO NEXT SECTION###
    #get only the top predictions per id
    shortlist = X_i_pred[:,:4]

    good = 1
    ix = np.in1d(y_true.ravel(), good).reshape(y_true.shape)
    true_loc = np.column_stack(np.where(ix))
    #print 'Predictions', X_i_pred
    #print 'Actual', true_loc
    #print 'Shortlist', shortlist
    #now want to iterate through values in pred and check if they match the ones in true loc
    rows = []
    for i in range(0,len(shortlist)):
        users_i = [users for users in true_loc if users[0]==i]
        for j in range(0,len(users_i)):
            if (shortlist[i][1] == users_i[j][1] or shortlist[i][2] == users_i[j][1] or shortlist[i][3] == users_i[j][1]):
                rows.append(1)
                #print "successful match!"
            else:
                rows.append(0)

    accuracy = sum(rows)/len(shortlist)
    print str(m) + ',' + str(accuracy), ' random = ', 1.0/len(shortlist)
