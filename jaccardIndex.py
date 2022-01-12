from __future__ import print_function


def Jaccard(y, y_pred, epsilon=1e-8):  
    if y.sum(0)==0:
        y = 1-y;
        y_pred= 1-y_pred
        
    TP = (y_pred * y).sum(0)
    FP = ((1-y_pred)*y).sum(0)
    FN = ((1-y)*y_pred).sum(0)
    jack = (TP+epsilon) / (TP+FP+FN+epsilon)
    return jack