from scipy.special import comb
from scipy.optimize import fsolve
from math import floor, ceil

#computes probability of being in pos x in the sample given pos r in ranking
def prob_sample(n, m, x, r):
    
    #julia comp differences?
    #use scipy.comb?
    t = comb(r-1, x-1) * comb(n-r, m-x) / comb(n-1, m-1)

    #float conversion issues?
    return t

#probability of being accepted by the algorithm given position r in the ranking
def prob_acc(n, m, k, r, eps=0):

    quota = k * m / n + eps

    q = sum([prob_sample(n, m, i, r) for i in range(1, floor(quota))]) + (quota - floor(quota)) * prob_sample(n, m, floor(quota) + 1, r)

    return sum([comb(m, i) * (q**i) * ((1-q)**(m-i)) for i in range(ceil(m/2), m)])

#expected accuraccy
def exp_acc(n, m, k, eps=0):
    return sum([prob_acc(n, m, k, r, eps) for r in range(1, k)]) / k

def exp_size(n, m, k, eps=0):
    return sum([prob_acc(n, m, k, r, eps) for r in range(1, n)])


#variance of true positives
def exp_tp(n, m, k, eps=0):
    return sum([prob_acc(n, m, k, r, eps) for r in range(1, k)])

def var_tp(n, m, k, eps=0):
    return sum([prob_acc(n, m, k, r, eps) * (1 - prob_acc(n, m, k, r, eps)) for r in range(1, k)])
            
#estimate epsilon given n, m, k to produce the correct expected size
def estimate_eps(n, m, k):
    f = lambda x : exp_size(n, m, k, x) - k

    return fsolve(f, 0)
