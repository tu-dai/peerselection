using Roots

"""
    prob_sample(n, m, x, r)

Probability of being in position x in the sample given position r in the ranking.
"""
function prob_sample(n, m, x, r)
    n = BigInt(n)
    m = BigInt(m)
    x = BigInt(x)
    r = BigInt(r)

    t = binomial(r-1, x-1)*binomial(n-r, m-x)/binomial(n-1, m-1)

    return convert(Float64, t)
end

"Probability of being accepted by the algorithm given position r in the ranking."
function prob_acc(n, m, k, r, eps=0)
    quota = k*m/n + eps
#     q = sum([prob_sample(n, m, i, r) for i in 1:round(k*m/n)])
    q = sum([prob_sample(n, m, i, r) for i in 1:floor(quota)]) + (quota-floor(quota))*prob_sample(n, m, floor(quota)+1, r)
    return sum([binomial(m, i)*q^i*(1-q)^(m-i) for i in Int64(ceil(m/2)):m])
end

"Expected accuracy of the algorithm."
exp_acc(n, m, k, eps=0) = sum([prob_acc(n, m, k, r, eps) for r in 1:k])/k
exp_size(n, m, k, eps=0) = sum([prob_acc(n, m, k, r, eps) for r in 1:n])

"Varance of true positives."
exp_tp(n, m, k, eps=0) = sum([prob_acc(n, m, k, r, eps) for r in 1:k])
var_tp(n, m, k, eps=0) = sum([prob_acc(n, m, k, r, eps)*(1-prob_acc(n, m, k, r, eps)) for r in 1:k])

"Estimate epsilon given n, m, k to produce the correct expected size."
function estimate_eps(n, m, k)
    f(x) = exp_size(n, m, k, x) - k
    return find_zero(f, 0)
end
