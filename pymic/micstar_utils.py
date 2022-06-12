# micstar-utils.py
#
# February 2021
import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats


# import matplotlib.pyplot as plt

# --------------------------------------------------------
# Normal density examples.
#
def bivariate_normal_density(x, y, mu_x, mu_y, sigma):
    # with correlation coefficient 0
    # and covariance matrix sigma^2 * I
    f = 1/(2*np.pi*np.power(sigma, 2))
    f *= np.exp(-0.5 * (np.power((x-mu_x)/sigma, 2) + np.power((y - mu_y)/sigma, 2)))
    return f

def uni_normal_density(x, mu, sigma):
    f = 1/(sigma * np.sqrt(2* np.pi))
    f *= np.exp(-0.5 * np.power((x - mu)/sigma, 2))
    return f


def uni_normal_mass(mu, xa, xb, sigma):
    F = integrate.quad(uni_normal_density, xa, xb, args=(mu, sigma))
    return F[0]


def covariance_density(x, y, x_mean, y_mean, x_loc, y_loc, sigma):
    bv = bivariate_normal_density(x, y, x_loc, y_loc, sigma)
    f =  bv * (x - x_mean) * (y - y_mean)
    return f


def pearson_corr(mu_xvals, mu_yvals, n, sigma):
    # compute pearson correlation (R2)
    x_mean = np.mean(mu_xvals)
    y_mean = np.mean(mu_yvals)

    cov = 0
    for i in range(n):
        print(i)
        x_loc = mu_xvals[i]; y_loc = mu_yvals[i]
        F = integrate.dblquad(
            covariance_density,
            -np.inf, np.inf,
            lambda x: -np.inf, lambda x: np.inf,
            args=(x_mean, y_mean, x_loc, y_loc, sigma)
        )
        cov += F[0]
    
    cov *= 1/n
    pearson = cov/(sigma * sigma)
    return pearson * pearson
        

# --------------------------------------------------------
# Density function definitions
#

# joint density function
def joint_density(x, y, mu_xvals, mu_yvals, n, sigma):
    # joint density formed by n conditional bivariate normals,
    # each with weight 1/n and each
    # centered at (x_vals[i], y_vals[i])
    # with cov matrix sigma * I
    f = 0
    for i, mu_x in enumerate(mu_xvals):
        mu_y = mu_yvals[i]
        f_i = 1/(2*np.pi*np.power(sigma, 2))
        f_i *= np.exp(-0.5*(np.power((x-mu_x)/sigma, 2) +
                            np.power((y-mu_y)/sigma, 2)))
        f_i *= 1/n
        f += f_i
    return f


# x marginal distribution
# only need to integrate univariate normal density
# given the n-list of x means
def x_marginal(xa, xb, mu_xvals, n, sigma):
    def px(x):
        val = 0
        for i, v in enumerate(mu_xvals):
            val += (1/n) * uni_normal_density(x, mu_xvals[i], sigma)
        return val
    
    Fx = integrate.quad(
        px,
        xa, xb,
    )
    return Fx[0]


def y_marginal(ya, yb, mu_yvals, n, sigma):
    def py(y):
        val = 0
        for i, v in enumerate(mu_yvals):
            val += (1/n) * uni_normal_density(y, mu_yvals[i], sigma)
        return val

    Fy = integrate.quad(
        py,
        ya, yb
    )
    return Fy[0]


# y marginal density
def y_marginal_density(y, mu_yvals, sigma, n):
    f = 0
    for i, mu_y in enumerate(mu_yvals):
        fi = 1/(sigma * np.sqrt(2* np.pi))
        fi *= np.exp(-0.5 * np.power((y - mu_y)/sigma, 2))
        fi *= 1/n
        f += fi
    return f


def bivariate_normal_mass(xa, xb, ya, yb, mu_x, mu_y, sigma):
    # with correlation coefficient 0
    # and covariance matrix sigma^2 * I
    Fxy = integrate.dblquad(
        bivariate_normal_density,
        xa, xb,
        lambda y: ya, lambda y: yb,
        args=(mu_x, mu_y, sigma)
    )
    return Fxy[0]


def xy_mass(xa, xb, ya, yb, mu_xvals, mu_yvals, n, sigma):
    total_mass = 0
    for i in range(n):
        mu_x = mu_xvals[i]; mu_y = mu_yvals[i]
        mass = bivariate_normal_mass(xa, xb, ya, yb, mu_x, mu_y, sigma)
        total_mass += mass

    total_mass *= 1/n
    return total_mass


def joint_mass(xa, xb, ya, yb, mu_xvals, mu_yvals, n, sigma, total_mass=1):
    """
    Compute normalized joint mass in region [[xa, xb], [ya, yb]] 
    determined by noise model of n bivariate normals cenetered
    at [mu_xvals, mu_yvals], each with covariance matrix sigma^2 * I. 

    Uses scipy multivariate_normal CDF type.

    Normalizes mass of region with respect to total_mass argument,
    which is 1 by default. 
    """
    if xa == -np.inf: xa = -100000000
    if ya == -np.inf: ya = -100000000

    mass = 0
    for k in range(n):
        mu_x = mu_xvals[k]; mu_y = mu_yvals[k]
        rv = stats.multivariate_normal(
            [mu_x, mu_y],
            (sigma*sigma) * np.identity(2),
        )
        z_mass = rv.cdf([xb, yb])
        z_mass -= rv.cdf([xb, ya])
        z_mass -= rv.cdf([xa, yb])
        z_mass += rv.cdf([xa, ya])

        mass += z_mass

    mass *= 1/n
    mass *= 1/total_mass

    if mass < 0: mass = 0
    
    return mass


# --------------------------------------------------
# Generate sample from distribution
# 
def generate_sample(n, mu_xvals, mu_yvals, k, sigma,
                    xbounds=(0, 1), ybounds=(0,1 )):
    """
    Sample n point from noise-model distribution of Reshef+16.
    Restricts sampling to region specified by xbounds and ybounds. 
    
    - mu_xvals: set of x-means of generating points
    - mu_yvals: set of y-means of generating points
    - k: number of generating points
    - sigma: std. dev of X and Y at every generating point
    """
    rng = np.random.default_rng()

    xa = xbounds[0]; xb = xbounds[1]
    ya = ybounds[0]; yb = ybounds[1]

    sample = []
    size = 0
    while size < n:
        point_index = rng.integers(0, k)
        mu_x = mu_xvals[point_index]; mu_y = mu_yvals[point_index]

        mean = [mu_x, mu_y]
        cov = (sigma*sigma)*np.identity(2)
        point = rng.multivariate_normal(mean, cov)

        # accept point only if within region
        # specified by xbounds and ybounds
        point_x = point[0]; point_y = point[1]
        if point_x >= xa and point_x <= xb:
            if point_y >= ya and point_y <= yb:
                size += 1
                sample.append(point)

    return sample


def generate_noisy_relationship_sample(n, f, mu_xvals,
                                       k, sigma, ybounds=(0, 1)):
    """
    Generate sample of the form
    
      (f(x + e1), f(x) + e2)

    where e1 and e2 are standard normals with std dev sigma. 

    """
    rng = np.random.default_rng()
    ya = ybounds[0]; yb = ybounds[1]

    sample = []
    size = 0
    while size < n:
        point_index = rng.integers(0, k)
        mu_x = mu_xvals[point_index]
        
        e1 = rng.normal(0, sigma)
        e2 = rng.normal(0, sigma)

        f1 = f(mu_x + e1)
        f2 = f(mu_x) + e2

        if (f1 >= ya and f1 <= yb) and (f2 >= ya and f2 <= yb):
            size += 1
            sample.append([f1, f2])

    return sample


def compute_sample_r2(sample):
    """
    Compute R^2 of large sample generated from noise model.
    """
    x_vals = [s[0] for s in sample]
    y_vals = [s[1] for s in sample]
    x_mean = np.mean(x_vals)
    y_mean = np.mean(y_vals)

    x_std = np.sqrt(sum([np.power(x - x_mean, 2) for x in x_vals]))
    y_std = np.sqrt(sum([np.power(y - y_mean, 2) for y in y_vals]))

    r = 0
    for i in range(len(sample)):
        r += (x_vals[i] - x_mean) * (y_vals[i] - y_mean)

    r *=  1 / (x_std * y_std)
    return r * r


def tune_sigma(target_r2, f, mu_xvals, k, ybounds):

    n = 10000
    sigma = 0.5
    sigma_low = 0
    sigma_hi = 1

    sample = generate_noisy_relationship_sample(
        n, f, mu_xvals, k, sigma, ybounds
    )

    sample_r2 = compute_sample_r2(sample)

    tolerance = 0.001

    iters = 1

    while np.abs(sample_r2 - target_r2) > tolerance:

        if sample_r2 > target_r2: sigma_low = sigma    # r2 too high: increase sigma
        elif sample_r2 < target_r2: sigma_hi = sigma # r2 too low: decrease sigma

        sigma = (sigma_low + sigma_hi)/2

        sample = generate_noisy_relationship_sample(
            n, f, mu_xvals, k, sigma, ybounds
        )

        sample_r2 = compute_sample_r2(sample)

        # DEBUG
        # iters += 1
        # s = "iter={}\t targetR2={}\t  sample_r2={}\t sigma={}\t sigma_hi={}\t sigma_low={}"
        # print(s.format(iters, target_r2, round(sample_r2, 4), round(sigma, 4), round(sigma_hi, 4), round(sigma_low, 4)))
        
    return sigma
    

def func_spaced_generating_points(func, k):

    gen_points = []
    xlen = 0.000001

    # find first point:
    # smallest x s.t. f(x) >= 0
    x_init = 0
    y_init = func(x_init)
    if y_init < 0: 
        while not (0 <= y_init <= 0.01):
            # print((x_init, y_init))
            x_init += 0.001
            y_init = func(x_init)

    # find last point:
    # largest x s.t. f(x) <= 1
    x_final = 1
    y_final = func(x_final)
    if y_final > 1:
        while not (1-0.01 <= y_final <= 1):
            # print((x_final, y_final))
            x_final -= 0.001
            y_final = func(x_final)

    curve_len = 0
    f0 = func(x_init)
    for xi in np.arange(x_init+xlen, x_final+xlen, xlen):
        f1 = func(xi)
        ydiff = f1 - f0
        arc = np.sqrt(ydiff*ydiff + xlen*xlen)
        curve_len += arc
        f0 = f1

    space_len = curve_len / k
    # print(space_len)
    gen_points.append(x_init)
    
    delta = xlen 
    for i in range(1, k):
        # print("gen point {}/{}".format(i, k))
        curve_len = 0
        x0 = gen_points[i-1]
        f0 = func(x0)
        for xi in np.arange(x0, 1+xlen, xlen):
            f1 = func(xi)
            ydiff = f1 - f0
            arc = np.sqrt(ydiff*ydiff + xlen*xlen)
            curve_len += arc
            # print("---> curve len {} / {}".format(curve_len, space_len))
            f0 = f1
            if space_len - curve_len <= xlen/2:
                gen_points.append(xi)
                # print("STOP -- xi = {}".format(xi))
                break
        
    return gen_points

