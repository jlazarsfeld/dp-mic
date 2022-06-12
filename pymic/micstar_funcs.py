# micstar_funcs.py
#
# JML, Spring 2021
#
# Definitions of functional relationships from [Reshef+11]
# used to evaluate bias/variance of MICe/r family of statistics.
#
# See page 17 of [Reshef+11] full supplement.
#
#
# [Reshef+11]:
#   - Reshef, David N., et al.
#     "Detecting novel associations in large data sets."
#     Science 334.6062 (2011): 1518-1524.
import numpy as np


# linear + periodic, low freq.
def f1_linear_periodic_lowfreq(x):
    y = 0.2 * np.sin(12 * x - 6)
    y += 1.1 * (x-1) + 1
    return y


# linear + periodic, medium freq.
def f2_linear_periodic_medfreq(x):
    y = 0.15 * np.sin(11 * np.pi * x)
    y += x + 0.05
    return y


# linear + periodic, high freq
def f3_linear_periodic_highfreq(x):
    y = 0.1 * np.sin(48 * x)
    y += 2 * x - 0.5
    return y


# linear + periodic, high freq v2
def f4_linear_periodic_highfreq_v2(x):
    y = 0.2 * np.sin(48 * x)
    y += 2 * x - 0.5
    return y


# non-fourier freq (low) cosine
def f5_non_fourier_freq_cosine(x):
    y = 0.4 * np.cos(7 * np.pi * x) + 0.5
    return y
    

# cosine high-freq
def f6_cosine_high_freq(x):
    y = 0.4 * np.cos(14 * np.pi * x) + 0.5
    return y


# cubic
def f7_cubic(x):
    y = 10 * np.power(x-0.6, 3) + 2 * np.power(x, 2)
    y += 1.5 - 3 * x
    return y


# cubic, y-stretched
def f8_cubic_ystretch(x):
    y = f7_cubic(x)
    y = 4*y - 1.4
    return y


# L-shaped
def f9_Lshape(x):
    if x <= 0.99: y = x / 99
    else: y = 99*x - 98
    return y


# exponential base 2
def f10_exp_base2(x):
    y = np.power(2, x) - 1
    return y


# exponential base 8
def f11_exp_base8(x):
    y = np.power(8, x-0.3) - 1
    return y


# line
def f12_line(x):
    y = x
    return y


# parabola
def f13_parabola(x):
    y = 4 * np.power(x - 0.5, 2) + 0.1
    return y


# non-fourier freq (low) sine
def f14_non_fourier_freq_sine(x):
    y = 0.4 * np.sin(9*np.pi*x) + 0.5
    return y


# sine, low freq
def f15_sine_low_freq(x):
    y = 0.4 * np.sin(8*np.pi*x) + 0.5
    return y
    

# sine, high freq
def f16_sine_high_freq(x):
    y = 0.4 * np.sin(16*np.pi*x) + 0.5
    return y


# sigmoid
def f17_sigmoid(x):
    if x < 0.491: y = 0.05
    elif x > 0.509: y = 0.95
    else: y = 50*(x-0.5) + 0.5
    return y


# varying freq (medium) cosine
def f18_varying_freq_cosine(x):
    y = 0.4 * np.cos(5*np.pi*x*(1+x)) + 0.5
    return y


# varying freq (medium) sine
def f19_varying_freq_sine(x):
    y = 0.4 * np.sin(6*np.pi*x*(1+x)) + 0.5
    return y


# spike
def f20_spike(x):
    if x <= 0.0528: y = 18*x
    elif x >= 0.1: y = -1*(x/9) + 1/9
    else: y = -18*x + 1.9
    return y


# lopsided L
def f21_lopsided_L(x):
    if x <= 0.0051: y = 190*x
    elif x >= 0.01: y = -1*(x/99) + 1/99
    else: y = -198*x + 1.99
    return y


def get_function(name):
    if name == "f1":
        return f1_linear_periodic_lowfreq
    if name == "f2":
        return f2_linear_periodic_medfreq
    if name == "f3":
        return f3_linear_periodic_highfreq
    if name == "f4":
        return f4_linear_periodic_highfreq_v2
    if name == "f5":
        return f5_non_fourier_freq_cosine
    if name == "f6":
        return f6_cosine_high_freq
    if name == "f7":
        return f7_cubic
    if name == "f8":
        return f8_cubic_ystretch
    if name == "f9":
        return f9_Lshape
    if name == "f10":
        return f10_exp_base2
    if name == "f11":
        return f11_exp_base8
    if name == "f12":
        return f12_line
    if name == "f13":
        return f13_parabola
    if name == "f14":
        return f14_non_fourier_freq_sine
    if name == "f15":
        return f15_sine_low_freq
    if name == "f16":
        return f16_sine_high_freq
    if name == "f17":
        return f17_sigmoid
    if name ==  "f18":
        return f18_varying_freq_cosine
    if name == "f19":
        return f19_varying_freq_sine
    if name == "f20":
        return f20_spike
    if name == "f21":
        return f21_lopsided_L
    
