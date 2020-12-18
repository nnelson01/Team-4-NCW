import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
from typing import Callable


## Payoff functions
def call_payoff(spot, strike):
    return np.maximum(spot - strike, 0.0)

def put_payoff(spot, strike):
    return np.maximum(strike - spot, 0.0)

## Pricing functions
def european_binomial_pricer(spot: float, strike: float, expiry: float, rate: float, div: float, vol: float, nper: int, payoff: Callable) -> float:
    nodes = nper + 1
    h = expiry / nper
    u = np.exp((rate-div) * h + vol * np.sqrt(h))
    d = np.exp((rate-div) * h - vol * np.sqrt(h))
    pu = (np.exp((rate - div) * h) - d) / (u - d)
    pd = 1.0 - pu   
    disc = np.exp(-rate * expiry)
    St = 0.0
    Ct = 0.0
    
    for t in range (nodes):
        St = spot * (u ** (nper-t)) * (d ** t)
        Ct += payoff(St, strike) * binom.pmf(nper-t, nper, pu)
    return round(disc * Ct, 2)

def american_binomial_pricer(spot: float, strike: float, expiry: float, rate: float, div: float, vol: float, nper: int, payoff: Callable) -> float:
    nodes = nper + 1
    h = expiry / nper 
    u = np.exp(((rate - div) * h) + vol * np.sqrt(h)) 
    d = np.exp(((rate - div) * h) - vol * np.sqrt(h))
    pu = (np.exp((rate - div) * h) - d) / (u - d)
    pd = 1 - pu
    disc = np.exp(-rate * h)
    dpu = disc * pu
    dpd = disc * pd

    Ct = np.zeros(nodes)
    St = np.zeros(nodes)

    for i in range(nodes):
        St[i] = spot * (u ** (nper - i)) * (d ** i)
        Ct[i] = payoff(St[i], strike)

    for i in range((nper - 1), -1, -1):
        for j in range(i+1):
            Ct[j] = dpu * Ct[j] + dpd * Ct[j+1]
            St[j] = St[j] / u
            Ct[j] = np.maximum(Ct[j], payoff(St[j], strike))
    return Ct[0]



def black_scholes_call(spot: float, strike: float, expiry: float, rate: float, div: float, vol: float) -> float:
    d1 = (np.log(spot/strike) + (rate - div + 0.5 * vol * vol) * expiry) / (vol * np.sqrt(expiry))
    d2 = d1 - vol * np.sqrt(expiry) 
    return (spot * np.exp(-div * expiry) * norm.cdf(d1)) - (strike * np.exp(-rate * expiry) * norm.cdf(d2))

def black_scholes_put(spot: float, strike: float, expiry: float, rate: float, div: float, vol: float) -> float:
    d1 = (np.log(spot/strike) + (rate - div + 0.5 * vol * vol) * expiry) / (vol * np.sqrt(expiry))
    d2 = d1 - vol * np.sqrt(expiry) 
    return (strike * np.exp(-rate * expiry) * norm.cdf(-d2)) - (spot * np.exp(-div * expiry) * norm.cdf(-d1))


## Delta
def black_scholes_call_delta(spot: float, strike: float, tau: float, rate: float, div: float, vol: float) -> float:
    d1 = (np.log(spot/strike) + (rate - div + 0.5 * vol * vol) * tau) / (vol * np.sqrt(tau))
    return np.exp(-div * tau) * norm.cdf(d1)


## Simulations
def binomial_path(spot: float, expiry: float, rate: float, div: float, vol: float, num: int) -> np.ndarray:
    h = expiry / num
    u = np.exp((rate - div) * h + np.sqrt(h) * vol)
    d = np.exp((rate - div) * h - np.sqrt(h) * vol)
    pstar = (np.exp((rate - div) * h) - d) / (u - d) 
    z = np.random.uniform(0,1,size=num)
    path = np.empty(num)
    path[0] = spot
    for i in range(1, num):
        if z[i] >= pstar: path[i] = u * path[i-1]
        else: z[i] = path[i] = d * path[i-1]

    return path


   