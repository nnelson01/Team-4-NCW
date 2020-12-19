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
def european_binomial_pricer(spot: float, strike: float, expiry: float, rate: float, div: float, vol: float, num: int, payoff: Callable) -> float:

    treenodes = num + 1
    h = expiry / num
    u = np.exp((rate-div) * h + vol * np.sqrt(h))
    d = np.exp((rate-div) * h - vol * np.sqrt(h))
    probup = (np.exp((rate - div) * h) - d) / (u-d)
    probdown = 1.0 - probup
    disc = np.exp(-rate * expiry)
    St = 0.0
    Ct = 0.0
    
    for i in range (treenodes):
        St = spot * (u**(num - i)) * (d**i)
        Ct += payoff(St, strike) * binom.pmf(num-i, num, probup)
    return round(disc*Ct, 2)

def american_binomial_pricer(spot: float, strike: float, expiry: float, rate: float, div: float, vol: float, num: int, payoff: Callable) -> float:

    treenodes = num + 1
    h = expiry / num
    u = np.exp(((rate-div) * h) + vol * np.sqrt(h))
    d = np.exp(((rate-div) * h) - vol * np.sqrt(h))
    probup = (np.exp((rate - div) * h) - d) / (u-d)
    probdown = 1.0 - probup
    disc = np.exp(-rate * h)
    
    discPu = disc * probup
    discPd = disc* probdown
    
    Ct = np.zeros(treenodes)
    St = np.zeros(treenodes)
    
    for i in range(treenodes):
        St[i] = spot * (u**(num - i)) * (d**i)
        Ct[i] = payoff(St[i], strike)
        
    for i in range((num - 1), -1, -1):
        for k in range(i + 1):
            Ct[k] = discPu * Ct[k] + discPd * Ct[k+1]
            St[k] = St[k] / u
            Ct[k] = np.maximum(Ct[k], payoff(St[k], strike))
    
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