"""
ez_diffusion.py

This file contains the implementation of the equations and functions for the EZ diffusion model.
- These functions were generated with the assistance of ChatGPT o3-mini-high and ChatGPT Diffusion Models for Python Code.

Author: Aiden Hai
Date: 03/20/2025
"""

import numpy as np
import math

def forward_stats(a, v, t):
    """
    Compute predicted summary statistics using the forward equations of the EZ diffusion model.
    
    Parameters:
        a (float): Boundary separation (range: 0.5 to 2)
        v (float): Drift rate (range: 0.5 to 2)
        t (float): Non-decision time (range: 0.1 to 0.5)
    
    Returns:
        r_pred (float): Predicted accuracy rate.
        m_pred (float): Predicted mean reaction time (RT).
        v_pred (float): Predicted variance of RT.
    """
    y = np.exp(-a * v)
    r_pred = 1 / (1 + y)
    m_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
    v_pred = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / (1 + y)**2)
    return r_pred, m_pred, v_pred

def simulate_data(r_pred, m_pred, v_pred, N):
    """
    Simulate observed data based on predicted summary statistics.
    
    Parameters:
        r_pred (float): Predicted accuracy rate.
        m_pred (float): Predicted mean RT.
        v_pred (float): Predicted RT variance.
        N (int): Number of trials.
    
    Returns:
        r_obs (float): Observed accuracy rate.
        m_obs (float): Observed mean RT.
        v_obs (float): Observed RT variance.
    """
    T_obs = np.random.binomial(N, r_pred)
    r_obs = T_obs / N
    
    # Simulate observed mean RT from a normal distribution.
    m_obs = np.random.normal(m_pred, np.sqrt(v_pred / N))
    # Simulate observed RT variance from a gamma distribution.
    shape = (N - 1) / 2.0
    scale = (2 * v_pred) / (N - 1) if N > 1 else v_pred
    v_obs = np.random.gamma(shape, scale)
    return r_obs, m_obs, v_obs

def inverse_stats(r_obs, m_obs, v_obs):
    """
    Recover model parameters using the inverse equations of the EZ diffusion model.
    
    Parameters:
        r_obs (float): Observed accuracy rate.
        m_obs (float): Observed mean RT.
        v_obs (float): Observed RT variance.
    
    Returns:
        v_est (float): Estimated drift rate.
        a_est (float): Estimated boundary separation.
        t_est (float): Estimated non-decision time.
    """

    # Prevent invalid values for r_obs (avoid log(0) or division by zero)
    epsilon = 1e-6
    r_obs = np.clip(r_obs, epsilon, 1 - epsilon)
    
    v_obs = max(v_obs, epsilon)  # Avoid division by zero

    # Compute log-odds (L)
    L = math.log(r_obs / (1 - r_obs))

    # Compute drift rate (ν_est) with proper numerical stability
    numerator = L * (r_obs**2 * L - r_obs * L + r_obs - 0.5)
    denominator = max(v_obs, epsilon)  # Ensure denominator is not zero
    v_est = np.sign(r_obs - 0.5) * np.power(max(numerator / denominator, epsilon), 0.25)  # Ensure non-negative value before root

    # Compute boundary separation (α_est)
    a_est = L / v_est if v_est != 0 else 0

    # Compute non-decision time (τ_est), preventing numerical instability
    if np.isnan(a_est) or np.isnan(v_est) or v_est == 0 or a_est == 0:
        t_est = m_obs  # If parameters are NaN or zero, fallback to mean RT
    else:
        exp_term = math.exp(-min(v_est * a_est, 700))  # Prevent overflow
        t_est = m_obs - (a_est / (2 * v_est)) * ((1 - exp_term) / (1 + exp_term))

    return v_est, a_est, t_est