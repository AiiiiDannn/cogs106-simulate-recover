# src/simulate_and_recover.py
import numpy as np
from .ez_diffusion import forward_stats, simulate_data, inverse_stats

def simulate_and_recover_iteration(N):
    """
    Perform one iteration of the simulate-and-recover process.
    
    Parameters:
        N (int): Number of trials for the simulation.
        
    Returns:
        bias_v (float): Bias in drift rate (true v - estimated v).
        bias_a (float): Bias in boundary separation (true a - estimated a).
        bias_t (float): Bias in non-decision time (true t - estimated t).
    """
    # Randomly generate true parameters within the specified ranges.
    a_true = np.random.uniform(0.5, 2)
    v_true = np.random.uniform(0.5, 2)
    t_true = np.random.uniform(0.1, 0.5)
    
    # Compute predicted summary statistics using forward equations.
    r_pred, m_pred, v_pred = forward_stats(a_true, v_true, t_true)
    
    # Simulate observed data based on the predicted statistics.
    r_obs, m_obs, v_obs = simulate_data(r_pred, m_pred, v_pred, N)
    
    # Recover estimated parameters using inverse equations.
    v_est, a_est, t_est = inverse_stats(r_obs, m_obs, v_obs)
    
    # Calculate biases between true and recovered parameters.
    bias_v = v_true - v_est
    bias_a = a_true - a_est
    bias_t = t_true - t_est
    return bias_v, bias_a, bias_t

def run_simulations(N, iterations=1000):
    """
    Run multiple iterations of simulate-and-recover and compute average bias and mean squared error.
    
    Parameters:
        N (int): Number of trials per simulation iteration.
        iterations (int): Total number of iterations.
        
    Returns:
        mean_bias (ndarray): Average bias for each parameter [drift, boundary, non-decision time].
        mse (ndarray): Mean squared error for each parameter.
    """
    biases = []
    for i in range(iterations):
        bias = simulate_and_recover_iteration(N)
        biases.append(bias)
    biases = np.array(biases)
    mean_bias = np.mean(biases, axis=0)    # b should be close to 0 on average
    mse = np.mean(biases ** 2, axis=0)    # b² should decrease when we increase N
    return mean_bias, mse

if __name__ == '__main__':
    # Define different sample sizes to test.
    sample_sizes = [10, 40, 4000]
    # for N in sample_sizes:
    #     mean_bias, mse = run_simulations(N)
    #     print(f"For sample size N = {N}:")
    #     print(f"Mean Bias: Drift Rate = {mean_bias[0]:.4f}, Boundary Separation = {mean_bias[1]:.4f}, Non-decision Time = {mean_bias[2]:.4f}")
    #     print(f"MSE: Drift Rate = {mse[0]:.4f}, Boundary Separation = {mse[1]:.4f}, Non-decision Time = {mse[2]:.4f}")
    #     print("========================================")

    print("\n" + "="*50)
    print(" EZ Diffusion Model - Simulate & Recover Results ")
    print("="*50 + "\n")
    
    for N in sample_sizes:
        mean_bias, mse = run_simulations(N)

        print(f" Sample Size (N) = {N} ".center(50, "="))
        print(f"| {'Metric':<25} | {'Drift Rate':>12} | {'Boundary Separation':>20} | {'Non-decision Time':>20} |")
        print("-" * 50)

        print(f"| {'Mean Bias':<25} | {mean_bias[0]:>12.4f} | {mean_bias[1]:>20.4f} | {mean_bias[2]:>20.4f} |")
        print(f"| {'Mean Squared Error':<25} | {mse[0]:>12.4f} | {mse[1]:>20.4f} | {mse[2]:>20.4f} |")

        print("="*50 + "\n")
