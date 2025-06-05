#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy.random as rand
import pandas as pd
from scipy.stats import poisson
import torch
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import poisson, multivariate_normal
from scipy.linalg import expm, inv
from scipy.optimize import linprog
from scipy.optimize import milp
import traceback
import time
import pickle 
from scipy.linalg import eigvals


# In[2]:


def generate_multistream_ST(n, d, T, a, omega, sigma2, mu, sigma_0, max_retries=50, verbose=False):
    """
    Generate multivariate spatio-temporal Hawkes process with optional verbosity.
    
    Parameters:
    n          -- number of nodes
    d          -- spatial dimension
    T          -- total time horizon
    a          -- interaction matrix (n x n)
    omega      -- scalar decay rate for all nodes
    sigma2     -- spatial variance for triggering kernel (n x n)
    mu         -- background intensity (n x 1)
    sigma_0    -- standard deviation for background intensity
    max_retries -- maximum number of retries in case of error
    verbose    -- If True, prints detailed messages during the generation process
    
    Returns:
    events -- list of events for each node or None if the generation failed after retries
    """
    for attempt in range(max_retries):
        try:
            if verbose:
                print(f"Attempt {attempt + 1}: Initializing variables")
            ST_hawkes = [None] * n
            Sigma_0 = sigma_0**2 * np.diag(np.ones(d))  # Covariance matrix for background intensity
            
            # Generate background events
            for i in range(n):
                if verbose:
                    print(f"Node {i}: Generating background events")
                ST_hawkes[i] = []
                N0 = poisson.rvs(mu=mu[i] * T, size=1)  # Number of background events for node i
                N0 = int(N0)
                G0 = np.zeros((N0, 1 + d))
                G0[:, 0] = rand.uniform(low=0, high=T, size=N0)  # Arrival times of background events
                for j in range(N0):
                    G0[j, 1:(d+1)] = rand.multivariate_normal(mean=np.zeros(d), cov=Sigma_0)
                    
                accepted = [i for i, v in enumerate(G0[:, 0]) if v <= T]
                if len(accepted) > 0:
                    ST_hawkes[i].append(G0[accepted, :])  # Store background events
                elif verbose:
                    print(f"Warning: Node {i} has no accepted background events.")
                
                if len(ST_hawkes[i]) == 0 and verbose:
                    print(f"Warning: Node {i} has no background events.")
            
            # Define offspring generation
            def offspring(v, i, j):
                try:
                    if verbose:
                        print(f"Generating offspring for node {i} from node {j}")
                    N = poisson.rvs(mu=a[i, j] / omega)  # Number of offspring
                    Sigma = sigma2 * np.diag(np.ones(d))
                    if N == 0:
                        return []
                    O = np.zeros((N, 1 + d))
                    O[:, 0] = v[0] + rand.exponential(size=N, scale=1 / omega)  # Temporal component
                    for i in range(N):
                        O[i, 1:(d+2)] = v[1:(d+2)] + rand.multivariate_normal(mean=np.zeros(d), cov=Sigma)
                    
                    accepted = [i for i, v in enumerate(O[:, 0]) if v <= T]
                    return O[accepted, :]
                except Exception as e:
                    if verbose:
                        print(f"Error in offspring generation for node {i} and parent node {j}: {e}")
                    raise
        
            # Generation-based sampling
            if verbose:
                print("Starting generation-based sampling")
            l, p, c = 0, 1, 0
            list_of_gens = [ST_hawkes]
            
            while p > 0:
                parents = list_of_gens[l]  # List of events of generation l
                if len(parents) == 0 and verbose:  # Check for empty parent lists
                    print(f"Warning: No parents found in generation {l}.")
                    break
                num_parents = sum(len(parents[i]) for i in range(n))
                if num_parents == 0 and verbose:
                    print(f"Warning: No parents to generate offspring from in generation {l}.")
                    break
                
                c += num_parents
                
                kids = [[] for _ in range(n)]  # Blank list for offspring
                for i in range(n):
                    for j in range(n):
                        if len(parents[j]) == 0 or len(parents[j][0]) == 0:
                            if verbose:
                                print(f"Warning: No events for node {j} in generation {l}.")
                            continue  # Skip if there are no parents for this node
                        
                        moms = parents[j][0]  # Get parent events for node j
                        for k in range(np.shape(moms)[0]):
                            vector = moms[k, :]
                            baby = offspring(vector, i, j)  # Generate offspring
                            if isinstance(baby, str):  # Catch errors
                                raise ValueError(f"Error in offspring generation for node {i}")
                            if len(baby) > 0:
                                kids[i].append(baby)
                
                s = sum(np.shape(kids[k][0])[0] for k in range(n) if len(kids[k]) > 0)
                if s == 0:
                    p = 0
                list_of_gens.append(kids)
                if verbose:
                    print(f"We are at generation {l}, Added {s} offsprings")
                l += 1
            
            # Finalize and sort events
            if verbose:
                print("Finalizing and sorting events")
            g = len(list_of_gens)  # Number of generations
            events = [None] * n
            for i in range(n):
                events[i] = np.zeros((1, d + 1))
                for j in range(g):
                    if len(list_of_gens[j][i]) > 0:
                        if len(list_of_gens[j][i][0]) == 0 and verbose:
                            print(f"Warning: No events to concatenate for node {i} in generation {j}.")
                            continue
                        Iwantthese = list_of_gens[j][i][0]
                        events[i] = np.concatenate((events[i], Iwantthese))
                
                if events[i].shape[0] > 1:
                    Myoutput = events[i][1:, :]
                    Myoutput = Myoutput[Myoutput[:, 0].argsort()]
                    events[i] = Myoutput
                elif verbose:
                    print(f"Warning: Node {i} has no valid events after sorting.")
            
            print("Event generation successful!")
            return events  # Return the events if successful

        except Exception as e:
            if verbose:
                print(f"Error during generation attempt {attempt + 1}: {e}")
            print("Retrying...")

    print(f"Generation failed after {max_retries} attempts.")
    return None  # Return None if the generation failed after max_retries


# def compute_S_vector(t, x, y, E, sigma2, omega):
#     """
#     Compute the state vector S(t, x, y) for all nodes.
#     
#     Parameters:
#     t      -- current time
#     x, y   -- spatial coordinates (scalar values)
#     E      -- list of events for each node, where E[i] contains events for node i
#     sigma2 -- variance for spatial kernel
#     omega  -- list/array of decay rates, one for each node
#     
#     Returns:
#     S_vector -- a vector of size n, where each entry S_vector[i] is the state for node i
#     """
#     n = len(E)  # Number of nodes
#     S_vector = np.zeros(n)
# 
#     for i in range(n):
#         node_i_events = E[i]  # Events for node i
#         S_value = 0
#         for event in node_i_events:
#             t_prime, x_prime, y_prime = event
#             if t_prime < t:  # Only consider past events
#                 temporal_decay = np.exp(-omega * (t - t_prime))  # Temporal decay
#                 spatial_decay = np.exp(-((x - x_prime)**2 + (y - y_prime)**2) / (2 * sigma2))  # Spatial decay
#                 S_value += temporal_decay * spatial_decay
#         S_vector[i] = S_value / (2 * np.pi * sigma2)
#     
#     return S_vector

# In[3]:


def compute_S_vector(t, E, sigma2, omega):
    """
    Compute the state vector S(t; E, sigma2, omega) for all nodes by integrating out x and y.
    
    Parameters:
    t      -- current time
    E      -- list of events for each node, where E[i] contains events for node i
    sigma2 -- variance for spatial kernel
    omega  -- list/array of decay rates, one for each node
    
    Returns:
    S_vector -- a vector of size n, where each entry S_vector[i] is the state for node i
    """
    n = len(E)  # Number of nodes
    S_vector = np.zeros(n)

    for i in range(n):
        node_i_events = E[i]  # Events for node i
        S_value = 0
        for event in node_i_events:
            t_prime, _, _ = event  # Extract only the time of the event
            if t_prime < t:  # Only consider past events
                temporal_decay = np.exp(-omega * (t - t_prime))  # Temporal decay
                S_value += temporal_decay
        S_vector[i] = S_value
    
    return S_vector


# In[4]:


from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
from scipy.linalg import expm, inv

def solve_optimization_sum_eta(A, omega, events, tau, sigma2, mu0, c, B, p, t):
    """
    Solve the global rate minimization problem with binary constraints using MILP.
    This version uses column sums for the cost vector in the optimization.
    
    Parameters:
    A      -- interaction matrix (n x n)
    omega  -- decay rate matrix (n x 1)
    events -- list of events for each node
    tau    -- intervention time
    sigma2 -- variance for spatial component
    mu0    -- background intensity (scalar or n x 1 vector)
    c      -- intervention costs for each node
    B      -- total intervention budget
    p      -- probability of survival post-intervention
    t      -- current time
    
    Returns:
    u_optimal -- optimal intervention strategy (vector of 0s and 1s)
    """
    n = len(A)
    
    # Step 1: Compute Xi(t - tau)
    #A_omega = A - np.diag(np.array([omega]))  # (A - omega * I)
    A_omega = A-omega*np.eye(n)
    Xi = expm(A_omega * (t - tau))  # Xi(t - tau)

    # Step 2: Compute S_tau (state vector at time tau)
    S_tau = compute_S_vector(tau, events, sigma2, omega)  # Compute S(tau) for each node
    
    # Step 3: Set up the objective function
    diag_S_tau = np.diag(S_tau)  # Make S_tau a diagonal matrix
    C = Xi @ A @ diag_S_tau  # Compute the matrix C
    
    # Compute the cost vector `c` as the **column sums** of matrix C
    c_vector = np.sum(C, axis=0)  # Column sums for the optimization cost

    # Step 4: MILP setup using LinearConstraint
    A_ub = -c.reshape(1, -1)  # Negated coefficients for the inequality constraint
    b_ub = [B - np.sum(c)]  # RHS for the inequality constraint

    # Binary constraint (integrality) on u
    integrality = np.ones(n)  # Ensure all u_i are binary (0 or 1)

    # Set lower and upper bounds for u: l=0, u=1 using scipy.Bounds
        # Set lower and upper bounds for u: l=0, u=1 using scipy.Bounds
    bounds = Bounds([0]*n, [1]*n)

    # Use MILP to minimize the objective function subject to the budget constraint
    res = milp(c=c_vector, integrality=integrality, constraints=LinearConstraint(A_ub, -np.inf, b_ub), bounds=bounds)

    u_optimal = np.round(res.x)  # Round to binary values
    return u_optimal


def solve_optimization_sum_EN(T, tau, A, omega, events, sigma2, mu0, c, B, p):
    """
    Solve the global invasion minimization problem to minimize the total expected number of events E[N(T; u)] using MILP.
    This version uses column sums for the cost vector in the optimization.
    
    Parameters:
    T      -- final time horizon
    A      -- interaction matrix (n x n)
    omega  -- decay rate matrix (n x 1 or scalar)
    events -- list of events for each node
    tau    -- intervention time
    sigma2 -- variance for spatial component
    mu0    -- background intensity (scalar or n x 1 vector)
    c      -- intervention costs for each node
    B      -- total intervention budget
    p      -- probability of survival post-intervention
    
    Returns:
    u_optimal -- optimal intervention strategy (vector of 0s and 1s)
    """
    n = len(A)

    # Step 1: Compute A - omega*I
    #A_omega = A - np.diag(np.array([omega]))
    A_omega = A-omega*np.eye(n)
    # Step 2: Compute Upsilon(T - tau) = inv(A - omega I) * (exp(A - omega I)*(T - tau) - I)
    Upsilon_T_tau = inv(A_omega).dot(expm(A_omega * (T - tau)) - np.eye(n))

    # Step 3: Compute S_tau (state vector at time tau)
    S_tau = compute_S_vector(tau,  events, sigma2, omega)  # Compute S(tau) for each node

    # Step 4: Compute the matrix C as Upsilon(T - tau) * A * diag(S_tau)
    diag_S_tau = np.diag(S_tau)
    C = Upsilon_T_tau.dot(A).dot(diag_S_tau)

    # Compute the cost vector `c` as the **column sums** of matrix C
    c_vector = np.sum(C, axis=0)  # Column sums for the optimization cost

    # Step 5: MILP setup using LinearConstraint
    A_ub = -c.reshape(1, -1)  # Negated coefficients for the inequality constraint
    b_ub = [B - np.sum(c)]  # RHS for the inequality constraint

    # Binary constraint (integrality) on u
    integrality = np.ones(n)  # Ensure all u_i are binary (0 or 1)

    # Set lower and upper bounds for u: l=0, u=1 using scipy.Bounds
    bounds = Bounds([0]*n, [1]*n)

    # Use MILP to minimize the objective function subject to the budget constraint
    res = milp(c=c_vector, integrality=integrality, constraints=LinearConstraint(A_ub, -np.inf, b_ub), bounds=bounds)

    u_optimal = np.round(res.x)  # Round to binary values
    return u_optimal


# In[5]:


from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
from scipy.linalg import expm, inv

def GAMMA_solve_optimization_sum_eta(gamma,A, omega, events, tau, sigma2, mu0, c, B, p, t):
    """
    Solve the global rate minimization problem with binary constraints using MILP.
    This version uses column sums for the cost vector in the optimization.
    
    Parameters:
    A      -- interaction matrix (n x n)
    omega  -- decay rate matrix (n x 1)
    events -- list of events for each node
    tau    -- intervention time
    sigma2 -- variance for spatial component
    mu0    -- background intensity (scalar or n x 1 vector)
    c      -- intervention costs for each node
    B      -- total intervention budget
    p      -- probability of survival post-intervention
    t      -- current time
    
    Returns:
    u_optimal -- optimal intervention strategy (vector of 0s and 1s)
    """
    n = len(A)
    I_n = np.eye(n)
    # Step 1: Compute Xi(t - tau)
    #A_omega = A - np.diag(np.array([omega]))  # (A - omega * I)
    A_omega = A-omega*np.eye(n)
    Xi = expm(A_omega * (t - tau))  # Xi(t - tau)

    # Step 2: Compute S_tau (state vector at time tau)
    S_tau = compute_S_vector(tau, events, sigma2, omega)  # Compute S(tau) for each node
    
    # Step 3: Set up the objective function
    diag_S_tau = np.diag(S_tau)  # Make S_tau a diagonal matrix
    C = Xi @ A @ diag_S_tau  # Compute the matrix C
    C = (1-p)*C
    #C += (1-gamma)*mu0
    Psi = I_n + A.dot(inv(A_omega)).dot(expm(A_omega * T) - I_n)  # n x n matrix
    C += (1-gamma)*mu0*Psi 
    
    # Compute the cost vector `c` as the **column sums** of matrix C
    c_vector = np.sum(C, axis=0)  # Column sums for the optimization cost

    # Step 4: MILP setup using LinearConstraint
    A_ub = -c.reshape(1, -1)  # Negated coefficients for the inequality constraint
    b_ub = [B - np.sum(c)]  # RHS for the inequality constraint

    # Binary constraint (integrality) on u
    integrality = np.ones(n)  # Ensure all u_i are binary (0 or 1)

    # Set lower and upper bounds for u: l=0, u=1 using scipy.Bounds
        # Set lower and upper bounds for u: l=0, u=1 using scipy.Bounds
    bounds = Bounds([0]*n, [1]*n)

    # Use MILP to minimize the objective function subject to the budget constraint
    res = milp(c=c_vector, integrality=integrality, constraints=LinearConstraint(A_ub, -np.inf, b_ub), bounds=bounds)

    u_optimal = np.round(res.x)  # Round to binary values
    return u_optimal


def GAMMA_solve_optimization_sum_EN(gamma,T, tau, A, omega, events, sigma2, mu0, c, B, p):
    """
    Solve the global invasion minimization problem to minimize the total expected number of events E[N(T; u)] using MILP.
    This version uses column sums for the cost vector in the optimization.
    
    Parameters:
    T      -- final time horizon
    A      -- interaction matrix (n x n)
    omega  -- decay rate matrix (n x 1 or scalar)
    events -- list of events for each node
    tau    -- intervention time
    sigma2 -- variance for spatial component
    mu0    -- background intensity (scalar or n x 1 vector)
    c      -- intervention costs for each node
    B      -- total intervention budget
    p      -- probability of survival post-intervention
    
    Returns:
    u_optimal -- optimal intervention strategy (vector of 0s and 1s)
    """
    n = len(A)

    # Step 1: Compute A - omega*I
    #A_omega = A - np.diag(np.array([omega]))
    A_omega = A-omega*np.eye(n)
    # Step 2: Compute Upsilon(T - tau) = inv(A - omega I) * (exp(A - omega I)*(T - tau) - I)
    Upsilon_T_tau = inv(A_omega).dot(expm(A_omega * (T - tau)) - np.eye(n))

    # Step 3: Compute S_tau (state vector at time tau)
    S_tau = compute_S_vector(tau,  events, sigma2, omega)  # Compute S(tau) for each node

    # Step 4: Compute the matrix C as Upsilon(T - tau) * A * diag(S_tau)
    diag_S_tau = np.diag(S_tau)
    C = Upsilon_T_tau.dot(A).dot(diag_S_tau)
    C = (1-p)*C
    I_n = np.eye(n)
    Upsilon_T = inv(A_omega).dot(expm(A_omega * T) - I_n)  # (A-wI)^{-1}*(e^(A-wI)*t-I)

    Gamma_T = I_n * T + A.dot(inv(A_omega)).dot(Upsilon_T - I_n * T)  # n x n matrix
    C += Gamma_T*mu0*(1-gamma)
    # Compute the cost vector `c` as the **column sums** of matrix C
    c_vector = np.sum(C, axis=0)  # Column sums for the optimization cost

    # Step 5: MILP setup using LinearConstraint
    A_ub = -c.reshape(1, -1)  # Negated coefficients for the inequality constraint
    b_ub = [B - np.sum(c)]  # RHS for the inequality constraint

    # Binary constraint (integrality) on u
    integrality = np.ones(n)  # Ensure all u_i are binary (0 or 1)

    # Set lower and upper bounds for u: l=0, u=1 using scipy.Bounds
    bounds = Bounds([0]*n, [1]*n)

    # Use MILP to minimize the objective function subject to the budget constraint
    res = milp(c=c_vector, integrality=integrality, constraints=LinearConstraint(A_ub, -np.inf, b_ub), bounds=bounds)

    u_optimal = np.round(res.x)  # Round to binary values
    return u_optimal


# In[6]:


def calculate_costs(E, constant_cost):
    """
    Calculate the intervention costs for each node based on the number of events.
    
    Parameters:
    E             -- list of events for each node, where E[i] contains the events for node i
    constant_cost -- constant component of the intervention cost for each node
    
    Returns:
    costs -- an array where costs[i] is the intervention cost for node i
    """
    n = len(E)  # Number of nodes
    costs = np.zeros(n)
    
    # For each node, calculate the cost based on the number of events up to time tau
    for i in range(n):
        num_events = len(E[i])  # Number of events at node i
        costs[i] = constant_cost + num_events  # Add constant cost to number of events
    
    return costs


def calculate_budget(costs, q):
    """
    Calculate the total intervention budget based on the percentage of total cost.
    
    Parameters:
    costs -- array of intervention costs for each node
    q     -- budget percentage (e.g., 20 for 20%, 40 for 40%, etc.)
    
    Returns:
    B -- total intervention budget
    """
    total_cost = np.sum(costs)  # Sum of intervention costs across all nodes
    B = total_cost * (q / 100)  # Budget is q% of the total cost
    return B


# In[7]:


# Functions to compute heuristic interventions
def heuristic_intervention_exogenous_intensity(mu, c, B):
    n = len(mu)
    u = np.ones(n)
    nodes_sorted = np.argsort(-mu)  # Sort by exogenous intensity
    total_cost = 0
    for node in nodes_sorted:
        if total_cost + c[node] <= B:
            u[node] = 0  # Set u_i = 0 for intervention
            total_cost += c[node]
        if total_cost >= B:
            break
    return u

def heuristic_intervention_based_on_events(E, c, B):
    """
    Perform heuristic intervention based on the number of events at each node up until time tau.
    
    Parameters:
    E -- list of events for each node, where E[i] contains the events for node i
    c -- intervention costs for each node
    B -- total intervention budget
    
    Returns:
    u -- optimal intervention strategy (vector of 0s and 1s)
         u_i = 0 if node i is intervened at, 1 otherwise
    """
    n = len(E)  # Number of nodes
    u = np.ones(n)  # Start with no interventions (all u_i = 1)
    
    # Calculate the number of events at each node
    num_events = [len(E[i]) for i in range(n)]
    
    # Sort nodes by number of events in descending order
    nodes_sorted = np.argsort(-np.array(num_events))
    
    total_cost = 0
    for node in nodes_sorted:
        if total_cost + c[node] <= B:
            u[node] = 0  # Set u_i = 0 for intervention at node i
            total_cost += c[node]
        if total_cost >= B:
            break
    
    return u


# In[8]:


def compute_eta(T, tau, A, omega, nu, S_tau, mu0):
    """
    Compute eta(t; u) as described in the paper.
    
    Parameters:
    T      -- current time
    tau    -- intervention time
    A      -- interaction matrix (n x n)
    omega  -- decay rate matrix (n x n)
    nu     -- vector (n x 1) for intervention effects (p + (1 - p) * u)
    S_tau  -- state vector (n x 1) at time tau
    mu0    -- background intensity (scalar or n x 1 vector)
    
    Returns:
    eta    -- a vector of size n representing the intensity at each node
    """
    n = len(A)  # number of nodes
    A_omega = A-omega*np.eye(n)
    #A - np.diag(np.array([omega]*n))
    Xi = expm(A_omega * (T - tau))  # n x n matrix
    
    #basically a bit of calculation shows Xi = expm(a*(T-tau))*np.exp(-omega*(T-tau))
    #Xi = expm(a*(T-tau))*np.exp(-omega*(T-tau)) #let me try this at first
     
    I_n = np.eye(n)
    Psi = I_n + A.dot(inv(A_omega)).dot(expm(A_omega * T) - I_n)  # n x n matrix
    eta_h = Xi.dot(A).dot(nu * S_tau)  # n x 1 vector (Hadamard product nu * S_tau)
    
    if np.isscalar(mu0):
        mu0 = np.full(n, mu0)
    
    eta_e = Psi.dot(mu0)  # n x 1 vector
    eta = eta_h + eta_e
    
    return eta


def compute_EN(T, tau, A, omega, nu, S_tau, mu0):
    """
    Compute the total expected number of events E[N(T; u)] till time T.
    
    Parameters:
    T      -- final time horizon
    tau    -- intervention time
    A      -- interaction matrix (n x n)
    omega  -- decay rate matrix (n x n)
    nu     -- vector (n x 1) for intervention effects (p + (1 - p) * u)
    S_tau  -- state vector (n x 1) at time tau
    mu0    -- background intensity (scalar or n x 1 vector)
    
    Returns:
    E_N    -- a vector of size n representing the expected number of events at each node
    """
    n = len(A)
    I_n = np.eye(n)
    A_omega = A-omega*I_n #A-wI
    
    #Upsilon(T-tau) = (A-wI)^{-1}.(expm(A_omega*(T-tau))-I)
    Upsilon_T_min_tau = inv(A_omega).dot(expm(A_omega * (T - tau)) - I_n)  # n x n matrix
    Upsilon_T =  inv(A_omega).dot(expm(A_omega * T) - I_n) #(A-wI)^{-1}*(e^(A-wI)*t-I)

    Gamma_T = I_n*T + A.dot(inv(A_omega)).dot(Upsilon_T - I_n*T)  # n x n matrix
    
    #these Upsilon's are different 
    E_N = Gamma_T.dot(mu0) + Upsilon_T_min_tau.dot(A).dot(nu * S_tau)
    #E[N(T;u)] = Gamma(T)*mu+ Upsilon(T-tau).A.(nu*S_tau)
    
    if np.all(E_N>0):
        print("Expected number of events is >0 - it is okay")
    
    return E_N


# In[9]:


def GAMMA_compute_eta(gamma,T, tau, A, omega, nu, S_tau, mu0):
    """
    Compute eta(t; u) as described in the paper.
    
    Parameters:
    T      -- current time
    tau    -- intervention time
    A      -- interaction matrix (n x n)
    omega  -- decay rate matrix (n x n)
    nu     -- vector (n x 1) for intervention effects (p + (1 - p) * u)
    S_tau  -- state vector (n x 1) at time tau
    mu0    -- background intensity (scalar or n x 1 vector)
    
    Returns:
    eta    -- a vector of size n representing the intensity at each node
    """
    n = len(A)  # number of nodes
    A_omega = A-omega*np.eye(n)
    #A - np.diag(np.array([omega]*n))
    Xi = expm(A_omega * (T - tau))  # n x n matrix
    
    #basically a bit of calculation shows Xi = expm(a*(T-tau))*np.exp(-omega*(T-tau))
    #Xi = expm(a*(T-tau))*np.exp(-omega*(T-tau)) #let me try this at first
     
    I_n = np.eye(n)
    Psi = I_n + A.dot(inv(A_omega)).dot(expm(A_omega * T) - I_n)  # n x n matrix
    eta_h = Xi.dot(A).dot(nu * S_tau)  # n x 1 vector (Hadamard product nu * S_tau)
    
    if np.isscalar(mu0):
        mu0 = np.full(n, mu0)
    
    u = (nu-p)/(1-p)
    phi = gamma+(1-gamma)*u
    mu0_damped = mu0*phi #this is a hadamard product of 2 vectors ideally
    
    eta_e = Psi.dot(mu0_damped)  # n x 1 vector
    eta = eta_h + eta_e
    
    return eta


def GAMMA_compute_EN(gamma,T, tau, A, omega, nu, S_tau, mu0):
    """
    Compute the total expected number of events E[N(T; u)] till time T.
    
    Parameters:
    T      -- final time horizon
    tau    -- intervention time
    A      -- interaction matrix (n x n)
    omega  -- decay rate matrix (n x n)
    nu     -- vector (n x 1) for intervention effects (p + (1 - p) * u)
    S_tau  -- state vector (n x 1) at time tau
    mu0    -- background intensity (scalar or n x 1 vector)
    
    Returns:
    E_N    -- a vector of size n representing the expected number of events at each node
    """
    n = len(A)
    I_n = np.eye(n)
    A_omega = A-omega*I_n #A-wI
    
    #Upsilon(T-tau) = (A-wI)^{-1}.(expm(A_omega*(T-tau))-I)
    Upsilon_T_min_tau = inv(A_omega).dot(expm(A_omega * (T - tau)) - I_n)  # n x n matrix
    Upsilon_T =  inv(A_omega).dot(expm(A_omega * T) - I_n) #(A-wI)^{-1}*(e^(A-wI)*t-I)

    Gamma_T = I_n*T + A.dot(inv(A_omega)).dot(Upsilon_T - I_n*T)  # n x n matrix
    
    #these Upsilon's are different 
    u = (nu-p)/(1-p)
    phi = gamma+(1-gamma)*u
    mu0_damped = mu0*phi #this is a hadamard product of 2 vectors ideally
    E_N = Gamma_T.dot(mu0_damped) + Upsilon_T_min_tau.dot(A).dot(nu * S_tau)
    #E[N(T;u)] = Gamma(T)*mu+ Upsilon(T-tau).A.(nu*S_tau)
    
   # if np.all(E_N>0):
   #     print("Expected number of events is >0 - it is okay")
    
    return E_N


# In[10]:


# Perform intervention and calculate minimized values of eta = E[lambda(T)] for each strategy
def perform_intervention_and_calculate_total_rate(E, costs, q, p):
    B = calculate_budget(costs, q)
    S_tau = compute_S_vector(tau,  E, sigma2, omega)
    
    #print("tau = ",tau)
    # 1. LP-based strategy minimizing eta
    u_eta = solve_optimization_sum_eta(a, omega, E, tau, sigma2, mu, costs, B, p, T)
    nu_eta = p + (1 - p) * u_eta  # Correctly compute nu
   # print("nu_eta = ",nu_eta)
    total_rate_eta = np.sum(compute_eta(T, tau, a, omega, nu_eta, S_tau, mu))

    # 2. Heuristic-based strategy using exogenous intensity (mu)
    u_heuristic_mu = heuristic_intervention_exogenous_intensity(mu, costs, B)
    nu_heuristic_mu = p + (1 - p) * u_heuristic_mu  # Correctly compute nu
    total_rate_heuristic_mu = np.sum(compute_eta(T, tau, a, omega, nu_heuristic_mu, S_tau, mu))
    
    # 3. Heuristic-based strategy based on number of events
    u_heuristic_N = heuristic_intervention_based_on_events(E, costs, B)
    nu_heuristic_N = p + (1 - p) * u_heuristic_N  # Correctly compute nu
    total_rate_heuristic_N = np.sum(compute_eta(T, tau, a, omega, nu_heuristic_N, S_tau, mu))
    
    return total_rate_eta, total_rate_heuristic_mu, total_rate_heuristic_N


# Perform intervention and calculate E[N(T)] for each strategy
def perform_intervention_and_calculate_total_events(E, costs, q, p):
    B = calculate_budget(costs, q)
    S_tau = compute_S_vector(tau,  E, sigma2, omega)

    # 1. LP-based strategy minimizing EN
    u_EN = solve_optimization_sum_EN(T, tau, a, omega, E, sigma2, mu, costs, B, p)
    nu_EN = p + (1 - p) * u_EN  # Correctly compute nu
    total_events_EN = np.sum(compute_EN(T, tau, a, omega, nu_EN, S_tau, mu))
    
    # 2. Heuristic-based strategy using exogenous intensity (mu)
    u_heuristic_mu = heuristic_intervention_exogenous_intensity(mu, costs, B)
    nu_heuristic_mu = p + (1 - p) * u_heuristic_mu  # Correctly compute nu
    total_events_heuristic_mu = np.sum(compute_EN(T, tau, a, omega, nu_heuristic_mu, S_tau, mu))
    
    # 3. Heuristic-based strategy based on number of events
    u_heuristic_N = heuristic_intervention_based_on_events(E, costs, B)
    nu_heuristic_N = p + (1 - p) * u_heuristic_N  # Correctly compute nu
    total_events_heuristic_N = np.sum(compute_EN(T, tau, a, omega, nu_heuristic_N, S_tau, mu))
    
    return total_events_EN, total_events_heuristic_mu, total_events_heuristic_N


# In[ ]:





# In[11]:


# GAMMA versions of perform_intervention_and_calculate_total_rate
def perform_intervention_and_calculate_total_rate_gamma(E, costs, q, p, gamma):
    B = calculate_budget(costs, q)
    S_tau = compute_S_vector(tau, E, sigma2, omega)

    # 1. LP-based strategy minimizing sum_eta without accounting for dimished background density
    u_eta = solve_optimization_sum_eta(a, omega, E, tau, sigma2, mu, costs, B, p, T)
    nu_eta = p + (1 - p) * u_eta
    total_rate_eta = np.sum(GAMMA_compute_eta(gamma,T, tau, a, omega, nu_eta, S_tau, mu))

    # 2. LP-based strategy minimizing eta with diminished base effects
    GAMMA_u_eta = GAMMA_solve_optimization_sum_eta(gamma,a, omega, E, tau, sigma2, mu, costs, B, p, T)
    GAMMA_nu_eta = p + (1 - p) * GAMMA_u_eta
    total_rate_eta_gamma = np.sum(GAMMA_compute_eta(gamma, T, tau, a, omega, nu_eta, S_tau, mu))
     
    # 2. Heuristic-based strategy using exogenous intensity (mu) with diminished base effects
    u_heuristic_mu = heuristic_intervention_exogenous_intensity(mu, costs, B)
    nu_heuristic_mu = p + (1 - p) * u_heuristic_mu
    total_rate_heuristic_mu_gamma = np.sum(GAMMA_compute_eta(gamma, T, tau, a, omega, nu_heuristic_mu, S_tau, mu))

    # 3. Heuristic-based strategy based on number of events with diminished base effects
    u_heuristic_N = heuristic_intervention_based_on_events(E, costs, B)
    nu_heuristic_N = p + (1 - p) * u_heuristic_N
    total_rate_heuristic_N_gamma = np.sum(GAMMA_compute_eta(gamma, T, tau, a, omega, nu_heuristic_N, S_tau, mu))

    return total_rate_eta, total_rate_eta_gamma, total_rate_heuristic_mu_gamma, total_rate_heuristic_N_gamma


# GAMMA versions of perform_intervention_and_calculate_total_events
def perform_intervention_and_calculate_total_events_gamma(E, costs, q, p, gamma):
    B = calculate_budget(costs, q)
    S_tau = compute_S_vector(tau, E, sigma2, omega)

    # 1. LP-based strategy minimizing E[N(T)] without accounting for diminished background density
    u_EN = solve_optimization_sum_EN(T, tau, a, omega, E, sigma2, mu, costs, B, p)
    nu_EN = p + (1 - p) * u_EN
    total_events_EN = np.sum(GAMMA_compute_EN(gamma,T, tau, a, omega, nu_EN, S_tau, mu))

    # 2. LP-based strategy minimizing EN with diminished base effects
    GAMMA_u_EN = GAMMA_solve_optimization_sum_EN(gamma, T, tau, a, omega, E, sigma2, mu, costs, B, p)
    GAMMA_nu_EN = p + (1 - p) * GAMMA_u_EN
    total_events_EN_gamma = np.sum(GAMMA_compute_EN(gamma, T, tau, a, omega, GAMMA_nu_EN, S_tau, mu))

    # 3. Heuristic-based strategy using exogenous intensity (mu) with diminished base effects
    u_heuristic_mu = heuristic_intervention_exogenous_intensity(mu, costs, B)
    nu_heuristic_mu = p + (1 - p) * u_heuristic_mu
    total_events_heuristic_mu_gamma = np.sum(GAMMA_compute_EN(gamma, T, tau, a, omega, nu_heuristic_mu, S_tau, mu))

    # 4. Heuristic-based strategy based on number of events with diminished base effects
    u_heuristic_N = heuristic_intervention_based_on_events(E, costs, B)
    nu_heuristic_N = p + (1 - p) * u_heuristic_N
    total_events_heuristic_N_gamma = np.sum(GAMMA_compute_EN(gamma, T, tau, a, omega, nu_heuristic_N, S_tau, mu))

    return total_events_EN, total_events_EN_gamma, total_events_heuristic_mu_gamma, total_events_heuristic_N_gamma




import matplotlib.pyplot as plt
import matplotlib

def plot_events(E):
    """
    Plot the events (x, y) for each node with different colors and a legend.
    
    Parameters:
    E -- list of events for each node (each element is a matrix with rows as events: [t, x, y])
    """
    n = len(E)  # Get the number of nodes based on the length of E
    cmap = matplotlib.colormaps.get_cmap('tab10')  # Get the 'tab10' colormap (default 10 colors)
    
    plt.figure(figsize=(10, 6))
    
    for i in range(n):
        # Extract x, y coordinates of the events for node i
        events = E[i]
        x_coords = events[:, 1]  # Second column is x
        y_coords = events[:, 2]  # Third column is y
        
        # Use modulo to cycle through colors if n > 10
        plt.scatter(x_coords, y_coords, color=cmap(i % 10), label=f'Node {i}', alpha=1)
    
    plt.title('Event Locations for Each Node')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
# plot_events(E)


# In[13]:


# Load the CSV file into a pandas DataFrame
file_path = 'Crime_Data_from_2020_to_Present_20241008.csv'
try:
    df = pd.read_csv(file_path)
    print(df.head())
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")

# Pre-process the data after loading
def preprocess_data(df):
    df['TIME OCC'] = df['TIME OCC'].astype(str).str.zfill(4)  # Zero-pad TIME OCC to ensure it is always 4 digits
    df['date'] = pd.to_datetime(df['DATE OCC'].str[:10])
    df['time'] = pd.to_timedelta(df['TIME OCC'].astype(int).astype(str).str.zfill(4).str[:2] + ':' + df['TIME OCC'].astype(str).str.zfill(4).str[2:] + ':00')
    df['t'] = df['date'] + df['time']
    df['t'] = (df['t'] - df['t'].min()).dt.total_seconds()
    df = df.sort_values(by='t').reset_index(drop=True)
    df_cleaned = df.rename(columns={'t': 't', 'LAT': 'x', 'LON': 'y', 'AREA': 'area'})[['t', 'x', 'y', 'area']]
    return df_cleaned

df_cleaned = preprocess_data(df)
df_cleaned['area'] = df_cleaned['area'] - 1  # Shift indices to start at 0
df_cleaned["t"] = df_cleaned["t"]/max(df_cleaned["t"])  # Scale t in [0,1]

def group_events_as_matrices(df_cleaned):
    # Determine the number of unique areas to initialize a list of arrays
    num_areas = df_cleaned['area'].nunique()
    
    # Initialize a list to store events for each area
    events_by_node = [[] for _ in range(num_areas)]

    # Iterate over the dataframe and append each event to the corresponding node list
    for _, row in df_cleaned.iterrows():
        t, x, y, area = row['t'], row['x'], row['y'], int(row['area'])
        events_by_node[area].append([t, x, y])

    # Convert each node's event list to a numpy array without any column names
    for i in range(len(events_by_node)):
        events_by_node[i] = np.array(events_by_node[i])
    
    return events_by_node

E = group_events_as_matrices(df_cleaned)  # This is the LA data


# In[14]:


def plot_events(E, area_names=None):
    """
    Plot the events (x, y) for each node with different colors and a legend.

    Parameters:
    E -- list of events for each node (each element is a matrix with rows as events: [t, x, y])
    area_names -- list of area names for each node (optional)
    """
    n = len(E)  # Get the number of nodes based on the length of E
    cmap = matplotlib.colormaps.get_cmap('tab10')  # Get the 'tab10' colormap (default 10 colors)

    plt.figure(figsize=(12, 8))

    for i in range(n):
        # Extract x, y coordinates of the events for node i
        events = E[i]
        if events.size > 0:  # Check if there are events for this node
            x_coords = events[:, 1]  # Second column is x
            y_coords = events[:, 2]  # Third column is y

            # Set the label for the legend, using the area name if provided
            label = f'Node {i}' if area_names is None else area_names[i]

            # Use modulo to cycle through colors if n > 10
            plt.scatter(x_coords, y_coords, color=cmap(i % 10), label=label, alpha=1)

    plt.title("Crime Event Locations for Each Precinct in Los Angeles in First Week of September")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(title='Precinct Areas', loc='upper right', bbox_to_anchor=(1.25, 1))  # Adjust legend position
    plt.grid(True)
    plt.show()

# Assuming you have the area names as a list
area_names = df['AREA NAME'].unique().tolist()

# Plot with area names
plot_events(E, area_names)


# In[22]:


# Load fitted parameters
with open('LA_crime_fitted_parameters.pkl', 'rb') as f:
    fitted_parameters = pickle.load(f)
    additional_parameters = pickle.load(f)
fitted_parameters.update(additional_parameters)

# Extract loaded parameters
sigma2 = fitted_parameters.get('sigma')
omega = fitted_parameters.get('omega')
eta_0 = fitted_parameters.get('eta_0')
a = fitted_parameters.get('A')
mu = fitted_parameters.get('mu_0')
sigma_0 = fitted_parameters.get('eta_0')

# Set constants
d = 2  # Working in x, y plane
n = a.shape[0]  # Number of nodes
tau = 1.0  # Time point till which we observe the time series
T = tau * 4.3 # Time horizon
constant_cost = 5

# Calculate intervention costs: cost for node j is constant_cost + number of events at node j
costs = np.array([constant_cost + len(E[j]) for j in range(n)])


#p - probability of survival after intervention
p = 0.1
gamma = 0.75 #how much does background rates scale by after intervention?


# In[23]:


# Budget percentages
q_values = np.linspace(0, 100, 11)

# Result storage for percentage reductions in rates and events
rate_reduction_gamma_eta = []
rate_reduction_gamma_heuristic_mu = []
rate_reduction_gamma_heuristic_N = []

event_reduction_gamma_EN = []
event_reduction_gamma_heuristic_mu = []
event_reduction_gamma_heuristic_N = []

# Compute the no-intervention baseline (all u_i = 1)
u_no_intervention = np.ones(n)  # No intervention
S_tau = compute_S_vector(tau, E, sigma2, omega)
rate_no_intervention = np.sum(compute_eta(T, tau, a, omega, u_no_intervention, S_tau, mu))
events_no_intervention = np.sum(compute_EN(T, tau, a, omega, u_no_intervention, S_tau, mu))

# Loop over different budget percentages
for q in q_values:
    print(f"Processing budget percentage q={q}%")

    # Calculate budget
    B = (q / 100.0) * np.sum(costs)

    # 1. LP-based Eta minimization with diminished effects
    u_eta_gamma = GAMMA_solve_optimization_sum_eta(gamma, a, omega, E, tau, sigma2, mu, costs, B, p, T)
    nu_eta_gamma = p + (1 - p) * u_eta_gamma  # Correct intervention effects
    total_rate_eta_gamma = np.sum(GAMMA_compute_eta(gamma, T, tau, a, omega, nu_eta_gamma, S_tau, mu))

    # 2. Heuristic-based Strategy using Exogenous Intensity with diminished effects
    u_heuristic_mu = heuristic_intervention_exogenous_intensity(mu, costs, B)
    nu_heuristic_mu = p + (1 - p) * u_heuristic_mu
    total_rate_heuristic_mu_gamma = np.sum(GAMMA_compute_eta(gamma, T, tau, a, omega, nu_heuristic_mu, S_tau, mu))

    # 3. Heuristic-based Strategy based on Number of Events with diminished effects
    u_heuristic_N = heuristic_intervention_based_on_events(E, costs, B)
    nu_heuristic_N = p + (1 - p) * u_heuristic_N
    total_rate_heuristic_N_gamma = np.sum(GAMMA_compute_eta(gamma, T, tau, a, omega, nu_heuristic_N, S_tau, mu))

    # Calculate percentage reduction in rates compared to no intervention
    rate_reduction_gamma_eta.append(100 * (rate_no_intervention - total_rate_eta_gamma) / rate_no_intervention)
    rate_reduction_gamma_heuristic_mu.append(100 * (rate_no_intervention - total_rate_heuristic_mu_gamma) / rate_no_intervention)
    rate_reduction_gamma_heuristic_N.append(100 * (rate_no_intervention - total_rate_heuristic_N_gamma) / rate_no_intervention)

    # 4. LP-based EN minimization with diminished effects
    u_EN_gamma = GAMMA_solve_optimization_sum_EN(gamma, T, tau, a, omega, E, sigma2, mu, costs, B, p)
    nu_EN_gamma = p + (1 - p) * u_EN_gamma
    total_events_EN_gamma = np.sum(GAMMA_compute_EN(gamma, T, tau, a, omega, nu_EN_gamma, S_tau, mu))

    # Calculate percentage reduction in events compared to no intervention
    event_reduction_gamma_EN.append(100 * (events_no_intervention - total_events_EN_gamma) / events_no_intervention)
    total_events_heuristic_mu_gamma = np.sum(GAMMA_compute_EN(gamma, T, tau, a, omega, nu_heuristic_mu, S_tau, mu))
    total_events_heuristic_N_gamma = np.sum(GAMMA_compute_EN(gamma, T, tau, a, omega, nu_heuristic_N, S_tau, mu))

    event_reduction_gamma_heuristic_mu.append(100 * (events_no_intervention - total_events_heuristic_mu_gamma) / events_no_intervention)
    event_reduction_gamma_heuristic_N.append(100 * (events_no_intervention - total_events_heuristic_N_gamma) / events_no_intervention)

    print(f"--- Budget {q}% ---")
    print(f"Total rate for LP-based Eta (Gamma): {total_rate_eta_gamma}")
    print(f"Total rate for Heuristic Mu (Gamma): {total_rate_heuristic_mu_gamma}")
    print(f"Total rate for Heuristic N (Gamma): {total_rate_heuristic_N_gamma}")
    print(f"Total rate at T with no intervention: {rate_no_intervention}")
    print("----------------------------------------------")



# In[24]:


# Plot Rate Reductions (Gamma comparison)
plt.figure(figsize=(10, 6))
plt.plot(q_values, rate_reduction_gamma_eta, label='LP-based Strategy (Eta, Gamma)', marker='x', linestyle='-')
plt.plot(q_values, rate_reduction_gamma_heuristic_mu, label='Heuristic Strategy (Mu, Gamma)', marker='^', linestyle='--')
plt.plot(q_values, rate_reduction_gamma_heuristic_N, label='Heuristic Strategy (N, Gamma)', marker='v', linestyle='--')
plt.xlabel('Budget as % of Total Cost')
plt.ylabel('Rate Reduction (%)')
plt.title('Comparison of Rate Reduction across Strategies (LA Crime Data with Gamma)')
plt.legend()
plt.grid(True)
plt.show()

# Plot Event Reductions (Gamma comparison)
plt.figure(figsize=(10, 6))
plt.plot(q_values, event_reduction_gamma_EN, label='LP-based Strategy (EN, Gamma)', marker='x', linestyle='-')
plt.plot(q_values, event_reduction_gamma_heuristic_mu, label='Heuristic Strategy (Mu, Gamma)', marker='^', linestyle='--')
plt.plot(q_values, event_reduction_gamma_heuristic_N, label='Heuristic Strategy (N, Gamma)', marker='v', linestyle='--')
plt.xlabel('Budget as % of Total Cost')
plt.ylabel('Event Reduction (%)')
plt.title('Comparison of Event Reduction across Strategies (LA Crime Data with Gamma)')
plt.legend()
plt.grid(True)
plt.show()


# In[25]:


# Create table with q values and reductions in rates
rate_reduction_table = pd.DataFrame({
    'Budget %': q_values,
    'Rate Reduction (Eta, Gamma)': rate_reduction_gamma_eta,
    'Rate Reduction (Heuristic Mu, Gamma)': rate_reduction_gamma_heuristic_mu,
    'Rate Reduction (Heuristic N, Gamma)': rate_reduction_gamma_heuristic_N
})

print(rate_reduction_table.to_string(index=False, line_width=50))



# In[26]:


# Create table with q values and reductions in rates
expected_num_event_reduction_table = pd.DataFrame({
    'Budget %': q_values,
    'Num Reduction (Eta, Gamma)': event_reduction_gamma_EN,
    'Num Reduction (Heuristic Mu, Gamma)':event_reduction_gamma_heuristic_mu,
    'Num Reduction (Heuristic N, Gamma)': event_reduction_gamma_heuristic_N
})

print(expected_num_event_reduction_table.to_string(index=False, line_width=50))


# In[49]:


# Display rate reduction table with lines between rows and columns
from tabulate import tabulate  # Ensure you have tabulate installed
print(tabulate(rate_reduction_table, headers='keys', tablefmt='grid'))

expected_num_event_reduction_table = pd.DataFrame({
    'Budget %': q_values,
    'Event Reduction (EN, Gamma)': event_reduction_gamma_EN,
    'Event Reduction (Heuristic Mu, Gamma)': event_reduction_gamma_heuristic_mu,
    'Event Reduction (Heuristic N, Gamma)': event_reduction_gamma_heuristic_N
})

# Display event reduction table with lines between rows and columns
print(tabulate(expected_num_event_reduction_table, headers='keys', tablefmt='grid'))


# In[40]:


# Updated function to plot events with intervention effect
def plot_events_intervention(E, area_names=None, p=0.3):
    """
    Plot the events (x, y) for each node with different colors and a legend after applying intervention effect.

    Parameters:
    E -- list of events for each node (each element is a matrix with rows as events: [t, x, y])
    area_names -- list of area names for each node (optional)
    p -- fraction of events that survive after intervention
    """
    n = len(E)  # Get the number of nodes based on the length of E
    cmap = matplotlib.colormaps.get_cmap('tab10')  # Get the 'tab10' colormap (default 10 colors)

    plt.figure(figsize=(15, 10))

    for i in range(n):
        # Extract x, y coordinates of the events for node i
        events = E[i]
        if events.size > 0:  # Check if there are events for this node
            # Randomly select which events survive after intervention
            survived_indices = np.random.rand(len(events)) < p
            survived_events = events[survived_indices]
            not_survived_events = events[~survived_indices]

            x_coords_survived = survived_events[:, 1]  # Second column is x
            y_coords_survived = survived_events[:, 2]  # Third column is y

            x_coords_not_survived = not_survived_events[:, 1]  # Second column is x
            y_coords_not_survived = not_survived_events[:, 2]  # Third column is y

            # Set the label for the legend, using the area name if provided
            label = f'Node {i}' if area_names is None else area_names[i]

            # Use modulo to cycle through colors if n > 10
            plt.scatter(x_coords_survived, y_coords_survived, color=cmap(i % 10), label=f'{label} (Active post-intervention)', alpha=0.7)
            plt.scatter(x_coords_not_survived, y_coords_not_survived, color=cmap(i % 10), marker='x', label=f'{label} (Inactive post-intervention)', alpha=0.7)

    plt.title(f"Crime Events in Intervened Precincts post intervention in Los Angeles (p = {p})")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(title='Precinct Areas', bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust legend position
    plt.grid(True)
    plt.show()



# In[20]:


# Create a dictionary mapping areas to their names from df_cleaned
area_name_dict = dict(zip(df['AREA'], df['AREA NAME']))
# Analysis of u_eta for budgets between 40% and 70%
for idx, q in enumerate(q_values):
    #u_eta_gamma = GAMMA_solve_optimization_sum_eta(gamma, a, omega, E, tau, sigma2, mu, costs, (q / 100.0) * np.sum(costs), p, T)
   
    u_EN_gamma = GAMMA_solve_optimization_sum_EN(gamma, T, tau, a, omega, E, sigma2, mu, costs, (q / 100.0) * np.sum(costs), p)
    if 40 <= q <= 70:
        print(f"Analysis for budget q={q}%:")
        nodes_intervention = np.where(u_EN_gamma== 0)[0]
        print(f"Nodes where we do intervention (u_i = 0): {nodes_intervention}")

        # Display the AREA NAME for nodes with no intervention
        areas = []
        for node in nodes_intervention:
            area_name = area_name_dict.get(node + 1, f"Unknown Area {node + 1}")
            areas.append(area_name)
        print("Intervened areas are:", areas)
        print("\n")


# In[ ]:





# In[ ]:


# Create a dictionary mapping areas to their names from df_cleaned
area_name_dict = dict(zip(df['AREA'], df['AREA NAME']))
# Analysis of u_eta for budgets between 40% and 70%
for idx, q in enumerate(q_values):
    u_eta_gamma = GAMMA_solve_optimization_sum_eta(gamma, a, omega, E, tau, sigma2, mu, costs, (q / 100.0) * np.sum(costs), p, T)
    if 40 <= q <= 70:
        print(f"Analysis for budget q={q}%:")
        nodes_intervention = np.where(u_eta_gamma == 0)[0]
        print(f"Nodes where we do intervention (u_i = 0): {nodes_intervention}")

        # Display the AREA NAME for nodes with no intervention
        areas = []
        for node in nodes_intervention:
            area_name = area_name_dict.get(node + 1, f"Unknown Area {node + 1}")
            areas.append(area_name)
        print("Intervened areas are:", areas)
        print("\n")


# In[43]:


#we discover some new regions! 

#'Olympic', 'Topanga'

# Filter events for selected areas
selected_area_indices = [0, 3, 4, 5, 6, 9, 19,20]  # Indices for 'Central', 'Rampart', 'Southwest', 'Harbor', 'Hollywood', 'West LA'
E_selected = [E[i] for i in selected_area_indices]
area_names_selected = [area_name_dict[i + 1] for i in selected_area_indices]

# Plot events with intervention effect
plot_events_intervention(E_selected, area_names=area_names_selected, p=0.1)


# In[ ]:




