#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy.random as rand
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
from scipy.linalg import eigvals
from scipy.optimize import milp, LinearConstraint, Bounds


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
            
          #  print("Event generation successful!")
            return events  # Return the events if successful

        except Exception as e:
            if verbose:
                print(f"Error during generation attempt {attempt + 1}: {e}")
            print("Retrying...")

    print(f"Generation failed after {max_retries} attempts.")
    return None  # Return None if the generation failed after max_retries


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


# In[5]:


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


# In[6]:


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


# In[7]:


def GAMMA_compute_eta(gamma,p,T, tau, A, omega, nu, S_tau, mu0):
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


def GAMMA_compute_EN(gamma,p,T, tau, A, omega, nu, S_tau, mu0):
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
    
  #  if np.all(E_N>0):
  #      print("Expected number of events is >0 - it is okay")
    
    return E_N


# In[8]:


# GAMMA versions of perform_intervention_and_calculate_total_rate
def perform_intervention_and_calculate_total_rate_gamma(E, costs, q, p, gamma):
    B = calculate_budget(costs, q)
    S_tau = compute_S_vector(tau, E, sigma2, omega)

    # 1. LP-based strategy minimizing sum_eta without accounting for dimished background density
    u_eta = GAMMA_solve_optimization_sum_eta(gamma,a, omega, E, tau, sigma2, mu, costs, B, p, T)
    nu_eta = p + (1 - p) * u_eta
    total_rate_eta = np.sum(GAMMA_compute_eta(gamma,p,T, tau, a, omega, nu_eta, S_tau, mu))

    # 2. LP-based strategy minimizing eta with diminished base effects
    GAMMA_u_eta = GAMMA_solve_optimization_sum_eta(gamma,a, omega, E, tau, sigma2, mu, costs, B, p, T)
    GAMMA_nu_eta = p + (1 - p) * GAMMA_u_eta
    total_rate_eta_gamma = np.sum(GAMMA_compute_eta(gamma,p,T, tau, a, omega, nu_eta, S_tau, mu))
     
    # 2. Heuristic-based strategy using exogenous intensity (mu) with diminished base effects
    u_heuristic_mu = heuristic_intervention_exogenous_intensity(mu, costs, B)
    nu_heuristic_mu = p + (1 - p) * u_heuristic_mu
    total_rate_heuristic_mu_gamma = np.sum(GAMMA_compute_eta(gamma,p, T, tau, a, omega, nu_heuristic_mu, S_tau, mu))

    # 3. Heuristic-based strategy based on number of events with diminished base effects
    u_heuristic_N = heuristic_intervention_based_on_events(E, costs, B)
    nu_heuristic_N = p + (1 - p) * u_heuristic_N
    total_rate_heuristic_N_gamma = np.sum(GAMMA_compute_eta(gamma, p,T, tau, a, omega, nu_heuristic_N, S_tau, mu))

    return total_rate_eta, total_rate_eta_gamma, total_rate_heuristic_mu_gamma, total_rate_heuristic_N_gamma


# GAMMA versions of perform_intervention_and_calculate_total_events
def perform_intervention_and_calculate_total_events_gamma(E, costs, q, p, gamma):
    B = calculate_budget(costs, q)
    S_tau = compute_S_vector(tau, E, sigma2, omega)

    # 1. LP-based strategy minimizing E[N(T)] without accounting for diminished background density
    u_EN = GAMMA_solve_optimization_sum_EN(T, tau, a, omega, E, sigma2, mu, costs, B, p)
    nu_EN = p + (1 - p) * u_EN
    total_events_EN = np.sum(GAMMA_compute_EN(gamma,p,T, tau, a, omega, nu_EN, S_tau, mu))

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


# In[9]:


def perform_intervention_and_calculate_total_rate_gamma(E, costs, q, p, gamma):
    B = calculate_budget(costs, q)
    S_tau = compute_S_vector(tau, E, sigma2, omega)
    
    u_eta = GAMMA_solve_optimization_sum_eta(gamma, a, omega, E, tau, sigma2, mu, costs, B, p, T)
    nu_eta = p + (1 - p) * u_eta
    total_rate_eta = np.sum(GAMMA_compute_eta(gamma, p, T, tau, a, omega, nu_eta, S_tau, mu))
    
    GAMMA_u_eta = GAMMA_solve_optimization_sum_eta(gamma, a, omega, E, tau, sigma2, mu, costs, B, p, T)
    GAMMA_nu_eta = p + (1 - p) * GAMMA_u_eta
    total_rate_eta_gamma = np.sum(GAMMA_compute_eta(gamma, p, T, tau, a, omega, GAMMA_nu_eta, S_tau, mu))
    
    u_heuristic_mu = heuristic_intervention_exogenous_intensity(mu, costs, B)
    nu_heuristic_mu = p + (1 - p) * u_heuristic_mu
    total_rate_heuristic_mu_gamma = np.sum(GAMMA_compute_eta(gamma, p, T, tau, a, omega, nu_heuristic_mu, S_tau, mu))
    
    u_heuristic_N = heuristic_intervention_based_on_events(E, costs, B)
    nu_heuristic_N = p + (1 - p) * u_heuristic_N
    total_rate_heuristic_N_gamma = np.sum(GAMMA_compute_eta(gamma, p, T, tau, a, omega, nu_heuristic_N, S_tau, mu))
    
    return total_rate_eta, total_rate_eta_gamma, total_rate_heuristic_mu_gamma, total_rate_heuristic_N_gamma


def perform_intervention_and_calculate_total_events_gamma(E, costs, q, p, gamma):
    B = calculate_budget(costs, q)
    S_tau = compute_S_vector(tau, E, sigma2, omega)
    
    u_EN = GAMMA_solve_optimization_sum_EN(gamma, T, tau, a, omega, E, sigma2, mu, costs, B, p)
    nu_EN = p + (1 - p) * u_EN
    total_events_EN = np.sum(GAMMA_compute_EN(gamma, p, T, tau, a, omega, nu_EN, S_tau, mu))
    
    GAMMA_u_EN = GAMMA_solve_optimization_sum_EN(gamma, T, tau, a, omega, E, sigma2, mu, costs, B, p)
    GAMMA_nu_EN = p + (1 - p) * GAMMA_u_EN
    total_events_EN_gamma = np.sum(GAMMA_compute_EN(gamma, p, T, tau, a, omega, GAMMA_nu_EN, S_tau, mu))
    
    u_heuristic_mu = heuristic_intervention_exogenous_intensity(mu, costs, B)
    nu_heuristic_mu = p + (1 - p) * u_heuristic_mu
    total_events_heuristic_mu_gamma = np.sum(GAMMA_compute_EN(gamma, p, T, tau, a, omega, nu_heuristic_mu, S_tau, mu))
    
    u_heuristic_N = heuristic_intervention_based_on_events(E, costs, B)
    nu_heuristic_N = p + (1 - p) * u_heuristic_N
    total_events_heuristic_N_gamma = np.sum(GAMMA_compute_EN(gamma, p, T, tau, a, omega, nu_heuristic_N, S_tau, mu))
    
    return total_events_EN, total_events_EN_gamma, total_events_heuristic_mu_gamma, total_events_heuristic_N_gamma


# #without any diminished background effect post-intrevnetion
# avg_rate_reduction_eta, avg_rate_reduction_heuristic_mu

# In[10]:


#example generation 
n, d, tau = 200, 2, 10.
T = tau*2
#tau is the time point till which we observe the time series
sigma_0 = 0.25


# Alternatively, using a list comprehension for a more concise implementation
a_max = 0.05
#a = np.array([[a_max * np.exp(-abs(i - j) ** 2) for j in range(n)] for i in range(n)])


a = 1.5*np.random.rand(n, n)  # Interaction matrix A_{ij}
# Calculate maximum row sum of A
omega = 0.2


# Ensure spectral radius of `a` is less than `omega`
eigenvalues = eigvals(a)  # Calculate the eigenvalues of `a`
spectral_radius = max(np.abs(eigenvalues))  # Spectral radius is the maximum absolute eigenvalue

if spectral_radius >= omega:
    scaling_factor = (omega / spectral_radius) * 0.9  # Apply a factor to ensure it is strictly less than omega
    a = a * scaling_factor
    print(f"Rescaling interaction matrix `a` by a factor of {scaling_factor:.4f} to ensure spectral radius < omega.")

# Print information for verification
else:
    print("Spectral radius of A/omega is :", max(np.abs(eigvals(a)))/omega)

sigma2 = 0.1
#np.random.rand(n, n)  # Spatial variance
mu = np.random.uniform(size=n,low=0.01,high=0.05)
#np.array([0.1]*n)  # Background intensity
constant_cost = 1


# In[ ]:





# In[ ]:


# Define p and gamma combinations
p_values = [0.1, 0.3]
gamma_values = [0.6, 0.8, 1.0]
p_gamma_list = [(p, gamma) for p in p_values for gamma in gamma_values]

# --- Rate Reduction Plot Setup (2x3 layout) ---
fig_rate, axes_rate = plt.subplots(2, 3, figsize=(18, 10))
axes_rate = axes_rate.flatten()

# --- Event Reduction Plot Setup (2x3 layout) ---
fig_event, axes_event = plt.subplots(2, 3, figsize=(18, 10))
axes_event = axes_event.flatten()

q_values = np.linspace(0, 100, 11)
n_realizations = 100

for idx, (p, gamma) in enumerate(p_gamma_list):
    start_time = time.time()

    rate_reduction_gamma_eta = {q: [] for q in q_values}
    rate_reduction_gamma_heuristic_mu = {q: [] for q in q_values}
    rate_reduction_gamma_heuristic_N = {q: [] for q in q_values}
    event_reduction_gamma_EN = {q: [] for q in q_values}
    event_reduction_gamma_heuristic_mu = {q: [] for q in q_values}
    event_reduction_gamma_heuristic_N = {q: [] for q in q_values}

    for realization_idx in range(n_realizations):
        E = generate_multistream_ST(n, d, tau, a, omega, sigma2, mu, sigma_0)
        costs = calculate_costs(E, constant_cost)
        u_no_intervention = np.ones(n)
        nu_no_intervention = p + (1 - p) * u_no_intervention
        S_tau = compute_S_vector(tau, E, sigma2, omega)
        rate_no_intervention = np.sum(GAMMA_compute_eta(gamma, p, T, tau, a, omega, nu_no_intervention, S_tau, mu))
        events_no_intervention = np.sum(GAMMA_compute_EN(gamma, p, T, tau, a, omega, nu_no_intervention, S_tau, mu))

        for q in q_values:
            _, r_eta, r_mu, r_N = perform_intervention_and_calculate_total_rate_gamma(E, costs, q, p, gamma)
            _, e_eta, e_mu, e_N = perform_intervention_and_calculate_total_events_gamma(E, costs, q, p, gamma)

            rate_reduction_gamma_eta[q].append(100 * (rate_no_intervention - r_eta) / rate_no_intervention)
            rate_reduction_gamma_heuristic_mu[q].append(100 * (rate_no_intervention - r_mu) / rate_no_intervention)
            rate_reduction_gamma_heuristic_N[q].append(100 * (rate_no_intervention - r_N) / rate_no_intervention)
            event_reduction_gamma_EN[q].append(100 * (events_no_intervention - e_eta) / events_no_intervention)
            event_reduction_gamma_heuristic_mu[q].append(100 * (events_no_intervention - e_mu) / events_no_intervention)
            event_reduction_gamma_heuristic_N[q].append(100 * (events_no_intervention - e_N) / events_no_intervention)

    avg_rate_reduction_gamma_eta = [np.mean(rate_reduction_gamma_eta[q]) for q in q_values]
    avg_rate_reduction_gamma_heuristic_mu = [np.mean(rate_reduction_gamma_heuristic_mu[q]) for q in q_values]
    avg_rate_reduction_gamma_heuristic_N = [np.mean(rate_reduction_gamma_heuristic_N[q]) for q in q_values]
    avg_event_reduction_gamma_EN = [np.mean(event_reduction_gamma_EN[q]) for q in q_values]
    avg_event_reduction_gamma_heuristic_mu = [np.mean(event_reduction_gamma_heuristic_mu[q]) for q in q_values]
    avg_event_reduction_gamma_heuristic_N = [np.mean(event_reduction_gamma_heuristic_N[q]) for q in q_values]

    # Plot Rate Reduction
    # --- Plot Rate Reduction ---
    ax_rate = axes_rate[idx]
    ax_rate.plot(q_values, avg_rate_reduction_gamma_eta, label=r'Optimal $\eta$ Strategy', marker='x', color='blue')
    ax_rate.plot(q_values, avg_rate_reduction_gamma_heuristic_mu, label=r'Heuristic $\mu$ Strategy', marker='^', color='orange')
    ax_rate.plot(q_values, avg_rate_reduction_gamma_heuristic_N, label=r'Heuristic $N(\tau)$ Strategy', marker='v', color='green')
    ax_rate.set_title(f'Rate Reduction (p={p}, γ={gamma})')
    ax_rate.set_xlabel('Budget (%)')
    ax_rate.set_ylabel('Rate Reduction (%)')
    ax_rate.legend()
    ax_rate.grid(True)

    # --- Plot Event Reduction ---
    ax_event = axes_event[idx]
    ax_event.plot(q_values, avg_event_reduction_gamma_EN, label=r'Optimal $E[N(T)]$ Strategy', marker='x', color='red')
    ax_event.plot(q_values, avg_event_reduction_gamma_heuristic_mu, label=r'Heuristic $\mu$ Strategy', marker='^', color='orange')
    ax_event.plot(q_values, avg_event_reduction_gamma_heuristic_N, label=r'Heuristic $N(\tau)$ Strategy', marker='v', color='green')
    ax_event.set_title(f'Event Reduction (p={p}, γ={gamma})')
    ax_event.set_xlabel('Budget (%)')
    ax_event.set_ylabel('Event Reduction (%)')
    ax_event.legend()
    ax_event.grid(True)

    elapsed_time = (time.time() - start_time) / 60
    print(f"Completed: (p, γ) = ({p}, {gamma}) in {elapsed_time:.2f} minutes")


# In[ ]:


fig_rate.tight_layout()
fig_event.tight_layout()
# Save the figures
fig_rate.savefig("p_gamma_rate_reduction.png", dpi=300, bbox_inches='tight')
fig_event.savefig("p_gamma_event_reduction.png", dpi=300, bbox_inches='tight')

# Display plots
plt.show()


# In[ ]:





# In[ ]:




