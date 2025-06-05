#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy.random as rand
from scipy.stats import poisson
import pandas as pd
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


# In[67]:


# Load the CSV file into a pandas DataFrame
file_path = 'Crime_Data_from_2020_to_Present_20241008.csv'
try:
    df = pd.read_csv(file_path)
    print(df.head())
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")


# In[68]:


np.sort(np.array(df["TIME OCC"]))[:100]


# In[69]:


# Pre-process the data after loading
def preprocess_data(df):
    # Extract the date from 'DATE OCC' and combine it with 'TIME OCC' to get a datetime value
    df['TIME OCC'] = df['TIME OCC'].astype(str).str.zfill(4)  # Zero-pad TIME OCC to ensure it is always 4 digits
    df['date'] = pd.to_datetime(df['DATE OCC'].str[:10])
    df['time'] = pd.to_timedelta(df['TIME OCC'].astype(int).astype(str).str.zfill(4).str[:2] + ':' + df['TIME OCC'].astype(str).str.zfill(4).str[2:] + ':00')
    df['t'] = df['date'] + df['time']
    df['t'] = (df['t'] - df['t'].min()).dt.total_seconds()

    # Sort the dataframe by time to ensure the rows are in increasing order of 't'
    df = df.sort_values(by='t').reset_index(drop=True)

    # Rename columns to (t, x, y) format
    df_cleaned = df.rename(columns={'t': 't', 'LAT': 'x', 'LON': 'y', 'AREA': 'area'})[['t', 'x', 'y', 'area']]
    
    return df_cleaned

# Apply the pre-processing function to the DataFrame
df_cleaned = preprocess_data(df)
df_cleaned['area'] = df_cleaned['area'] - 1  # Shift indices to start at 0
df_cleaned["t"] = df_cleaned["t"]/max(df_cleaned["t"]) # Scale t in [0,1]


# In[70]:


# Pre-calculate centers for each area
def calculate_centers(df_array, n):
    centers = []
    for area in range(n):
        events = df_array[df_array[:, 3] == area]
        if len(events) > 0:
            x_mean = events[:, 1].mean()
            y_mean = events[:, 2].mean()
            centers.append((x_mean, y_mean))
        else:
            centers.append((0, 0))  # Default value if no events
    return centers

# Convert DataFrame to NumPy array for faster processing
df_array = df_cleaned.to_numpy()

# Number of areas (nodes)
n = len(np.unique(df_array[:, 3]))

# Calculate centers
centers = calculate_centers(df_array, n)


# centers

# In[71]:


#Initialize parameters for the EM algorithm
def initialize_parameters(num_areas):
   # Initial guesses for parameters
   K = np.random.rand(num_areas, num_areas)  # Mean number of events triggered between areas
   beta = np.random.rand(num_areas, num_areas)  # Extent to which events contribute to background rate
   sigma = 1.0  # Standard deviation for spatial triggering kernel
   omega = 1.0  # Decay rate for temporal triggering kernel
   mu_0 = 0.5  # Initial background rate
   eta_0 = 1.0  # Spatial scale for background rate
   p_ij = np.random.rand(len(E), len(E))  # Initial probability matrix for E-step
   return K, beta, sigma, omega, mu_0, eta_0, p_ij

# Number of areas (nodes)
num_areas = len(set(df_cleaned['area']))


#this give mu_u(t,x,y) at teh uth node
#def background_intensity(x, y, u, eta_0, beta, centers, T):
#    return np.sum([beta[j, int(u)] / (2 * np.pi * eta_0**2 * T) * np.exp(-((x - centers[j][0])**2 + (y - centers[j][1])**2) / (2 * eta_0**2)) for j in range(n)])


def background_intensity(x, y, u, eta_0, beta, df_array, T):
   small_tol = 1e-10  # Small value to prevent division by zero
   intensity = 0
   
   # Sum over all events (from 1 to N)
   for i in range(len(df_array)):
       u_i, t_i, x_i, y_i = int(df_array[i, 3]), df_array[i, 0], df_array[i, 1], df_array[i, 2]
       #if u_i == u:
       dist_sq = (x - x_i)**2 + (y - y_i)**2
       intensity += beta[u, u_i] / (2 * np.pi * eta_0**2 * T + small_tol) * np.exp(-dist_sq / (2 * eta_0**2 + small_tol))

   return intensity


# Triggering kernel
def triggering_kernel(t, x, y, omega, sigma):
   g1_t = omega * np.exp(-omega * t)
   g2_xy = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x**2 + y**2) / (2 * sigma**2)))
   return g1_t * g2_xy


# In[ ]:





# In[100]:


def em_algorithm(df_array, K, beta, sigma, omega, eta_0, p,  epsilon=1e-3, max_iter=10):
        # EM Algorithm
    t_1 = time.time()
    delta_values = []
    log_likelihood_values = []
    k = 0
    T = df_array[:, 0].max()  # Time window
    prev_log_likelihood = -np.inf
    while k < max_iter:
        # E-step
        lambda_i =[] #I'd save lambda(t_i,x_i,y_i) here
        for i in range(len(df_array)):
            u_i, t_i, x_i, y_i = int(df_array[i, 3]), df_array[i, 0], df_array[i, 1], df_array[i, 2]
            mu_u = background_intensity(x_i, y_i, u_i, eta_0, beta, df_array , T)
            lambda_i.append(mu_u)
            
            # Vectorized calculation for all j < i
            t_j, x_j, y_j, u_j = df_array[:i, 0], df_array[:i, 1], df_array[:i, 2], df_array[:i, 3].astype(int)
            g_ij_array = K[u_i, u_j] * triggering_kernel(t_i - t_j, x_i - x_j, y_i - y_j, omega, sigma)
            lambda_ij = mu_u + np.sum(g_ij_array)
            
            # Probability of event i being triggered by event j (for all j < i)
            p[i, :i] = np.where(lambda_ij > 0, g_ij_array /(small_tol+lambda_ij), 0)
            
            # Update p[i, i] to ensure probabilities sum to 1
            p[i, i] = 1 - np.sum(p[i, :i])
            if p[i, i] < 0:
                p[i, i] = 0  # Ensure p[i, i] is non-negative

        # Update Pb matrix based on Matlab code (Pb calculation)
        Pb = np.zeros_like(p)
        for i in range(len(df_array)):
            u_i = int(df_array[i, 3])
            for j in range(i):
                u_j = int(df_array[j, 3])
                dist_ij = (df_array[i, 1] - df_array[j, 1])**2 + (df_array[i, 2] - df_array[j, 2])**2
                Pb[i, j] = beta[u_i, u_j] * np.exp(-dist_ij / (2 * eta_0**2)) / (2 * np.pi * eta_0**2 * T * lambda_i[i] +small_tol)
            Pb[i, i] = 1 - np.sum(Pb[i, :i])  # Ensure Pb[i, i] is non-negative
            if Pb[i, i] < 0:
                Pb[i, i] = 0
        
        # M-step
        # Update omega
        sum_pij = np.sum(np.tril(p, k=-1))
        sum_pij_t_diff = np.sum(np.tril(p, k=-1) * (df_array[:, 0][:, None] - df_array[:, 0])[:len(df_array), :len(df_array)])
        K_sum = np.sum(np.tril(K[df_array[:, 3].astype(int)[:, None], df_array[:, 3].astype(int)], k=-1) * (1 - np.exp(-omega * (T - df_array[:, 0])[:, None]))[:len(df_array), :len(df_array)])
        omega_new = sum_pij / (sum_pij_t_diff + K_sum + small_tol)
        
          # Update K and beta
        for u in range(M):
            for v in range(M):
                mask_uv = (df_array[:, 3] == u)[:, None] & (df_array[:, 3] == v)[None, :]
                mask_uv = mask_uv[:len(df_array), :len(df_array)]  # Ensure mask dimensions match p dimensions
                sum_pij = np.sum(p[mask_uv])
                beta[u, v] = np.sum([Pb[i, j] for i in range(len(df_array)) for j in range(len(df_array)) if df_array[i, 3] == u and df_array[j, 3] == v]) / (np.sum(df_array[:, 3] == u) + small_tol)
       

        # Update sigma and eta_0
        sigma_new = np.sqrt(np.sum([p[i, j] * ((df_array[i, 1] - df_array[j, 1])**2 + (df_array[i, 2] - df_array[j, 2])**2) for i in range(len(df_array)) for j in range(len(df_array))]) / (2 * np.sum([p[i, j] for i in range(len(df_array)) for j in range(len(df_array))]) + small_tol)) if np.sum([p[i, j] for i in range(len(df_array)) for j in range(len(df_array))]) > 0 else sigma
        eta_0_new = sigma_new
        
        # Update A and B (K and beta) based on Matlab code
        #for u in range(M):
        #    for v in range(M):
        #        mask_uv = (df_array[:, 3] == u)[:, None] & (df_array[:, 3] == v)[None, :]
        #        sum_p_uv = np.sum(Pb[mask_uv])
        #        K[u, v] = sum_p_uv / (np.sum(Pb) + small_tol) if sum_p_uv > 0 else 0
        #        beta[u, v] = sum_p_uv / (2 * np.pi * eta_0**2 * T + small_tol) if sum_p_uv > 0 else 0

        # Clamp K and beta within bounds
        K = np.clip(K, lb, ub)
        beta = np.clip(beta, lb, ub)
        
        # Calculate Q(Θ) (Complete Log-Likelihood)
        complete_log_likelihood = 0
        for i in range(len(df_array)):
            for j in range(len(df_array)):
                if i != j:
                    dist_ij = (df_array[i, 1] - df_array[j, 1])**2 + (df_array[i, 2] - df_array[j, 2])**2
                    term1 = Pb[i, j] * np.log(beta[int(df_array[i, 3]), int(df_array[j, 3])] / (2 * np.pi * eta_0**2 * T + small_tol) * np.exp(-dist_ij / (2 * eta_0**2)) + small_tol)
                    complete_log_likelihood += term1
            for j in range(i):
                dist_ij = (df_array[i, 1] - df_array[j, 1])**2 + (df_array[i, 2] - df_array[j, 2])**2
                term2 = p[i, j] * (np.log(omega * K[int(df_array[i, 3]), int(df_array[j, 3])] * np.exp(-omega * (df_array[i, 0] - df_array[j, 0])) / (2 * np.pi * sigma**2) * np.exp(-dist_ij / (2 * sigma**2)) + small_tol))
                complete_log_likelihood += term2
        for u in range(M):
            for i in range(len(df_array)):
                if df_array[i, 3] == u:
                    complete_log_likelihood -= K[u, u] * (1 - np.exp(-omega * (T - df_array[i, 0])))
        log_likelihood_values.append(complete_log_likelihood)

        # Convergence check based on complete log-likelihood
        if complete_log_likelihood < prev_log_likelihood:
            print("Warning: Log-likelihood decreased. Potential issue with convergence.")
        prev_log_likelihood = complete_log_likelihood

        k += 1
        
         # Clamp K, beta, omega, sigma, and eta_0 within bounds
        K = np.clip(K, lb, ub)
        beta = np.clip(beta, lb, ub)
        omega = np.clip(omega, lb, ub)
        sigma = np.clip(sigma, lb, ub)
        eta_0 = np.clip(eta_0, lb, ub)
        
        print(f"Iteration {k}: Complete Log-Likelihood = {complete_log_likelihood:.3e}")
        print("Time spent : ", round((time.time() - t_1) / 60, 2), " minutes")
   
    # Plot log-likelihood values to track convergence
    plt.plot(log_likelihood_values)
    plt.xlabel('Iteration')
    plt.ylabel('Complete Log-Likelihood')
    plt.title('Complete Log-Likelihood over Iterations')
    plt.grid()
    plt.show()
    
    return K, beta, sigma, omega, eta_0, p, Pb


# In[101]:


# Initialize parameters
M = len(np.unique(df_cleaned['area']))  # Number of areas
para = np.random.rand(1, 2 * M**2 + 2).flatten()
K = para[:M**2].reshape(M, M)  # Mean number of events triggered between areas
beta = para[M**2:2 * M**2].reshape(M, M)  # Extent to which events contribute to background rate
omega = para[2 * M**2]  # Decay rate for temporal triggering kernel
sigma = para[2 * M**2 + 1]  # Standard deviation for spatial triggering kernel based on observed data
eta_0 = sigma  # Spatial scale for background rate

# Initialize p matrix
p = np.zeros((len(df_cleaned), len(df_cleaned)))
for i in range(len(df_cleaned)):
    p[i, :i+1] = 1 / (i + 1)  # Initialize each row of the lower triangle with 1/(i+1)

# Set tolerance and maximum iterations
epsilon = 1e-3
max_iter = 10
lb, ub = 1e-5, 1e7 # Lower and upper bounds for clamping parameters
small_tol = 1e-10  # Small tolerance to prevent division by zero


#Run the EM algorithm
K, beta, sigma, omega, eta_0, p, Pb = em_algorithm(df_array, K, beta, sigma, omega, eta_0, p)

# Display final parameters
print("Final Parameters:")
print("K:", K)
print("beta:", beta)
print("sigma:", sigma)
print("omega:", omega)
print("eta_0:", eta_0)


# In[ ]:


# Save final parameters to a pickle file 
with open('LA_crime_fitted_parameters.pkl', 'wb') as f:
    pickle.dump({'K': K, 'beta': beta, 'sigma': sigma, 'omega': omega, 'eta_0': eta_0, 'p': p, 'Pb': Pb}, f)
    print("Fitted parameters saved to fitted_parameters.pkl")


# In[ ]:


# Create matrix A with A[i, j] = omega * K[i, j]
A = omega * K
print("Matrix A:", A)

# Calculate vector mu_0 whose i-th entry is the sum of the i-th column of beta matrix divided by T
T = np.max(np.array(df_cleaned[["t"]]))
mu_0 = np.sum(beta, axis=0) / T
print("Vector mu_0:", mu_0)

# Save A and mu_0 to the same pickle file
with open('LA_crime_fitted_parameters.pkl', 'ab') as f:
    pickle.dump({'A': A, 'mu_0': mu_0}, f)
    print("Matrix A and vector mu_0 saved to fitted_parameters.pkl")


# In[ ]:




