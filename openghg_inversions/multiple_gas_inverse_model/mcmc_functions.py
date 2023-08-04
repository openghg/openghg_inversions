#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:25:00 2023

@author: cv18710

Functions to perform a Markov Chain Monte Carlo algorithm as part of an 
atmospheric inverse model. 

This version created to work with the 'multiple-tracers' version of the inverse model,
where a list of tracers can be used.

y_sig_all()
    - Expands the current/proposed y_sig (model error) values out to the size 
      of the Q array, based on the y_sig_index.

print_out_convergence_test()
    - Prints out information to help with confirming that the MCMC chains have converged.

update_x()
    - Proposes a new value for all components of x, using a random number generator. 
    
create_P_inv_logdet()
    - Calculates the inverse and log determinant of the emissions uncertainty matrix.
    
create_Q_inv_logdet()
    - Calculates the inverse and log determinant of the model-measurement uncertainty matrix.
    
normal_prior_ln()
    - Calculates the Gaussian probability distribution for a parameter.
    
truncnormal_prior_ln()
    - Calculates a trucnated Gaussian probability distribution for a parameter.
    
uniform_ln() 
    - Calculates the probability for uniform distribution.

normal_ln_slow()
    - Slow part of probability calculation, separated out to use numba.
    
acceptance_ratio()
    - Accepts or rejects propoposed paramater values based on their relative probability.

update_step_size()
    - Step size optimisation.
    
mcmc()
    - MCMC process for producing posterior scaling factors or posterior parameter values.

"""

from copy import deepcopy
import numpy as np
import time
import numba
import data_functions as data_func
import isotopes as isotopes

def y_sig_all(y_sig_index,y_sig_values,y_shape,pymc_inputs=False):
    """
    Expands the current/proposed y_sig (model error) values out to the size 
    of the Q array, based on the y_sig_index.
    
    y_sig_index is based on the number of obs per site (if using monthly y_sig) or 
    the number of obs per time period per site (if using daily or e.g. 5 daily y_sig).
    
    Inputs:
        y_sig_index(array): 
            Index values used to split array up into different sites or timeframes.
        y_sig_values (array): 
            Values to apply to each index paritioning.
        y_shape (int): 
            Length of arrays to apply partitioning over.
        pymc_inputs (bool):
            If True, converts y_sig_values from a TensorVariable object to something that
                     is readable by this code.
        
    Returns:
        y_sig_all (array): Expanded version of y_sig_values, applied to whole array of shape y_shape.
    """

    y_sig_all = np.zeros(y_shape)
    
    for i,index in enumerate(y_sig_index):
        if i == 0:
            if pymc_inputs == True:
                y_sig_all[:index] = y_sig_values[i].eval()
            else:
                y_sig_all[:index] = y_sig_values[i]
        else:
            if pymc_inputs == True:
                y_sig_all[y_sig_index[i-1]:index] = y_sig_values[i].eval()
            else:
                y_sig_all[y_sig_index[i-1]:index] = y_sig_values[i]
            
    return y_sig_all

def print_out_convergence_test(i,total_iter,trace,var_name=None):
    """
    Finds the mean value of each variable in the MCMC trace,
    for the last 40-20% and 20% of the trace.
    Prints the absolute and percentage difference between the 
    last 40-20% of interations and the last 20% of interations.
    
    Inputs:
        i (int): current iteration
        total_iter (int): total number of interations
        trace (array): mcmc trace for single variable
        var_name (str): descriptive name for variable
        
    Returns:
        Nothing, only prints to the command line/output file.
    """
    
    np.set_printoptions(precision=4,suppress=True)
    
    min_iter = int(i-(total_iter* 0.4))
    max_iter = int(i-(total_iter* 0.2))
    
    trace1 = trace[min_iter:max_iter,:]
    trace2 = trace[max_iter:i,:]

    past_mean = np.nanmean(trace1,axis=0)
    current_mean = np.nanmean(trace2,axis=0)
    
    diff = np.round(past_mean - current_mean,3)
    perc_diff = np.round(diff/current_mean * 100,3)
    
    print(f'\n*** {var_name} diff between iterations {min_iter} to {max_iter} = {np.round(diff,4)} ***')
    print(f'*** {var_name} perc diff betweeen iterations {max_iter} to {i} = {np.round(perc_diff,4)} ***\n ')
    
def update_x(x,step_size):
    """
    Proposes a new value for all components of x, using a random number generator. 
    
    Inputs:
        x (array): Parameter values.
        step_size (array): Step size to inform random walk.
        
    Returns:
        x_new (array): Current parameter values + a perturbation.
    """
    
    #dx = np.zeros(x.shape)
    x_new = np.zeros(x.shape)
        
    for i in range(x.shape[0]):
        
        if step_size[i] != 0.:
        
            #dx[i] = np.random.normal(loc=0,scale=step_size[i],size=1)
            dx = np.random.normal(loc=0,scale=step_size[i],size=1)

            x_new[i] = x[i] + dx
            
        else:
            
            x_new[i] = x[i]

    return x_new

def create_P_inv_logdet(P):
    """
    Calculates the inverse of the emissions uncertainty matrix.
    Only for diagonal matrices.
    """          
                     
    P_diagonal = P.diagonal()
    
    P_logdet = np.sum(np.log(P_diagonal)) # only correct for diagonal matrices
    #P_logdet = np.linalg.slogdet(P)[1] # use for off-diagonal matrices
    
    P_inv = np.zeros(P.shape)
    
    np.fill_diagonal(P_inv,P_diagonal**(-1))
    
    return P_inv,P_logdet

def create_Q_inv_logdet(Q,Q_off_diagonal=False):
    """
    Calculates the inverse of the model-measurement uncertainty matrix.
    Options to include off-diagonal terms for the covariance between y1 and y2
    or for time-correlated uncertainties.
    I need to create a faster version for use with Q with off-diagonal terms.
    """
    
    if Q_off_diagonal == True:
        
        #start_time_ld = time.time()
        
        Q_logdet = np.linalg.slogdet(Q)[1]
        
        #L = np.linalg.cholesky(Q)
        #Q_logdet = np.sum(2*np.log(np.diagonal(L)))
        
        #end_time_ld = time.time() - start_time_ld
        #print(f'\nLogdet = {end_time_ld}')
        
        #start_time_inv = time.time()
        
        Q_inv = np.linalg.inv(Q)
        
        #I = np.identity(Q.shape[0])
        #Q_inv = np.linalg.solve(Q,I)
        
        #L = np.linalg.cholesky(Q)
        #L_inv = np.linalg.inv(L)
        #Q_inv = np.dot(np.transpose(L_inv),L_inv)
        
        #end_time_inv = time.time() - start_time_inv
        #print(f'\nInv = {end_time_inv}')
        
        #Q_inv[np.where(Q != 0.)] = Q[np.where(Q != 0.)]**(-1)
        
    else:
        
        Q_diagonal = Q.diagonal().astype(float)
        
        #start_time_ld = time.time()

        Q_logdet = np.sum(np.log(Q_diagonal)) #old method for diagonal matrices
        
        #end_time_ld = time.time() - start_time_ld
        #print(f'\nLogdet = {end_time_ld}')
        
        #start_time_inv = time.time()
        
        Q_inv = np.zeros(Q.shape)
        
        np.fill_diagonal(Q_inv,Q_diagonal**(-1))
        
        #end_time_inv = time.time() - start_time_inv
        #print(f'\nInv = {end_time_inv}')
        
        #Q_inv[indices_j==indices_i+y1_shape] = Q[indices_j==indices_i+y1_shape]**(-1)
        #Q_inv[indices_i==indices_j+y1_shape] = Q[indices_i==indices_j+y1_shape]**(-1)
    
    return Q_inv,Q_logdet

def normal_prior_ln(x,x_prior_mean,P_inv,P_logdet):
    """
    Calculates a Gaussian probability distribution of x.
    Logged version, to reduce use of very large numbers.
    Only works with a diagonal covariance matrix. 
    
    Inputs:
        x (array): Parameter values.
        x_prior_mean (array): Parameter a priori means.
        P_inv (array): Inverse of the apriori uncertainty matrix.
        P_logdet (float): Log determinant of the uncertainty matrix.
    
    Returns:
        prob (array): Gaussian multi-variate PDF
    
    """
    
    x_diff = x - x_prior_mean            #difference between the current x and the prior mean
    
    if P_inv.shape[0] > 1:
        
        prob = normal_ln_slow(P_logdet,P_inv,x_diff) 
        
    else:
        # workaround for single value matrics (e.g. when using spatialR == False)
        prob = (0.5*P_logdet) - (0.5 * (np.transpose(x_diff) * P_inv * x_diff))
    
    return prob

def truncnormal_prior_ln(x,x_prior_mean,x_range,P_inv,P_logdet):
    """
    Calculates a trucnated Gaussian probability distribution of x.
    Logged version, to reduce use of very large numbers.
    Only works with a diagonal covariance matrix. 
    
    Args:
        x (array):  Parameter values.
        x_prior_mean (array): Parameter a priori means.
        x_range (array): Min and max values for truncation.
        P_inv (array): Inverse of the apriori uncertainty matrix.
        P_logdet (float): Log determinant of the uncertainty matrix.
    
    Returns:
        prob (array): Truncated Gaussian multi-variate PDF
    """
    
    for n in range(x.shape[0]):
        
        if x[n] < x_range[0] or x[n] > x_range[1]:
            
            return - np.inf
    
    x_diff = x - x_prior_mean            #difference between the current x and the prior mean

    if P_inv.shape[0] > 1:
        
        prob = normal_ln_slow(P_logdet,P_inv,x_diff) 
        
    else:
        # workaround for single value matrics (e.g. when using spatialR == False)
        prob = (0.5*P_logdet) - (0.5 * (np.transpose(x_diff) * P_inv * x_diff))
    
    return prob

def uniform_ln(x,uniform_range):
    """
    Calculates the probability of a uniform distribution, logged.
    
    Inputs:
        x (array): Parameter scaling factors.
        uniform_range (array): Min and max values.
        
    Returns: 
        prob (array): Uniform probability.
    """
    
    for n in range(x.shape[0]):
        
        if x[n] < uniform_range[0] or x[n] > uniform_range[1]:

            return - np.inf
            
    prob = - np.log(uniform_range[1]-uniform_range[0])
            
    return prob

@numba.jit(nopython=True,cache=True)
def normal_ln_slow(uncert_logdet,uncert_inv,v_diff):
    """
    Slow part of MCMC calculations, when the actual probability is calculated.
    This has to use np.dot, instead of np.linalg.multi_dot, as only np.dot is supported by numba.
    Inputs:
        uncert_logdet (float): Log determinant of the uncertainty matrix.
        uncert_inv (array): Inverse of the uncertainty matrix.
        v_diff (array): Difference between the current and prior values of the parameter.
    """
    
    prob = - (0.5*uncert_logdet) - (0.5 * (np.dot(np.dot(np.transpose(v_diff),uncert_inv),v_diff)))
    return prob
    
def acceptance_ratio(prob_proposed,prob_current,accept_count,reject_count,
                         x_proposed,x_current):
    """ 
    Accepts proposed x if they are more likely than current x.
    Also randomly accepts some failures.
    Recorded accepts and rejects. 
    Logged version.
    
    Inputs:
        prob_proposed (array): New probability to test.
        prob_current (array): Probability of previous set of values.
        accept_count (int): Total number of accepts.
        reject_count (int): Total number of rejects.
        x_proposed (array): New values to test.
        x_current (array): Previous set of values.
        
    Returns:
        x_trace (array): Set of x values to carry forwards.
        prob_trace (array): Set of probabilities to store.
        accept_count (int): Total number of accepts.
        reject_count  (int): Total number of rejects.
    
    """
    
    a = prob_proposed - prob_current                    #for use with logged forms of equations
    
    u = (np.random.uniform(0,1))
    
    #if np.abs(a) < 1e-05:
    #    u = (np.random.uniform(0,1))            #acceptance constraint
    #else:
    #    u = (np.random.uniform(0.999999,1))            #acceptance constraint
    
    if np.log(u) < a:
    
    #if 0. < a:
        
        if np.isfinite(prob_proposed) == True:         #always accepts the proposed values if the new probability is lower
        
            prob_trace = prob_proposed
            x_trace = x_proposed
                    
            accept_count = accept_count + 1
                
        else:

            prob_trace = prob_current
            x_trace = x_current
            
            reject_count = reject_count + 1
    
    else:                                   #accepts some failures randomly, based on the acceptance constraint
        
        prob_trace = prob_current
        x_trace = x_current
        
        reject_count = reject_count + 1

    return x_trace,prob_trace,accept_count,reject_count
    
def update_step_size(current_ss,const_1,const_2,accepts,iter_num,target_accept_ratio):
    """
    Andreiu & Thomas 2008, Algorithm 4, adapted by Luke Western.
    
    Inputs:
        current_ss (array): Current step sizes.
        const_1 (float): Tuned constant 1.
        const_2 (float): Tuned constant 2.
        accepts (int): Total number of accepts so far.
        iter_num (int): Current number of interations.
        target_accept_ratio (float): Target proportion of accepts to rejects.
        
    Returns:
        new_ss (array): Updated step sizes.
    
    """
    
    change = (const_1) * (iter_num**(-const_2)) * ((accepts/iter_num) - target_accept_ratio)
    
    new_ss = np.exp(np.log(current_ss) + change)
    
    return new_ss

def mcmc(mcmc_type,species,species_type,sectors_dict,total_iter,sites,
         x_pdf,xem_range,xem_mu_all,Pem,step_size_xem,xall_mu_all,Hall,
         y,y_times,y_err,Q,
         H,Hbc=None,
         xbc_pdf=None,xbc_range=None,xbc_mu_all=None,Pbc=None,step_size_xbc=None,
         R_pdf=None,R_min=None,R_max=None,R_mu_dict=None,R_mu_all=None,R_mu_allsectors=None,PR=None,step_size_R=None,
         Rbc_pdf=None,Rbc_min=None,Rbc_max=None,Rbc_mu_all=None,PRbc=None,step_size_Rbc=None,
         y_sig_pdf=None,y_sig_range=None,y_sig_mu_dict=None,y_sig_mu_all=None,step_size_y_sig=None,y_sig_index=None,
         Q_sig=None,sites_array_all=None,
         spatialR=True,y_cov=False,
         ac_timescale_hours=None):
    """
    MCMC process for producing posterior scaling factors for emissions, emission ratios,
    source signatures and model error hyper-parameters.
    Based on a standard Metropolis-Hastings MCMC method, with options for different prior PDFs
    and step size optimisation.
    For information on inputs, see the .ini file and create_mcmc_inputs in data_functions().
    """
    
    delta_indices = [i for i,e in enumerate(species_type) if e == 'delta_value']
    delta_species = [species[i] for i,e in enumerate(species_type) if e == 'delta_value']
    mf_species = [species[i] for i,e in enumerate(species_type) if e == 'mf']
    
    if y_cov == False and ac_timescale_hours is None:
        Q_off_diagonal = False
    else:
        Q_off_diagonal = True
    
    #---------------------------------------------------------------------------------
    # Emissions inputs ---------------------------------------------------------------
    
    xem_trace = np.zeros((total_iter,xem_mu_all.shape[0]))
    prob_trace_em = np.zeros((total_iter))
    
    accept_count_em, reject_count_em = 0,0
    
    xem_mu = deepcopy(xem_mu_all)
    xem_current = deepcopy(xem_mu_all)
    
    Pem_inv,Pem_logdet = create_P_inv_logdet(Pem)
    
    H_current = deepcopy(H)
    H_proposed = deepcopy(H)
    
    #---------------------------------------------------------------------------------
    # Boundary conditions inputs -----------------------------------------------------
    
    xall_current = deepcopy(xall_mu_all)
    xall_proposed = deepcopy(xall_mu_all)
    Hall_current = deepcopy(Hall)
    Hall_proposed = deepcopy(Hall)
    
    if xbc_pdf is not None:
        
        Hbc_current = deepcopy(Hbc)
        
        Pbc_inv,Pbc_logdet = {},{}
        
        xbc_trace = {sp:np.zeros((total_iter,xbc_mu_all[sp].shape[0])) for sp in mf_species}
        prob_trace_bc = {sp:np.zeros((total_iter,xbc_mu_all[sp].shape[0])) for sp in mf_species}
        
        accept_count_bc = {sp:0 for sp in mf_species}
        reject_count_bc = {sp:0 for sp in mf_species}
        
        xbc_mu = deepcopy(xbc_mu_all)
        xbc_current = deepcopy(xbc_mu_all)
        
        for s_name in mf_species:
            
            Pbc_inv[s_name],Pbc_logdet[s_name] = create_P_inv_logdet(Pbc[s_name])
        
        use_bc = True
        xbc_proposed = deepcopy(xbc_current)

    else:
        
        xbc_current,Hbc_current = {},{}
        
        for s_name in mf_species:
            xbc_current[s_name] = None
            Hbc_current[s_name] = None
        
        use_bc = False
        xbc_proposed = deepcopy(xbc_current)
 
    #---------------------------------------------------------------------------------
    # Emission ratio/source signature inputs -----------------------------------------
    
    if len(species) > 1:
        
        R_current = deepcopy(R_mu_dict)
        R_proposed = deepcopy(R_mu_dict)
        
        if use_bc == True and Rbc_pdf is not None:
            Rbc_current = deepcopy(Rbc_mu_all)
        else:
            Rbc_current = None
            
        R_current_allsectors = deepcopy(R_mu_allsectors)
            
        R_trace_allsectors = {sp:np.repeat(np.expand_dims(R_current_allsectors[sp],0),total_iter,axis=0) for sp in species[1:]}
        
        if step_size_R is not None:
            
            R_trace = {sp:{se:np.zeros((total_iter,R_current[sp][se].shape[0])) for se in sectors_dict[sp]} for sp in species[1:]}
            prob_trace_R = {sp:{se:np.zeros((total_iter,R_current[sp][se].shape[0])) for se in sectors_dict[sp]} for sp in species[1:]}
            
            accept_count_R = {sp:{se:0 for se in sectors_dict[sp]} for sp in species[1:]}
            reject_count_R = {sp:{se:0 for se in sectors_dict[sp]} for sp in species[1:]}
                    
            R_proposed_allsectors = deepcopy(R_current_allsectors)
                    
            PR_inv = {sp:{se:np.array([]) for se in sectors_dict[sp]} for sp in species[1:]}
            PR_logdet = {sp:{se:np.array([]) for se in sectors_dict[sp]} for sp in species[1:]}
        
            for s,s_name in enumerate(species[1:]):
                for sector in sectors_dict[s_name]:
                    if 'uniform' not in R_pdf[s]:
                        PR_inv[s_name][sector],PR_logdet[s_name][sector] = create_P_inv_logdet(PR[s_name][sector])
                        
        else:
            R_trace = {sp:{se:np.ones((total_iter,R_current[sp][se].shape[0])) * R_current[sp][se][0] for se in sectors_dict[sp]} for sp in species[1:]}
                 
        if step_size_Rbc is not None:
            
            Rbc_trace = {sp:np.zeros((total_iter,Rbc_mu_all[sp].shape[0])) for sp in delta_species}
            prob_trace_Rbc = {sp:np.zeros((total_iter,Rbc_mu_all[sp].shape[0])) for sp in delta_species}
            
            accept_count_Rbc = {sp:0 for sp in delta_species}
            reject_count_Rbc = {sp:0 for sp in delta_species}
            
            Rbc_proposed = deepcopy(Rbc_mu_all)
            
            PRbc_inv,PRbc_logdet = {},{}
        
            for s in delta_indices:
                if 'uniform' not in Rbc_pdf[s-1]:
                    PRbc_inv[species[s]],PRbc_logdet[species[s]] = create_P_inv_logdet(PRbc[species[s]])
                    
        elif use_bc == True:
            Rbc_trace = {sp:np.ones((total_iter,Rbc_mu_all[sp].shape[0])) * Rbc_mu_all[sp][0] for sp in delta_species}
        else:
            Rbc_trace = None
            
    else:
        R_trace,Rbc_trace,R_trace_allsectors = None,None,None
                
    #---------------------------------------------------------------------------------
    # Obs and model-measurement error inputs -----------------------------------------
    
    y_mod_trace = {sp:np.zeros((total_iter,y[sp].shape[0])) for sp in species}
    
    Q_current = deepcopy(Q)
    
    Q_inv,Q_logdet = {},{}
    
    for s_name in species:

        Q_inv[s_name],Q_logdet[s_name] = create_Q_inv_logdet(Q_current[s_name],Q_off_diagonal=Q_off_diagonal)
    
    y_mod_current = {}
    
    for s,s_name in enumerate(species):
        if species_type[s] == 'mf':
            y_mod_current[s_name] = np.matmul(Hall_current[s_name],xall_current[s_name])
        elif species_type[s] == 'delta_value':
            y_mod0,y_mod_current[s_name] = isotopes.modelled_ch4_delta_obs(s_name,R_current_allsectors,xem_current,
                                                                    H_current[species[0]],sites[0],sites[s],
                                                                    sites_array_all[species[0]],sites_array_all[s_name],
                                                                    y_times[species[0]],y_times[s_name],delta_bc_sample=Rbc_current,
                                                                    xbc_sample=xbc_current[species[0]],Hbc_sample=Hbc_current[species[0]],
                                                                    return_bc_separately=False)
        
    y_mod_proposed = deepcopy(y_mod_current)
                    
    if step_size_y_sig is not None:
        
        y_sig_mu_all = deepcopy(y_sig_mu_all)
        y_sig_current = deepcopy(y_sig_mu_dict)
        y_sig_current_all = deepcopy(y_sig_mu_all)
        
        y_sig_proposed = deepcopy(y_sig_mu_dict)
        y_sig_proposed_all = deepcopy(y_sig_mu_all)
        
        y_sig_trace = {sp:np.zeros((total_iter,y_sig_current[sp].shape[0])) for sp in species}
        prob_trace_y_sig = {sp:np.zeros(total_iter) for sp in species}
        
        accept_count_y_sig = {sp:0 for sp in species}
        reject_count_y_sig = {sp:0 for sp in species}

        Q_sig_current = deepcopy(Q_sig)
        
        Q_sig_inv,Q_sig_logdet = {},{}
        
        for s_name in species:
            
            Q_sig_inv[s_name],Q_sig_logdet[s_name] = create_Q_inv_logdet(Q_sig_current[s_name],Q_off_diagonal=Q_off_diagonal)

    #---------------------------------------------------------------------------------
    # MCMC loops ---------------------------------------------------------------------
    
    start_time = time.time() 
    
    for i in range(total_iter):
                
        if i == 0:
            print('\nFirst iteration started')
                        
        if i == total_iter//2:
            print(f'{i} iterations completed')
            
        #---------------------------------------------------------------------------------
        # x ------------------------------------------------------------------------------
        
        prob_current = 0        
        
        for s_name in species:
            prob_current += normal_ln_slow(Q_logdet[s_name],Q_inv[s_name],y[s_name]-y_mod_current[s_name])
            
        if 'normal' in x_pdf:
            prob_current = prob_current + normal_prior_ln(x=xem_current,x_prior_mean=xem_mu,
                                                            P_inv=Pem_inv,P_logdet=Pem_logdet)
        elif 'truncnorm' in x_pdf:
            prob_current = prob_current + truncnormal_prior_ln(x=xem_current,x_prior_mean=xem_mu,
                                                            x_range=xem_range,P_inv=Pem_inv,P_logdet=Pem_logdet)
        xem_proposed = update_x(x=xem_current,step_size=step_size_xem)
        
        for s_name in mf_species:
            if use_bc == True and xbc_mu[s_name] is not None:
                xall_proposed[s_name] = np.hstack((xem_proposed,xbc_current[s_name]))
            else:
                xall_proposed[s_name] = xem_proposed.copy()
        
        for s,s_name in enumerate(species):
            if species_type[s] == 'mf':
                y_mod_proposed[s_name] = np.matmul(Hall_current[s_name],xall_proposed[s_name])
                
            elif species_type[s] == 'delta_value':
                y_mod0,\
                y_mod_proposed[s_name] = isotopes.modelled_ch4_delta_obs(s_name,R_current_allsectors,xem_proposed,
                                                                         H_current[species[0]],sites[0],sites[s],
                                                                         sites_array_all[species[0]],sites_array_all[s_name],
                                                                        y_times[species[0]],y_times[s_name],
                                                                        delta_bc_sample=Rbc_current,
                                                                        xbc_sample=xbc_current[species[0]],
                                                                        Hbc_sample=Hbc_current[species[0]],
                                                                        return_bc_separately=False)
                
        prob_proposed = 0
        
        for s_name in species:
            
            prob_proposed += normal_ln_slow(Q_logdet[s_name],Q_inv[s_name],y[s_name]-y_mod_proposed[s_name])
            
        if 'normal' in x_pdf:
            prob_proposed += normal_prior_ln(x=xem_proposed,x_prior_mean=xem_mu,
                                                            P_inv=Pem_inv,P_logdet=Pem_logdet)
        elif 'truncnorm' in x_pdf:
            prob_proposed += truncnormal_prior_ln(x=xem_proposed,x_prior_mean=xem_mu,
                                                  x_range=xem_range,P_inv=Pem_inv,P_logdet=Pem_logdet)
            
        xem_trace0,prob_trace_em0,accept_count_em,\
        reject_count_em = acceptance_ratio(prob_proposed=prob_proposed,prob_current=prob_current,
                                           accept_count=accept_count_em,reject_count=reject_count_em,
                                           x_proposed=xem_proposed,x_current=xem_current)
        
        prob_trace_em[i] = prob_trace_em0
        xem_trace[i,:] = xem_trace0

        xem_current = xem_trace0
        
        for s_name in mf_species:
            if use_bc == True and xbc_mu[s_name] is not None:
                xall_current[s_name] = np.hstack((xem_current,xbc_current[s_name]))
            else:
                xall_current[s_name] = xem_current.copy()
            
        for s,s_name in enumerate(species):
            if species_type[s] == 'mf':
                y_mod_current[s_name] = np.matmul(Hall_current[s_name],xall_current[s_name])
            elif species_type[s] == 'delta_value':
                y_mod0,\
                y_mod_current[s_name] = isotopes.modelled_ch4_delta_obs(s_name,R_current_allsectors,xem_current,
                                                                    H_current[species[0]],sites[0],sites[s],
                                                                    sites_array_all[species[0]],sites_array_all[s_name],
                                                                    y_times[species[0]],
                                                                    y_times[s_name],delta_bc_sample=Rbc_current,
                                                                    xbc_sample=xbc_current[species[0]],
                                                                    Hbc_sample=Hbc_current[species[0]],
                                                                    return_bc_separately=False)

        if i > 0:
            step_size_xem = update_step_size(step_size_xem,1.,0.8,accept_count_em,i,0.30)
                
        if i > total_iter*0.4 and i%int(total_iter/4) == 0:
            print_out_convergence_test(i,total_iter,xem_trace,var_name='x')
        
        #---------------------------------------------------------------------------------
        # xbc ----------------------------------------------------------------------------

        if step_size_xbc is not None:
            
            for s_name in mf_species:
            
                prob_current = 0
                
                #for s in mf_species:
                for s in species:
                        
                    prob_current += normal_ln_slow(Q_logdet[s],Q_inv[s],y[s]-y_mod_current[s])
                
                if 'normal' in xbc_pdf:
                    prob_current += normal_prior_ln(x=xbc_current[s_name],x_prior_mean=xbc_mu[s_name],
                                                                P_inv=Pbc_inv[s_name],P_logdet=Pbc_logdet[s_name])
                elif 'truncnorm' in xbc_pdf:
                    prob_current += truncnormal_prior_ln(x=xbc_current[s_name],x_prior_mean=xbc_mu[s_name],
                                                                        x_range=xbc_range,P_inv=Pbc_inv[s_name],
                                                                        P_logdet=Pbc_logdet[s_name])
                    
                xbc_proposed[s_name] = update_x(x=xbc_current[s_name],step_size=step_size_xbc[s_name])
            
                xall_proposed[s_name] = np.hstack((xem_current,xbc_proposed[s_name]))
                
                #y_mod_proposed[s_name] = np.matmul(Hall_current[s_name],xall_proposed[s_name])
                
                for a,sp in enumerate(species):
                    if species_type[a] == 'mf':
                        y_mod_proposed[sp] = np.matmul(Hall_current[sp],xall_proposed[sp])
                    elif species_type[a] == 'delta_value':
                        y_mod0,\
                        y_mod_proposed[sp] = isotopes.modelled_ch4_delta_obs(sp,R_current_allsectors,xem_current,
                                                                             H_current[species[0]],sites[0],sites[a],
                                                                    sites_array_all[species[0]],sites_array_all[sp],
                                                                                y_times[species[0]],y_times[sp],
                                                                                delta_bc_sample=Rbc_current,
                                                                                xbc_sample=xbc_proposed[species[0]],
                                                                                Hbc_sample=Hbc_current[species[0]],
                                                                                return_bc_separately=False)
                
                prob_proposed = 0
                
                #for s in mf_species:
                for s in species:
                    prob_proposed += normal_ln_slow(Q_logdet[s],Q_inv[s],y[s]-y_mod_proposed[s])
                    
                if 'normal' in xbc_pdf:
                    prob_proposed += normal_prior_ln(x=xbc_proposed[s_name],x_prior_mean=xbc_mu[s_name],
                                                    P_inv=Pbc_inv[s_name],P_logdet=Pbc_logdet[s_name])
                    
                elif 'truncnorm' in xbc_pdf:
                    prob_proposed += truncnormal_prior_ln(x=xbc_proposed[s_name],x_prior_mean=xbc_mu[s_name],
                                                        x_range=xbc_range,P_inv=Pbc_inv[s_name],
                                                        P_logdet=Pbc_logdet[s_name])
                    
                xbc_trace0,prob_trace_bc0,accept_count_bc[s_name],\
                reject_count_bc[s_name] = acceptance_ratio(prob_proposed=prob_proposed,prob_current=prob_current,
                                                    accept_count=accept_count_bc[s_name],
                                                    reject_count=reject_count_bc[s_name],x_proposed=xbc_proposed[s_name],
                                                    x_current=xbc_current[s_name]) 
                    
                prob_trace_bc[s_name][i] = prob_trace_bc0
                xbc_trace[s_name][i,:] = xbc_trace0
                
                xbc_current[s_name] = xbc_trace0
            
                xall_current[s_name] = np.hstack((xem_current,xbc_current[s_name]))
                
                for a,sp in enumerate(species):
                    if species_type[a] == 'mf':
                        y_mod_current[sp] = np.matmul(Hall_current[sp],xall_current[sp])
                    elif species_type[a] == 'delta_value':
                        y_mod0,\
                        y_mod_current[sp] = isotopes.modelled_ch4_delta_obs(sp,R_current_allsectors,xem_current,
                                                                            H_current[species[0]],sites[0],sites[a],
                                                                    sites_array_all[species[0]],sites_array_all[sp],
                                                                                y_times[species[0]],y_times[sp],
                                                                                delta_bc_sample=Rbc_current,
                                                                                xbc_sample=xbc_current[species[0]],
                                                                                Hbc_sample=Hbc_current[species[0]],
                                                                                return_bc_separately=False)
                
                if i > 0:
                    step_size_xbc[s_name] = update_step_size(step_size_xbc[s_name],1.,0.8,accept_count_bc[s_name],i,0.30)
                
                if i > total_iter*0.4 and i%int(total_iter/4) == 0:
                    print_out_convergence_test(i,total_iter,xbc_trace[s_name],var_name=f'xbc_{s_name}')

        #---------------------------------------------------------------------------------
        # R ------------------------------------------------------------------------------
                
        if step_size_R is not None:
            
            for s,s_name in enumerate(species[1:]):
                for a,sector_name in enumerate(sectors_dict[s_name]):
                
                    if sector_name is not None:
                    
                        prob_current = 0

                        #include mod CH4 as this changes when new delta values are sampled
                        if species_type[s+1] == 'delta_value':
                            
                            #prob_current += normal_ln_slow(Q_logdet[species[0]],Q_inv[species[0]],y[species[0]]-y_mod_current_species0_test)
                            prob_current += normal_ln_slow(Q_logdet[species[0]],Q_inv[species[0]],
                                                           y[species[0]]-y_mod_current[species[0]])
                            
                        prob_current += normal_ln_slow(Q_logdet[s_name],Q_inv[s_name],y[s_name]-y_mod_current[s_name])
                        
                        if 'uniform' in R_pdf[s]:
                            prob_current += uniform_ln(x=R_current[s_name][sector_name],
                                                                    uniform_range=[R_min[s][a],
                                                                                    R_max[s][a]])
                        elif 'normal' in R_pdf[s]:
                            prob_current += normal_prior_ln(x=R_current[s_name][sector_name],
                                                                        x_prior_mean=R_mu_dict[s_name][sector_name],
                                                                P_inv=PR_inv[s_name][sector_name],P_logdet=PR_logdet[s_name][sector_name])
                        elif 'truncnorm' in R_pdf[s]:
                            prob_current += truncnormal_prior_ln(x=R_current[s_name][sector_name],
                                                                            x_prior_mean=R_mu_dict[s_name][sector_name],
                                                                    x_range=[R_min[s][a],R_max[s][a]],P_inv=PR_inv[s_name][sector_name],
                                                                    P_logdet=PR_logdet[s_name][sector_name])
                
                        R_proposed[s_name][sector_name] = update_x(x=R_current[s_name][sector_name],step_size=step_size_R[s_name][sector_name])
                
                        for b,sector_name_loop in enumerate(sectors_dict[s_name]):
                            if sector_name_loop == sector_name:
                                R_update = R_proposed[s_name][sector_name_loop].copy()
                            else:
                                R_update = R_current[s_name][sector_name_loop].copy()
                            if b == 0:
                                R_proposed_allsectors[s_name] = np.array([])
                            if spatialR == True:
                                R_proposed_allsectors[s_name] = np.hstack((R_proposed_allsectors[s_name],R_update))
                            elif spatialR == False:
                                R_proposed_allsectors[s_name] = np.hstack((R_proposed_allsectors[s_name],
                                                                            np.ones(R_mu_all[s_name][sector_name_loop].shape[0]) * R_update))
            
                        if species_type[s+1] == 'mf':
                            H_proposed[s_name] = H[s_name] * R_proposed_allsectors[s_name]
                            if use_bc == True:
                                Hall_proposed[s_name] = np.hstack((H_proposed[s_name],Hbc_current[s_name]))
                            else:
                                Hall_proposed[s_name] = H_proposed[s_name].copy()

                            y_mod_proposed[s_name] = np.matmul(Hall_proposed[s_name],xall_current[s_name])

                        elif species_type[s+1] == 'delta_value':
                            H_proposed[s_name] = H_current[s_name].copy()
                            y_mod0,\
                            y_mod_proposed[s_name] = isotopes.modelled_ch4_delta_obs(s_name,R_proposed_allsectors,xem_current,
                                                                                     H_proposed[species[0]],sites[0],sites[s+1],
                                                                                            sites_array_all[species[0]],sites_array_all[s_name],
                                                                                            y_times[species[0]],y_times[s_name],
                                                                                            delta_bc_sample=Rbc_current,
                                                                                            xbc_sample=xbc_current[species[0]],
                                                                                            Hbc_sample=Hbc_current[species[0]],
                                                                                            return_bc_separately=False)
                        prob_proposed = 0

                        #include mod CH4 as this changes when new delta values are sampled
                        if species_type[s+1] == 'delta_value':
                            prob_proposed += normal_ln_slow(Q_logdet[species[0]],Q_inv[species[0]],
                                                            y[species[0]]-y_mod_proposed[species[0]])
                                                        
                        prob_proposed += normal_ln_slow(Q_logdet[s_name],Q_inv[s_name],y[s_name]-y_mod_proposed[s_name])
                        
                        if 'uniform' in R_pdf[s]:
                            prob_proposed += uniform_ln(x=R_proposed[s_name][sector_name],
                                                                    uniform_range=[R_min[s][a],
                                                                                    R_max[s][a]])
                        elif 'normal' in R_pdf[s]:
                            prob_proposed += normal_prior_ln(x=R_proposed[s_name][sector_name],
                                                                        x_prior_mean=R_mu_dict[s_name][sector_name],
                                                                P_inv=PR_inv[s_name][sector_name],P_logdet=PR_logdet[s_name][sector_name])
                        elif 'truncnorm' in R_pdf[s]:
                            prob_proposed += truncnormal_prior_ln(x=R_proposed[s_name][sector_name],
                                                                            x_prior_mean=R_mu_dict[s_name][sector_name],
                                                                    x_range=[R_min[s][a],R_max[s][a]],P_inv=PR_inv[s_name][sector_name],
                                                                    P_logdet=PR_logdet[s_name][sector_name])
                            
                        R_trace0,prob_trace_R0,accept_count_R[s_name][sector_name],\
                        reject_count_R[s_name][sector_name] = acceptance_ratio(prob_proposed=prob_proposed,prob_current=prob_current,
                                                        accept_count=accept_count_R[s_name][sector_name],
                                                            reject_count=reject_count_R[s_name][sector_name],
                                                            x_proposed=R_proposed[s_name][sector_name],
                                                            x_current=R_current[s_name][sector_name])
                                
                        prob_trace_R[s_name][sector_name][i] = prob_trace_R0    
                        R_trace[s_name][sector_name][i,:] = R_trace0
                    
                        R_current[s_name][sector_name] = R_trace0
                        
                        for b,sector_name_loop in enumerate(sectors_dict[s_name]):
                            if b == 0:
                                R_current_allsectors[s_name] = np.array([])
                            if spatialR == True:
                                R_current_allsectors[s_name] = np.hstack((R_current_allsectors[s_name],R_current[s_name][sector_name_loop]))
                            elif spatialR == False:
                                R_current_allsectors[s_name] = np.hstack((R_current_allsectors[s_name],
                                                                        np.ones(R_mu_all[s_name][sector_name_loop].shape[0]) * R_current[s_name][sector_name_loop]))
        
                        R_trace_allsectors[s_name][i,:] = R_current_allsectors[s_name]
                        
                        if species_type[s+1] == 'mf':
                            H_current[s_name] = H[s_name] * R_current_allsectors[s_name]
                            if use_bc == True:
                                Hall_current[s_name] = np.hstack((H_current[s_name],Hbc_current[s_name]))
                            else:
                                Hall_current[s_name] = H_current[s_name].copy()
                            y_mod_current[s_name] = np.matmul(Hall_current[s_name],xall_current[s_name])
                        
                        elif species_type[s+1] == 'delta_value':
                            y_mod0,\
                            y_mod_current[s_name] = isotopes.modelled_ch4_delta_obs(s_name,R_current_allsectors,xem_current,
                                                                                    H_current[species[0]],sites[0],sites[s+1],
                                                                                    sites_array_all[species[0]],sites_array_all[s_name],
                                                                                    y_times[species[0]],y_times[s_name],
                                                                                    delta_bc_sample=Rbc_current,
                                                                                    xbc_sample=xbc_current[species[0]],
                                                                                    Hbc_sample=Hbc_current[species[0]],
                                                                                    return_bc_separately=False)
                            
                        if i > 0:
                            step_size_R[s_name][sector_name] = update_step_size(step_size_R[s_name][sector_name],1.,0.8,
                                                                                    accept_count_R[s_name][sector_name],i,0.30)
                                        
                        if i > total_iter*0.4 and i%int(total_iter/4) == 0:
                            print_out_convergence_test(i,total_iter,R_trace[s_name][sector_name],var_name=f'R_{s_name}_{sector_name}')
                                                     
        else:

            for s,s_name in enumerate(species[1:]):
                for a,sector_name in enumerate(sectors_dict[s_name]):
                    R_trace[s_name][sector_name][i,:] = R_current[s_name][sector_name]

        #---------------------------------------------------------------------------------
        # Rbc ----------------------------------------------------------------------------
        
        if step_size_Rbc is not None:
            
            for s in delta_indices:
                
                s_name = species[s]
                
                prob_current = (normal_ln_slow(Q_logdet[species[0]],Q_inv[species[0]],y[species[0]]-y_mod_current[species[0]]) + 
                                normal_ln_slow(Q_logdet[s_name],Q_inv[s_name],y[s_name]-y_mod_current[s_name]))
            
                if 'uniform' in Rbc_pdf[s-1]:
                    prob_current = prob_current + uniform_ln(x=Rbc_current[s_name],
                                                            uniform_range=[Rbc_min[s-1],
                                                                            Rbc_max[s-1]])
                elif 'normal' in Rbc_pdf[s-1]:
                    prob_current = prob_current + normal_prior_ln(x=Rbc_current[s_name],
                                                                x_prior_mean=Rbc_mu_all[s_name],
                                                        P_inv=PRbc_inv[s_name],P_logdet=PRbc_logdet[s_name])
                elif 'truncnorm' in Rbc_pdf[s-1]:
                    prob_current = prob_current + truncnormal_prior_ln(x=Rbc_current[s_name],
                                                                    x_prior_mean=Rbc_mu_all[s_name],
                                                            x_range=[Rbc_min[s-1],Rbc_max[s-1]],P_inv=PRbc_inv[s_name],
                                                            P_logdet=PRbc_logdet[s_name])
                        
                Rbc_proposed[s_name] = update_x(x=Rbc_current[s_name],step_size=step_size_Rbc[s_name])
                
                y_mod0,\
                y_mod_proposed[s_name] = isotopes.modelled_ch4_delta_obs(s_name,R_current_allsectors,xem_current,
                                                                    H_current[species[0]],sites[0],sites[s],
                                                                    sites_array_all[species[0]],sites_array_all[s_name],
                                                                    y_times[species[0]],
                                                                    y_times[s_name],delta_bc_sample=Rbc_proposed,
                                                                    xbc_sample=xbc_current[species[0]],
                                                                    Hbc_sample=Hbc_current[species[0]],
                                                                    return_bc_separately=False)
                
                prob_proposed = (normal_ln_slow(Q_logdet[species[0]],Q_inv[species[0]],y[species[0]]-y_mod_proposed[species[0]]) + 
                                normal_ln_slow(Q_logdet[s_name],Q_inv[s_name],y[s_name]-y_mod_proposed[s_name]))

                if 'uniform' in Rbc_pdf[s-1]:
                    prob_proposed = prob_proposed + uniform_ln(x=Rbc_proposed[s_name],
                                                                uniform_range=[Rbc_min[s-1],
                                                                            Rbc_max[s-1]])
                elif 'normal' in Rbc_pdf[s-1]:
                    prob_proposed = prob_proposed + normal_prior_ln(x=Rbc_proposed[s_name],
                                                                    x_prior_mean=Rbc_mu_all[s_name],
                                                            P_inv=PRbc_inv[s_name],P_logdet=PRbc_logdet[s_name])
                elif 'truncnorm' in Rbc_pdf[s-1]:
                    prob_proposed = prob_proposed + truncnormal_prior_ln(x=Rbc_proposed[s_name],
                                                                        x_prior_mean=Rbc_mu_all[s_name],
                                                                x_range=[Rbc_min[s-1],Rbc_max[s-1]],P_inv=PRbc_inv[s_name],
                                                                P_logdet=PRbc_logdet[s_name])
                
                Rbc_trace0,prob_trace_Rbc0,\
                accept_count_Rbc[s_name],reject_count_Rbc[s_name] = acceptance_ratio(prob_proposed=prob_proposed,prob_current=prob_current,
                                                accept_count=accept_count_Rbc[s_name],
                                                    reject_count=reject_count_Rbc[s_name],x_proposed=Rbc_proposed[s_name],
                                                    x_current=Rbc_current[s_name])
                      
                prob_trace_Rbc[s_name][i] = prob_trace_Rbc0    
                Rbc_trace[s_name][i,:] = Rbc_trace0

                Rbc_current[s_name] = Rbc_trace0
                
                y_mod0,\
                y_mod_current[s_name] = isotopes.modelled_ch4_delta_obs(s_name,R_current_allsectors,xem_current,
                                                                    H_current[species[0]],sites[0],sites[s],
                                                                    sites_array_all[species[0]],sites_array_all[s_name],
                                                                    y_times[species[0]],
                                                                    y_times[s_name],delta_bc_sample=Rbc_current,
                                                                    xbc_sample=xbc_current[species[0]],
                                                                    Hbc_sample=Hbc_current[species[0]],return_bc_separately=False)

                if i > 0:
                    step_size_Rbc[s_name] = update_step_size(step_size_Rbc[s_name],1.,0.8,accept_count_Rbc[s_name],i,0.30)
                            
                if i > total_iter*0.4 and i%int(total_iter/4) == 0:
                    print_out_convergence_test(i,total_iter,Rbc_trace[s_name],var_name=f'Rbc_{s_name}')
                                                                         
        #---------------------------------------------------------------------------------
        # Model error uncertainty (y_sig) ------------------------------------------------
        
        if step_size_y_sig is not None:
            
            for s,s_name in enumerate(species):
                
                prob_current = 0
                
                prob_current += normal_ln_slow(Q_logdet[s_name],Q_inv[s_name],y[s_name]-y_mod_current[s_name])
                
                if 'normal' in y_sig_pdf:
                    prob_current += normal_prior_ln(x=y_sig_current_all[s_name],x_prior_mean=y_sig_mu_all[s_name],
                                                                    P_inv=Q_sig_inv[s_name],P_logdet=Q_sig_logdet[s_name])
            
                elif 'truncnorm' in y_sig_pdf:
                    prob_current += truncnormal_prior_ln(x=y_sig_current_all[s_name],x_prior_mean=y_sig_mu_all[s_name],
                                                                        x_range=y_sig_range[s],
                                                                        P_inv=Q_sig_inv[s_name],P_logdet=Q_sig_logdet[s_name])              
            
                elif 'uniform' in y_sig_pdf:
                    prob_current += uniform_ln(x=y_sig_current[s_name],uniform_range=y_sig_range[s])
                    
                y_sig_proposed[s_name] = update_x(x=y_sig_current[s_name],step_size=step_size_y_sig[s_name])
                
                y_sig_proposed_all[s_name] = y_sig_all(y_sig_index[s_name],y_sig_proposed[s_name],y[s_name].shape[0])
                
                Q_proposed,Q_sig0 = data_func.create_model_measurement_uncertainty_matrix(mcmc_type,y[s_name],y_times[s_name],
                                                                                                  y_err[s_name],y_sig_proposed_all[s_name],
                                                                 y_cov,ac_timescale_hours,y_sig_pdf=None,
                                                                 sites_all=sites_array_all[s_name])
                
                Q_inv[s_name],Q_logdet[s_name] = create_Q_inv_logdet(Q_proposed,Q_off_diagonal=Q_off_diagonal)
                #Q_sig_inv[s_name],Q_sig_logdet[s_name] = create_Q_inv_logdet(Q_sig_proposed,Q_off_diagonal=Q_off_diagonal)
                
                prob_proposed = 0

                prob_proposed += normal_ln_slow(Q_logdet[s_name],Q_inv[s_name],y[s_name]-y_mod_current[s_name])
                
                if 'normal' in y_sig_pdf:
                    prob_proposed += normal_prior_ln(x=y_sig_proposed_all[s_name],x_prior_mean=y_sig_mu_all[s_name],
                                                                    P_inv=Q_sig_inv[s_name],P_logdet=Q_sig_logdet[s_name])
            
                elif 'truncnorm' in y_sig_pdf:
                    prob_proposed += truncnormal_prior_ln(x=y_sig_proposed_all[s_name],x_prior_mean=y_sig_mu_all[s_name],
                                                                        x_range=y_sig_range[s],
                                                                        P_inv=Q_sig_inv[s_name],P_logdet=Q_sig_logdet[s_name])              
            
                elif 'uniform' in y_sig_pdf:
                    prob_proposed += uniform_ln(x=y_sig_proposed[s_name],uniform_range=y_sig_range[s])
                
                y_sig_trace0,prob_trace_y_sig0,accept_count_y_sig[s_name],\
                reject_count_y_sig[s_name] = acceptance_ratio(prob_proposed=prob_proposed,prob_current=prob_current,
                                                accept_count=accept_count_y_sig[s_name],
                                                reject_count=reject_count_y_sig[s_name],x_proposed=y_sig_proposed[s_name],
                                                x_current=y_sig_current[s_name])
                
                y_sig_trace[s_name][i] = y_sig_trace0
                prob_trace_y_sig[s_name][i] = prob_trace_y_sig0
                
                y_sig_current[s_name] = y_sig_trace0
                
                y_sig_current_all[s_name] = y_sig_all(y_sig_index[s_name],y_sig_current[s_name],y[s_name].shape[0])
                
                Q_current[s_name],Q_sig_current[s_name] = data_func.create_model_measurement_uncertainty_matrix(mcmc_type,y[s_name],y_times[s_name],
                                                                                                                y_err[s_name],y_sig_current_all[s_name],
                                                                 y_cov,ac_timescale_hours,y_sig_pdf=None,
                                                                 sites_all=sites_array_all[s_name])
                
                Q_inv[s_name],Q_logdet[s_name] = create_Q_inv_logdet(Q_current[s_name],Q_off_diagonal=Q_off_diagonal)
                #Q_sig_inv[s_name],Q_sig_logdet[s_name] = create_Q_inv_logdet(Q_sig_current[s_name],Q_off_diagonal=Q_off_diagonal)
                
                if i > 0:
                    step_size_y_sig[s_name] = update_step_size(step_size_y_sig[s_name],1.,0.8,accept_count_y_sig[s_name],i,0.30)
                    
                if i > total_iter*0.4 and i%int(total_iter/4) == 0:
                    print_out_convergence_test(i,total_iter,y_sig_trace[s_name],var_name=f'y_sig_{s_name}')

        #---------------------------------------------------------------------------------
        # Save y_mod trace ---------------------------------------------------------------
        
        for s in species:
            
            y_mod_trace[s][i,:] = y_mod_current[s]

    #---------------------------------------------------------------------------------
    # Outputs ------------------------------------------------------------------------
    
    total_time = time.time() - start_time
    
    print('\nMCMC chain finished.')
    print(f'{total_iter} interations completed in: {round(total_time,4)} seconds.')
    print('\n')
    
    ratio = float(accept_count_em)/float(reject_count_em)
    print(f'--- {round(ratio,4)} x acceptance ratio ---')   
    print('\n')
    
    if step_size_xbc is not None:
        for s_name in mf_species:
            ratio_bc = float(accept_count_bc[s_name])/float(reject_count_bc[s_name])
            print(f'--- {round(ratio_bc,4)} {s_name} xbc acceptance ratio ---') 
        print('\n')   
    else:
        xbc_trace,prob_trace_bc = None,None
      
    if step_size_R is not None:
        for s_name in species[1:]:
            for sector_name in sectors_dict[s_name]:
                if sector_name is not None:
                    ratio_R = float(accept_count_R[s_name][sector_name])/float(reject_count_R[s_name][sector_name])
                    print(f'--- {round(ratio_R,4)} {s_name} {sector_name} R acceptance ratio ---') 
            print('\n')   
    else:
        prob_trace_R = None,None
              
    if step_size_Rbc is not None:
        for s_name in delta_species:
            ratio_R = float(accept_count_Rbc[s_name])/float(reject_count_Rbc[s_name])
            print(f'--- {round(ratio_R,4)} {s_name} Rbc acceptance ratio ---') 
        print('\n')   
         
    else:
        prob_trace_Rbc = None,None
    
    if y_sig_pdf is not None:
        for s_name in species:
            ratio_y_sig = float(accept_count_y_sig[s_name])/float(reject_count_y_sig[s_name])
            print(f'--- {round(ratio_y_sig,4)} {s_name} sig acceptance ratio ---')   
        print('\n') 
    else:
        y_sig_trace,prob_trace_y_sig = None,None
        
    
    return (y_mod_trace,xem_trace,xbc_trace,R_trace,R_trace_allsectors,Rbc_trace,y_sig_trace,
            prob_trace_em,prob_trace_bc,prob_trace_R,prob_trace_Rbc,prob_trace_y_sig)