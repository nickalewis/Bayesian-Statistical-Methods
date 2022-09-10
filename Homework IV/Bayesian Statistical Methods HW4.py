# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:11:01 2020

@author: nlewis
"""

import numpy as np #useful math functions and everything else
import pandas as pd #for dataframe manipulation
import matplotlib.pyplot as plt #for plotting our histograms and contours
from numpy.random import rand #Uniform(0,1). just makes writing more succinct

#lets us use probability distributions like t, beta, gamma,etc.
from scipy.stats import poisson,t, binom, multivariate_normal, multivariate_t
from scipy.stats import dirichlet, multinomial, norm, nbinom, gamma, chi2, uniform

from scipy import stats, optimize #use this package to maximize posteriors
from scipy.special import gammaln #usful function
import time #so we can time how long it takes to run our mcmc methods


def param_stats(X,p,names,sig):
    '''
    PARAMETERS:
        X - data matrix
        p - the quantiles you wish to calculate
        names - names you want to give the parameters
        sig - number of digits to report
    Return:
        stats - a datframe detailing statistics about the parameters
    '''
    import pandas #declare this here in the case that we forget to apriori
    #we always specify the mean and standard deviation
    stats  = {
        'Mean': X.mean(axis=0),
        'Standard Deviation': X.std(axis=0,ddof=1)
                   } 
    #add quantile columns
    quant_names = [str(100*x) +'%' for x in p]
    for x in p: stats[ str(100*x) +'%'] = np.quantile(X,x,axis=0)
    
    return np.round(pd.DataFrame(stats,index = names),sig)

### Problem 1

def ImpSampler(nsamples,logTarget,logProposal,proposal,ControlConstant = 'NULL'):
    '''
    Parameters
    ----------
    nsamples : # of samples one wishes to collect
    logTarget : log of the target distribution (aka the posterior)
    logProposal : log of the proposal distribution
    proposal : samples of the proposal distribution
    ControlConstant : default is NULL. Use this to implement rejection/importance
                        sampling hybrid.
    
    Returns
    -------
    f_samples : number of samples used in calculating the weights
    f_LogWeights : weights
    n_eff : effective sample size
    acceptanceRate : acceptance rate of the samples
    '''
    
    #importance sampling to calculate weights
    samples = np.array([proposal() for j in range(nsamples)])
    logweights = np.array([logTarget(x) - logProposal(x) for x in samples])
    if ControlConstant == 'NULL':
        weights = np.exp( logweights)
        weights_norm = weights / weights.sum() #normalize weights
        #Calculation of effective sample size
        n_eff = np.ceil(1 / np.sum( (weights_norm)**2   ) )
        return samples, weights_norm, n_eff
    
    elif(ControlConstant > 0):
        log_u = np.log( rand(nsamples) ) #uniforms we use to reject or sample
        probs = np.minimum(0,logweights - np.log(ControlConstant))
        accepted = probs > log_u
        acc_samples, acc_logweights = samples[accepted], (logweights-probs)[accepted]
        acc_weights = np.exp(acc_logweights)
        acc_weights_norm = acc_weights / acc_weights.sum()
        n_eff = np.ceil(1 / np.sum( (acc_weights_norm)**2   )) #ESS
        accept = len(acc_samples) / nsamples #acceptance rate
        return acc_samples, acc_weights_norm, n_eff, accept
    
    else:
        print("Control Constant Must be Greater than 0")

#Part B/ PART C

log_f = lambda y:np.log((1/3)*norm.pdf(x  = y, loc = -2, scale = 1) + 
                        (2/3)*norm.pdf(x = y, loc = 2, scale = 1))
log_g = lambda y: norm.logpdf(x= y, loc = 0, scale = 3)

x_values = np.linspace(start = -8, stop = 10, num = 1000)
funcs = np.exp(np.c_[log_f(x_values),log_g(x_values)])
logl1,logl2 = plt.plot(x_values, funcs, linewidth=0.8) 
plt.legend((logl1,logl2),('Target Density', 'Proposal Density'))
plt.show()

# Part E

proposer = lambda: norm.rvs(loc=0,scale=3,size=None)

null_samples , null_weights , null_eff = ImpSampler(nsamples = 5000,
                                                    logTarget = log_f,
                                                    logProposal = log_g,
                                                    proposal = proposer)
#calculate E(X), E(X^2) and E(e^X)
est_vals = np.array([np.sum(null_samples*null_weights),
                     np.sum((null_samples**2)*null_weights),
                     np.sum(np.exp(null_samples)*null_weights)])

#The actual values of these expectations, not the estimated ones
real_vals = np.array([2/3,5,(1/3)*np.exp(-3/2) + (2/3)*np.exp(5/2)])

print(f"The Estimated Values of our expected values are {est_vals.round(4)}")
print(f"The True Values of our expected values are {real_vals.round(4)}")
print(f"This is done with an effective sample size of {null_eff}")

# Part F
c = 10
reported = [None]*c #store info

for i in range(c):
   draws,weights, ESS, a_c = ImpSampler(nsamples = 5000,logTarget = log_f,
                        logProposal = log_g,proposal = proposer,
                        ControlConstant = i+1 )
    
   est_vals = np.array([np.sum(draws*weights),
                        np.sum((draws**2)*weights),
                        np.sum(np.exp(draws)*weights)])

   error = np.abs(est_vals - real_vals)
   reported[i] = np.append(error,np.array([ESS,a_c])).round(3)   

#need to turn list to array
reported = np.array(reported)

#turns matrix to dataframe. index = rownames, columns = columns names
reported_df = pd.DataFrame(reported, 
                        index = ["c = " + str(j+1) for j in range(10)])

reported_df.columns = ['\u03BC 1 Error','\u03BC 2 Error','\u03B8 Error',
           "Effective Sample Size","Acceptance Rate"]

print(reported_df)

'NOTE: This portion needs to be run all at the same time'

#Sets up grid for our five figures
plt.figure(figsize=(14, 8))
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)

ax = [ax1,ax2,ax3,ax4,ax5]
col = ['red', 'blue','orange','magenta','green']

for j in range(5):
    ax[j].plot(range(1,c+1),reported[:,j],'o',color=col[j])
    ax[j].set_title('Control Constant' + ' vs ' + reported_df.columns[j])
    ax[j].set_xlabel("Control Constant")
    ax[j].set_ylabel(reported_df.columns[j])
plt.subplots_adjust(hspace=0.5, wspace=1)

### PROBLEM 2 (BDA 3rd Ed. Problem 10.5)
    
#PART A

#number of trials
J = 10
#having loc = 1 means we can't draw a zero
n = poisson.rvs(5, loc = 1, size = J, random_state = 123)
print(f"Our n values are {n}")
#initial pulls of our x's
x = uniform.rvs(size = J, random_state = 123 )
print(f"Our x values are {x.round(3)} ")
alpha_t = 2*t.rvs(df = 4, random_state = 123); beta_t = t.rvs(df=4, random_state=123)
y = binom.rvs(size = J, n = n, p =  1 / (np.exp(-alpha_t-beta_t*x) + 1), random_state=123)
print(f"Our y values are {y} ")

sim_data  = {'Trials': n,'Events': y,'Dose': x} 
sim_data = pd.DataFrame(sim_data)

#PART B

t_prior = lambda x: t.pdf(x[0], df = 4,loc=0,scale=2)*t.pdf(x[1],loc=0,scale=1,df = 4)
#function of bioassay posterior,p(alpha,beta | y, n ,x)
def bioassay(z,data,prior):
    '''
    Parameters
    ----------
    z : input point (i.e. (alpha,beta) tuple)
    data : dataframe of simulated datapoints
    prior : prior function for this problem
    Returns
    -------
    logpost : natural log of unormalized density

    '''
    a,b = z[0], z[1]
    
    x,y,n = data['Dose'].to_numpy(),data['Events'].to_numpy(),data['Trials'].to_numpy()
    logprior = np.log(prior(z) )
    def loglik(a,b,x,y,n):
        theta = a+b*x
        return y*theta - n*np.log1p(np.exp(theta))
    logpost = logprior+sum(loglik(a,b,x[j],y[j],n[j]) for j in range(len(y)) )
    return logpost 

alphas = np.linspace(start = -7, stop = 5, num = 1000)
betas =  np.linspace(start = -10, stop = 10, num = 1000)

#logpost
logit_ = bioassay(z=list([alphas[:,None],betas[:,None,None]]),
                        data = sim_data,
                        prior = t_prior)

#turn log posterior back to posterior. subtract to avoid overflow
logit_ = logit_.sum(axis=2)
logit_ = np.exp(logit_ - logit_.max())

#Rejection Sampling Technique
w0 = np.array([0,0])
optim = optimize.minimize(lambda x: 
                               -(bioassay(x,data=sim_data,prior=t_prior)-
                                 np.log(t_prior(x)) ), w0)
M = (bioassay(optim['x'],data=sim_data,prior=t_prior)-np.log(t_prior(optim['x'])) )

num = 1000 #Number of Samples we want

post_samples = np.zeros((num,2))
counter = 0 # tells us when to stop
iters = 0 #tells us how many times we sample
while counter < num-1:
    draw = np.append(2*t.rvs(df=4), t.rvs(df=4))
    p = (bioassay(z=draw,data = sim_data,prior = t_prior) - np.log(t_prior(draw)) - M)
    if p > np.log(rand()):
        post_samples[counter,:] = draw
        counter +=1
    iters += 1
print(f"It took {iters} runs to get {num} samples")

#plots contours and simulated points
plt.figure(figsize = (10, 8))
lev = [ 0.001, 0.01,.025,0.05,0.25,0.50,0.75,0.90,0.95]
cont = np.quantile(np.linspace(logit_.min(),logit_.max(),10000),lev)
plt.contour(alphas, betas, logit_, colors='red',levels=cont, zorder = 2)
plt.scatter(post_samples[:,0],post_samples[:,1])
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.title(r'Posterior Distribution of p($\alpha$, $\beta$ | y)')
plt.show()

# PART C

# initial guess
w0 = np.array([0.0, 0.0])
# finds the mode and covariance matrix for normal approx
optim_norm = optimize.minimize(lambda x: -bioassay(x,data=sim_data,prior=t_prior), w0)
# extract desired results
norm_mu = optim_norm['x'];  norm_Sig = optim_norm['hess_inv']

#Now let's set up our multivariate normal
#Create grid and multivariate normal
A, B = np.meshgrid(alphas,betas)
pos = np.empty(A.shape + (2,))
pos[:, :, 0] = A; pos[:, :, 1] = B
norm_approx = multivariate_normal.pdf(x=pos, mean=norm_mu, cov=norm_Sig)
norm_cont = np.quantile(np.linspace(norm_approx.min(),norm_approx.max(),10000),lev)

fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(10,5))

#contour w/o replacement
ax1.contour(alphas,betas,logit_,levels=cont,colors='red')
ax1.set_ylabel(r'$\beta$', fontsize = 20)
ax1.set_xlabel( r'$\alpha$', fontsize = 20)
ax1.set_title('Posterior Distribution', fontsize = 18)

#contour w/ replacement
ax2.contour(alphas,betas,norm_approx,levels=norm_cont,colors='red')
ax2.set_ylabel(r'$\beta$', fontsize = 20)
ax2.set_xlabel( r'$\alpha$', fontsize = 20)
ax2.set_title('Normal Approximation', fontsize = 18)

fig.suptitle('Simulated Data Distributions', y = 1.05,fontsize=20)
fig.tight_layout()

# PART D
param_stats(X = post_samples,p = [0.025,0.25,0.50,0.75,0.975],
            names = ['\u03B1','\u03B2'], sig = 2)
#We use the importance sampler function in [Problem 1] to solve d and e

mvt_samps = lambda : multivariate_t.rvs(loc=norm_mu, shape=norm_Sig, df=4, size=1)
post_target = lambda x: bioassay(x,data=sim_data,prior = t_prior)
post_proposal = lambda x: multivariate_t.pdf(x,loc=norm_mu, shape=norm_Sig, df=4)

#Importance Sampling
samps, weights, ESS = ImpSampler(nsamples = 1000, logTarget = post_target,
                            logProposal = post_proposal, proposal = mvt_samps)

print(f"Our Posterior mean E(\u03B1|y) = {np.round((samps[:,0]*weights).sum(),4)}")
print(f"Our Posterior mean E(\u03B2|y) = {np.round((samps[:,1]*weights).sum(),4)}")

# PART E
print(f"Our Effective Sample Size is {ESS}")

### PROBLEM 3 (BDA 3rd Ed, Exercise 10.8)

bioassay_data  = {
        'Trials': np.array([5,5,5,5]),
        'Events': np.array([0,1,3,5]),
        'Dose': np.array([-0.86,-0.30,-0.05,0.73])
                   } 

bioassay_data = pd.DataFrame(bioassay_data)

#function of bioassay posterior,p(alpha,beta | y, n ,x)
def bioassay(z,data,prior):
    '''
    Parameters
    ----------
    z : input point (i.e. (alpha,beta) tuple)
    data : dataframe of simulated datapoints
    prior : prior function for this problem
    Returns
    -------
    logpost : natural log of unormalized density

    '''
    a,b = z[0], z[1]
    
    x,y,n = data['Dose'].to_numpy(),data['Events'].to_numpy(),data['Trials'].to_numpy()
    logprior = np.log(prior(z) )
    def loglik(a,b,x,y,n):
        theta = a+b*x
        return y*theta - n*np.log1p(np.exp(theta))
    logpost = logprior+sum(loglik(a,b,x[j],y[j],n[j]) for j in range(len(y)) )
    return logpost 

#flat prior p(alpha,beta) propto 1
ind_prior = lambda x: 1

# initial guess
w0 = np.array([0.0, 0.0])
# optimise
optim_res = optimize.minimize(lambda x: -bioassay(x,data=bioassay_data,
                                              prior = ind_prior), w0)
# extract desired results
mu_w = optim_res['x']; I_w = optim_res['hess_inv'] 
np.linalg.eig(I_w)[0] #convince yourself I_w is positive definite

S = 10000
bio_target = lambda x: bioassay(x,data=bioassay_data,prior = ind_prior)
bio_proposal = lambda x: np.log(multivariate_normal.pdf(x, mean=mu_w, cov=I_w))
bio_proposer = lambda : multivariate_normal.rvs(mean=mu_w, cov=I_w, size = 1)
#importance re-sampling

#Importance Sampling
bio_samps, bio_wts, bio_ = ImpSampler(nsamples = S,logTarget = bio_target,
                            logProposal = bio_proposal, proposal = bio_proposer)

#This'll give us 1000 random draws
N1 = 1000
Picks_F = np.random.choice(S, size = N1,replace = False, p = bio_wts)
Picks_T = np.random.choice(S, size = N1,replace = True, p = bio_wts)

#new draws
alpha_draws_F, beta_draws_F = bio_samps[Picks_F,0], bio_samps[Picks_F,1]
alpha_draws_T, beta_draws_T = bio_samps[Picks_T,0], bio_samps[Picks_T,1]

#everything plotted on posterior
#Grid for plotting the unnormalized posterior
alpha_bio = np.linspace(start = -5, stop = 10, num = 1000)
beta_bio =  np.linspace(start = -10, stop = 40, num = 1000)

#logpost
bioassay_post = bioassay(z=list([alpha_bio[:,None],beta_bio[:,None,None]]),
                        data = bioassay_data,
                        prior = ind_prior)
#subtract to avoid overflow
bioassay_post = np.exp( np.sum(bioassay_post,axis=2)) 

#contour levels
biolev = [0.01,.025,0.05,0.25,0.50,0.75,0.90,0.95]
bio_cont = np.quantile(np.linspace(bioassay_post.min(),bioassay_post.max(),10000),biolev)

fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(10,5))

#contour w/o replacement
ax1.contour(alpha_bio,beta_bio,bioassay_post,levels=bio_cont,colors='red')
ax1.scatter(alpha_draws_F, beta_draws_F , zorder = 1)
ax1.set_ylabel(r'$\beta$', fontsize = 20)
ax1.set_ylim(-10,40)
ax1.set_xlabel( r'$\alpha$', fontsize = 20)
ax1.set_xlim(-5,10)
ax1.set_title('w/o Replacement', fontsize = 18)

#contour w/ replacement
ax2.contour(alpha_bio,beta_bio,bioassay_post,levels=bio_cont,colors='red')
ax2.scatter(alpha_draws_T, beta_draws_T, zorder = 1)
ax2.set_ylabel(r'$\beta$', fontsize = 20)
ax2.set_ylim(-10,40)
ax2.set_xlabel( r'$\alpha$', fontsize = 20)
ax2.set_xlim(-5,10)
ax2.set_title('w/ Replacement', fontsize = 18)

fig.suptitle('Bioassay Posterior Distribution', y = 1.05,fontsize=20)
fig.tight_layout()

fig, (ax3,ax4) = plt.subplots(1, 2,figsize=(10,5))

#Histogram for distirbution of wts
ax3.hist(x = bio_wts, bins='auto', color='green', alpha=0.7)
ax3.set_ylabel('Frequency', fontsize = 20)
ax3.set_xlabel(r'w($\theta$)', fontsize = 20)
ax3.set_title('Histogram of Importance Ratios', fontsize = 18)

#Boxplot for distirbution of wts
ax4.boxplot(x = bio_wts)
ax4.set_ylabel('Frequency', fontsize = 20)
ax4.set_xlabel(r'w($\theta$)', fontsize = 20)
ax4.set_title('Boxplot of Importance Ratios', fontsize = 18)

fig.suptitle('Distribution of Importance Ratios', y = 1.05,fontsize=20)
fig.tight_layout()

### Problem 4 (BDA 3rd Ed, Exercise 11.2)

bioassay_data  = {
        'Trials': np.array([5,5,5,5]),
        'Events': np.array([0,1,3,5]),
        'Dose': np.array([-0.86,-0.30,-0.05,0.73])
                   } 

#turns data from dictionary format to dataframe format
bioassay_data = pd.DataFrame(bioassay_data)

# uniform prior for this problem
ind_prior = lambda x: 1

#function of bioassay posterior,p(alpha,beta | y, n ,x)
def bioassay(z,data,prior):
    '''
    Parameters
    ----------
    z : input point (i.e. (alpha,beta) tuple)
    data : dataframe of simulated datapoints
    prior : prior function for this problem
    Returns
    -------
    logpost : natural log of unormalized density

    '''
    #The function is written like this so we can generalize to grid sampling,
    #and so that we can plug in points
    a,b = z[0], z[1]
    
    #make it easier to write function. .to_numpy() turns dataframe column to numpy array
    x,y,n = data['Dose'].to_numpy(),data['Events'].to_numpy(),data['Trials'].to_numpy()
    
    #putting prior in log format
    logprior = np.log(prior(z) )
    
    #log liklihood function
    def loglik(a,b,x,y,n):
        return y*(a+b*x) - n*np.log1p(np.exp(a+b*x))
    
    #natural log of the posterior = log of prior + log-liklihood
    logpost = logprior+sum(loglik(a,b,x[j],y[j],n[j]) for j in range(len(y)) )
    return logpost 

### GET MAP for this posterior, and also get the Inverse Fisher info associated with it.

# initial guess
w0 = np.array([0.0, 0.0])
# optimise (NOTE: This only minimizes so we make it negative to maximize)
optim_res = optimize.minimize(lambda x: -bioassay(x,data=bioassay_data,
                                              prior = ind_prior), w0)

# extract desired results
mu_w = optim_res['x']; I_w = optim_res['hess_inv'] 

np.linalg.eig(I_w)[0] #convince yourself I_w is positive definite (pos def means it's a max)
print(f"The mode of our posterior is {mu_w}")
print(f"The Fisher Information (Covariance Matrix) associated with the mode is {I_w}")

### METROPOLIS HASTINGS ALGORITHM

init = mu_w # initial starting position

# do this externally so we don't have to continuously do it in the loop process
r_bottom = bioassay(z = init, data = bioassay_data, prior = ind_prior)

c = 2.4 / np.sqrt(2) # variance scaling parameter. improves sampling
D = 50000 # number of samples to draw
u = np.log( rand(D) ) # on log scale. Calling outside loop saves time.
count = 0 # used to count the number of acceptances
MC  = [None]*D #pre-allocating space will save you precious computational time

# records speed
t1 = time.time()

for i in range(D):
     
    #sample proposal theta* using jumping distribution J_t(theta* | theta^{t-1})
    #here, proposal = theta*, and theta^{t-1} = init

    proposal = multivariate_normal(mean=init, cov=(c**2)*I_w).rvs()
    
    '''calculate log ratio of densities where the ratio is
    r = [ p(theta* | y)/J_t(theta* | theta^{t-1}) ] / [ p(theta^{t-1} | y)/J_t(theta^{t-1} | theta*) ].
    NOTE: For metropolis hastings, the jumping rule need not be symmetric
    '''

    r_top = bioassay(z = proposal, data = bioassay_data, prior = ind_prior)
    
    r = min(r_top-r_bottom,0) #use this to accept or reject
    
    #if r > u, accept. otherwise reject
    if r > u[i]:
        init = proposal
        r_bottom = r_top #explained above
        count += 1 #counts the number of samples that we've accepted
    MC[i] = init #regardless of accept/reject, append this to your samples
    
t2 = time.time()

#This is called an "f string". Superior printing for when you have numbers to include
print(f"It took about {t2-t1} seconds to perform the MH algorithm")
print(f"Our acceptance rate is {100*count / D}%")

MC = np.array(MC) #turns from list to array (must be array to do plotting)

## Trace plots

fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(10,5))

ax = [ax1,ax2]
MH_parameters = [r'$\alpha$', r'$\beta$']
MH_colors = ['red','blue']
for j in range(2):
    plt.tight_layout()
    ax[j].plot(np.arange(0,D), MC[:,j],color=col[j])
    ax[j].set_title('Trace Plot for ' + MH_parameters[j])
    ax[j].set_xlabel('iterations')
    ax[j].set_ylabel(MH_parameters[j])
fig.tight_layout()

## Mean Plots

## Mean Plots
fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(10,5))

ax = [ax1,ax2]
for j in range(2):
    plt.tight_layout()
    ax[j].plot(np.arange(0,D), np.cumsum(MC[:,j])/np.arange(1,D+1),color=col[j])
    ax[j].set_title('Trace Plot for ' + MH_parameters[j])
    ax[j].set_xlabel('iterations')
    ax[j].set_ylabel(MH_parameters[j])
fig.tight_layout()

## Autocorrelation plots

fig = plt.figure(figsize=(8, 16))
ax1 = fig.add_subplot(3, 1, 3)
maxlag = 100 # maximum lag for autocorrelation
sampsc = MC - MC.mean(axis=0) # scale the samples by subtracting the mean
acorlags = np.arange(maxlag+1) # lags from 0 to maxlag

# calculate autocorrelation for all different lags
for i in range(2):
    auto = np.correlate(sampsc[:,i], sampsc[:,i], mode = 'full') 
    auto = auto[-len(sampsc):-len(sampsc)+maxlag+1] / auto[-len(sampsc)] 
    ax1.plot(acorlags, auto)
ax1.set_xlabel('Lag')
ax1.set_title('Autocorrelation for Parameters')
ax1.legend([r'$\alpha$',r'$\beta$'])
fig.tight_layout()

#accepted samples (throws away the first 60%, considered here to be burnin)
accepted_bio = MC[-int(0.4*D):,]

# range for the grid to sample from
alpha_bio = np.linspace(start = -5, stop = 10, num = 1000)
beta_bio =  np.linspace(start = -10, stop = 40, num = 1000)

#logpost
logit_ = bioassay(z=list([alpha_bio[:,None],beta_bio[:,None,None]]),
                        data = bioassay_data,
                        prior = ind_prior)

#turn log posterior back to posterior. subtract to avoid overflow
logit_ = logit_.sum(axis=2)
logit_ = np.exp(logit_ - logit_.max())


#contour levels
biolev = [0.01,.025,0.05,0.25,0.50,0.75,0.90,0.95]
bio_cont = np.quantile(np.linspace(logit_.min(),logit_.max(),10000),biolev)

fig, (ax2,ax3) = plt.subplots(1, 2,figsize=(10,5))

#contour w/o samples
ax2.contour(alpha_bio,beta_bio,logit_,levels=bio_cont,colors='red')
ax2.set_ylabel(r'$\beta$', fontsize = 20)
ax2.set_ylim(-10,40)
ax2.set_xlabel( r'$\alpha$', fontsize = 20)
ax2.set_xlim(-5,10)
ax2.set_title('w/o Posterior Draws', fontsize = 18)

#contour w samples
ax3.contour(alpha_bio,beta_bio,logit_,levels=bio_cont,colors='red')
ax3.scatter(accepted_bio[:,0], accepted_bio[:,1] , zorder = 1)
ax3.set_ylabel(r'$\beta$', fontsize = 20)
ax3.set_ylim(-10,40)
ax3.set_xlabel( r'$\alpha$', fontsize = 20)
ax3.set_xlim(-5,10)
ax3.set_title('w/ Posterior Draws', fontsize = 18)

fig.suptitle('Bioassay Posterior Distribution', y = 1.05,fontsize=20)
fig.tight_layout()

#LD50 plot
plt.hist(x = -accepted_bio[:,0]/accepted_bio[:,1], 
         bins='auto', color='blue', alpha=0.7, rwidth=0.85)
plt.ylabel('Frequency')
plt.xlabel("LD50")
plt.show()

### PROBLEM 5 (BDA 3rd Ed., Exercise 11.3)

machine = { 'Machine 1': np.array([83,92,92,46,67]),
      'Machine 2':np.array([117,109,114,104,87]),
      'Machine 3': np.array([101,93,92,86,67]),
      'Machine 4': np.array([105,119,116,102,116]),
      'Machine 5':np.array([79,97,103,79,92]),
      'Machine 6':np.array([57,92,104,77,100])
      }

#statistics about the data
machine_n = np.array([len(x) for x in machine.values()])
machine_mean = np.array([(x).mean() for x in machine.values()])
machine_var = np.array([(x).var(ddof=1) for x in machine.values()])

### (i) Pooled Model

pooled = np.concatenate(list(machine.values())).ravel()
pooled_var = pooled.var(ddof=1)
pooled_mean = pooled.mean()

B = 1000

#marginal posterior pdf's for mu and sigma.
sig_pooled = np.sqrt( (len(pooled)-1)*pooled_var / (chi2.rvs(df = len(pooled) - 1, size = B)) )
theta_pooled = norm.rvs(loc = pooled_mean, scale = sig_pooled/len(pooled))

param_stats(X = np.c_[theta_pooled,sig_pooled],p = [0.025,0.25,0.50,0.75,0.975],
                       names = ['\u03B8','\u03C3'], sig = 2)

#Histograms of theta_6 |y, tilde y_6 | y, and theta_7 | y.

pooled_hist = np.c_[theta_pooled,
                    norm.rvs(loc = theta_pooled, scale = sig_pooled),
                    norm.rvs(loc = pooled_mean, scale = sig_pooled/len(pooled))]

#Sets up grid for our three figures
ax1 = plt.subplot2grid(shape=(4,4), loc=(0,1), colspan=2,rowspan = 2)
ax2 = plt.subplot2grid((4,4), (2,0), colspan=2,rowspan = 2)
ax3 = plt.subplot2grid((4,4), (2,2), colspan=2,rowspan = 2)

ax = [ax1,ax2,ax3]
col = ['purple', 'gold','green']
posterior_labels = [r'$\theta_6$ | y',r'$\tilde{y_6}$ | y',r'$\theta_7$ | y']

for j in range(3):
    plt.tight_layout()
    ax[j].hist(x = pooled_hist[:,j],bins='auto', color=col[j], alpha=0.7, rwidth=0.85)
    ax[j].set_ylabel("Frequency")
    ax[j].set_xlabel(posterior_labels[j])
    
### (ii) Separate Model

sig_sep = np.sqrt( sum( (machine_n-1)*machine_var ) / chi2.rvs(size=B,df = sum(machine_n) ))
theta_sep = [norm.rvs(loc = np.mean(x), scale = sig_sep/len(x), size = B) 
             for x in machine.values()]

theta_sep = np.array(theta_sep).T

sep_names =  np.concatenate([['\u03B8'+str(j) for j in range(1,len(machine_n)+1)],
         ['\u03C3']])

param_stats(X = np.c_[theta_sep,sig_sep],p = [0.025,0.25,0.50,0.75,0.975],
                       names = sep_names, sig = 2)

#Histograms of theta_6 |y, tilde y_6 | y, and theta_7 | y.

sep_hist = np.c_[theta_sep[:,5],
                 norm.rvs(loc = theta_sep[:,5], scale = sig_sep)]

#Sets up grid for our three figures
ax1 = plt.subplot2grid((4,4), (2,0), colspan=2,rowspan = 2)
ax2 = plt.subplot2grid((4,4), (2,2), colspan=2,rowspan = 2)

ax = [ax1,ax2]

for j in range(2):
    plt.tight_layout()
    ax[j].hist(x = sep_hist[:,j],bins='auto', color=col[j], alpha=0.7, rwidth=0.85)
    ax[j].set_ylabel("Frequency")
    ax[j].set_xlabel(posterior_labels[j])
    
###  (iii) Hierarchical Model
N5 = 50000 #How many values we want to draw up

#list to append our samples to
theta_gibbs = [None]*N5
sigma_gibbs = [None]*N5
mu_gibbs = [None]*N5
tau_gibbs = [None]*N5
#initial values
theta = np.array([72,75,87,91,69,73])
mu = theta.mean()

#helpful things
J = len(theta)

t1 = time.time()
for j in range(N5):
    
    # Sigma draws
    s2 = machine_n*(machine_mean - theta)**2 + (machine_n - 1)*machine_var 
    sigma = np.sqrt( s2.sum() / chi2.rvs(df = machine_n.sum() ))
    sigma_gibbs[j] = sigma
    
    # Tau draw
    tau = np.sqrt( ((theta - mu)**2).sum() / chi2.rvs(df = J - 1) )
    tau_gibbs[j] = tau

    # Theta draws
    V = 1 / ( (1/tau)**2 + (machine_n/(sigma**2)) )
    theta_hat = ((1/tau**2)*mu + (1/sigma**2)*(machine_mean*machine_n))*V
    theta = norm.rvs(loc = theta_hat, scale = np.sqrt(V))
    theta_gibbs[j] = theta
    
    # Mu Draw
    mu = norm.rvs(loc = theta.mean(), scale = tau / np.sqrt(J) )
    mu_gibbs[j] = mu
    
t2 = time.time()
print(f"It took approximately {round(t2-t1,4)} seconds for this to run {N5} samples")

theta_gibbs = np.array(theta_gibbs)
sigma_gibbs = np.array(sigma_gibbs)
mu_gibbs = np.array(mu_gibbs)
tau_gibbs = np.array(tau_gibbs)

P5_gibbs = np.c_[theta_gibbs,sigma_gibbs,mu_gibbs,tau_gibbs]
P5_titles = np.concatenate([['\u03B8'+str(j) for j in range(1,7)],
                         ['\u03C3','\u03BC','\u03C4'],])

param_stats(X = P5_gibbs[-int(0.4*N5):,] ,
            p = [0.025,0.25,0.50,0.75,0.975], 
            names = P5_titles, sig = 2)

iters = np.arange(0, N5+1)

# Trace Plots
colors = ['red', 'blue','orange','magenta','cyan','green',
          'brown','hotpink','yellow']
theta_titles = [r'$\theta_1$',r'$\theta_2$',r'$\theta_3$',
          r'$\theta_4$',r'$\theta_5$',r'$\theta_6$',
          r'$\sigma$',r'$\mu$',r'$\tau$']

f,a = plt.subplots(3,3, figsize = (12,12))
a = a.ravel()
for idx,ax in enumerate(a):
    plt.tight_layout()
    ax.plot(P5_gibbs[:,idx], color = colors[idx])
    ax.set_title('Trace Plot for' + ' ' + theta_titles[idx])
    ax.set_xlabel("Iterations")
    ax.set_ylabel(theta_titles[idx])

## Autocorrelation plots

fig = plt.figure(figsize=(8, 16))
ax1 = fig.add_subplot(3, 1, 3)
maxlag = 100 # maximum lag for autocorrelation
sampsc = P5_gibbs - P5_gibbs.mean(axis=0) # scale the samples by subtracting the mean
acorlags = np.arange(maxlag+1) # lags from 0 to maxlag

# calculate autocorrelation for all different lags
for i in range(P5_gibbs.shape[1]):
    auto = np.correlate(sampsc[:,i], sampsc[:,i], mode = 'full') 
    auto = auto[-len(sampsc):-len(sampsc)+maxlag+1] / auto[-len(sampsc)] 
    ax1.plot(acorlags, auto)
ax1.set_xlabel('Lag')
ax1.set_title('Autocorrelation for Parameters')
ax1.legend(theta_titles)
fig.tight_layout()

#Histograms of theta_6 |y, tilde y_6 | y, and theta_7 | y.

hier_hist = np.c_[theta_gibbs[:,5],
                    norm.rvs(loc = theta_gibbs[:,5], scale = sigma_gibbs),
                    norm.rvs(loc = mu_gibbs, scale = tau_gibbs)]

#Sets up grid for our three figures
ax1 = plt.subplot2grid(shape=(4,4), loc=(0,1), colspan=2,rowspan = 2)
ax2 = plt.subplot2grid((4,4), (2,0), colspan=2,rowspan = 2)
ax3 = plt.subplot2grid((4,4), (2,2), colspan=2,rowspan = 2)

ax = [ax1,ax2,ax3]

for j in range(3):
    plt.tight_layout()
    ax[j].hist(x = hier_hist[:,j],bins='auto', color=col[j], alpha=0.7, rwidth=0.85)
    ax[j].set_ylabel("Frequency")
    ax[j].set_xlabel(posterior_labels[j])

### PROBLEM 6 (BDA 3rd Ed., Exercise 11.4)

N6 = 50000 #How many values we want to draw up

machine = { 'Machine 1': np.array([83,92,92,46,67]),
      'Machine 2':np.array([117,109,114,104,87]),
      'Machine 3': np.array([101,93,92,86,67]),
      'Machine 4': np.array([105,119,116,102,116]),
      'Machine 5':np.array([79,97,103,79,92]),
      'Machine 6':np.array([57,92,104,77,100])
      }

#statistics about the data
machine_n = np.array([len(x) for x in machine.values()])
machine_mean = np.array([(x).mean() for x in machine.values()])
machine_var = np.array([(x).var(ddof=1) for x in machine.values()])

#list to append our samples to
P6_theta_gibbs = [None]*N6
P6_sigma_gibbs = [None]*N6
P6_mu_gibbs = [None]*N6
P6_tau_gibbs = [None]*N6
P6_sig0_gibbs = [None]*N6

#initial values
theta = np.array([72,75,87,91,69,73])
sig0 = 12
mu = theta.mean()
nu = 3

#Functions for sig0 pdf 
def sig0_pdf(x,df,s,J): 
     p = ((J*df)/2 - 1)*np.log(x) - ( (df*x)/2)*sum(1/s**2)
     return p
 
#Grid for sigma 0 
sig2_0_grid = np.linspace(start = 1, stop = 500, num = 2000)
d_sig0 = np.diff(sig2_0_grid)[0]/2

#helpful things
J = len(theta)

t3 = time.time()
for j in range(N5):
    
    # Sigma draws
    big = machine_n*(machine_mean - theta)**2 + (machine_n - 1)*machine_var
    sigma_hat_sq = (nu*sig0**2 + big ) 
    sigma = np.sqrt(sigma_hat_sq / chi2.rvs(df = nu +  machine_n ))
    P6_sigma_gibbs[j] = sigma
    
    # sig_0 draws
    sig2_0_pdf = sig0_pdf(x=sig2_0_grid,df = nu,s = sigma, J = 5)
    sig2_0_pdf = np.exp(sig2_0_pdf) / (np.exp(sig2_0_pdf)).sum()
    sig0 = np.sqrt( np.random.choice(sig2_0_grid,p = sig2_0_pdf) + (d_sig0)*rand() )
    P6_sig0_gibbs[j] = sig0

    # Tau draw
    tau_hat_sq = ((theta - mu)**2).sum()
    tau = np.sqrt( tau_hat_sq / chi2.rvs(df = J - 1) )
    P6_tau_gibbs[j] = tau
    
    # Theta draws
    V = 1 / ((1/tau)**2 + (machine_n/(sigma**2)))
    theta_hat = ((mu / tau**2) + ( (machine_mean*machine_n)/sigma**2))*V
    theta = norm.rvs(loc = theta_hat, scale = np.sqrt(V))
    P6_theta_gibbs[j] = theta
    
    # Mu Draw
    mu = stats.norm.rvs(loc = theta.mean(), scale = np.sqrt(tau**2 / J))
    P6_mu_gibbs[j] = mu
    
t4 = time.time()
print(f"It took approximately {round(t4-t3,4)} seconds for this to run {N5} samples")

#turns list into arrays
P6_theta_gibbs = np.array(P6_theta_gibbs)
P6_sigma_gibbs = np.array(P6_sigma_gibbs)
P6_mu_gibbs = np.array(P6_mu_gibbs)
P6_tau_gibbs = np.array(P6_tau_gibbs)
P6_sig0_gibbs = np.array(P6_sig0_gibbs)

#combine to make it easier to calculate statistics
P6_gibbs = np.c_[P6_theta_gibbs,P6_sigma_gibbs,P6_mu_gibbs,P6_tau_gibbs,P6_sig0_gibbs]
P6_hyper_gibbs = np.c_[P6_mu_gibbs,P6_tau_gibbs,P6_sig0_gibbs]

P6_titles = np.concatenate([['\u03B8'+str(j) for j in range(1,7)],
                            ['\u03C3'+str(j) for j in range(1,7)],
                            ['\u03BC','\u03C4','\u03C3'+str(0)]])

param_stats(X = P6_gibbs[-int(0.4*N6):,],
            p = [0.025,0.25,0.50,0.75,0.975], 
            names = P6_titles, sig = 2)

#Trace plots

#our iterations
iters = np.arange(0, N6+1)

colors = ['red', 'blue','orange','magenta','cyan','green']
theta_titles = [r'$\theta_1$',r'$\theta_2$',r'$\theta_3$',
          r'$\theta_4$',r'$\theta_5$',r'$\theta_6$']
sigma_titles = [r'$\sigma_1$',r'$\sigma_2$',r'$\sigma_3$',
          r'$\sigma_4$',r'$\sigma_5$',r'$\sigma_6$']
hyper_titles = [r'$\mu$',r'$\sigma_0$',r'$\tau$']
hyper_colors = ['yellow', 'purple','green']

f,a = plt.subplots(3,2, figsize = (12,8))
a = a.ravel()
for idx,ax in enumerate(a):
    plt.tight_layout()
    ax.plot(P6_theta_gibbs[:,idx], color = colors[idx])
    ax.set_title('Trace Plot for' + ' ' + theta_titles[idx])
    ax.set_xlabel("Iterations")
    ax.set_ylabel(theta_titles[idx])
    
f,a = plt.subplots(3,2, figsize = (12,8))
a = a.ravel()
for idx,ax in enumerate(a):
    plt.tight_layout()
    ax.plot(P6_sigma_gibbs[:,idx], color = colors[idx])
    ax.set_title('Trace Plot for' + ' ' + sigma_titles[idx])
    ax.set_xlabel("Iterations")
    ax.set_ylabel(sigma_titles[idx])

f,a = plt.subplots(3,1,figsize = (12,8))
a = a.ravel()
for idx,ax in enumerate(a):
    plt.tight_layout()
    ax.plot(P6_hyper_gibbs[:,idx], color = hyper_colors[idx])
    ax.set_title('Trace Plot for' + ' ' + hyper_titles[idx])
    ax.set_xlabel("Iterations")
    ax.set_ylabel(hyper_titles[idx])

## Autocorrelation plots

fig = plt.figure(figsize=(8, 16))
ax1 = fig.add_subplot(3, 1, 3)
maxlag = 100 # maximum lag for autocorrelation
sampsc = P6_gibbs - P6_gibbs.mean(axis=0) # scale the samples by subtracting the mean
acorlags = np.arange(maxlag+1) # lags from 0 to maxlag

# calculate autocorrelation for all different lags
for i in range(P6_gibbs.shape[1]):
    auto = np.correlate(sampsc[:,i], sampsc[:,i], mode = 'full') 
    auto = auto[-len(sampsc):-len(sampsc)+maxlag+1] / auto[-len(sampsc)] 
    ax1.plot(acorlags, auto)
ax1.set_xlabel('Lag')
ax1.set_title('Autocorrelation for Parameters')
ax1.legend(np.concatenate([theta_titles,sigma_titles,hyper_titles]))
fig.tight_layout()

#Histograms of theta_6 |y, tilde y_6 | y, and theta_7 | y.

hier_hist = np.c_[P6_theta_gibbs[:,5],
                    norm.rvs(loc = P6_theta_gibbs[:,5], scale = P6_sigma_gibbs[:,5]),
                    norm.rvs(loc = P6_mu_gibbs, scale = P6_tau_gibbs)]

#Sets up grid for our three figures
ax1 = plt.subplot2grid(shape=(4,4), loc=(0,1), colspan=2,rowspan = 2)
ax2 = plt.subplot2grid((4,4), (2,0), colspan=2,rowspan = 2)
ax3 = plt.subplot2grid((4,4), (2,2), colspan=2,rowspan = 2)

ax = [ax1,ax2,ax3]
col = ['purple', 'gold','green']
posterior_labels = [r'$\theta_6$ | y',r'$\tilde{y_6}$ | y',r'$\theta_7$ | y']

for j in range(3):
    plt.tight_layout()
    ax[j].hist(x = hier_hist[:,j],bins='auto', color=col[j], alpha=0.7, rwidth=0.85)
    ax[j].set_ylabel("Frequency")
    ax[j].set_xlabel(posterior_labels[j])

### Extra Credit: PROBLEM 7 (BDA 3rd Ed., Exercise 13.5)

#data
captured = np.array([118,74,44,24,29,22,20,14,20,15,
                 12,14,6,12,6,9,9,6,10,10,11,5,3,3])
times = np.arange(1,25) 

#Grid for sampling
R = 1000
alphas = np.linspace(start = 0.0001, stop = 17, num = R)
betas = np.linspace(start = 0.0001, stop = 0.6, num = R)

ind_prior = lambda x,y :1
#Posterior Function. It is of form seen in derivation
def animal(w,y,prior):
    '''
    PARAMETERS:
    ----------
    w - alpha, beta parameter
    x - beta parameter
    y - number of species caught
    prior - prior on alpha, beta
    Returns:
    -------
    logpost : natural logarithm of unnormalized posterior density
    '''
    a,b = w[0], w[1]
    #prior distribution used for this problem
    logprior = np.log(prior(a,b))

    # for brevity, split the likelihood into a numerator term and denominator
    def loglik(a,b,x):
     l = gammaln(a+x) + a*np.log(b) - gammaln(a) - gammaln(x+1) - (x+a)*np.log1p(b)
     return l

    logpost = logprior + sum(y[j]*loglik(a,b,j+1) for j in range(len(y)) )
    return logpost

ani = animal(w=list([alphas[:,None],betas[:,None,None]]),
             y = captured,prior = ind_prior)

ani = np.sum(ani, axis=2)
ani = np.exp(ani - ani.max())

#Draw Samples via Grid Sampling (In python, probs need to sum to q)
ani = ani / ani.sum()

alpha_grid =  np.repeat(alphas,len(betas))
beta_grid = np.tile(betas,len(alphas))

N7 = 10000 #Number of Samples to draw
samples = np.random.choice(ani.size, size=N7, p = ani.ravel(order="F"))

#add some random jitter so the variables are continous random variables
d_alpha = np.diff(alphas)[0]/2
d_beta = np.diff(betas)[0]/2

a_post = alpha_grid[samples] -d_alpha + (d_alpha)*rand(N7)
b_post = beta_grid[samples] -d_beta + (d_beta)*rand(N7)

#plots contours and simulated points
plt.figure(figsize = (10, 8))

lev = [ 0.001, 0.01,.025,0.05,0.25,0.50,0.75,0.90,0.95]
cont = np.quantile(np.linspace(ani.min(),ani.max(),10000),lev)

fig, (ax3,ax4) = plt.subplots(1, 2,figsize=(10,5))

#contour w samples
ax3.contour(alphas, betas, ani, colors='red',levels=cont, zorder = 2)
ax3.scatter(a_post,b_post)
ax3.set_ylabel(r'$\beta$', fontsize = 20)
ax3.set_ylim(0.10,0.40)
ax3.set_xlabel( r'$\alpha$', fontsize = 20)
ax3.set_xlim(0,4)
ax3.set_title('w/ Posterior Draws', fontsize = 18)

#contour w/o samples
ax4.contour(alphas, betas, ani, colors='red',levels=cont, zorder = 2)
ax4.set_ylabel(r'$\beta$', fontsize = 20)
ax4.set_ylim(0.10,0.40)
ax4.set_xlabel( r'$\alpha$', fontsize = 20)
ax4.set_xlim(0,4)
ax4.set_title('w/o Posterior Draws', fontsize = 18)

fig.suptitle('Animal Data Posterior Distribution', y = 1.05,fontsize=20)
fig.tight_layout()

#PART C
# initial guess
w0 = np.array([1,5])
# optimise
optim_res = optimize.minimize(lambda x: -animal(x,y=captured,prior = ind_prior),
                              x0 = w0,bounds = ((0.001,np.inf),(0.001,np.inf)))
# extract desired results
mu_p7 = optim_res['x']; S_7 = optim_res['hess_inv'].todense()
print(f"The mode of our posterior is {mu_p7}")
print(f"The Fisher Information (Covariance Matrix) associated with the mode is {S_7}")
np.linalg.eig(S_7)[0] #convince yourself I_w is positive definite

# PART D
B = [np.min(np.where(nbinom.rvs(n  = a_post, p = b_post/(1+b_post)).cumsum()>10000))
     for x in range(1000)]
extra_species = (np.percentile(np.array(B),[2.5,97.5])).round()
print(f"Our 95% CI for the additional number of species observed is {extra_species}")

# PART E
#e Posterior Predictive Checks
animal_reps = nbinom.rvs(size = (10000,496), n = a_post[:,None], 
                 p = b_post[:,None]/(1+b_post[:,None]))

Test1 = animal_reps.sum(axis=1)
Test2 = animal_reps.max(axis=1)

fig, (ax3,ax4) = plt.subplots(1, 2,figsize=(10,5))

#(i) compare the sums
ax3.hist(x = Test1, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
ax3.grid(axis='y', alpha=0.75)
ax3.set_xlabel('Test Statistic')
ax3.set_ylabel('Frequency')
ax3.axvline((captured*times).sum(), color='red', linewidth=1)
ax3.text(2550, 300,'p = ' + str((Test1 > (captured*times).sum()).mean()), fontsize = 18)
ax3.set_title('Expected number of total animals seen')

#(ii)
ax4.hist(x = Test2, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
ax4.grid(axis='y', alpha=0.75)
ax4.set_xlabel('Test Statistic')
ax4.set_ylabel('Frequency')
ax4.axvline(max(times), color='red', linewidth=1)
ax4.text(70, 350,'p = ' + str((Test2 > max(times)).mean()), fontsize = 18)
ax4.set_title('Maximum Number of Times Seen')

fig.suptitle('Animal Test Statistics', y = 1.05,fontsize=20)
fig.tight_layout()

### Problem 8

#set seed for reproducibility
np.random.seed(82)

#True parameter values
mu_true = np.array([0,-2,3])
sig_true = np.array([1,np.sqrt(2),4])
p_true = np.array([0.1,0.3,0.6])

r = np.random.choice(np.arange(len(mu_true)),size=120, p = p_true)
samples = norm(loc = mu_true[r], scale = sig_true[r]).rvs()

#mus, probs and sigmas should be initial guesses
def EM(samples, probs, mus, sigmas, tol):
    #initial log-likelihood
    n, m = len(samples), len(probs)
    count = 0
    
    ### NOTE: Calculate this external to the loop to save time
    
    #### E-step: Calculate the expectation of log p(y,z|theta)
        
    #calculating our weights
    w = probs*norm(mus, sigmas).pdf(samples[:,None])
    #weighting our weights
    w_tilde = w / w.sum(axis=1)[:,np.newaxis]
    #calculating Q(theta | theta^t)
    l_old = ( np.log(w.sum(axis=1)) ).sum()
    loglik = [l_old] #list so we can check algorithm is working properly
    while True:
        
        ### M-step: calculating our new MLE's
        
        #probability calculation
        probs  = w_tilde.sum(axis=0) / n
        
        #mean calculation
        mus = (w_tilde.T).dot(samples) / w_tilde.sum(axis=0)
        
        #standard deviation calculation
        mass = (w_tilde*(samples[:,None] - mus[np.newaxis,:])**2 ).sum(axis=0) 

        sigmas =  np.sqrt(mass / w_tilde.sum(axis=0))
        
        # update complete log likelihoood
        w = probs*norm(mus, sigmas).pdf(samples[:,None])
        w_tilde = w / w.sum(axis=1)[:,np.newaxis]

        #l_new = np.sum(W,axis = 1)
        #calculating Q(theta | theta^(t+1) )

        l_new = ( np.log(w.sum(axis=1)) ).sum()
                
        count +=1 #how many times it takes to reach convergence        
        if np.abs(l_new - l_old)  < tol:
            break
        l_old = l_new #so we don't have to repeatedly calculate l_old
        loglik.append(l_old)
    return count,loglik,probs, mus, sigmas 

init_p = np.array([1/2,1/4,1/4])  
init_mu = np.array([2,-1,0.7])
init_sigma = np.array([1,1,1])

t3 = time.time()
count,loglik,probs, mus,sigmas = EM(samples, init_p, init_mu, init_sigma,tol=1e-20)
t4 = time.time()

print(f"It took approximately {round(t4-t3,4)} seconds for this to run")

#check to make sure log-likelihood is monotonically decreasing
plt.plot(np.arange(count),np.array(loglik))
print(f"It took approximately {count} iterations to reach convergence")
print(f"The MLE for the weights are {probs}")
print(f"The MLE for the means are {mus}")
print(f"The MLE for the standard deviations are {sigmas}")


#Now let's see how our results work with

x_grid = np.linspace(start = -15, stop = 20, num = 1000)
true_curve = norm.pdf(x_grid[:,None],mu_true, sig_true).dot(p_true)
EMcurve_approx = norm.pdf(x_grid[:,None],mus, sigmas).dot(probs)

plt.plot(x_grid, true_curve, linewidth=0.8) # create lines for both theta1 and theta2 samples
plt.plot(x_grid, EMcurve_approx, linewidth=0.8) # create lines for both theta1 and theta2 samples
plt.title("True Gaussian Mixture plotted against the simulated values from EM Algorithm")
plt.xlabel("y")
plt.ylabel(r'f(y | $\theta$)')
plt.legend(['True Curve', 'E-M Curve'])
plt.show()

#plot of the histograms on top of each other
separates = norm.pdf(x = x_grid[:,None],loc = mu_true, scale = sig_true)
plt.plot(x_grid, separates, linewidth=0.8) # create lines for both theta1 and theta2 samples

#Gibbs Sampler Parameters

#mu posterior 
alpha = 3
#sigma posterior
eta, nu = 2, 2
eps = 0
# Number of iterations
N = 50000
#Number of mixtures to consider
n = len(mu_true)

# vectors to save our values
lmbda_gibbs,mu_gibbs,sigma_gibbs= [None]*N, [None]*N, [None]*N

lmbda = np.array([1/3,1/3,1/3])  
mu = np.array([2,-1,0.7])
sig = np.array([1,1,1])

t5 = time.time()
for i in range(N):
    #first get initial probabilities to draw z
    p = norm.pdf(x = samples[:,None], loc = mu, scale = sig)*lmbda
    p = p / p.sum(axis=1)[:,np.newaxis]
    
    #draw z for initial probabilities
    z = (p.cumsum(axis = 1) > rand(p.shape[0])[:,None]).argmax(axis=1)
    #How many values belong to 1, 2, and 3
    y_k = [samples[z==j] for j in range(n)]; n_k = np.array([len(x) for x in y_k])
    #draw lambda values, and append it to list
    lmbda = dirichlet.rvs(n_k+1, size=1).reshape(n,)
    lmbda_gibbs[i] = lmbda
    
    #draw mu values
    mu_gibbs_var = 1 / ((1/alpha)**2 + (n_k/sig**2) ) 
    mu_gibbs_mean = [sum(x) for x in y_k]/sig**2 + eps / alpha**2
    mu_gibbs_mean = mu_gibbs_mean*mu_gibbs_var
    
    mu = norm.rvs(loc = mu_gibbs_mean, scale = np.sqrt(mu_gibbs_var))
    mu_gibbs[i] = mu
    
    #draw sigma^2 values
    sums = np.array([sum( (y_k[x] - mu[x])**2 ) for x in range(n)])
    sig = np.sqrt( 1 / gamma.rvs(a = eta + 0.5*n_k, scale =  1/(nu + 0.5*sums)) )
    sigma_gibbs[i] = sig

t6 = time.time()
print(f"It took approximately {round(t6-t5,4)} seconds for this to run {N} samples")

#change back to arrays for ease of use
lmbda_gibbs = np.array(lmbda_gibbs)
mu_gibbs = np.array(mu_gibbs)
sigma_gibbs = np.array(sigma_gibbs)

gibbs_samples = np.c_[mu_gibbs,sigma_gibbs,lmbda_gibbs]
colors = ['red','blue','orange',
          'magenta','cyan','green',
          'yellow','gray','purple']

titless = [r'$\mu_1$',r'$\mu_2$',r'$\mu_3$',
           r'$\sigma_1$',r'$\sigma_2$',r'$\sigma_3$',
           r'$\lambda_1$',r'$\lambda_2$',r'$\lambda_3$']

f,a = plt.subplots(3,3,figsize = (8,8))
a = a.ravel()
for idx,ax in enumerate(a):
    plt.tight_layout()
    ax.plot(gibbs_samples[:,idx], color = colors[idx])
    ax.set_title('Trace Plot for' + ' ' + titless[idx])
    ax.set_xlabel("Iterations")
    ax.set_ylabel(titless[idx])

## Autocorrelation plots

fig = plt.figure(figsize=(8, 16))
ax1 = fig.add_subplot(3, 1, 3)
maxlag = 15000 # maximum lag for autocorrelation
sampsc = gibbs_samples - gibbs_samples.mean(axis=0) #scale samples by subtracting mean
acorlags = np.arange(maxlag+1) # lags from 0 to maxlag

# calculate autocorrelation for all different lags
for i in range(sampsc.shape[1]): # loop for parameters
    auto = np.correlate(sampsc[:,i], sampsc[:,i], mode = 'full')
    auto = auto[-len(sampsc):-len(sampsc)+maxlag+1] / auto[-len(sampsc)] 
    ax1.plot(acorlags, auto)
ax1.set_xlabel('Lag')
ax1.set_title('Autocorrelations')
ax1.legend(titless)
fig.tight_layout()

### EXTRA CREDIT: PROBLEM 9 (BDA 3rd Ed., Exercise 15.3)

#The data
#Reactor Temperature
x1 = np.array([1300,1300,1300,1300,1300,1300,1200,1200,1200,1200,
               1200,1200,1100,1100,1100,1100])
#Ratio of Hydrogen to n-heptane
x2 = np.array([7.5,9,11,13.5,17,23,5.3,7.5,11,13.5,17,23,5.3,7.5,11,17])
#contact time (sec)
x3 = np.array([0.012,0.012,0.0115,0.013,0.0135,0.012,0.04,0.038,0.032,0.026,
                0.034,0.041,0.084,0.098,0.092,0.086])
#conversion of heptane to acetlylene (%)
y = np.array([49,50.2,50.5,48.5,47.5,44.5,28,31.5,34.5,35,
              38,38.5,15,17,20.5,19.5])

#combine data into data matrix
X = np.c_[x1,x2,x3,x1*x2,x1*x3,x2*x3,x1**2,x2**2,x3**2]
#standardize our covariates for easier computation
X = (X-X.mean(axis=0)) / X.std(ddof=1,axis=0)
#add constant term
X_star = np.c_[np.ones(len(x1)),X]
#MLE
gram_mat = (X_star.T).dot(X_star); inv_gram_mat = np.linalg.inv(gram_mat)
MLE = inv_gram_mat.dot((X_star.T).dot(y))

#Number of simulations
N9 = 2000
#simulates sigma
s2 = ((y - X_star.dot(MLE)).T).dot((y - X_star.dot(MLE))) 
q = s2 / chi2.rvs(df = X_star.shape[0] - X_star.shape[1], size =  N9) #sigma^2
beta_post = [multivariate_normal.rvs(mean = MLE, cov = q[j]*inv_gram_mat)
             for j in range(N9)]
post_parameters = np.c_[np.array(beta_post),q]

names = ["Intercept","x1","x2","x3","x1x2","x1x3","x2x3","x1^2","x2^2","x3^2"]
param_names = np.concatenate([['\u03B2 '+str(j) for j in names],['\u03C3']])
part_a_stats = param_stats(X = post_parameters,
                           p = [0.025,0.25,0.50,0.75,0.975],
                           names = param_names, sig = 2)
pd.set_option("display.max_columns", 9)
print(part_a_stats)


### PART B

K = 100000 #How many values we want to draw up

#list to append our samples to
beta_gibbs = [None]*K
sigma_gibbs = [None]*K
mu_gibbs = [None]*K
tau_gibbs = [None]*K

#initial values
beta = MLE #This is good as any guess
mu = beta.mean()

#helpful things
N,J = X_star.shape[0], X_star.shape[1]
a = 1 #this represent how many values have improper prior

y_tilde = (y.T).dot(X_star) # do this beforehand so we don't have to do this N9 times
Id = np.concatenate([np.zeros(a),np.ones(J-a)]); Id_mat = np.diag(Id)

t1 = time.time()
for j in range(K):
    
    # Sigma draws
    s2 = ((y - X_star.dot(beta) ).T).dot( y - X_star.dot(beta) )
    sigma = np.sqrt( s2 / chi2(df = N ).rvs() )
    sigma_gibbs[j] = sigma
    
    # Tau draw
    tau = np.sqrt( ((beta[a:] - mu)**2).sum() / chi2(df = J - a).rvs() )
    tau_gibbs[j] = tau

    # Theta draws
    V_beta = np.linalg.inv(Id_mat/tau**2 + gram_mat/sigma**2)
    beta_tilde = V_beta.dot(Id*(mu/tau**2) + y_tilde/sigma**2)
    beta = multivariate_normal(mean=beta_tilde, cov=V_beta).rvs()
    beta_gibbs[j] = beta
    
    # Mu Draw
    mu = norm(loc = (beta[a:]).mean(),scale = tau / np.sqrt(J-a)).rvs()
    mu_gibbs[j] = mu
    
t2 = time.time()
print(f"It took approximately {round(t2-t1,4)} seconds for this to run {K} samples")

beta_gibbs = np.array(beta_gibbs)
sigma_gibbs = np.array(sigma_gibbs)
mu_gibbs = np.array(mu_gibbs)
tau_gibbs = np.array(tau_gibbs)

Mixed_Model_gibbs = np.c_[beta_gibbs,sigma_gibbs,mu_gibbs,tau_gibbs]
MM_titles = np.concatenate([['\u03B2'+str(j) for j in range(0,J)],
                                ['\u03C3','\u03BC','\u03C4']])
param_stats(X = Mixed_Model_gibbs[-int(0.4*K):,],
                       p = [0.025,0.25,0.50,0.75,0.975],
                       names = MM_titles, sig = 2)

### PART D

K = 100000 #How many values we want to draw up

#list to append our samples to
beta_gibbs = [None]*K
sigma_gibbs = [None]*K
lmbda_gibbs = [None]*K

#initial values
beta = MLE #This is good as any guess

#helpful things
N,J = X_star.shape[0], X_star.shape[1]
a = 1 #this represent how many values have improper prior
nu,mu, tau = 4 ,0,1 #t distribution parameters
y_tilde = (y.T).dot(X_star) # do this beforehand so we don't have to do this N9 times
Id = np.concatenate([np.zeros(a),np.ones(J-a)]); Id_mat = np.diag(Id)

t1 = time.time()
for j in range(K):
    
    # Sigma draws
    s2 = ((y - X_star.dot(beta) ).T).dot( y - X_star.dot(beta) )
    sigma = np.sqrt( s2 / chi2(df = N ).rvs() )
    sigma_gibbs[j] = sigma
    
    # Lambda draw
    lmbda_hat_sq = ((beta[a:] - mu)**2).sum() + nu*tau**2
    lmbda = np.sqrt( lmbda_hat_sq / chi2(df = nu+J).rvs() )
    lmbda_gibbs[j] = lmbda

    # Theta draws
    V_beta = np.linalg.inv(Id_mat/lmbda**2 + gram_mat/sigma**2)
    beta_tilde = V_beta.dot(Id*(mu/lmbda**2) + y_tilde/sigma**2)
    beta = multivariate_normal(mean=beta_tilde, cov=V_beta).rvs()
    beta_gibbs[j] = beta
    
t2 = time.time()
print(f"It took approximately {round(t2-t1,4)} seconds for this to run {K} samples")

beta_gibbs = np.array(beta_gibbs)
sigma_gibbs = np.array(sigma_gibbs)
lmbda_gibbs = np.array(lmbda_gibbs)

tprior_gibbs = np.c_[beta_gibbs,sigma_gibbs,lmbda_gibbs]
tprior_titles = np.concatenate([['\u03B2'+str(j) for j in range(0,J)],
                            ['\u03C3','\u03BB']])
param_stats(X = tprior_gibbs[-int(0.4*K):,],
                            p=[0.025,0.25,0.50,0.75,0.975],
                            names = tprior_titles, sig = 2)

# Generalized t4 prior
#list to append our samples to
beta_gibbs2 = [None]*K
sigma_gibbs2 = [None]*K
lmbda_gibbs2 = [None]*K
mu_gibbs = [None]*K
tau_gibbs = [None]*K

#initial values
beta = MLE #This is good as any guess
lmbda = 3
mu = beta.mean()

def tau_pdf(x,df,s): 
     p = (df/2)*np.log(x) - ( (df*x)/2)*(1/s**2)
     return p
#Grid for sigma 0 and nu
tau_grid = np.linspace(start = 1, stop = 500, num = 2000)
d_tau = np.diff(tau_grid)[0]/2

#helpful things
N,J = X_star.shape[0], X_star.shape[1]
a = 1 #this represent how many values have improper prior
nu = 4 #t distribution parameters
y_tilde = (y.T).dot(X_star) # do this beforehand so we don't have to do this N9 times
Id = np.concatenate([np.zeros(a),np.ones(J-a)]); Id_mat = np.diag(Id)

t1 = time.time()
for j in range(K):
    
    # Sigma draws
    s2 = ((y - X_star.dot(beta) ).T).dot( y - X_star.dot(beta) )
    sigma = np.sqrt( s2 / chi2(df = N ).rvs() )
    sigma_gibbs2[j] = sigma
    
    # tau draws
    tau_p = tau_pdf(x=tau_grid,df = nu,s = sigma)
    tau_p = np.exp(tau_p) / (np.exp(tau_p)).sum()
    tau = np.sqrt( np.random.choice(tau_grid,p = tau_p) + (d_tau)*rand() )
    tau_gibbs[j] = tau

    # Lambda draw
    lmbda_hat_sq = ((beta[a:] - mu)**2).sum() + nu*tau**2
    lmbda = np.sqrt( lmbda_hat_sq / chi2(df = nu+J).rvs() )
    lmbda_gibbs2[j] = lmbda
    
    # Mu Draw
    mu = norm(loc = (beta[1:]).mean(),scale = tau / np.sqrt(J)).rvs()
    mu_gibbs[j] = mu

    # Beta draws
    V_beta = np.linalg.inv(Id_mat/lmbda**2 + gram_mat/sigma**2)
    beta_tilde = V_beta.dot(Id*(mu/lmbda**2) + y_tilde/sigma**2)
    beta = multivariate_normal(mean=beta_tilde, cov=V_beta).rvs()
    beta_gibbs2[j] = beta
    
t2 = time.time()
print(f"It took approximately {round(t2-t1,4)} seconds for this to run {K} samples")

beta_gibbs2 = np.array(beta_gibbs2)
sigma_gibbs2 = np.array(sigma_gibbs2)
lmbda_gibbs2 = np.array(lmbda_gibbs2)
mu_gibbs = np.array(mu_gibbs)
tau_gibbs = np.array(tau_gibbs)

gtprior_gibbs = np.c_[beta_gibbs2,sigma_gibbs2,lmbda_gibbs2,mu_gibbs,tau_gibbs]

gtprior_titles = np.concatenate([['\u03B2'+str(j) for j in range(0,J)],
                            ['\u03C3','\u03BB','\u03BC','\u03C4']])

param_stats(X = gtprior_gibbs[-int(0.4*K):,],
                            p=[0.025,0.25,0.50,0.75,0.975],
                            names = gtprior_titles, sig = 2)





