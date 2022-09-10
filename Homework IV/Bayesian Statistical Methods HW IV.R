library(ggplot2) #this makes better looking plots in R
theme_set(theme_minimal())
library(patchwork) #for plot manipulatios (i.e. subplots)
library(Cairo) #Windows is bad at makinf good ggplots so this helps with resolution
library(tidyr)
library(mvtnorm) #let's us use multivariate normal (could also use MASS library)
library(DirichletReg) #for use in the Gibbs Sampler in Problem 8
library(gridExtra) #For printing dataframes nicely

'
Use this pieces of code to get smooth ggplots for windows. I dont like this 
method since it uses too much memory in declaring, but do this to save your plots
'
 #sort( sapply(ls(),function(x){object.size(get(x))})) 

#ggsave(county.plot,path="~/Bayesian Statistical Methods/HW4/Figures",
#       filename = 'County plots.png', dpi = 300, type = 'cairo',
#       width = 11, height = 8, units = 'in',bg="white")


#This function is used to make computing our statistics easier
quantities <- function(x,p,params,sigfig){
  '
    PARAMETERS:
        X - data matrix
        p - the quantiles you wish to calculate
        names - names you want to give the parameters
        sigfig - number of significant digits you want
    Return:
        stats - a datframe detailing statistics about the parameters
    '
  stats <- data.frame(mean = apply(x,2,mean),
                      std = apply(x,2,sd) )
  
  #add quantile columns
  for (y in p){stats[paste0(100*y,"%")] <- apply(x,2,quantile,probs=y)}
  rownames(stats) <- params
  return( round(stats,sigfig) )
}

### Problem 1

ImpSampler <- function(nsamples,logTarget,logProposal,proposal,ControlConstant='NULL'){
  '
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
    n_eff : effective smaple size
    acceptanceRate : acceptance rate of the samples
  '

#importance sampling to calculate weights
samples <- sapply(1:nsamples, function(x) proposal())
#use this to generalize dimensions
if(length(proposal())==1){
  logweights <- sapply(samples,function(x) logTarget(x) - logProposal(x))
}
else{
  logweights <- apply(samples,2,function(x) logTarget(x) - logProposal(x))
}

if (missing(ControlConstant)){
  weights <- exp( logweights )
  weights.norm <- weights/ sum(weights) #return the normalized weights
  #Calculation of effective sample size
  ESS <- ceiling( 1 / sum( (weights.norm)^2   ) )
  return( list(samples=samples,weights=weights.norm,ess=ESS) )
  }
else if(ControlConstant > 0){
  log.u <- log(runif(nsamples)) #uniforms we use to reject or sample
  log.probs <- pmin(0,logweights - log(ControlConstant) )
  accepted <- log.probs > log.u
  acc.samples <- samples[accepted]; acc.logweights <- (logweights-log.probs)[accepted]
  acc.weights <- exp(acc.logweights) 
  acc.weights.norm <- acc.weights/ sum(acc.weights) #return the normalized weights
  ESS <- ceiling( 1 / sum( (acc.weights.norm)^2   ) )
  accept <- length(acc.samples) / nsamples
  return(list(samples=acc.samples,weights=acc.weights.norm,ess=ESS,a_c = accept) )
}

else{
  print("ControlConstant must be a positive number.")
}

}

#Part B
# Target density
log.f <- function(x){ log( 1/3*dnorm(x,mean=-2,sd=1)+2/3*dnorm(x,mean=2,sd=1) ) }
# Proposal density
log.g <- function(x){dnorm(x,0,3,log=TRUE) }
# Simulate from proposal density
proposer <- function() { rnorm(1,mean=0,sd=3) }

x <- seq(from=-10, to = 10, length= 1000)
impdata <- data.frame(x = x, f = exp(log.f(x)), g = exp( log.g(x) ))
impdata <- impdata %>% gather(densities,p,-x)

ggplot(data=impdata) + aes(x=x)+
  geom_line(aes(x=x,y = p,color=densities,group=densities),size=2) + 
  theme(axis.line = element_line(colour = "black",size=2),
        text = element_text(size=20),
        axis.text = element_text(colour = "black",size = 20,face="bold"),
        axis.title = element_text(size = 24,face="bold"),
        axis.ticks.length=unit(.25, "cm"),
        axis.ticks = element_line(colour = "black", size = 1.5),
        legend.background = element_blank())+
  scale_color_manual(name=" Density Functions ", 
                     values=c("blue","red"),
                     labels=c("Target (f(x))","Proposal (g(x))"))+
  ylab("Probability Density")+
  xlab("x")

#PART E
null.sampler <- ImpSampler(nsamples = 5000,logTarget = log.f,
           logProposal = log.g,proposal = proposer)

#Calculate E(X), E(X^2) and E(e^X)
est.vals <- c(sum(null.sampler[["samples"]]*null.sampler[["weights"]]),
              sum(null.sampler[["samples"]]^2*null.sampler[["weights"]]),
              sum(exp(null.sampler[["samples"]])*null.sampler[["weights"]]))

#actual values, not the estimated ones.
real.vals <- c(2/3,5,(1/3)*exp(-3/2) + (2/3)*exp(5/2))

sprintf("The Estimated Values of our expected values are %s",
        paste0(round(est.vals,4), collapse = ', '))

sprintf("The True Values of our expected values are %s ",
        paste0(round(real.vals,4), collapse = ', '))

sprintf("The Effective Sample Size is %s ",null.sampler[["ess"]])

# PART F
c <- 10
reported <- matrix(NA,ncol = 5, nrow = c)
colnames(reported) <- c("\u03BC 1 error", "\u03BC 2 error",
                        "\u03B8 error","Effective Sample Size",
                        "Acceptance Rate")
rownames(reported) <- sapply(1:c, function(x) paste0("c = ", x))
for (j in 1:c){
  imp <- ImpSampler(nsamples = 5000,logTarget = log.f,logProposal = log.g,
                    proposal = proposer, ControlConstant = j)
  #estimated values
  est.vals <- c(sum(imp[["samples"]]*imp[["weights"]]),
                sum(imp[["samples"]]^2*imp[["weights"]]),
                sum(exp(imp[["samples"]])*imp[["weights"]]) ) 
  error <- round(abs(est.vals - real.vals),3)
  
  reported[j,] <- c(error,imp[["ess"]],imp[["a_c"]])
}
grid.table(reported)

labels <- c(expression(mu[1]~Error),expression(mu[2]~Error),
            expression(theta~Error),"ESS","Acceptance Rate")
col <- c("red","blue","yellow","green","orange")

Impsampler.plots <- lapply(1:ncol(reported), function(j){
  ggplot() +
    aes(x = 1:c,y = reported[,j])+
    scale_x_continuous(breaks=seq(1,10,by=2))+ #breaks the x-axis into pieces
    geom_point(color=col[j],size=4)+
    theme(axis.line = element_line(colour = "black",size=2),
          text = element_text(size=20),
          axis.text = element_text(colour = "black",size = 20,face="bold"),
          axis.title = element_text(size = 24,face="bold"),
          axis.ticks.length=unit(.25, "cm"),
          axis.ticks = element_line(colour = "black", size = 1.5))+
    ylab(labels[j])+
    xlab("Rejection Constant")}
)

(Impsampler.plots[[1]] |Impsampler.plots[[2]] | Impsampler.plots[[3]]) / 
    (Impsampler.plots[[4]] |  Impsampler.plots[[5]] ) +
  plot_annotation(tag_levels = 'A')

### PROBLEM 2 (BDA 3rd Ed., Exercise 10.5)

set.seed(123) #do this for reproducibility
J <- 10
alpha.t <- 2*rt(n=1,df=4) #recall t_n(mu,sigma^2) is mu+sigma*t_n
beta.t <- rt(n=1,df=4)
x.sample <- runif(J, min = 0, max = 1)
n.sample <- rpois(J, 5) + 1 #shifts so all values are positive
y.sample <- rbinom(J, size = n.sample, prob = 1 / (1+exp(-alpha.t-beta.t*x.sample)) )

### STORES OUR INFORMATION
sim.data  = data.frame(
  Trials  = n.sample,
  Events = y.sample,
  Dose =  x.sample
)

# PART B

#prior distribution for independent t's
t.prior <- function(x){
  f.alpha <- (1/2)*dt(x[1]/2,df=4); f.beta <- dt(x[2],df=4)
  return(f.alpha*f.beta)
}

#posterior function (generalized as to apply to problem's 10.5, 10.8, 11.2)
bioassay <- function(v, data,prior) {
  '
    MODEL PARAMETERS:
        v - (alpha,beta) tuple
        data - the y, n and x values
        prior - prior distribution to be used for this model
        
    Returns:
      natural log of the unnormalized posterior density
    '
  a <- v[1]; b <- v[2] #our parameters
  x <- data[,"Dose"]; y <- data[,"Events"]; n <- data[,"Trials"] #our data
  theta <- a+b*data[,"Dose"]
  loglik <- sum(data[,"Events"]*theta - data[,"Trials"]*log1p( exp(theta) ) )
  logpost <- log(prior(v)) + loglik
  return(logpost)
}

optim.output <- optim(c(0,0), 
                function(x) bioassay(x,sim.data,t.prior)-log(t.prior(x)),
                control=list(fnscale=-1), hessian=T)
M <- optim.output$value

num <-  1000 #how many samples we want
counter <- 0 #count to tell us when to stop
iters <- 0 #Tells us how many times it took to get to 1000
sample.draws <- as.data.frame(matrix(NA,nrow=num,ncol=2) )#ALWAYS PRE-ALLOCATE
colnames(sample.draws) <- c("x","y")
system.time(while( counter < num){
  draw  <- c(2*rt(n=1,df=4),rt(n=1,df=4)) #your draw from your proposal dist.
  p <- bioassay(draw,sim.data,t.prior) - log(t.prior(draw)) - M
  
  if(p > log(runif(1,0,1)) ){
    counter <- counter + 1 #counts how many successes so we know when to stop
    sample.draws[counter,] <- draw #ADD DRAW TO matrix
  }
  iters <- iters + 1
})

sprintf("It took %s runs to produce %s samples",iters,counter)

#graph the posterior distribution

alphas <- seq(from = -10, to = 10, length = 500)
betas <- seq(from = -10, to = 10, length = 500)

sim.grid <- data.frame(alpha = rep(alphas, times = length(betas)), 
                       beta =rep(betas, each = length(alphas)))
#calculates log posterior
sim.grid["post"] <-  exp(apply(sim.grid[,1:2],1,bioassay,data=sim.data,prior=t.prior))
sample.draws["z"] <- exp(apply(sample.draws[,1:2],1,bioassay,data=sim.data,prior=t.prior))

#contour levels
levels <- c(0.001,0.01,0.05,0.25,0.50,0.75,0.95)
sim.cont <- quantile(seq(min(sim.grid$post),max(sim.grid$post),
                         length.out = 1e5),levels)

sim.plot <- ggplot(sim.grid, aes(x=alpha, y= beta, z=post))+
  stat_contour(breaks= sim.cont,color="black",size = 2)+ #contour levels
  coord_cartesian(xlim = c(-10,5), ylim = c(-10,10)) +
  scale_fill_gradient(low = 'yellow', high = 'red', guide = "none") +
  scale_alpha(range = c(0, 1), guide = "none")+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), 
        axis.line = element_line(colour = "black",size=2),
        text = element_text(size=20),
        axis.text = element_text(colour = "black",size = 20,face="bold"),
        axis.title = element_text(size = 24,face="bold"),
        axis.ticks.length=unit(.25, "cm"),
        axis.ticks = element_line(colour = "black", size = 1.5))+
  ylab(~ paste(beta))+
  xlab(~ paste(alpha))

(sim.plot | sim.plot+geom_point(aes(x=x,y=y,z=z),sample.draws,colour="red",size=4)) + 
  plot_annotation(tag_levels = 'A')

quantities(x = sample.draws[,1:2], 
           p = c(0.025,0.25,0.50,0.75,0.975),
           params = c('\u03B1','\u03B2'),sigfig=2)

# PART C

optimus <- optim(c(0,0), 
                      function(x) bioassay(x,sim.data,t.prior),
                      control=list(fnscale=-1), hessian=T)
#set up the normal approximation parameters
norm.mu <- optimus$par
norm.sig <- solve(-optimus$hessian)
sim.grid["norm"] <- dmvnorm(sim.grid[,1:2],mean = norm.mu, sigma = norm.sig)          

sim.norm.cont <- quantile(seq(min(sim.grid$norm),max(sim.grid$norm),
                         length.out = 1e5),levels)

sim.norm.plot <- ggplot(sim.grid, aes(x=alpha, y= beta, z=norm))+
  stat_contour(breaks = sim.norm.cont,color="blue",size = 2)+ #contour levels
  coord_cartesian(xlim = c(-10,5), ylim = c(-10,10)) +
  scale_fill_gradient(low = 'yellow', high = 'red', guide = "none") +
  scale_alpha(range = c(0, 1), guide = "none")+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), 
        axis.line = element_line(colour = "black",size=2),
        text = element_text(size=20),
        axis.text = element_text(colour = "black",size = 20,face="bold"),
        axis.title = element_text(size = 24,face="bold"),
        axis.ticks.length=unit(.25, "cm"),
        axis.ticks = element_line(colour = "black", size = 1.5))+
  ylab(~ paste(beta))+
  xlab(~ paste(alpha))

(sim.plot | sim.norm.plot) + plot_annotation(tag_levels = 'A')

# PART D
sim.target <- function(x) bioassay(x,sim.data,t.prior)
sim.proposal <- function(x) dmvt(x, delta = norm.mu, sigma = norm.sig,df=4,log=T)
sim.proposer <- function() rmvt(n=1,delta = norm.mu, sigma = norm.sig,df=4)

sim.sampler <- ImpSampler(nsamples = 1000,logTarget = sim.target,
                           logProposal = sim.proposal,proposal = sim.proposer)

alpha.mean <- sum(sim.sampler[["samples"]][1,]*sim.sampler[["weights"]])
beta.mean <- sum(sim.sampler[["samples"]][2,]*sim.sampler[["weights"]])

sprintf("The Estimated value for E(\u03B1|y) is %s",round(alpha.mean,4))
sprintf("The Estimated value for E(\u03B2|y) is %s",round(beta.mean,4))

# PART E

sprintf("The Effective Sample Size is %s",sim.sampler[["ess"]])

### PROBLEM 3 (BDA 3rd Ed., Exercise 10.8)

#Bioassay data
bioassay.df  = data.frame(
  Trials  = c(5,5,5,5),
  Events = c(0,1,3,5),
  Dose =  c(-0.86,-0.30,-0.05,0.73)
)

#posterior function (generalized as to apply to problem's 10.5, 10.8, 11.2)
bioassay <- function(v, data,prior) {
  '
    MODEL PARAMETERS:
        v - (alpha,beta) tuple
        data - the y, n and x values
        prior - prior distribution to be used for this model
        
    Returns:
      natural log of the unnormalized posterior density
    '
  a <- v[1]; b <- v[2] #our parameters
  x <- data[,"Dose"]; y <- data[,"Events"]; n <- data[,"Trials"] #our data
  theta <- a+b*data[,"Dose"]
  loglik <- sum(data[,"Events"]*theta - data[,"Trials"]*log1p( exp(theta) ) )
  logpost <- log(prior(v)) + loglik
  return(logpost)
}

#prior for bioassay in section 3.7
ind.prior <- function(x) 1

optim.output <- optim(c(0,0), function(x) bioassay(x,bioassay.df,ind.prior),
                      control=list(fnscale=-1), hessian=T)

bio.mu <- optim.output$par
bio.sig <- solve(-optim.output$hessian)

S <- 10000
bio.target <- function(x) bioassay(x,bioassay.df,ind.prior)
bio.proposal <- function(x) dmvnorm(x, mean=bio.mu, sigma = bio.sig,log=T)
bio.proposer <- function(x) rmvnorm(n=1, mean=bio.mu, sigma = bio.sig)
#importance re-sampling

#Importance Sampling
bio.sampler <- ImpSampler(nsamples = S,logTarget = bio.target,
                           logProposal = bio.proposal,proposal = bio.proposer)

without <- sample(1:S,size=1000,replace=F,prob = bio.sampler[["weights"]])
with <- sample(1:S,size=1000,replace=T,prob = bio.sampler[["weights"]])

#subset based on chosen indices
imp.with <- as.data.frame( t(bio.sampler[["samples"]])[with,] )
imp.without <- as.data.frame( t(bio.sampler[["samples"]])[without,] )
colnames(imp.with) <- colnames(imp.without) <- c("x","y")

bio.alphas <- seq(from = -5, to = 10, length = 500)
bio.betas <- seq(from = -10, to = 45, length = 500)

bio.grid <- data.frame(alpha = rep(bio.alphas, times = length(bio.betas)), 
                       beta = rep(bio.betas, each = length(bio.alphas)))
#add posterior to dataframes
bio.grid["post"] <- exp(apply(bio.grid,1,bioassay,data=bioassay.df,prior=ind.prior))
imp.with["z"] <- exp(apply(imp.with[,1:2],1,bioassay,data=bioassay.df,
                           prior=ind.prior))
imp.without["z"] <- exp(apply(imp.without[,1:2],1,bioassay,data=bioassay.df,
                              prior=ind.prior))

# This is another Posterior plot, but this has the simulated points built on top of it

#contour levels for flight posterior
bio.cont <- quantile(seq(min(bio.grid$post),max(bio.grid$post),length.out=1e5),levels)

bio.plot <- ggplot(bio.grid, aes(x = alpha, y = beta, z=post))+
  stat_contour(breaks= bio.cont,color="black",size = 2)+ #contour levels
  scale_fill_gradient(low = 'yellow', high = 'red', guide = "none") +
  scale_alpha(range = c(0, 1), guide = "none")+
  theme(axis.line = element_line(colour = "black",size=2),
        text = element_text(size=20),
        axis.text = element_text(colour = "black",size = 20,face="bold"),
        axis.title = element_text(size = 24,face="bold"),
        axis.ticks.length=unit(.25, "cm"),
        axis.ticks = element_line(colour = "black", size = 1.5))+
  ylab(~ paste(beta))+
  xlab(~ paste(alpha))

(bio.plot+geom_point(aes(x=x,y=y,z=z),imp.with,colour="blue",size=4)
 | bio.plot+geom_point(aes(x=x,y=y,z=z),imp.without,colour="red",size=4))+
  plot_annotation(tag_levels = 'A')

#Distribution of importance ratios
bio_hist <- ggplot() +aes(x=bio.sampler[["weights"]]) + 
  geom_histogram(color="black", fill="yellow")+
  theme(axis.line = element_line(colour = "black",size=2),
        text = element_text(size=20),
        axis.text = element_text(colour = "black",size = 20,face="bold"),
        axis.title = element_text(size = 24,face="bold"),
        axis.ticks.length=unit(.25, "cm"),
        axis.ticks = element_line(colour = "black", size = 1.5))+
  ylab("Frequency")+
  xlab(~ paste(tilde(w),"(",theta,")"))

bio_boxplot <- ggplot() + aes(y=bio.sampler[["weights"]]) + 
  geom_boxplot(color="black", fill="yellow")+
  theme(axis.line = element_line(colour = "black",size=2),
        text = element_text(size=20),
        axis.text = element_text(colour = "black",size = 20,face="bold"),
        axis.title = element_text(size = 24,face="bold"),
        axis.ticks.length=unit(.25, "cm"),
        axis.ticks = element_line(colour = "black", size = 1.5))+
  ylab("Frequency")+
  xlab(~ paste(tilde(w),"(",theta,")"))
(bio_hist | bio_boxplot)+
  plot_annotation(tag_levels = 'A')

### PROBLEM 4 (BDA 3rd Ed., 11.2)


#posterior function (generalized as to apply to problem's 10.5, 10.8, 11.2)
bioassay <- function(v, data,prior) {
  '
    MODEL PARAMETERS:
        v - (alpha,beta) tuple
        data - the y, n and x values
        prior - prior distribution to be used for this model
        
    Returns:
      natural log of the unnormalized posterior density
    '
  a <- v[1]; b <- v[2] #our parameters
  x <- data[,"Dose"]; y <- data[,"Events"]; n <- data[,"Trials"] #our data
  theta <- a+b*data[,"Dose"]
  loglik <- sum(data[,"Events"]*theta - data[,"Trials"]*log1p( exp(theta) ) )
  logpost <- log(prior(v)) + loglik
  return(logpost)
}

#prior for bioassay in section 3.7
ind.prior <- function(x) 1

optim.output <- optim(c(0,0), function(x) bioassay(x,bioassay.df,ind.prior),
                      control=list(fnscale=-1), hessian=T)

bio.mu <- optim.output$par
bio.sig <- solve(-optim.output$hessian)

#initial draw 
init.draw <- bio.mu
#variance scaling parameter
c <- 2.4 / sqrt(2) # this is supposed to help with computation
D <- 50000 #How many samples we are going to draw
count <- 0 #Tells us how many values we accept

#pre-allocate spaceas this will make it go faster than having to append
bio.draws <- matrix(NA,ncol=2,nrow=D)
#do this externally so we don't have to continuously do this in the loop process
r.bottom <- bioassay(v = init.draw, data = bioassay.df, prior = ind.prior)
log.u <- log( runif(D) ) # call this external to the loop

###METROPOLIS HASTINGS ALGORITHM

#LOOP FOR HOWEVER MANY SAMPLES YOU WANT TO GAIN. HERE I LOOP FOR D (=20,000)
system.time( for (j in 1:D){
  
  #sample proposal theta* using jumping distribution J_t(theta* | theta^{t-1})
  #here, proposal = theta*, and theta^{t-1} = init.draw
  proposal <- rmvnorm(1, mean = init.draw, sigma = c^2*bio.sig)
  
  'calculate log ratio of densities where r is defined as
  r = [ p(theta* | y)/J_t(theta* | theta^{t-1}) ] / [ p(theta^{t-1} | y)/J_t(theta^{t-1} | theta*) ].
  NOTE: For metropolis hastings, the jumping rule need not be symmetric
  '
  
  r.top <- bioassay(v = proposal, data = bioassay.df, prior = ind.prior)
  #with this ratio probability, we either accept or reject.  
  r <- min(r.top - r.bottom, 0)
  
  #If r > log( runif(1) ), we accept the new draw
  if (r > log.u[j] ){
    init.draw <- proposal
    r.bottom <- r.top #if accepted we move on. If not, we stay at same place until we accept
    count <- count +1 #points for acceptance
    
  }
  #otherwise we reject, and try again
  bio.draws[j,]  <- init.draw #regardless of acc/rej, append new sample
} )

sprintf("Our acceptance rate is %.4s", 100*count/D)

# TRACE PLOTS

col <- c("cyan","magenta")
P4.labels <- c(expression(alpha),expression(beta))

MCMC.trace_plots <- lapply(1:ncol(bio.draws), function(j){
  ggplot() +
    aes(x = 1:D,y = bio.draws[,j])+
    geom_line(color=col[j],size=2)+
    theme(axis.line = element_line(colour = "black",size=2),
          text = element_text(size=20),
          axis.text = element_text(colour = "black",size = 20,face="bold"),
          axis.title = element_text(size = 24,face="bold"),
          axis.ticks.length=unit(.25, "cm"),
          axis.ticks = element_line(colour = "black", size = 1.5))+
    ylab(P4.labels[j])+
    xlab("Iterations")}
)

(MCMC.trace_plots[[1]] | MCMC.trace_plots[[2]]) + 
  plot_annotation(tag_levels = 'A')

# Cumulative Average Plots
mean.bio.draws <- apply(bio.draws,2,function(x) cumsum(x) / seq_along(x) )

MCMC.cumavg_plots <- lapply(1:ncol(mean.bio.draws), function(j){
  ggplot() +
    aes(x = 1:D,y = mean.bio.draws[,j])+
    geom_line(color=col[j],size=2)+
    theme(axis.line = element_line(colour = "black",size=2),
          text = element_text(size=20),
          axis.text = element_text(colour = "black",size = 20,face="bold"),
          axis.title = element_text(size = 24,face="bold"),
          axis.ticks.length=unit(.25, "cm"),
          axis.ticks = element_line(colour = "black", size = 1.5))+
    ylab(P4.labels[j])+
    xlab("Iterations")}
)

(MCMC.cumavg_plots[[1]] | MCMC.cumavg_plots[[2]]) + 
  plot_annotation(tag_levels = 'A')

# Autocorrelation Plots
autocor.bio.draws <- apply(bio.draws,2,function(x) acf(x,plot=F,lag.max=100)$acf)

autocor.bio.draws <- data.frame(x = 1:101, alpha = autocor.bio.draws[,1],
                                beta = autocor.bio.draws[,2])

autocor.bio.draws <- autocor.bio.draws %>% gather(params,p,-x)

ggplot(data=autocor.bio.draws) + aes(x=1:101)+
  geom_line(aes(x=x,y = p,color=params,group=params),size=2) + 
  theme(axis.line = element_line(colour = "black",size=2),
        text = element_text(size=20),
        axis.text = element_text(colour = "black",size = 20,face="bold"),
        axis.title = element_text(size = 24,face="bold"),
        axis.ticks.length=unit(.25, "cm"),
        axis.ticks = element_line(colour = "black", size = 1.5),
        legend.background = element_blank(),
        legend.text = element_text(size = 28, face = "bold"),
        legend.title = element_text(size = 22,face="bold"))+
  scale_color_manual(name=" Parameter ", values=c("cyan","magenta"),
                     labels=c('\u03B1','\u03B2'))+
  ylab("Autocorrelation Plot")+
  xlab("Iterations")

# Contour Plots + Statistics

#add posterior to dataframe
accepted.bio <- as.data.frame(bio.draws[tail(1:D,ceiling(0.4*D)),])
colnames(accepted.bio) <- c("x","y")
accepted.bio["z"] <- exp(apply(accepted.bio[,1:2],1,bioassay,data=bioassay.df,
                           prior=ind.prior))

(bio.plot | bio.plot+geom_point(aes(x=x,y=y,z=z),accepted.bio,colour="red",size=4))+
  plot_annotation(tag_levels = 'A')

# LD50 Histogram

ggplot() + aes(x=-accepted.bio[,1]/accepted.bio[,2]) + 
  geom_histogram(color="black", fill="blue",binwidth=0.02)+
  coord_cartesian(xlim = c(-0.5, 0.5))+
  theme(axis.line = element_line(colour = "black",size=2),
        text = element_text(size=24),
        axis.text = element_text(colour = "black",size = 24,face="bold"),
        axis.title = element_text(size = 30,face="bold"),
        axis.ticks.length=unit(.25, "cm"),
        axis.ticks = element_line(colour = "black", size = 1.5))+
  ylab("Frequency")+
  xlab("LD50")

### PROBLEM 5 (BDA 3rd Ed., Exercise 11.3)

machine <- list(M1 = c(83,92,92,46,67),
                     M2 = c(117,109,114,104,87),
                     M3 = c(101,93,92,86,67),
                     M4 = c(105,119,116,102,116),
                     M5 = c(79,97,103,79,92),
                     M6 = c(57,92,104,77,100) )

#statistics about the data
machine.n <-  sapply(machine,length) # number of data points in each machine
machine.mean <-  sapply(machine,mean)
machine.var <-  sapply(machine, var)

## Pooled Model

pooled <- unlist(machine)
pooled.var <- var(pooled)
pooled.mean <- mean(pooled)

#marginal posterior pdf's for mu and sigma.
sig.pooled <- sqrt(( (length(pooled)-1)*pooled.var) / (rchisq(1000, length(pooled)-1)) )
theta.pooled <- rnorm(1000, pooled.mean, sig.pooled/sqrt(length(pooled) ) )


#Quantiles+ other statistics
quantities(x = cbind(theta.pooled,sig.pooled), 
           p = c(0.025,0.25,0.50,0.75,0.975),
           params = c('\u03B8','\u03C3'), sigfig = 2)


#Histograms of theta_6 |y, tilde y_6 | y, and theta_7 | y.
post_labels <- c(expression(paste(theta[6],"|y")),
                 expression(paste(tilde(y)[6],"|y")),
                 expression(paste(theta[7],"|y")) )

col <- c('purple','gold','green')

pooled_dist <- cbind(theta.pooled,
                 rnorm(n=1000, mean = theta.pooled, sd = sig.pooled),
                 rnorm(n=1000, mean = pooled.mean, sd = sig.pooled/ sqrt(length(pooled))))

pooled_plots <- lapply(1:3, function(j){
  ggplot()+
  aes(x=pooled_dist[,j]) + 
    geom_histogram(color="black", fill=col[j]) +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(), 
          axis.line = element_line(colour = "black",size=2),
          text = element_text(size=20),
          axis.text = element_text(colour = "black",size = 20,face="bold"),
          axis.title = element_text(size = 24,face="bold"),
          axis.ticks.length=unit(.25, "cm"),
          axis.ticks = element_line(colour = "black", size = 1.5))+
    ylab("Frequency") +
    xlab(post_labels[j])
  } )


( (pooled_plots[[1]] | pooled_plots[[2]]) / 
    (plot_spacer() + pooled[[3]] + plot_spacer() + plot_layout(widths = c(1,2,1))) ) + 
  plot_annotation(tag_levels = 'A')

## Separate Model
sig.sep <- sqrt( sum( (machine.n-1)*machine.var ) / rchisq(n=1000,df = sum(machine.n) ))
mu.sep <- sapply(machine, function(x) rnorm(1000, mean(x), sig.sep/sqrt(length(x)) ) )

quantities(x = cbind(mu.sep,sig.sep), 
           p = c(0.025,0.25,0.50,0.75,0.975),
           params = c(sapply(1:length(machine), function(x) paste0('\u03B8',x))
                      ,'\u03C3'), sigfig = 2)

sep_dist <- cbind(mu.sep[,6], rnorm(n=1000, mean = mu.sep[,6], sd = sig.sep) )

sep_plots <- lapply(1:2, function(j){
  ggplot()+
    aes(x=sep_dist[,j]) + 
    geom_histogram(color="black", fill=col[j]) +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(), 
          axis.line = element_line(colour = "black",size=2),
          text = element_text(size=20),
          axis.text = element_text(colour = "black",size = 20,face="bold"),
          axis.title = element_text(size = 24,face="bold"),
          axis.ticks.length=unit(.25, "cm"),
          axis.ticks = element_line(colour = "black", size = 1.5))+
    ylab("Frequency") +
    xlab(post_labels[j])
} )


(sep_plots[[1]] | sep_plots[[2]]) + plot_annotation(tag_levels = 'A')

## Hierarchical Model

N5 <- 50000 #How many values we want to draw up
P5_indices <- tail(1:N5,ceiling(0.4*N5)) #indices for values after burnin

#list to append our samples to
theta.gibbs <- matrix(NA,nrow=N5,ncol=6)
sigma.gibbs <-NULL
mu.gibbs <- NULL
tau.gibbs <- NULL

#initial values
theta <- c(72,75,87,91,69,73)
mu <- mean(theta)

#helpful things
J <- length(theta)

system.time( for (j in 1:N5){
  # Sigma draws
  sigma.hat.sq <- sum( mapply ("-", machine , theta )^2)
  sigma <- sqrt( sigma.hat.sq / rchisq(n=1,df = sum(machine.n) ))
  sigma.gibbs[j] <- sigma
  
  # Tau draw
  tau <- sqrt( sum((theta - mu)^2) / (rchisq(n=1,df = J - 1)) )
  tau.gibbs[j] <- tau
  
  # Theta draws
  V <- 1 / ( (1/tau)^2 + (machine.n/(sigma^2)) )
  theta.hat <- (mu*(1/tau)^2 + machine.mean*machine.n*(1/sigma)^2)*V
  theta <- rnorm(n = J,mean = theta.hat, sd = sqrt(V))
  theta.gibbs[j,] <- theta
  
  # Mu Draw
  mu <- rnorm(n = 1, mean = mean(theta), sd = tau / sqrt(J) )
  mu.gibbs[j] <- mu
} )

gibbs.samples <- cbind(theta.gibbs,sigma.gibbs,mu.gibbs,tau.gibbs)

#use this to show posterior plots
colnames(gibbs.samples) <- c(sapply(1:length(machine), function(x) paste0("theta ",x)),
                             "sigma","mu", "tau")

P5.labels <- c(expression(theta[1]),expression(theta[2]),expression(theta[3]),
               expression(theta[4]),expression(theta[5]),expression(theta[6]),
               expression(sigma),expression(mu), 
               expression(tau))

gibbs.col <- c("cyan","magenta","red","blue","yellow","green","orange","hotpink","brown")

gibbs.plots <- lapply(1:ncol(gibbs.samples), function(j){
 ggplot() +
  aes(x = 1:N5,y = gibbs.samples[,j])+
  geom_line(color=gibbs.col[j],size=2)+
  theme(axis.line = element_line(colour = "black",size=2),
        text = element_text(size=20),
        axis.text = element_text(colour = "black",size = 20,face="bold"),
        axis.title = element_text(size = 24,face="bold"),
        axis.ticks.length=unit(.25, "cm"),
        axis.ticks = element_line(colour = "black", size = 1.5))+
  ylab( P5.labels[j] )+
  xlab("Iterations")}
)

( (gibbs.plots[[1]] |gibbs.plots[[2]] | gibbs.plots[[3]]) / 
     (gibbs.plots[[4]] | gibbs.plots[[5]] | gibbs.plots[[6]] ) )+ 
  plot_annotation(tag_levels = 'A')

( (gibbs.plots[[7]] |gibbs.plots[[8]] | gibbs.plots[[9]]) )+ 
  plot_annotation(tag_levels = 'A')

#Autocorrelation Plot
gibbs.auto <-  apply(gibbs.samples,2,function(x) acf(x,plot=F,lag.max=100)$acf) %>%
  as.data.frame() %>% cbind(x = 1:101) %>% gather(params, p, -x)

ggplot(data=gibbs.auto) + aes(x=1:101) +
  geom_line(aes(x=x,y = p,color=params,group=params),size=2) + 
  theme(axis.line = element_line(colour = "black",size=2),
        text = element_text(size=20),
        axis.text = element_text(colour = "black",size = 20,face="bold"),
        axis.title = element_text(size = 24,face="bold"),
        axis.ticks.length=unit(.25, "cm"),
        axis.ticks = element_line(colour = "black", size = 1.5),
        legend.background = element_blank(),
        legend.text = element_text(size = 28, face = "bold"),
        legend.title = element_text(size = 22,face="bold"))+
  scale_color_manual(name=" Parameter ", values=col,
                     labels=P5.labels)+
  ylab("Autocorrelation Plot")+
  xlab("Iterations")


#Summary Statistics
quantities(x = gibbs.samples[P5_indices,], 
           p = c(0.025,0.25,0.50,0.75,0.975),
           params = P5.labels,sigfig=2)

hier_dist <- cbind(theta.gibbs[P5_indices,6],
                   rnorm(n=N5*0.4, mean = theta.gibbs[P5_indices,6], 
                         sd = sigma.gibbs[P5_indices]),
                   rnorm(n=N5*0.4, mean = mu.gibbs[P5_indices], 
                         sd = tau.gibbs[P5_indices]))

hier_plots <- lapply(1:3, function(j){
  ggplot()+
    aes(x=hier_dist[,j]) + 
    geom_histogram(color="black", fill=col[j]) +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(), 
          axis.line = element_line(colour = "black",size=2),
          text = element_text(size=20),
          axis.text = element_text(colour = "black",size = 20,face="bold"),
          axis.title = element_text(size = 24,face="bold"),
          axis.ticks.length=unit(.25, "cm"),
          axis.ticks = element_line(colour = "black", size = 1.5))+
    ylab("Frequency") +
    xlab(post_labels[j])
} )


( (hier_plots[[1]] | hier_plots[[2]]) / 
    (plot_spacer() + hier_plots[[3]] + plot_spacer() + plot_layout(widths = c(1,2,1))) ) + 
  plot_annotation(tag_levels = 'A')

### PROBLEM 6 (BDA 3rd Ed., Exercise 11.4)

N6 <- 50000 #How many values we want to draw up
P6_indices <- tail(1:N6,ceiling(0.4*N6)) #indices for values after burnin

#list to append our samples to
P6.theta.gibbs <- matrix(NA,nrow=N5,ncol=6)
P6.sigma.gibbs <- matrix(NA,nrow=N5,ncol=6)
P6.mu.gibbs <- NULL
P6.tau.gibbs <- NULL
sig0.gibbs <- NULL

#initial values
theta <- c(72,75,87,91,69,73)
mu <- mean(theta)
sig0 <- 8
nu <- 3

sig0.pdf<- function(x,df,s,J){
  p <- (J*df/2 - 1)*log(x) -(x*df/2)*sum(1/s^2)
return( p )
}
#Grid for sigma 0
sig2.0.grid <- seq(from = 1, to = 500, length = 2000)
d.sig0 <- diff(sig2.0.grid)[1]/2

#helpful things
J <- length(theta)

system.time( for (j in 1:N5){
  # Sigma draws
  sigma.hat.sq <- ( nu*sig0**2 + colSums(mapply ("-", machine , theta )^2) ) 
  sigma <- sqrt(sigma.hat.sq / (rchisq(n=length(sigma),df = nu +  machine.n )))
  P6.sigma.gibbs[j,] <- sigma
  
  # sig_0 draws
  sig2.0.pdf <- exp( sig0.pdf(x=sig2.0.grid,df = nu,s = sigma, J = 5) )
  sig2.0 <- sample(sig2.0.grid,size=1,replace=T,prob = sig2.0.pdf) + runif(1,0,d.sig0)
  sig0 <- sqrt(sig2.0)
  sig0.gibbs[j] <- sig0
  
  # Tau draw
  tau.hat.sq <- sum((theta - mu)^2)
  tau <- sqrt( (tau.hat.sq) / (rchisq(n=1,df = J - 1)) )
  P6.tau.gibbs[j] <- tau
  
  # Theta draws
  V <- 1 / ( (1/tau)^2 + (machine.n/(sigma^2)) )
  theta.hat <- ((mu / (tau^2)) + ( (machine.mean*machine.n)/sigma^2))*V
  theta <- rnorm(n = length(theta),mean = theta.hat, sd = sqrt(V))
  P6.theta.gibbs[j,] <- theta

  # Mu Draw
  mu <- rnorm(n = 1, mean = mean(theta), sd = sqrt(tau^2 / J))
  P6.mu.gibbs[j] <- mu
} )

P6.gibbs.samples <- cbind(P6.theta.gibbs,P6.sigma.gibbs,P6.mu.gibbs,
                       P6.tau.gibbs,sig0.gibbs)
#use this to show posterior plots
colnames(P6.gibbs.samples) <- c(sapply(1:length(machine), function(x) paste0("theta ",j)),
                            sapply(1:length(machine), function(x) paste0("sigma ",j)),
                            "mu", "tau", "sigma0")

P6.labels <- c(expression(theta[1]),expression(theta[2]),expression(theta[3]),
               expression(theta[4]),expression(theta[5]),expression(theta[6]),
               expression(sigma[1]),expression(sigma[2]),expression(sigma[3]),
               expression(sigma[4]),expression(sigma[5]),expression(sigma[6]),
               expression(mu),expression(tau),expression(sigma[0]))

P6.col <- c( rep(c("cyan","magenta","red","blue","yellow","green"),2),"orange"
          ,"hotpink","brown")

P6.gibbs.plots <- lapply(1:ncol(P6.gibbs.samples), function(j){
  ggplot() +
    aes(x = 1:N5,y = P6.gibbs.samples[,j])+
    geom_line(color=P6.col[j],size=2)+
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(), 
          axis.line = element_line(colour = "black",size=2),
          text = element_text(size=20),
          axis.text = element_text(colour = "black",size = 20,face="bold"),
          axis.title = element_text(size = 24,face="bold"),
          axis.ticks.length=unit(.25, "cm"),
          axis.ticks = element_line(colour = "black", size = 1.5))+
    ylab( P6.labels[j] )+
    xlab("Iterations")}
)

( (P6.gibbs.plots[[1]] |P6.gibbs.plots[[2]] | P6.gibbs.plots[[3]]) / 
                   (P6.gibbs.plots[[4]] | P6.gibbs.plots[[5]] | P6.gibbs.plots[[6]] ))+ 
  plot_annotation(tag_levels = 'A')

( (P6.gibbs.plots[[7]] |P6.gibbs.plots[[8]] | P6.gibbs.plots[[9]]) / 
                   (P6.gibbs.plots[[10]] | P6.gibbs.plots[[11]] | P6.gibbs.plots[[12]] ))+ 
  plot_annotation(tag_levels = 'A')

( (P6.gibbs.plots[[13]] |P6.gibbs.plots[[14]] | P6.gibbs.plots[[15]]) )+ 
  plot_annotation(tag_levels = 'A')

#Summary Statistics
quantities(x = P6.gibbs.samples[P6_indices,], 
           p = c(0.025,0.25,0.50,0.75,0.975),
           params = P6.labels, sigfig=2)


P6_hier_dist <- cbind(P6.theta.gibbs[P6_indices,6],
                   rnorm(n=N6*0.4, mean = P6.theta.gibbs[P6_indices,6], 
                         sd = P6.sigma.gibbs[P6_indices]),
                   rnorm(n=N6*0.4, mean = P6.mu.gibbs[P6_indices], 
                         sd = P6.tau.gibbs[P6_indices]))

P6_hier_plots <- lapply(1:3, function(j){
  ggplot()+
    aes(x=P6_hier_dist[,j]) + 
    geom_histogram(color="black", fill=col[j]) +
    theme(axis.line = element_line(colour = "black",size=2),
          text = element_text(size=20),
          axis.text = element_text(colour = "black",size = 20,face="bold"),
          axis.title = element_text(size = 24,face="bold"),
          axis.ticks.length=unit(.25, "cm"),
          axis.ticks = element_line(colour = "black", size = 1.5))+
    ylab("Frequency") +
    xlab(post_labels[j])
} )

( (P6_hier_plots[[1]] | P6_hier_plots[[2]]) / 
    (plot_spacer() + P6_hier_plots[[3]] + plot_spacer() + plot_layout(widths = c(1,2,1))) ) + 
  plot_annotation(tag_levels = 'A')

### EXTRA CREDIT: PROBLEM 7 (BDA 3rd Ed., Exercise 13.5)

#data
captured <- c(118,74,44,24,29,22,20,14,20,15,
              12,14,6,12,6,9,9,6,10,10,11,5,3,3)
times <- 1:24 

#Grid for sampling
R <- 500
alphas.animals <- seq(from = 0.001, to = 5, length = R)
betas.animals <- seq(from = 0.001, to = 0.5, length = R)

#Posterior Function. It is of form seen in derivation
animal <- function(w,y,x,prior){
  '
    PARAMETERS:
    ----------
    w - alpha, beta parameter
    x - beta parameter
    y - number of species caught
    prior - prior on alpha, beta
    Returns:
    -------
    logpost : natural logarithm of unnormalized posterior density
  '
  a <- w[1]; b <- w[2] #our parameters
  logpost <- function(a,b,w,x,y){
    logprior <- log(prior(w))
    loglik <- sum(y*(lgamma(a+x) + a*log(b) - lgamma(a) - lgamma(x+1) - (x+a)*log1p(b)))
    return( loglik + logprior)
  }
  #prior distribution used for this problem
  return( ifelse( a >0 & b > 0, logpost(a,b,w,x,y),-Inf) )
}

animal.grid <- data.frame(alpha = rep(alphas.animals, times = length(betas.animals)), 
                          beta = rep(betas.animals, each = length(alphas.animals)))

#prior for bioassay in section 3.7
ind.prior <- function(x) 1

#calculates log posterior
animal.grid["post"] <-  apply(animal.grid[,1:2],1,animal,y=captured,
                              x = times,prior=ind.prior)
animal.grid["post"] <-  exp(animal.grid["post"] - max(animal.grid["post"]))

### Draw samples using grid smapling
#posterior used to select random numbers that correspond to indices of our grid
samples <- sample(nrow(animal.grid), size = 10000,replace = T,
                  prob = c(animal.grid$post) )

#Random jitter
#plug sample values in to give us result + random jitter
d.alpha <- diff(alphas.animals)[1]/2
d.beta <- diff(betas.animals)[1]/2

#I don't declare these anymore to save on memory
animal.samples <- animal.grid[samples,1:2]
animal.samples["alpha"] <- animal.samples["alpha"] +runif(10000,-d.alpha,d.alpha)
animal.samples["beta"] <- animal.samples["beta"] +runif(10000,-d.beta,d.beta)
animal.samples["z"] <- exp(apply(animal.samples[,1:2],1,animal,y=captured,
                                 x = times,prior=ind.prior))

#contour levels
levels <- c(0.001,0.01,0.05,0.25,0.50,0.75,0.95)
animal.cont <- quantile(seq(min(animal.grid$post),max(animal.grid$post),
                            length.out = 1e5),levels)

animal.plot <- ggplot(animal.grid, aes(x=alpha, y= beta, z=post))+
  stat_contour(breaks= animal.cont,color="black",size = 2)+ #contour levels
  scale_fill_gradient(low = 'yellow', high = 'red', guide = "none") +
  scale_alpha(range = c(0, 1), guide = "none")+
  theme(axis.line = element_line(colour = "black",size=2),
        text = element_text(size=20),
        axis.text = element_text(colour = "black",size = 20,face="bold"),
        axis.title = element_text(size = 24,face="bold"),
        axis.ticks.length=unit(.25, "cm"),
        axis.ticks = element_line(colour = "black", size = 1.5))+
  ylab(~ paste(beta))+
  xlab(~ paste(alpha))

(animal.plot | animal.plot+geom_point(aes(x=alpha,y=beta,z=z),animal.samples,colour="red",size=4)) + 
  plot_annotation(tag_levels = 'A')

#PART C

# optimise
optim.animal <- optim(c(1,5), function(w) animal(w,y = captured,x = times,prior = ind.prior),
                      control=list(fnscale=-1),hessian=T)
animal.mu <- optim.animal$par
sprintf("The mode of the posterior is (%.5s, %.5s)", animal.mu[1],animal.mu[2])

animal.sig <- solve(-optim.animal$hessian)
print("The Fisher Information matrix is: ")
round(animal.sig,3)

#PART D

animal_species <- replicate(1000, rnbinom(n=10000,size = animal.samples$alpha,
                                          prob= animal.samples$beta/(animal.samples$beta+1)) )
N <- apply(animal_species,2,function(x) min(which(cumsum(x) > 10000)) )

sprintf("The 95 percent Credible Interval for the new species observed is [%s]",
        paste0(round(quantile(N,probs = c(0.025,0.975))), collapse = ', '))

#PART E
animal_reps <- replicate(496, rnbinom(n=10000,size = animal.samples$alpha,
                                      prob= animal.samples$beta/(animal.samples$beta+1)) )
Test1 <- apply(animal_reps,1,sum)
Test2 <-  apply(animal_reps,1,max)


Test1.hist <- ggplot() +aes(x=Test1) + 
  geom_histogram(color="black", fill="blue")+
  geom_vline(xintercept = sum(times*captured),size=2)+
  annotate("text",x = 3750, y=2000, size = 6,label =
             paste0("p = ",mean(Test1 >= sum(times*captured) )))+
  theme(axis.line = element_line(colour = "black",size=2),
        text = element_text(size=24),
        axis.text = element_text(colour = "black",size = 24,face="bold"),
        axis.title = element_text(size = 30,face="bold"),
        axis.ticks.length=unit(.25, "cm"),
        axis.ticks = element_line(colour = "black", size = 1.5))+
  ylab("Frequency")+
  xlab(~ paste(T(y^{rep})))

Test2.hist <- ggplot() +aes(x=Test2) + 
  geom_histogram(color="black", fill="red")+
  geom_vline(xintercept = max(times),size=2)+
  annotate("text",x = 60, y=1500, size = 6,label =
             paste0("p = ",mean(Test2 >= max(times) )))+
  theme(axis.line = element_line(colour = "black",size=2),
        text = element_text(size=24),
        axis.text = element_text(colour = "black",size = 24,face="bold"),
        axis.title = element_text(size = 30,face="bold"),
        axis.ticks.length=unit(.25, "cm"),
        axis.ticks = element_line(colour = "black", size = 1.5))+
  ylab("Frequency")+
  xlab(~ paste(T(y^{rep})))

(Test1.hist | Test2.hist)  + plot_annotation(tag_levels = 'A')

### PROBLEM 8
#set seed for reproducibility
set.seed(11)
mu.true <- c(0,-2,3)
sig.true <- c(1,sqrt(2),4)
p.true <- c(0.1,0.3,0.6)

nsamp <- 120 #number of samples to draw
r <- sample(1:3,size=nsamp, replace=TRUE, prob = p.true)
samples <- rnorm(n=nsamp, mean = mu.true[r], sd = sig.true[r])

#mus, probs and sigmas should be initial guesses
EM <- function(samples, probs, mus, sigmas, tol){
  #initial log-likelihood
  n <- length(samples); m <- length(probs)
  count <- 0
  #### E-step: Calculate the expectation of log p(y,z|theta)
  
  #calculating our weights
  w <- t( sapply(samples, function(x) probs*dnorm(x,mean = mus, sd = sigmas) ) )
  #weighting our weights
  w.tilde <- sweep(w,1,rowSums(w),FUN = "/")
  #calculating Q(theta | theta^t)
  l.old <- sum( log( rowSums(w) ) )
  
  while(TRUE){

  ### M-step: calculating our new MLE's

  #probability calculation
  probs  <- colSums(w.tilde) / n

  #mean calculation
  #mus <- ( t(w.tilde) %*% samples) / colSums(w.tilde)
  #mus <- as.numeric(mus)
  mus <- colSums( sweep(w.tilde,1,samples,FUN = "*") ) / colSums(w.tilde)
  #standard deviation calculation
  mass <- sapply(1:m, function(x) sum(w.tilde[,x]*(samples - mus[x])**2 ) ) 
  #mass <- colSums(w.tilde*sweep(w.tilde,2,mus,FUN = "-")^2)
  sigmas <- sqrt(mass / colSums(w.tilde) )
  
  #calculating our weights
  w <- t( sapply(samples, function(x) probs*dnorm(x,mean = mus, sd = sigmas) ) )
  #weighting our weights
  w.tilde <- sweep(w,1,rowSums(w),FUN = "/")
  
  #calculating Q(theta | theta^(t+1) )
  l.new <- sum( log( rowSums(w) ) )
  
  count <- count + 1
  if(abs(l.new - l.old) < tol) { break }
  l.old <- l.new
  
  }
  return(list(iters = count, p = round(probs,3), mu = round(mus,3), sd = round(sigmas,3)))
}

init.p <- c(1/3,1/3,1/3)  
init.mu <- c(2,-1,0.7)
init.sigma <- c(1,1,1)
count <- EM(samples, probs = init.p, mus = init.mu, sigmas = init.sigma,tol=1e-18)
sprintf("It took approximately %s iterations to reach convergence", count["iters"])
sprintf("The MLE for the weights are [%s]",paste0(count["p"], collapse = ', '))
sprintf("The MLE for the means are [%s]",paste0(count["mu"], collapse = ', '))
sprintf("The MLE for the std dev are [%s]",paste0(count["sd"], collapse = ', '))

#curve plotting
x.grid <- seq(from = -20, to = 20, length = 1000) #range of x values to plot

true.curve <- p.true %*% sapply(x.grid, dnorm, mean = mu.true, sd = sig.true)
true.curve <-c(true.curve)

EM.curve <- count[["p"]] %*% sapply(x.grid, dnorm, count[["mu"]], count[["sd"]])
EM.curve <- c(EM.curve)

curves <-cbind(true.curve,EM.curve) %>%
  as.data.frame() %>% cbind(x.grid) %>% gather(curves, p, -x.grid)

emcurvs <- ggplot(data=curves) + aes(x=x.grid)+
  geom_line(aes(x=x.grid,y = p,color=curves,group=curves),size=2) + 
  theme(axis.line = element_line(colour = "black",size=2),
        text = element_text(size=20),
        axis.text = element_text(colour = "black",size = 20,face="bold"),
        axis.title = element_text(size = 24,face="bold"),
        axis.ticks.length=unit(.25, "cm"),
        axis.ticks = element_line(colour = "black", size = 1.5),
        legend.background = element_blank(),
        legend.text = element_text(size = 16, face = "bold"),
        legend.title = element_text(size = 18,face="bold"))+
  scale_color_manual(name=" Gaussian Distributions ", values=c("purple","gold"),
                     labels=c('EM Curve (n=120)','True Curve'))+
  ylab("Probability Density")+
  xlab("x")

# Individual curves
indy <- t(sapply(x.grid, function(x) dnorm(x,mean = mu.true, sd = sig.true))) %>%
  as.data.frame() %>% cbind(x.grid) %>% gather(curves, p, -x.grid)

ind.curvs <- ggplot(data=indy) + aes(x=x.grid) +
  geom_line(aes(x=x.grid,y = p,color=curves,group=curves),size=2) + 
  theme(axis.line = element_line(colour = "black",size=2),
        text = element_text(size=20),
        axis.text = element_text(colour = "black",size = 20,face="bold"),
        axis.title = element_text(size = 24,face="bold"),
        axis.ticks.length=unit(.25, "cm"),
        axis.ticks = element_line(colour = "black", size = 1.5),
        legend.background = element_blank(),
        legend.text = element_text(size = 18, face = "bold"),
        legend.title = element_text(size = 22,face="bold"))+
  scale_color_manual(name = " Gaussians ", values=c("red","green","blue"),
  labels=expression(paste("(",mu[1],", ", sigma[1],")",'=',"(",0,", ",1,")"),
                    paste("(",mu[2],", ", sigma[2],")",'=',"(-",2,", ",paste(sqrt(2)),")"),
                    paste("(",mu[3],", ", sigma[3],")",'=',"(",3,", ",4,")")))+
  ylab("Probability Density")+
  xlab("x")

( emcurvs | ind.curvs )+ 
   plot_annotation(tag_levels = 'A')
 
#Gibbs Sampler Parameters

#mu posterior 
alpha <- 3
#sigma posterior
eta <- 2; nu <- 2
eps <- 0
# Number of iterations
N <- 50000
#Number of mixtures to consider
n <- 3

# vectors to save our values
lmbda.gibbs <- matrix(NA,nrow = N, ncol = n)
mu.gibbs <- matrix(NA,nrow = N, ncol = n)
sigma.gibbs <- matrix(NA,nrow = N, ncol = n)

#starting values
lmbda <- c(1/3,1/3,1/3)  
mu <- c(2,-1,0.7)
sig <- c(1,1,1)

system.time(for(i in 1:N){
  #first get initial probabilities to draw z
  #calculating our weights
  p <- t( sapply(samples, function(x) lmbda*dnorm(x,mean = mu, sd = sig) ) )
  #weighting our weights
  p <- sweep(p,1,rowSums(p),FUN = "/")


  #draw z for initial probabilities
  z <- apply(t(apply(p,1,cumsum)) > matrix(rep(runif(nsamp),n),ncol=n),1,which.max) 
  #How many values belong to 1, 2, and 3
  y.k <- lapply(1:n,function(x) samples[z==x])
  n.k <- lengths(y.k)
  #draw lambda values, and append it to list
  lmbda <- as.numeric(rdirichlet(n=1,alpha=n.k+1))
  lmbda.gibbs[i, ] <- lmbda
  
  #draw mu values
  mu.gibbs.var <- 1 / ((1/alpha)^2 + (n.k/sig^2) ) 
  mu.gibbs.mean <- as.numeric(lapply(y.k,sum))/sig^2 + eps / alpha^2
  mu.gibbs.mean <- mu.gibbs.mean*mu.gibbs.var

  mu <- rnorm(n=n, mean = mu.gibbs.mean, sd = sqrt(mu.gibbs.var))
  mu.gibbs[i, ] <- mu

  #draw sigma^2 values
  sums <-sapply(1:n, function(x) sum((y.k[[x]]-mu[x])^2) )
  sig <- sqrt( 1 / rgamma(n=3, shape = eta + 0.5*n.k, scale =  1/(nu + 0.5*sums)) )
  sigma.gibbs[i, ] <- sig
})


gibbs.samples <- cbind(mu.gibbs,log(sigma.gibbs),lmbda.gibbs)

colnames(gibbs.samples) <- c(sapply(1:n, function(x) paste0("mu ",j)),
                             sapply(1:n, function(x) paste0("sigma ",j)),
                             sapply(1:n, function(x) paste0("lambda ",j)))

P8.labels <- c(expression(mu[1]),expression(mu[2]),expression(mu[3]),
               expression(sigma[1]),expression(sigma[2]) ,expression(sigma[3]),
               expression(lambda[1]),expression(lambda[2]),expression(lambda[3]))

P8.col <- c("cyan","magenta","red","blue","yellow","green","orange"
             ,"hotpink","brown")

gibbs.trace.plots <- lapply(1:ncol(gibbs.samples), function(j){
  ggplot() +
    aes(x = 1:N,y = gibbs.samples[,j])+
    geom_line(color=P8.col[j],size=2)+
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(), 
          axis.line = element_line(colour = "black",size=2),
          text = element_text(size=20),
          axis.text = element_text(colour = "black",size = 20,face="bold"),
          axis.title = element_text(size = 24,face="bold"),
          axis.ticks.length=unit(.25, "cm"),
          axis.ticks = element_line(colour = "black", size = 1.5))+
    ylab( P8.labels[j] )+
    xlab("Iterations")}
)

( (gibbs.trace.plots[[1]] |gibbs.trace.plots[[2]] | gibbs.trace.plots[[3]]) / 
                   (gibbs.trace.plots[[4]] | gibbs.trace.plots[[5]] | gibbs.trace.plots[[6]] ) /
                   (gibbs.trace.plots[[7]] | gibbs.trace.plots[[8]] | gibbs.trace.plots[[9]] ))+ 
  plot_annotation(tag_levels = 'A')

# Autocorrelation Plots
autocor.gibbs <- apply(gibbs.samples,2,function(x) acf(x,plot=F,lag.max=2000)$acf)

autocor.gibbs <- as.data.frame(autocor.gibbs)
autocor.gibbs["x"] <- 1:2001
autocor.gibbs <- autocor.gibbs %>% gather(params,p,-x)

ggplot(data=autocor.gibbs) + aes(x=1:2001)+
  geom_line(aes(x=x,y = p,color=params,group=params),size=2) + 
  theme(axis.line = element_line(colour = "black",size=2),
        text = element_text(size=20),
        axis.text = element_text(colour = "black",size = 20,face="bold"),
        axis.title = element_text(size = 24,face="bold"),
        axis.ticks.length=unit(.25, "cm"),
        axis.ticks = element_line(colour = "black", size = 1.5),
        legend.background = element_blank(),
        legend.text = element_text(size = 28, face = "bold"),
        legend.title = element_text(size = 22,face="bold"))+
  scale_color_manual(name=" Parameter ", values=P8.col,
                     labels=P8.labels)+
  ylab("Autocorrelation Plot")+
  xlab("Iterations")

### EXTRA CREDIT: PROBLEM 9 (BDA 3rd Ed., Exercise 15.3)

#The data
#Reactor Temperature
x1 <- c(1300,1300,1300,1300,1300,1300,1200,1200,1200,1200,
        1200,1200,1100,1100,1100,1100)
#Ratio of Hydrogen to n-heptane
x2 <- c(7.5,9,11,13.5,17,23,5.3,7.5,11,13.5,17,23,5.3,7.5,11,17)
#contact time (sec)
x3 <- c(0.012,0.012,0.0115,0.013,0.0135,0.012,0.04,0.038,0.032,0.026,
               0.034,0.041,0.084,0.098,0.092,0.086)
#conversion of heptane to acetlylene (%)
y <- c(49,50.2,50.5,48.5,47.5,44.5,28,31.5,34.5,35,38,38.5,15,17,20.5,19.5)

#combine data into data matrix
X <- cbind(x1,x2,x3,x1*x2,x1*x3,x2*x3,x1^2,x2^2,x3^2)
#standardize our covariates for easier computation
X <- apply(X,2, function(x) (x - mean(x))/sd(x))
#add constant term
X.star <- cbind(rep(1,nrow(X)),X)

#MLE
gram.mat <- t(X.star) %*% X.star; inv.gram.mat <- solve(gram.mat)
MLE <-  inv.gram.mat  %*% t(X.star) %*% y
#Number of simulations
N9 <- 2000
#simulates sigma
s2 <- as.numeric( t( y - X.star %*% MLE) %*%  ( y - X.star %*% MLE) )
q <- s2 / rchisq(n = N9, df = nrow(X.star) - qr(X.star)$rank ) #sigma^2
beta.post <- sapply(q, function(x) rmvnorm(n=1, mean = MLE, sigma = x*inv.gram.mat))
post.params <- cbind(t(beta.post),q)
names <- c("Intercept","x1","x2","x3","x1x2","x1x3","x2x3","x1^2","x2^2","x3^2")
param.names <- c(sapply(names, function(x) paste0('\u03B2 ', x)),'\u03C3')

quantities(x = post.params, 
           p = c(0.025,0.25,0.50,0.75,0.975),
           params = param.names,sigfig=2)
d <- lm(y~.,data = as.data.frame(X))

### PART B

K <- 100000 #How many values we want to draw up
MM_indices <- tail(1:K,ceiling(0.4*K)) #mixed model indices after burnin
#list to append our samples to
beta.gibbs <- matrix(NA,ncol = ncol(X.star), nrow = K)
sigma.gibbs <- NULL
mu.gibbs <- NULL
tau.gibbs <- NULL

#initial values
beta <- MLE #This is good as any guess
mu <- mean(MLE)

#helpful things
N <- nrow(X.star); J <- ncol(X.star)
a <- 1 #this represent how many values have improper prior

y.tilde <- t(X.star) %*% y # do this beforehand so we don't have to do this N9 times
Id <- c(rep(0,a),rep(1,J-a)); Id.mat <- diag(Id)

for (j in 1:K){
  # Sigma draws
  s2 <- t(y - (X.star %*% beta) ) %*% ( y - (X.star %*% beta) )
  sigma <- sqrt( as.numeric(s2) / rchisq(n=1,df = N ) )
  sigma.gibbs[j] <- sigma
  
  # Tau draw
  tau <- sqrt( sum((beta[(a+1):J] - mu)^2) / rchisq(n=1,df = J - a) )
  tau.gibbs[j]<- tau
  
  # Theta draws
  V.beta <- solve(Id.mat/tau^2 + gram.mat/sigma^2)
  beta.tilde <- V.beta %*% (Id*(mu/tau^2) + y.tilde/sigma^2)
  beta <- t( rmvnorm(n=1,mean=beta.tilde, sigma=V.beta) )
  beta.gibbs[j,] <- beta
  
  # Mu Draw
  mu <- rnorm(n=1, mean = mean(beta[(a+1):J]), sd = tau / sqrt(J-a))
  mu.gibbs[j] <- mu
  
}

#summary statistics for parameters

post.params.partB <- cbind(beta.gibbs,sigma.gibbs,mu.gibbs,tau.gibbs)
param.names.partB <- c(sapply(names, function(x) paste0('\u03B2 ', x)),'\u03C3',
                       '\u03BC','\u03C4')

quantities(x = post.params.partB[MM_indices,], 
           p = c(0.025,0.25,0.50,0.75,0.975),
           params = param.names.partB,sigfig=2)

### PART D

K <- 100000 #How many values we want to draw up
T4_indices <- tail(1:K,ceiling(0.4*K)) #indices for t4 model after burnin
#initial values
beta <- MLE #This is good as any guess
mu <- mean(MLE)

#helpful things
N <- nrow(X.star); J <- ncol(X.star)
a <- 1 #this represent how many values have improper prior
nu <- 4; mu <- 0; tau <- 1 #t distribution parameters
y.tilde <- t(y) %*% X.star # do this beforehand so we don't have to do this N9 times
Id <- c(rep(0,a),rep(1,J-a)); Id.mat <- diag(Id)

#t4 prior
beta.gibbs <- matrix(NA,nrow = K, ncol = ncol(X.star))
sigma.gibbs <- NULL
lmbda.gibbs <- NULL

for (j in 1:K){
  # Sigma draws
  s2 <- t(y - (X.star %*% beta )) %*% ( y - (X.star %*% beta ) )
  sigma <- sqrt( as.numeric(s2) / rchisq(n=1,df = N ) )
  sigma.gibbs[j] <- sigma
  
  #Lambda draw
  lmbda <- sqrt( (sum((beta[(a+1):J] - mu)^2) + nu*tau^2 ) / rchisq(n=1,df = J + nu) )
  lmbda.gibbs[j] <- lmbda

  # beta draws
  V.beta <- solve(Id.mat/lmbda^2 + gram.mat/sigma^2)
  beta.tilde <- V.beta %*% (Id*(mu/lmbda^2) + t(y.tilde)/sigma^2)
  beta <- as.numeric( rmvnorm(n=1,mean=beta.tilde, sigma=V.beta) )
  beta.gibbs[j,] <- beta
}

post.params.partD <- cbind(beta.gibbs,sigma.gibbs,lmbda.gibbs)
param.names.partD <- c(sapply(names, function(x) paste0('\u03B2 ', x)),'\u03C3'
                        ,'\u03BB')

quantities(x = post.params.partD[T4_indices,], 
           p = c(0.025,0.25,0.50,0.75,0.975),
           params = param.names.partD,sigfig=2)

# Generalized t4 prior
#list to append our samples to
beta.gibbs2 <- matrix(NA,nrow = K, ncol = ncol(X.star))
sigma.gibbs2 <- NULL
lmbda.gibbs2 <- NULL
mu.gibbs <- NULL
tau.gibbs <- NULL

#initial values
beta <- MLE #This is good as any guess
lmbda <- 3
mu <- mean(beta)

tau.pdf <- function(x,df,s){
  return( (df/2)*log(x) - ( (df*x)/2)*(1/s^2) )
}

#Grid for sigma 0 and nu
tau.grid <- seq(from = 1, to = 500, length = 2000)
d.tau <- diff(tau.grid)[1]/2

for (j in 1:K){
  # Sigma draws
  s2 <- t(y - (X.star %*% beta )) %*% ( y - (X.star %*% beta ) )
  sigma <- sqrt( as.numeric(s2) / rchisq(n=1,df = N ) )
  sigma.gibbs2[j] <- sigma
  
  #Tau draw
  tau.p <- tau.pdf(x=tau.grid,df = nu,s = sigma)
  tau.p <- exp(tau.p) 
  tau <- sqrt( sample(tau.grid,size=1,replace=T,p = tau.p) + runif(1,0,d.tau) )
  tau.gibbs[j] <- tau
  
  #Lambda draw
  lmbda <- sqrt( (sum((beta[(a+1):J] - mu)^2) + nu*tau^2 ) / rchisq(n=1,df = J + nu) )
  lmbda.gibbs2[j] <- lmbda
  
  # Mu Draw
  mu <- rnorm(n=1, mean = mean(beta[(a+1):J]), sd = tau / sqrt(J))
  mu.gibbs[j] <- mu
  
  # Theta draws
  V.beta <- solve(Id.mat/lmbda^2 + gram.mat/sigma^2)
  beta.tilde <- V.beta %*% (Id*(mu/lmbda^2) + t(y.tilde)/sigma^2)
  beta <- as.numeric( rmvnorm(n=1,mean=beta.tilde, sigma=V.beta) )
  beta.gibbs2[j,] <- beta
}


post.params.partD2 <- cbind(beta.gibbs2,sigma.gibbs2,lmbda.gibbs2,mu.gibbs,tau.gibbs)
param.names.partD2 <- c(sapply(names, function(x) paste0('\u03B2 ', x)),'\u03C3'
                       ,'\u03BB','\u03BC','\u03C4')

quantities(x = post.params.partD2[T4_indices,], 
           p = c(0.025,0.25,0.50,0.75,0.975),
           params = param.names.partD2,sigfig=2)