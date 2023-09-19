%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                  CVEXAMPLE.M
%                         � Michael J. Prerau, Ph.D. 2011
%
%   This code is used in thethe algorithm described in:
%   Prerau M.J., Eden U.T. 
%   "A General Likelihood Framework for Characterizing the Time Course of Neural Activity", 
%   Journal of Neuroscience, 2011
%
%   An example of how to use the cross-validation kernel smoother in
%   MATLAB. In this example we simulate a "true rate" which is a scaled
%   sinusoid. From this rate we generate spikes using an inhomogeneous
%   Poisson model. Using only the spikes, we estimate the true rate. Our
%   estimate is shown in red, and the true rate is shown in dashhed gray.
%
%   We show and example of estimating rates from two simulated rate
%   functions--one with a large bandwidth, and one with a small bandwidth
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% close all;
clc;
figure;

%Try two different bandwidths
for p=[1.5 5]
    %Simulate a true firing rate as a sin function
    dt=5/300;
    t=dt:dt:5;
    lambda=sin(p*t);
    
    %Scale the function to be an accurate rate
    lambda=(lambda-min(lambda))*150+5;
    
    %Generate spikes with inhomogenous Poisson model
    spikecount=poissrnd(lambda*dt);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Run the cross-validated kernel smoother--Just one command :)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [estimate kmax loglikelihoods bandwidths CI]=cvkernel(spikecount, dt,[],1);
    
    %Plot the data and the estimate of the true value
    axs=get(gcf,'children');
    subplot(axs(3));
    hold on
    plot(t,lambda,'--','linewidth',2,'color',[.7 .7 .7]);
    
    %Make likelihood easy to view
    subplot(212)
    xlim([0 2]);
end
