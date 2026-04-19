%CVEXAMPLE  Example driver demonstrating the cross-validated kernel smoother
%
%   Usage:
%       cvexample
%
%   Inputs:
%       none
%
%   Outputs:
%       none (side effects only — generates a demo figure)
%
%   Notes:
%       Simulates a true firing rate as a scaled sinusoid, generates spikes
%       from an inhomogeneous Poisson model, and calls cvkernel to recover
%       the rate. Runs at two different ground-truth bandwidths for
%       comparison. Implements the algorithm described in:
%           Prerau M.J., Eden U.T. "A General Likelihood Framework for
%           Characterizing the Time Course of Neural Activity", Journal of
%           Neuroscience, 2011.
%
%   See also: cvkernel, kconv
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿
%        Source: https://github.com/preraulab/labcode_main

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
