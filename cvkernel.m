%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                   CVKERNEL.M
%                         � Michael J. Prerau, Ph.D. 2011
%
%   This code is derived from the algorithm in:
%   Prerau M.J., Eden U.T. 
%   "A General Likelihood Framework for Characterizing the Time Course of Neural Activity", 
%   Journal of Neuroscience, 2011
%
%   Performs hanning kernel smoothing using cross validation on the
%   kernel smoother used on spiking
%
%   USAGE:
%       [estimate kmax loglikelihoods bandwidths]=cvkernel(spikecounts, dt)
%                                      or
%       [estimate kmax loglikelihoods bandwidths]=cvkernel(spikecounts, dt, range)
%                                      or
%       [estimate kmax loglikelihoods bandwidths]=cvkernel(spikecounts, dt, range, ploton)
%
%   INPUTS:
%       spikecounts is the 1xN vector of spike counts
%       dt is the sampling rate (in seconds) of spikecounts
%
%     Optional:
%       range is a 1xV vector that specifies the range of bandwidths to use in the
%           cross-validation. This vector the units of this vector is number of
%           dt sized time bins and all values should be odd. (Default: [3:2:N])
%       ploton is 1 for a figure showing the estimate and bandwidth
%           likelihood with confidence bounds, (Default: 0)
%
%   OUTPUTS:
%       estimate is the estimate of the nonparametric regression/rate
%       kmax is the bandwidth with the maximum likelihood
%       loglikelihoods is a vector of log-likelihood values for each kernel
%           bandwidth
%       bandwidths is a 1xV vector number of dt sized time bins used for the bandwith
%           at each iteration of the cross-validation
%       CI is a 1x2 vector of estimated 95% confidence bounds from the Fisher
%           information on the bandwidth size
%
%   EXAMPLE:
%       Run "cvexample.m" for an example of the cross-validated kernel
%       smoother used on spiking data.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [estimate, kmax, loglikelihoods, bandwidths, CI]=cvkernel(spikecounts, dt, range, ploton)

if nargin<4
    ploton=false;
end

if ~any(spikecounts)
    estimate=zeros(1,length(spikecounts));
    kmax=-1;
    loglikelihoods=[];
    bandwidths=[];
    CI=[];
    return;
end

%Make sure spikecounts isn't logical
if ~isa(spikecounts,'double')
    spikecounts=double(spikecounts);
end

spikecounts=spikecounts(:)';

%Set dt based on input, default dt=1
if nargin==1
    dt=1;
end

%Get spikecounts length
N=length(spikecounts);
L=round(N/2);

%Set kernel range and adjust to make the first odd < N
if mod(N,2)==0
    L=N-1;
end

%Set bandwidths if not specified
if nargin==2 || isempty(range)
        bandwidths=3:2:3*L;
%         bandwidths=round(linspace(3,3*L,50))+mod(round(linspace(3,3*L,50)
%         ),2)  -1;
%     bandwidths=unique(round(logspace(log10(3),log10(3*L),30))+mod(round(l
%     ogspace(log10(3),log10(3*L),30)),2)-1);
else
    bandwidths=range;
end

%Allocate mean square error
loglikelihoods=zeros(1,length(bandwidths));

%Loop through kernel sizes, do a leave one out filter, and find loglikelihoods
parfor wn=1:length(bandwidths)
    %Set window size
    if ~mod(bandwidths(wn),2)
        bandwidths(wn)=bandwidths(wn)+1;
    end
    w=bandwidths(wn);
    
    %Set center point to zero for leave one out filter
    mid=(w-1)/2+1;
    k=hanning(w);
    k(mid)=0;
    
    %Normalize the notch kernel
    k=k/sum(k);
    
    %Perform lave one out convolution
    l1o=kconv(spikecounts,k,dt);
    
    %Fix log(0) problem
    l1o(~l1o)=1e-5;
    
    %Calculate the likelihood
    loglikelihoods(wn)=sum(-l1o*dt+spikecounts.*log(l1o)+spikecounts*log(dt)-log(factorial(spikecounts)));
    
    %     progressbar(wn/length(bandwidths));
end

%Calculate the maximum likelihood bandwidth
[~, ki]=max(loglikelihoods);
kmax=bandwidths(ki);

%Fix last bandwidth
if (ki==length(loglikelihoods)) || (ki==1)
    ki=length(loglikelihoods)-1;
    kmax=bandwidths(end);
end

%Calculate confidence bounds using Fisher information
a=loglikelihoods(ki-1);
b=loglikelihoods(ki);
c=loglikelihoods(ki+1);
pstd=sqrt(-1/((c+a-2*b)/(dt^2)));

CI(1)=kmax*dt-2*pstd;
CI(2)=kmax*dt+2*pstd;

%Calculate the full convolution with the best kernel
if kmax<length(loglikelihoods)
    k=hanning(kmax)/sum(hanning(kmax));
    estimate=kconv(spikecounts,k,dt);
else
    warning('No peak in likelihood found, estimating as flat rate');
    estimate=ones(1,length(spikecounts))*(sum(spikecounts)/(dt*length(spikecounts)));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          PLOT UP THE ESTIMATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ploton
    figure
    %Exponentiate the likelihood for easy viewing
    likelihood=exp(loglikelihoods-max(loglikelihoods));
    lmax=max(likelihood);
    t=dt:dt:(length(spikecounts)*dt);
    
    %Plot the data and the estimate of the true value
    ax=subplot(211);
    hold on
    plot(t,estimate,'r','linewidth',2);
    
    axis tight
    xlabel('Time (s)');
    ylabel('Rate (Hz)');
    title('Cross-Validated Kernel Smoother Estimate');
    
    ax2=axes('position',get(ax,'position'));
    subplot(ax2);
    stem(t,spikecounts,'marker','none');
    set(gca,'color','none','yaxislocation','right','xticklabel','')
    ylabel('Spike Count');
    
    %Plot the likelihood and confidence bounds for the bandwidth estimate
    subplot(212)
    hold on
    fill([CI fliplr(CI)],[lmax lmax 0 0],'g','edgecolor','none');
    plot(bandwidths*dt,likelihood,'k','linewidth',2);
    plot(kmax*dt,lmax,'r.','markersize',20);
    axis([0 kmax*dt*2, 0, lmax]);
    set(gca,'yticklabel','');
    xlabel('Hanning Bandwidth Size (s)');
    ylabel('Likelihood');
    title('Bandwidth Likelihood');
end



