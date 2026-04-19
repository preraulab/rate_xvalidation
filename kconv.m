%KCONV  Kernel convolution that corrects end effects and returns same-size output
%
%   Usage:
%       result = kconv(data, k)
%       result = kconv(data, k, dt)
%
%   Inputs:
%       data : 1xN double - input signal -- required
%       k    : 1xW double - odd-length kernel / window function -- required
%       dt   : double - time resolution (default: 1)
%
%   Outputs:
%       result : 1xN double - smoothed output, same length as data
%
%   Notes:
%       Runs a standard 'same' convolution and then re-weights the first and
%       last W samples so boundary points use only in-range kernel mass
%       (removes tapering artifacts at the edges).
%       From: Prerau M.J., Eden U.T. "A General Likelihood Framework for
%       Characterizing the Time Course of Neural Activity", Journal of
%       Neuroscience, 2011.
%
%   See also: cvkernel, conv
%
%   ∿∿∿  Prerau Laboratory MATLAB Codebase · sleepEEG.org  ∿∿∿

function result=kconv(data,k,dt)

data=data(:)';
k=k(:)';

%Assume dt=1 if none specified
if nargin==2
    dt=1;
end

%Require an odd length window
w=length(k);
if mod(w,2)==0
    error('Window must be of an odd length');
end

%Normalize k
k=k/sum(k);

%Perform the standard convolution
result=conv(data,k/dt,'same');

%Define the overlap size and window midpoint
snip=(w-1)/2;
mid=snip+1;



%Fix the ends to remove the end effects
if w<length(data)
wval=[1:w (length(data)-w):length(data)];
else
    wval=1:length(data);
end

for wsize=wval
    %Calculate data start and end, dealing with boundaries
    ds=max(wsize-snip,1);
    de=min(wsize+snip,length(data));

    %Calculate kernel start and end, dealing with boundaries
    ks=max(mid-(wsize-1),1);
    ke=min(mid+length(data)-wsize,w);

    %Calculate the leave-one out convolution
    result(wsize)=sum(data(ds:de).*k(ks:ke)/sum(k(ks:ke))/dt);
end
