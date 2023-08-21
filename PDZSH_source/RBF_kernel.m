function [K,bandwidth,mvec] = RBF_kernel(Xq,Xr,bandwidth,mvec)
% RBF_fast: generate the RBF kernel

D = sqdist(Xq',Xr');

if ~exist('bandwidth','var')
    bandwidth = 2*mean(D(:));
end

K = exp(-D/bandwidth);
if ~exist('mvec','var')
    mvec = mean(K);
end

K = K-repmat(mvec,size(Xq,1),1);