function d=sqdist(a,b)
% SQDIST - computes squared Euclidean distance matrix  计算平方欧几里得距离矩阵
%          computes a rectangular matrix of pairwise distances  计算成对距离的矩形矩阵
% between points in A (given in columns) and points in B
% input:         a:p*n     b:p*m
% output:        d:n*m
%       dij是a第i列和b第j列的平方欧几里得距离

% NB: very fast implementation taken from Roland Bunschoten

aa = sum(a.^2,1); %a.^2将矩阵a所有项平方，然后sum函数将矩阵沿第1维度相加
bb = sum(b.^2,1); 
ab = a'*b; 
d = (repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);
    
