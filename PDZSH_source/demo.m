close all; clear; clc;
addpath(genpath('./utils/'));
addpath(genpath('./datasets/'));
addpath(genpath('./PDZSH/'));

seed = 0;
rng('default');
rng(seed); 
param.seed = seed;

param.top_R = 5000;
param.top_K = 2000;
param.pr_ind = [1:50:1000,1001];
param.pn_pos = [1:100:2000,2000];

% 'APY'  'AWA2' 'ImageNet'
db_name = 'AWA2';
param.db_name = db_name;
loopnbits = [8,12,16,24,32,48,64,96,128,192,256];
%% load data
if strcmp(db_name,'APY') || strcmp(db_name,'AWA2')
%     load(['datasets/xlsa17/',db_name,'res101.m','.mat'],'features','labels');
    load(['datasets/xlsa17/data/',db_name,'/res101.mat'],'features','labels');
%     load(['datasets/xlsa17/data/',db_name,'att_splits','.mat'],'att');
    [XTrain,LTrain,XTest,LTest,XRetr,LRetr]=split_data(features,labels,db_name);
    clear features labels
elseif strcmp(db_name,'ImageNet')
    load('datasets/ImageNet_subset.mat','XTrain','LTrain','XTest','LTest','XRetr','LRetr','original_att','att');
end
    

%% kernelization
param.nXanchors = 1000; 
anchor_idx = randsample(size(XTrain,1), param.nXanchors);
XAnchors = XTrain(anchor_idx,:);  
[XKTrain,bandwidth,mvec] = RBF_kernel(XTrain,XAnchors);
[XKTest,~,~] = RBF_kernel(XTest,XAnchors,bandwidth,mvec);
[XKRetr,~,~] = RBF_kernel(XRetr,XAnchors,bandwidth,mvec);
%% evaluate
for ii = 1:length(loopnbits)
    param.nbits = loopnbits(ii);
    fprintf("======%s: start %d bits encoding======\n\n",db_name,loopnbits(ii));
    eva_info_ = evaluate_PDZSH(XKTrain,LTrain,XKTest,LTest,XKRetr,LRetr,param);
    MAP{ii} = eva_info_.MAP;
    trainT{ii} = eva_info_.trainT;
end

save([db_name '_results.mat'],'MAP','trianT');