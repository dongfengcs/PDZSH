%---------------------------------------------------------
%------- construct ImageNet subset for evaluation
%------- choose 100 seen classes from ImageNet 1K
%------- choose 10 unseen classes from ImageNet 20K
%------- replace attributes with corresponding w2v
%---------------------------------------------------------
clear; clc;
rng('default'); rng(0); 

load('datasets/ILSVRC2012_res101_feature.mat', 'features', 'labels');
% choose seen class
class_num = zeros(1000,1);
for ii = 1:1000
    class_num(ii) = sum(labels==ii);
end
m_b = max(class_num);
temp_class = randperm(1000);
ii = 1; jj = 1;
seen_classes = [];
while ii <= 100
    if class_num(temp_class(jj)) ~= m_b
        jj = jj + 1;
        continue;
    end
    seen_classes(ii) = temp_class(jj);
    ii = ii + 1;
    jj = jj + 1;
end
% split training samples
features = double(features);
trn = 500; 
XTrain = [];  XRetr = [];
lTrain = []; lRetr = [];
for ii = 1:length(seen_classes)
    fea_temp = features(labels == seen_classes(ii),:);
    temp = randperm(m_b); 
    XTrain = [XTrain; fea_temp(temp(1:trn),:)]; 
    lTrain = [lTrain; ones(trn,1)*ii];
    XRetr = [XRetr; fea_temp(temp(trn + 1:end),:)];
    lRetr = [lRetr; ones(m_b - trn,1)*ii];
end
clear features labels

% split test samples
load("datasets/imagenet_class_splits.mat",'mp500');
ten = 200; 
unseen_classes = mp500(1:10); 
XTest = []; lTest = [];
for ii = 1:length(unseen_classes)
    file_path = ['datasets/ImageNet2011_res101_feature/res101_1crop_feature/' num2str(unseen_classes(ii)) '.bin'];
    fea_temp = load_X(file_path);
    temp = randperm(size(fea_temp,1)); 
    XTest = [XTest; fea_temp(temp(1:ten),:)];
    lTest = [lTest; ones(ten,1)*(100 + ii)];
    XRetr = [XRetr; fea_temp(temp(ten + 1:end),:)];
    lRetr = [lRetr; ones(size(fea_temp,1) - ten,1)*(100 + ii)];
end

LTrain = full(sparse(1:length(lTrain), double(lTrain), 1)); 
LTrain = [LTrain, zeros(size(LTrain,1),10)]; 
LTest = full(sparse(1:length(lTest), double(lTest), 1)); 
LRetr = full(sparse(1:length(lRetr), double(lRetr), 1)); 

% replace attributes with word2vec embedding
load('datasets/ImageNet_w2v.mat','w2v');
original_att = w2v(seen_classes,:)'; original_att = [original_att, w2v(unseen_classes,:)'];
att = NormalizeFea(original_att,0);
clear w2v
save("datasets/ImageNet_subset.mat",'XTrain','LTrain','XTest','LTest','XRetr','LRetr','original_att','att','seen_classes','unseen_classes');
