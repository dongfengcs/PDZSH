function [XTrain,LTrain,XTest,LTest,XVal,LVal]=split_data(features,labels,dsname)
%---------------------------------------------------------
%------- construct train set, test set, and retrieval set
%---------------------------------------------------------
switch(dsname)
    case 'AWA1'
        c = 50; trn = 80; ten = 80;
        class_unseen = [7	9	23	24	30	31	34	41	47	50];
    case 'AWA2'
        c = 50; trn = 100; ten = 100;
        class_unseen = [7	9	23	24	30	31	34	41	47	50];
    case 'APY'
        c = 32; trn = 50; ten = 50;
%         class_unseen = [10	13	14	15	16	17	19	20	21	23	25	29];
        class_unseen = [13	14	15	16	17	19	20  25];
end

class_seen = setdiff(1:c,class_unseen);

all_loc = 1:size(features,2);all_loc = all_loc';
all_loc_tag = zeros(size(features,2),1);

for i = 1:length(class_seen)
    temp_loc = all_loc(labels == class_seen(i));
    train_loc_i = randperm(length(temp_loc),trn);
    all_loc_tag(temp_loc(train_loc_i)) = 1;
end

for i = 1:length(class_unseen)
    temp_loc = all_loc(labels == class_unseen(i));
    test_loc_i = randperm(length(temp_loc),ten);
    all_loc_tag(temp_loc(test_loc_i)) = 2;
end

train_loc = all_loc(all_loc_tag == 1);
test_loc = all_loc(all_loc_tag == 2);
val_loc = all_loc(all_loc_tag == 0);


features = features';
LAll = full(sparse(1:length(labels), double(labels), 1));
XTrain = features(train_loc,:); LTrain = full(LAll(train_loc,:));
XTest = features(test_loc,:); LTest = full(LAll(test_loc,:));
XVal = features(val_loc,:); LVal = full(LAll(val_loc,:));

class_train_loc = unique(labels(train_loc));
class_test_loc = unique(labels(test_loc));

if ~isempty(intersect(class_train_loc,class_test_loc)) 
    fprintf('division wrong!\n');
end

end