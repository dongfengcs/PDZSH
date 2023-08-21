function evaluation_info=evaluate_PDZSH(XKTrain,LTrain,XKTest,LTest,XKRetr,LRetr,param)
    %% parameters setting
    if strcmp(param.db_name,"APY") 
        if param.nbits < 64
            param.sigma = 10; 
        else
            param.sigma = 5; 
        end
    elseif strcmp(param.db_name,"AWA1") || strcmp(param.db_name,"AWA2")
        if param.nbits < 64
            param.sigma = 8; 
        else
            param.sigma = 4; 
        end
    elseif strcmp(param.db_name,"ImageNet")
        if param.nbits < 64
            param.sigma = 8; 
        else
            param.sigma = 4; 
        end
    else
        param.max_iter = 10;
        param.sigma = 8; 
    end
    param.max_iter = 5;
    param.tau = 1; 
    param.thre = param.nbits/param.sigma;
    %% hash learning
    id_seenclass = find(sum(LTrain,1)~=0);
    LTrain_s = LTrain(:,id_seenclass);
    param.id_seenclass = id_seenclass;
    param.c = length(id_seenclass);

    tic;
    %class prototype learning
    YTrain = train_PDZSH_Y(param);
    
    %hash codes learning
    BTrain = train_PDZSH_B(LTrain_s,YTrain,param);
    
    %hash functions    
    XP = (XKTrain'*XKTrain+param.tau*eye(size(XKTrain,2))) \ (XKTrain'*BTrain);
    evaluation_info.trainT=toc;
    
    %% image retrieval
    BRetr = compactbit(sign(XKRetr*XP)>0);
    BTest = compactbit(sign(XKTest*XP)>0);
    DHamm = hammingDist(BTest, BRetr);
    
    Bm = zeros(param.c,param.nbits);
    for ci = 1:param.c
        Bm(ci,:)  = sign(mean(BTrain(LTrain_s(:,ci)==1,:),1));
    end
    Bm = compactbit(Bm>0);
    preseen_idx = min(hammingDist(Bm, BRetr)) <= param.thre;
    DHamm(:,preseen_idx) = param.nbits+1;
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.MAP = mAP(orderH', LRetr, LTest,param.top_R);
    fprintf('MAP: %f\n',evaluation_info.MAP);
    [evaluation_info.precision,evaluation_info.recall] = precision_recall(orderH', LRetr, LTest);
    evaluation_info.Precision = precision_at_k(orderH',LRetr, LTest,param.top_K);
    
    
    evaluation_info.param = param;
    
end