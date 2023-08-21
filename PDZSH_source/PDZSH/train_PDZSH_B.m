function finalB = train_PDZSH_B(LTrain_s,YTrain,param)
    % parameters
    nbits = param.nbits;
    n = size(LTrain_s,1);
    
    % initization
    H = randn(n,nbits);
    B = sign(randn(n,nbits)); B(B==0)=-1;

    % iteartive optimization
    sLY = sign(LTrain_s*YTrain);
    for i = 1:param.max_iter
        % update B
        B = sign(sign(nbits*((2+0)*LTrain_s*(LTrain_s'*H)-ones(n,1)*(ones(1,n)*H))) + sLY);
        B(B==0) = -1;
        
        % update H
        HT = nbits*((2+0)*LTrain_s*(LTrain_s'*B)-ones(n,1)*(ones(1,n)*B));
        
        Temp = HT'*HT-1/n*(HT'*ones(n,1)*(ones(1,n)*HT));
        [~,Lmd,HVV] = svd(Temp); clear Temp
        idx = (diag(Lmd)>1e-6);
        HV = HVV(:,idx); HV_ = orth(HVV(:,~idx)); 
        HU = (HT-1/n*ones(n,1)*(ones(1,n)*HT)) *  (HV / (sqrt(Lmd(idx,idx))));
        HU_ = orth(randn(n,nbits-length(find(idx==1))));
        H = sqrt(n)*[HU HU_]*[HV HV_]';
        clear HT HU HV HVV
        
    end
    finalB = B;
end