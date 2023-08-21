function YTrain = train_PDZSH_Y(param)
    
    nbits = param.nbits;
    c = param.c;
    
    C = eye(c)*2-1;
    
    % initialize Y: Hadamard matrix
    set = [1,2,4,8,12,16,20,24,32,40,48,64,80,96,128,160,192,256,320,384,512,640,768,1024];
    
    if nbits==c
        Hadm = hadamard(nbits);
        YTrain = Hadm(randperm(nbits),:);
    elseif nbits<c
        Hadm = hadamard(set(find(set>c,1)));
        YTrain = Hadm(randperm(set(find(set>c,1)),c),1:nbits);
    else
        Hadm = hadamard(nbits);
        YTrain = Hadm(randperm(nbits,c),:);
    end
    
    % class-wise update Y
    for ci = 1:c
        tmp = 1:c;
        tmp(ci) = [];
        YTrain(ci,:) = sign(C(ci,tmp)*YTrain(tmp,:)); YTrain(YTrain==0) = -1;
    end
    
end