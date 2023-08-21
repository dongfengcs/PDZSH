function [XKTrain,XKTest,XKVal]=Kernel_Feature(XTrain,XTest,XVal,Anchors)
    %核化操作
    [nX,Xdim]=size(XTrain);

    [nXT,XTdim]=size(XTest);
    
    [nXV,XVdim] = size(XVal);


    XKTrain = sqdist(XTrain',Anchors');
    Xsigma = mean(mean(XKTrain,2)); %核宽σ的平方，因为前面是平方欧几里得距离？
    XKTrain = exp(-XKTrain/(2*Xsigma));
    Xmvec = mean(XKTrain);
    XKTrain = XKTrain-repmat(Xmvec,nX,1);
    
    XKTest = sqdist(XTest',Anchors');
    XKTest = exp(-XKTest/(2*Xsigma));
    XKTest = XKTest-repmat(Xmvec,nXT,1);
    
    XKVal = sqdist(XVal',Anchors');
    XKVal = exp(-XKVal/(2*Xsigma));
    XKVal = XKVal-repmat(Xmvec,nXV,1);
end