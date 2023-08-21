function [XKTrain,XKTest,XKVal]=Kernel_Feature(XTrain,XTest,XVal,Anchor)
    
Dis = EuDist2(XTrain,Anchor,0);
bandwidth = mean(mean(Dis)).^0.5;
clear Dis;
XKTrain = exp(-sqdist(XTrain',Anchor')/(bandwidth*bandwidth));
XKTest = exp(-sqdist(XTest',Anchor')/(bandwidth*bandwidth));
XKVal = exp(-sqdist(XVal',Anchor')/(bandwidth*bandwidth));




% %     
%     [nX,Xdim]=size(XTrain);
%     [nXT,XTdim]=size(XTest);
%     [nXV,~] = size(XVal);
% 
%     XKTrain = sqdist(XTrain',Anchors');
%     if ~exist('Xsigma','var') || Xsigma == -1000
%         Xsigma = mean(mean(XKTrain,2));
%     end
%     XKTrain = exp(-XKTrain/(2*Xsigma^2));
% %     XKTrain = exp(-XKTrain/(2*Xsigma));
%     Xmvec = mean(XKTrain);
%     XKTrain = XKTrain-repmat(Xmvec,nX,1);
%     
%     XKTest = sqdist(XTest',Anchors');
%     XKTest = exp(-XKTest/(2*Xsigma^2));
% %     XKTest = exp(-XKTest/(2*Xsigma));
%     XKTest = XKTest-repmat(Xmvec,nXT,1);
%     
%     XKVal = sqdist(XVal',Anchors');
%     XKVal = exp(-XKVal/(2*Xsigma^2));
% %     XKVal = exp(-XKVal/(2*Xsigma));
%     XKVal = XKVal-repmat(Xmvec,nXV,1);
end