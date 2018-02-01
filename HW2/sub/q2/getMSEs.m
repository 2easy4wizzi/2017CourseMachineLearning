function [w, MSEtrain, MSEtest] = getMSEs(trainData, testData)
    [Rrows, Rcols] = size(trainData);
    X = trainData(:, 2:Rcols);%chop train samples
    y = trainData(:,1);%chop first column
    w = inv(X'*X)*X'*y;%find w
     
    MSEtrain = 0;
    for smpInd = 1 : Rrows
        yt = y(smpInd);%get real y
        xCurrnet = X(smpInd, :);%get xi
        yCurrent = fxw(xCurrnet, w);%use y=f(x,w)
        MSEtrain = MSEtrain + (yt - yCurrent)^2;%calc MSE
    end
    MSEtrain = MSEtrain / Rrows;%normalize
    
    [Rrows, Rcols] = size(testData);
    X = testData(:, 2:Rcols);
    y = testData(:,1);    
    
    MSEtest = 0;
    for smpInd = 1 : Rrows
        yt = y(smpInd);%get real y
        xCurrnet = X(smpInd, :);%get xi
        yCurrent = fxw(xCurrnet, w);%use y=f(x,w)
        MSEtest = MSEtest + (yt - yCurrent)^2;%calc MSE
    end
    MSEtest = MSEtest / Rrows;%normalize

end

