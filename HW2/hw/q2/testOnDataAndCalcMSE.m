function MSE = testOnDataAndCalcMSE(testData,w)
    [Rrows, Rcols] = size(testData);
    X = testData(:, 2:Rcols);
    y = testData(:,1);    
    
    MSE = 0;
    for smpInd = 1 : Rrows
        yt = y(smpInd);%get real y
        xCurrnet = X(smpInd, :);%get xi
        yCurrent = fxw(xCurrnet, w);%use y=f(x,w)
        MSE = MSE + (yt - yCurrent)^2;%calc MSE
    end
    MSE = MSE / Rrows;%normalize
end