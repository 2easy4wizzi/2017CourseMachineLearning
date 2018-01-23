%authors: 
% Matan Finch, id 300895315
% Gilad Eini , id 034744920
function [w,MSE] = trainOnDataAndCalcWandMSE(R)
    [Rrows, Rcols] = size(R);
    X = R(:, 2:Rcols);%chop train samples
    y = R(:,1);%chop first column
    w = inv(X'*X)*X'*y;%find w
     
    MSE = 0;
    for smpInd = 1 : Rrows
        yt = y(smpInd);%get real y
        xCurrnet = X(smpInd, :);%get xi
        yCurrent = fxw(xCurrnet, w);%use y=f(x,w)
        MSE = MSE + (yt - yCurrent)^2;%calc MSE
    end
    MSE = MSE / Rrows;%normalize
end
