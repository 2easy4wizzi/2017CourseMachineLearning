function runLinearRegression 
    clear;
    close all;
    clc;
    load('regdata.mat', 'R');
    
    %%clause a
    [w, MSE] = myTrainOnDataAndCalcWandMSE(R);
    disp(['The MSE:'  num2str(MSE) '. The optimal W: [' num2str(w(:).') ']']) ;
    %%clause b - split and: calc W and MSE on trainData,calc MSE on testData
    [Rrows, ~] = size(R);
    splits = [10 50 100 200];
    for i=splits
        disp(['case split is ' num2str(i) ' and ' num2str(Rrows-i)]);
       [trainData, testData] = myPermAndSplit(R,i); 
       [w, MSE] = myTrainOnDataAndCalcWandMSE(trainData);
       disp(['       On |trainData| =' num2str(i)]);
       disp(['       The MSE:'  num2str(MSE) '. The optimal W: [' num2str(w(:).') ']']) ;
       MSE = myTestOnDataAndCalcMSE(testData,w);
       disp(['       On |testData| =' num2str(Rrows-i)]);
       disp(['       The MSE:'  num2str(MSE)]) ;
    end
end

function [w,MSE] = myTrainOnDataAndCalcWandMSE(R)
    [Rrows, Rcols] = size(R);
    X = R(:, 2:Rcols);%chop train samples
    y = R(:,1);%chop first column
    w = inv(X'*X)*X'*y;%find w
     
    MSE = 0;
    for smpInd = 1 : Rrows
        yt = y(smpInd);%get real y
        xCurrnet = X(smpInd, :);%get xi
        yCurrent = myfxw(xCurrnet, w);%use y=f(x,w)
        MSE = MSE + (yt - yCurrent)^2;%calc MSE
    end
    MSE = MSE / Rrows;%normalize
end

function MSE = myTestOnDataAndCalcMSE(testData,w)
    [Rrows, Rcols] = size(testData);
    X = testData(:, 2:Rcols);
    y = testData(:,1);    
    
    MSE = 0;
    for smpInd = 1 : Rrows
        yt = y(smpInd);%get real y
        xCurrnet = X(smpInd, :);%get xi
        yCurrent = myfxw(xCurrnet, w);%use y=f(x,w)
        MSE = MSE + (yt - yCurrent)^2;%calc MSE
    end
    MSE = MSE / Rrows;%normalize
end


function y=myfxw(x,w)
    y = w(1);
    [~,c] = size(x);
    for i=1 : c
        y = y + w(i)*x(i);
    end
end

function [trainData, testData] = myPermAndSplit(R,trainSize)
    %permute R rows
    [Rrows, ~] = size(R);
    idx = randperm(Rrows);
    X = R(idx,:);
    %split 
    trainData = X(1:trainSize, :);
    testData = X((trainSize+1):end, :);
end