function runLinearReg 
    clear;
    close all;
    clc;
    load('regdata.mat', 'R');
    
    %%clause a
    [w, MSE] = trainOnDataAndCalcWandMSE(R);
    disp(['The MSE:'  num2str(MSE) '. The optimal W: [' num2str(w(:).') ']']) ;
    %%clause b - split and: calc W and MSE on trainData,calc MSE on testData
    [Rrows, ~] = size(R);
    splits = [10 50 100 200];
    for i=splits
        disp(['case split is ' num2str(i) ' and ' num2str(Rrows-i)]);
       [trainData, testData] = permAndSplit(R,i); 
       [w, MSE] = trainOnDataAndCalcWandMSE(trainData);
       disp(['       On |trainData| =' num2str(i)]);
       disp(['       The MSE:'  num2str(MSE) '. The optimal W: [' num2str(w(:).') ']']) ;
       MSE = testOnDataAndCalcMSE(testData,w);
       disp(['       On |testData| =' num2str(Rrows-i)]);
       disp(['       The MSE:'  num2str(MSE)]) ;
    end
end








