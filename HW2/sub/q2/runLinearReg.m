%authors: 
% Matan Finch, id 300895315
% Gilad Eini , id 034744920
function runLinearReg 
    clear;
    close all;
    clc;
    load('regdata.mat', 'R');
    
    %%clause b - split and: calc W and MSE on trainData,calc MSE on testData
    [Rrows, ~] = size(R);
    splits = [10 50 100 200];
    for i=splits
        disp(['case split is ' num2str(i) ' and ' num2str(Rrows-i)]);
       [trainData, testData] = permAndSplit(R,i); 
       [w, MSEtrain, MSEtest] = getMSEs(trainData, testData);
       disp(['       On |trainData| =' num2str(i)]);
       disp(['       The MSE:'  num2str(MSEtrain) '. The optimal W: [' num2str(w(:).') ']']) ;
       disp(['       On |testData| =' num2str(Rrows-i)]);
       disp(['       The MSE:'  num2str(MSEtest)]) ;
    end
end








