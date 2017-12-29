%authors: 
% Matan Finch, id 300895315
% Gilad Eini , id 034744920
function q3solSigmaEst()
    clear;
    close all;
    clc;

    load('dataAB.mat','train_dataA','train_dataB','valid_dataA','valid_dataB','test_dataA','test_dataB');

    trainA = train_dataA;
    trainB = train_dataB;
    validA = valid_dataA;
    validB = valid_dataB;
    testA = test_dataA;
    testB = test_dataB;
    clear train_dataA train_dataB valid_dataA valid_dataB test_dataA test_dataB;
    
%     %use valid data to get the gausian window function - sigma    
%     max = -1000000;
%     bestSig = [];
%     for i=0.1 : 0.1 : 3
%         suc = pnn(trainA, trainB, validA, validB, i );
%         if suc >= max
%            max = suc; 
%            bestSig = [bestSig i];
%            disp(['succes rate(in fraction) for valid data: ' num2str(suc) ' ' num2str(i)]);
%         end
%     end
%     %test all sigmas that were good on valid data stage:
%     max = -1;
%     for elm = bestSig
%         suc = pnn(trainA, trainB, testA, testB, elm);
%         disp(['succes rate(in fraction) for test data: ' num2str(suc) ' ' num2str(elm)]);
%         if suc > max
%             maxSig = elm;
%             max=suc;
%         end
%     end
%     disp(['best: ' num2str(max) ' ' num2str(maxSig)]);

    sig = 1.3;
    suc = pnn(trainA, trainB, testA, testB, sig);
    disp(['clasifier results(fraction) on testData: ' num2str(suc) ' ,sigama: ' num2str(sig)]);
    disp('done');
end

function suc = pnn(trainA, trainB, testedDataA, testedDataB, sigma )  

    [trainAsize, ~] = size(trainA);
    [trainBsize, ~] = size(trainB);
    [testedAsize, ~] = size(testedDataA);
    [testedBsize, ~] = size(testedDataB);
    totalCorrect = 0; %total clasified correct
  
    %calculate algoritm for class A. count only if Ya > Yb

    for i=1 : testedAsize %go over all tested samples (could be valid data or test data)
        valueFromA = 0;
        %calculate Ya(tested sample, trainDataA). for each train sample calculate the gausian with the
        %current tested sample and sum it up
        for j=1 : trainAsize
            valueFromA = valueFromA + gaus(trainA(j,:), testedDataA(i,:), sigma);
        end
        %normalize with the number of train samples.
        valueFromA = valueFromA/trainAsize;
        
        %do to this current tested sample the same with trainDataB - Yb(tested sample, trainDataB)
        valueFromB = 0;
        for j=1 : trainBsize
            valueFromB = valueFromB + gaus(trainB(j,:), testedDataA(i,:), sigma);
        end
        valueFromB = valueFromB/trainBsize;
        
        if(valueFromA > valueFromB)% if Ya > Yb -> we clasified correctly
            totalCorrect = totalCorrect + 1;
        end
    end
    
    %the same as explaind up just for testedDataB
    
    for i=1 : testedBsize
        valueFromA = 0;
        for j=1 : trainAsize
            valueFromA = valueFromA + gaus(trainA(j,:), testedDataB(i,:), sigma);
        end
        valueFromA = valueFromA/trainAsize;
        
        valueFromB = 0;
        for j=1 : trainBsize
            valueFromB = valueFromB + gaus(trainB(j,:), testedDataB(i,:), sigma);
        end
        valueFromB = valueFromB/trainBsize;
        
        if(valueFromA < valueFromB)%if Ya < Yb -> we clasified correctly
            totalCorrect = totalCorrect + 1;
        end
    end
    
    suc = totalCorrect / (testedAsize + testedBsize);  
end

function scalar = gaus(sampleVecValue, testedVecValue, sig)
%      c = (1/sig*sqrt(2*pi));
    euclidinDistance = norm(sampleVecValue-testedVecValue);%euclidian distance from 0 (lenght of the vector)
    exponent = exp(-(euclidinDistance^2)/(2*sig^2));
    scalar = exponent;
end


