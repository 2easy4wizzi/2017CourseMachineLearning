%authors: 
% Matan Finch, id 300895315
% Gilad Eini , id 034744920
function q3sol()
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


    % windowSize = 2;
    % suc = pnn(trainA, trainB, validA, validB);
    % disp(['succes rate(in fraction) for valid data: ' num2str(suc)]);


    suc = pnn(trainA, trainB, testA, testB);
    disp(['succes rate(in fraction) for test data: ' num2str(suc)]);
    disp('done');
end

function suc = pnn(trainA, trainB, testedDataA, testedDataB)  
    [trainAsize, ~] = size(trainA);
    [trainBsize, ~] = size(trainB);
    [testedAsize, ~] = size(testedDataA);
    [testedBsize, ~] = size(testedDataB);
    totalCorrect = 0; %total clasified correct
    
    %calculate algoritm for class A. count only if Ya > Yb
    covMatA = cov(trainA);%cov matrix - used in the multiDimGaus 
    for i=1 : testedAsize %go over all tested samples (could be valid data or test data)
        valueFromA = 0;
        %calculate Ya(tested sample, trainDataA). for each train sample calculate the gausian with the
        %current tested sample and sum it up
        for j=1 : trainAsize
            valueFromA = valueFromA + multiDimGaus(trainA(j,:), testedDataA(i,:), covMatA);
        end
        %normalize with the number of train samples.
        valueFromA = valueFromA/trainAsize;
        
        %do to this current tested sample the same with trainDataB - Yb(tested sample, trainDataB)
        valueFromB = 0;
        for j=1 : trainBsize
            valueFromB = valueFromB + multiDimGaus(trainB(j,:), testedDataA(i,:), covMatA);
        end
        valueFromB = valueFromB/trainBsize;
        if(valueFromA > valueFromB)% if Ya > Yb -> we clasified correctly
            totalCorrect = totalCorrect + 1;
        end
    end
    
    %the same as explaind up just for testedDataB
    covMatB = cov(trainB);
    for i=1 : testedBsize
        valueFromA = 0;
        for j=1 : trainAsize
            valueFromA = valueFromA + multiDimGaus(trainA(j,:), testedDataB(i,:), covMatB);
        end
        valueFromA = valueFromA/trainAsize;
        valueFromB = 0;
        for j=1 : trainBsize
            valueFromB = valueFromB + multiDimGaus(trainB(j,:), testedDataB(i,:), covMatB);
        end
        valueFromB = valueFromB/trainBsize;
        if(valueFromA < valueFromB)%if Ya < Yb -> we clasified correctly
            totalCorrect = totalCorrect + 1;
        end
    end
    
    suc = totalCorrect / (testedAsize + testedBsize);  
end

function scalar = multiDimGaus(sampleVecValue, testedVecValue, covMat)
     scalar = (1/(sqrt(det(2*pi*covMat)))) * exp(-1/2*(sampleVecValue-testedVecValue)*inv(covMat)*(sampleVecValue-testedVecValue)');
end





