function suc = pnn(trainA, trainB, testedDataA, testedDataB)  
    [trainAsize, ~] = size(trainA);
    [trainBsize, ~] = size(trainB);
    [testedAsize, ~] = size(testedDataA);
    [testedBsize, ~] = size(testedDataB);
    totalCorrect = 0; %total clasified correct
    
    %calculate algoritm for class A. count only if Ya > Yb
    covMatA = cov(trainA);%cov matrix - used in the multiDimGaus 
    covMatB = cov(trainB);
    for i=1 : testedAsize %go over all tested samples (could be valid data or test data)
        if i == 3
           x=2; 
        end
        valueFromA = 0;
        %calculate Ya(tested sample, trainDataA). for each train sample calculate the gausian with the
        %current tested sample and sum it up
        for j=1 : trainAsize
            valueFromA = valueFromA + multiDimGaus(trainA(j,:), testedDataA(i,:), covMatA);
        end
        %normalize with the number of train samples.
        %valueFromA = valueFromA/trainAsize;
        
        %do to this current tested sample the same with trainDataB - Yb(tested sample, trainDataB)
        valueFromB = 0;
        p2 = testedDataA(i,:);
        for j=1 : trainBsize
            valueFromB = valueFromB + multiDimGaus(trainB(j,:), p2, covMatB);
        end
        %valueFromB = valueFromB/trainBsize;
        if(valueFromA > valueFromB)% if Ya > Yb -> we clasified correctly
            totalCorrect = totalCorrect + 1;
        end
    end
    
    %the same as explaind up just for testedDataB
    for i=1 : testedBsize
        valueFromA = 0;
        for j=1 : trainAsize
            valueFromA = valueFromA + multiDimGaus(trainA(j,:), testedDataB(i,:), covMatA);
        end
        %valueFromA = valueFromA/trainAsize;
        valueFromB = 0;
        for j=1 : trainBsize
            valueFromB = valueFromB + multiDimGaus(trainB(j,:), testedDataB(i,:), covMatB);
        end
        %valueFromB = valueFromB/trainBsize;
        if(valueFromA < valueFromB)%if Ya < Yb -> we clasified correctly
            totalCorrect = totalCorrect + 1;
        end
    end
    
    suc = totalCorrect / (testedAsize + testedBsize);  
end