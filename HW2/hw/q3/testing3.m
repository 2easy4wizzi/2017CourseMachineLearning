function testing3 
    clear;close all;clc;
    rng('default');  load ('mnist_svm','trainX', 'trainY');
    tr = ones(1, size(trainX,1)); tr(1:5:size(trainX,1)) = 0; tr = logical(tr);
    trX = trainX( tr,:); trY = trainY( tr); %training set that you use for training and cross validation
    teX = trainX(~tr,:); deY = trainY(~tr); %test set
    allC = [1 5]; allG = [0.005 0.01 0.05]; numFolds=5;
    
    [models,cv_acc]=myTrainLinearSVM(trX,trY,numFolds,allC);
    fprintf('Cross-validation accuracy is %2.2f\n***\n',cv_acc);
    
    acc=myTestSVM(models, teX,deY);
    fprintf('\n***Test accuracy is %2.4f***\n',acc);
    
    [models,cv_acc]=myTrainRBFSVM(trX,trY,numFolds, allC,allG);
    fprintf('Cross-validation accuracy of RBF SVM is %2.2f\n***\n',cv_acc);
    
    acc=myTestSVM(models, teX,deY);
    fprintf('\n***Test accuracy of RBF SVM is %2.4f***\n',acc);
    
    fprintf('ALL DONE\n');
end  

function [models,cv_acc]=myTrainLinearSVM(X,Y,K, allC)
labels =  0 : 9;%10 classes - zero to nine
c1 = 0;%meassure cross-validation accuracy rate with C=1
c5 = 0;%meassure cross-validation accuracy rate with C=5
options = '-q -v %d -c %d -b 1';

for i=labels
    [newX,newY] = sortDataPerClass(X,Y,i);
    c1 = c1 + svmtrain(newY, newX, sprintf(options, K, allC(1))); 
    c5 = c5 + svmtrain(newY, newX, sprintf(options, K, allC(2)));
end
c1 = c1 / numel(labels);
c5 = c5 / numel(labels);
sprintf(options,1,1);
% fprintf('with c1: %3.4f and with c5: %3.4f\n',c1,c5);

if(c1 > c5)
    bestC = allC(1);
    cv_acc = c1;
else
    bestC = allC(2);
    cv_acc = c5;
end

options = '-q -c %d -b 1';
for i=labels%calculate models using best C found from allC
    [newX,newY] = sortDataPerClass(X,Y,i);
    newModel = svmtrain(newY, newX, sprintf(options, bestC));
    models(i+1) = newModel;
end
fprintf('\n***\nLinearSvm was trained with bestC = %d \n',bestC);

end

function [newX,newY] = sortDataPerClass(samples,labels,classI)
    mat = [labels samples];
    ind = (mat(:,1) == classI);
    newMat = [ mat(ind,:); mat(~ind,:)];
    newY = newMat(:,1);
    newY(newY ~= classI) = -1;
    newY(newY == classI) = 1;
    newX = newMat(:,2:end);
end

function acc=myTestSVM(models, X, Y)
labels =  0 : 1 : 9;

probability = zeros(size(Y,1),numel(labels));
newY = zeros(size(Y));
for i=labels
    newY(Y ~= i) = -1;
    newY(Y == i) = 1;
    [~,~,p] = svmpredict(newY, X, models(i+1), '-b 1');
    probability(:,i+1) = p(:,models(i+1).Label==1);   
end

acc=0;
for i=1 : size(Y,1)
    [~,I] = max(probability(i,:));
    if(Y(i) == double(I(1) -1))
        acc = acc + 1;
    end
end
acc = acc/size(Y,1);
end

function [models,cv_acc]=myTrainRBFSVM(X,Y,K, allC,allG)
sumMat = zeros(2,3);% i.e sumMat(1,3): allC(1) and allG(3)
classes =  0 : 1 : 9;
options = '-q -v %d -c %d -g %f -b 1';

bar=waitbar(0,'Cross Val RBF SVM...');
for i=classes
    [newX,newY] = sortDataPerClass(X,Y,i);
    sumMat(1,1) = sumMat(1,1) + svmtrain(newY, newX, sprintf(options, K, allC(1), allG(1)) );
    sumMat(1,2) = sumMat(1,2) + svmtrain(newY, newX, sprintf(options, K, allC(1), allG(2)) );
    sumMat(1,3) = sumMat(1,3) + svmtrain(newY, newX, sprintf(options, K, allC(1), allG(3)) );
    sumMat(2,1) = sumMat(2,1) + svmtrain(newY, newX, sprintf(options, K, allC(2), allG(1)) );
    sumMat(2,2) = sumMat(2,2) + svmtrain(newY, newX, sprintf(options, K, allC(2), allG(2)) );
    sumMat(2,3) = sumMat(2,3) + svmtrain(newY, newX, sprintf(options, K, allC(2), allG(3)) );
    waitbar((i+1)/10);
end
close(bar);
sumMat = sumMat/10;
[~,I] = max(sumMat(:));
[maxRow, maxCol] = ind2sub(size(sumMat),I);

cv_acc = sumMat(maxRow,maxCol);
bestC = allC(maxRow);
bestG = allG(maxCol);

options = '-q  -c %d -g %f -b 1';
for i=classes
    [newX,newY] = sortDataPerClass(X,Y,i);
    newModel = svmtrain(newY, newX, sprintf(options, bestC, bestG) );
    models(i+1) = newModel;
end
fprintf('\n***\nRBF SVM was trained with bestC = %d, bestG = %1.3f\n', bestC, bestG);
end


