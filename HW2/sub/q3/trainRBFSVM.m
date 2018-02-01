function [models,cv_acc]=trainRBFSVM(X,Y,K, allC,allG)
%Train RBF kernel SVM and use K-fold cross validation to find the best parameters (one set for all classes)  from allC and allG
% X is Nxd matrix of  training data, N is the number of items, d is the size of the feature vector
% Y is Nx1 vector of labels from 0 to 9 (meaning that there are 10 classes)
% models is the array of model structes for the 10 classes obtained from running linear SVM training (using LIBSVM)
%K is the number of folds in cross validation.
% cv_acc is the cross-validation accuracy rate (number of times the predicted label is equal to the true label devided by the number of validation points.)
% The rate should be averaged over K folds. 
%Implement your code here. Note that the multi-class classifiers should be trained in one-against-all manner.

sumMat = zeros(2,3);% i.e sumMat(1,3): allC(1) and allG(3)
classes =  0 : 1 : 9;
options = '-q -v %d -c %d -g %f -b 1';

bar=waitbar(0,'Cross Val RBF SVM...');
for i=classes%calculate cross validation rate. for each class with all c's and all g's
    [newX,newY] = sortDataPerClass(X,Y,i);%sort data. on top: feature vectors and labels that belong to the ith class. the bottom: all the rest. 
    sumMat(1,1) = sumMat(1,1) + svmtrain(newY, newX, sprintf(options, K, allC(1), allG(1)) );
    sumMat(1,2) = sumMat(1,2) + svmtrain(newY, newX, sprintf(options, K, allC(1), allG(2)) );
    sumMat(1,3) = sumMat(1,3) + svmtrain(newY, newX, sprintf(options, K, allC(1), allG(3)) );
    sumMat(2,1) = sumMat(2,1) + svmtrain(newY, newX, sprintf(options, K, allC(2), allG(1)) );
    sumMat(2,2) = sumMat(2,2) + svmtrain(newY, newX, sprintf(options, K, allC(2), allG(2)) );
    sumMat(2,3) = sumMat(2,3) + svmtrain(newY, newX, sprintf(options, K, allC(2), allG(3)) );
    waitbar((i+1)/numel(classes));
end
close(bar);
sumMat = sumMat/numel(classes);%normalize by classes number
[~,I] = max(sumMat(:));%get index of max
[maxRow, maxCol] = ind2sub(size(sumMat),I);%use index to get row and col of max

cv_acc = sumMat(maxRow,maxCol);%save best acc
bestC = allC(maxRow);%bestC
bestG = allG(maxCol);%bestG

options = '-q  -c %d -g %f -b 1';
for i=classes%calculate models using best C found from allC and bestG found from allG 
    [newX,newY] = sortDataPerClass(X,Y,i);
    newModel = svmtrain(newY, newX, sprintf(options, bestC, bestG) );
    models(i+1) = newModel;
end
fprintf('\n***\nRBF SVM was trained with bestC = %d, bestG = %1.3f\n***\n', bestC, bestG);
end