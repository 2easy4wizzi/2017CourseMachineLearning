%authors: 
% Matan Finch, id 300895315
% Gilad Eini , id 034744920
function [models,cv_acc]=trainLinearSVM(X,Y,K, allC)
%Train linear SVM and use K-fold cross validation to find the best parameter (one for all classes)  from allC
% X is Nxd matrix of  training data, N is the number of items, d is the size of the feature vector
% Y is Nx1 vector of labels from 0 to 9 (meaning that there are 10 classes).
%K is the number of folds in cross validation.
% models is the array of model structes for the 10 classes obtained from running linear SVM training (using LIBSVM)
% cv_acc is the cross-validation accuracy rate (number of times the predicted label is equal to the true label devided by the number of validation points.)
% The rate should be averaged over K folds. 
%Implement your code here. Note that the multi-class classifiers should be trained in one-against-all manner.

labels =  0 : 9;%10 classes - zero to nine
c1 = 0;%meassure cross-validation accuracy rate with C=1
c5 = 0;%meassure cross-validation accuracy rate with C=5
options = '-q -v %d -c %d -b 1';

for i=labels%calculate cross validation rate. for each class with c=1 and c=5
    [newX,newY] = sortDataPerClass(X,Y,i);%sort data. on top: feature vectors and labels that belong to the ith class. the bottom: all the rest. 
    c1 = c1 + svmtrain(newY, newX, sprintf(options, K, allC(1))); 
    c5 = c5 + svmtrain(newY, newX, sprintf(options, K, allC(2)));
end
c1 = c1 / numel(labels); %normalize by classes number
c5 = c5 / numel(labels);

% fprintf('with c1: %3.4f and with c5: %3.4f\n',c1,c5);
%choose best c and take the CV accuracy
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
fprintf('\n***\nLinearSvm was trained with bestC = %d \n***\n',bestC);
end