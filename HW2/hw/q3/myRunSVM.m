%authors: 
% Matan Finch, id 300895315
% Gilad Eini , id 034744920
function myRunSVM 
    clear;
    close all;
    clc;
    
    rng('default');     
    load ('mnist_svm','trainX', 'trainY');
    % extract 
    tr = ones(1, size(trainX,1));
    tr(1:5:size(trainX,1)) = 0;
    tr = logical(tr);
    trX = trainX( tr,:); trY = trainY( tr); %training set that you use for training and cross validation
    deX = trainX(~tr,:); deY = trainY(~tr); %test set
    allC = [1 5];
    allG = [0.005 0.01 0.05];
    numFolds=5;
    
    %%%LINIAR REGRESSION
    % For each class learn models for this class in one-against-all manner 
    % train linear SVM and use 5-fold cross-validation to find the best C parameter
    % from allC.  
    [models,cv_acc]=myTrainLinearSVM(trX,trY,numFolds,allC);
    fprintf('Cross-validation accuracy is %2.2f\n',cv_acc);
    mypause();
    %test linear SVM for each class in one-against-all manner and report the
    %accuracy rate for each class
    acc=myTestSVM(models, deX,deY);
    fprintf('Test accuracy is %2.2f\n',acc);
    mypause();
    
    %%%RBF KERNEL
    %train kernel SVM with RBF kernel and cross-validation to find the best
    %C and G parameters from allC and allG.
    [models,cv_acc]=myTrainRBFSVM(trX,trY,numFolds, allC,allG);
    fprintf('Cross-validation accuracy of RBF SVM is %2.2f\n',cv_acc);
    mypause();
    %test kernel SVM for each class in one-against-all manner and report the
    %accuracy rate for each class
    acc=myTestSVM(models, deX,deY);
    fprintf('Test accuracy of RBF SVM is %2.2f\n',acc);
    fprintf('ALL DONE\n');
end  

function [models,cv_acc]=myTrainLinearSVM(X,Y,K, allC)
%Train linear SVM and use K-fold cross validation to find the best parameter (one for all classes)  from allC
% X is Nxd matrix of  training data, N is the number of items, d is the size of the feature vector
% Y is Nx1 vector of labels from 0 to 9 (meaning that there are 10 classes).
%K is the number of folds in cross validation.
% models is the array of model structes for the 10 classes obtained from running linear SVM training (using LIBSVM)
% cv_acc is the cross-validation accuracy rate (number of times the predicted label is equal to the true label devided by the number of validation points.)
% The rate should be averaged over K folds. 
%Implement your code here. Note that the multi-class classifiers should be trained in one-against-all manner.

classes =  0 : 9;%10 classes - zero to nine
sumFirstC = 0;%meassure cross-validation accuracy rate with C=1
sumSecondC = 0;%meassure cross-validation accuracy rate with C=5

for i=classes
    [sortXperClassI,sortYperClassI] = sortDataPerClass(X,Y,i);
    op1 = ['-q -v ',num2str(K),' -c ', num2str(allC(1))];
    op2 = ['-q -v ',num2str(K),' -c ', num2str(allC(2))];
    sumFirstC  = sumFirstC  + svmtrain(sortYperClassI, sortXperClassI, op1); %#ok<*SVMTRAIN>
    sumSecondC = sumSecondC + svmtrain(sortYperClassI, sortXperClassI, op2);
end

sumFirstC = sumFirstC / 10;
sumSecondC = sumSecondC / 10;
disp(['sumFirstC: ' num2str(sumFirstC) ' sumSecondC:' num2str(sumSecondC)]);

if(sumFirstC > sumSecondC)
    cChosenStr = num2str(allC(1));
    cv_acc = sumFirstC;
else
    cChosenStr = num2str(allC(2));
    cv_acc = sumSecondC;
end

for i=classes%calculate models using best C found from [1 5]
    [sortXperClassI,sortYperClassI] = sortDataPerClass(X,Y,i);
    op = ['-q -c ', num2str(cChosenStr)];
    modelI = svmtrain(sortYperClassI, sortXperClassI, op);
    models(i+1) = modelI;
end

end

function [sortXperClassI,sortYperClassI] = sortDataPerClass(samples,labels,classI)
    [rows,~] = size(labels);
    indexOnes = 1;
    indexMinusOnes = 1;
    for r=1 : rows
        if(labels(r) == classI)
           onesYArray(indexOnes) = 1;
           onesXArray(indexOnes,:) = samples(r,:);
           indexOnes = indexOnes + 1;
        else
           minOnesYArray(indexMinusOnes) = -1;
           minOnesXArray(indexMinusOnes,:) = samples(r,:);
           indexMinusOnes = indexMinusOnes + 1;
        end
    end   
    sortXperClassI = vertcat(onesXArray,minOnesXArray);
    sortYperClassI = vertcat(onesYArray',minOnesYArray');
end

function acc=myTestSVM(models, X,Y)
% Function for testing SVM.
% X is nxd matrix, where n is the number of tested samples and d is the dimension of the features vectors.
% Y is nx1 vector, containing the true labels. Use it to compute the accuracy.
% models is an array of model structes for the 10 classes obtained from running SVM training (using LIBSVM).
% acc is the tested accuracy rate: (the number of times the predicted label is equal to the true label devided by the number of tested points.)
%Implement the function here. Note that the multi-class classification should be implemented in one-against-all manner.
acc = 0; 
classes =  0 : 1 : 9;

for i=classes
    [sortXperClassI,sortYperClassI] = sortDataPerClass(X,Y,i);
    [predicted_label] = svmpredict(sortYperClassI, sortXperClassI, models(i+1));
end

% if(sumFirstC > sumSecondC)
%     cChosenStr = cFirstStr;
%     cv_acc = sumFirstC;
% else
%     cChosenStr = cSecondStr;
%     cv_acc = sumSecondC;
% end
end

function [models,cv_acc]=myTrainRBFSVM(X,Y,K, allC,allG)
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

bar=waitbar(0,'Cross Val RBF SVM...');
for i=classes
    [sortXperClassI,sortYperClassI] = sortDataPerClass(X,Y,i);
    
    op11 = ['-q -v ',num2str(K),' -c ', num2str(allC(1)), ' -g ' , num2str(allG(1))];
    sumMat(1,1) = sumMat(1,1) + svmtrain(sortYperClassI, sortXperClassI, op11);
    
    op12 = ['-q -v ',num2str(K),' -c ', num2str(allC(1)), ' -g ' , num2str(allG(2))];
    sumMat(1,2) = sumMat(1,2) + svmtrain(sortYperClassI, sortXperClassI, op12);
    
    op13 = ['-q -v ',num2str(K),' -c ', num2str(allC(1)), ' -g ' , num2str(allG(3))];
    sumMat(1,3) = sumMat(1,3) + svmtrain(sortYperClassI, sortXperClassI, op13);
    
    op21 = ['-q -v ',num2str(K),' -c ', num2str(allC(2)), ' -g ' , num2str(allG(1))];
    sumMat(2,1) = sumMat(2,1) + svmtrain(sortYperClassI, sortXperClassI, op21);
    
    op22 = ['-q -v ',num2str(K),' -c ', num2str(allC(2)), ' -g ' , num2str(allG(2))];
    sumMat(2,2) = sumMat(2,2) + svmtrain(sortYperClassI, sortXperClassI, op22);
    
    op23 = ['-q -v ',num2str(K),' -c ', num2str(allC(2)), ' -g ' , num2str(allG(3))];
    sumMat(2,3) = sumMat(2,3) + svmtrain(sortYperClassI, sortXperClassI, op23);
    waitbar((i+1)/10);
end
close(bar);

sumMat = sumMat/10;
[~,I] = max(sumMat(:));
[maxRow, maxCol] = ind2sub(size(sumMat),I);

cv_acc = sumMat(maxRow,maxCol);
cChosenStr = allC(maxRow);
gChosenStr = allG(maxCol);

for i=classes
    [sortXperClassI,sortYperClassI] = sortDataPerClass(X,Y,i);
    op = ['-q -c ', num2str(cChosenStr), ' -g ' , num2str(gChosenStr)];
    modelI = svmtrain(sortYperClassI, sortXperClassI, op);
    models(i+1) = modelI;
end

end
