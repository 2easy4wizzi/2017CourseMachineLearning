function runSVM 
    clear;
    close all;
    clc;
    
    rng('default');     
    load mnist_svm;

    % extract 
    tr = ones(1, size(trainX,1));
    tr(1:5:size(trainX,1)) = 0;
    tr = logical(tr);
    trX = trainX( tr,:); trY = trainY( tr); %training set that you use for training and cross validation
    deX = trainX(~tr,:); deY = trainY(~tr); %test set

    allC = [1 5];
    allG = [0.005 0.01 0.05];
    numFolds=5;
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
    %train kernel SVM with RBF kernel and cross-validation to find the best
    %C and G parameters from allC and allG.
    [models,cv_acc]=trainRBFSVM(trX,trY,numFolds, allC,allG);
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
firstC  = allC(1); sumFirstC = 0; cFirstStr  = [' -c ' num2str(firstC)  ' '];
secondC = allC(2); sumSecondC = 0;cSecondStr = [' -c ' num2str(secondC) ' '];  

classes =  0 : 1 : 9;
kstr    = [' -v ' num2str(K) ' '];
temp1    = [' -s  ' num2str(0) ' '];
temp2    = [' -t  ' num2str(2) ' '];
for i=classes
    [sortXperClassI,sortYperClassI] = sortDataPerClass(X,Y,i);
    op1 = [kstr cFirstStr temp1 temp2];
    op2 = [kstr cSecondStr temp1 temp2];
    sumFirstC  = sumFirstC  + svmtrain(sortYperClassI, sortXperClassI, op1);
    sumSecondC = sumSecondC + svmtrain(sortYperClassI, sortXperClassI, op2);
end
sumFirstC = sumFirstC / 10;
sumSecondC = sumSecondC / 10;

if(sumFirstC > sumSecondC)
    cChosenStr = cFirstStr;
    cv_acc = sumFirstC;
else
    cChosenStr = cSecondStr;
    cv_acc = sumSecondC;
end

for i=classes
    [sortXperClassI,sortYperClassI] = sortDataPerClass(X,Y,i);
    op = [cChosenStr];
    modelI = svmtrain(sortYperClassI, sortXperClassI, op);
    models(i+1) = modelI;
end

end

function [sortXperClassI,sortYperClassI] = sortDataPerClass(X,Y,i)
    [rows,~] = size(Y);
    indexOnes = 1;
    indexMinusOnes = 1;
    for r=1 : rows
        if(Y(r) == i)
           onesYArray(indexOnes) = 1;
           onesXArray(indexOnes,:) = X(r,:);
           indexOnes = indexOnes + 1;
        else
           minOnesYArray(indexMinusOnes) = -1;
           minOnesXArray(indexMinusOnes,:) = X(r,:);
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

function [models,cv_acc]=trainRBFSVM(X,Y,K, allC,allG)
%Train RBF kernel SVM and use K-fold cross validation to find the best parameters (one set for all classes)  from allC and allG
% X is Nxd matrix of  training data, N is the number of items, d is the size of the feature vector
% Y is Nx1 vector of labels from 0 to 9 (meaning that there are 10 classes)
% models is the array of model structes for the 10 classes obtained from running linear SVM training (using LIBSVM)
%K is the number of folds in cross validation.
% cv_acc is the cross-validation accuracy rate (number of times the predicted label is equal to the true label devided by the number of validation points.)
% The rate should be averaged over K folds. 
%Implement your code here. Note that the multi-class classifiers should be trained in one-against-all manner.
firstC  = allC(1); cFirstStr  = [' -c ' num2str(firstC)  ' '];
secondC = allC(2); cSecondStr = [' -c ' num2str(secondC) ' '];  
firstG = allG(1);  firstGStr   = [' -g ' num2str(firstG) ' '];
secondG = allG(2); secondGStr = [' -g ' num2str(secondG) ' '];
thirdG = allG(3);  thirdGStr   = [' -g ' num2str(thirdG) ' '];
sum11 = 0; sum12 = 0; sum13 = 0; sum21 = 0; sum22 = 0; sum23 = 0;
classes =  0 : 1 : 9;
kstr       = [' -v ' num2str(K) ' '];

for i=classes
    [sortXperClassI,sortYperClassI] = sortDataPerClass(X,Y,i);
    
    op11 = [kstr cFirstStr firstGStr];
    sum11  = sum11  + svmtrain(sortYperClassI, sortXperClassI, op11);
    
    op12 = [kstr cFirstStr secondGStr];
    sum12  = sum12  + svmtrain(sortYperClassI, sortXperClassI, op12);
    
    op13 = [kstr cFirstStr thirdGStr];
    sum13  = sum13  + svmtrain(sortYperClassI, sortXperClassI, op13);
    
    op21 = [kstr cSecondStr firstGStr];
    sum21 = sum21 + svmtrain(sortYperClassI, sortXperClassI, op21);
    
    op22 = [kstr cSecondStr secondGStr];
    sum22 = sum22 + svmtrain(sortYperClassI, sortXperClassI, op22);
    
    op23 = [kstr cSecondStr thirdGStr];
    sum23 = sum23 + svmtrain(sortYperClassI, sortXperClassI, op23);
    
end
sum11 = sum11/10;
sum12 = sum12/10;
sum13 = sum13/10;
sum21 = sum21/10;
sum22 = sum22/10;
sum23 = sum23/10;

vec = [sum11 sum12 sum13 sum21 sum22 sum23];
maxSum = max(vec);
switch maxSum
case sum11
    cChosenStr = cFirstStr;
    gChosenStr = firstGStr;
    cv_acc = sum11;
case sum12
    cChosenStr = cFirstStr;
    gChosenStr = secondGStr;
    cv_acc = sum12;
case sum13
    cChosenStr = cFirstStr;
    gChosenStr = thirdGStr;
    cv_acc = sum13;
case sum21
    cChosenStr = cSecondStr;
    gChosenStr = firstGStr;
    cv_acc = sum21;
case sum22
    cChosenStr = cSecondStr;
    gChosenStr = secondGStr;
    cv_acc = sum22;
case sum23
    cChosenStr = cSecondStr;
    gChosenStr = thirdGStr;
    cv_acc = sum23;
otherwise
 disp('');
end

for i=classes
    [sortXperClassI,sortYperClassI] = sortDataPerClass(X,Y,i);
    op = [cChosenStr gChosenStr];
    modelI = svmtrain(sortYperClassI, sortXperClassI, op);
    models(i+1) = modelI;
end

end
