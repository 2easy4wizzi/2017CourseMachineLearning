function acc=testSVM(models, X,Y)
% Function for testing SVM.
% X is nxd matrix, where n is the number of tested samples and d is the dimension of the features vectors.
% Y is nx1 vector, containing the true labels. Use it to compute the accuracy.
% models is an array of model structes for the 10 classes obtained from running SVM training (using LIBSVM).
% acc is the tested accuracy rate: (the number of times the predicted label is equal to the true label devided by the number of tested points.)
%Implement the function here. Note that the multi-class classification should be implemented in one-against-all manner.
labels =  0 : 1 : 9;

%create probability matrix: a row for each test. a column for each class
%so each test will have a row and the maxium value will correlate him to
%that class(the colums are classes).
probability = zeros(size(Y,1),numel(labels));
newY = zeros(size(Y));
for i=labels
    newY(Y ~= i) = -1;%change Y to +1 and -1 by class i
    newY(Y == i) = 1;
    [~,~,p] = svmpredict(newY, X, models(i+1), '-b 1');%predict and use only the probability
    probability(:,i+1) = p(:,models(i+1).Label==1);%p is 2:testNumber. each of the columns belongs to models(i+1).Label. we care just for the label 1.
end

acc=0;
for i=1 : size(Y,1)
    [~,I] = max(probability(i,:));%get the index of the max
    if(Y(i) == double(I(1) -1))%check if Yi is the same like the index of the max. (the -1 is cus class 0 to 9 and models 1-10)
        acc = acc + 1;
    end
end
acc = acc/size(Y,1);
end