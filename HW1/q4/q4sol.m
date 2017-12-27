%authors: 
% Matan Finch, id 300895315
% Gilad Eini , id 034744920
clear;
close all;
clc;

load('ABCD_data.mat','train_data','valid_data','test_data');
train = train_data;
valid = valid_data;
test = test_data;
clear train_data valid_data test_data;
% trainA = train(1:500,:);
% trainB = train(501:1000,:);
% trainC = train(1001:1500,:);
% trainD = train(1501:2000,:);
% 
% validA = valid(1:100,:);
% validB = valid(101:200,:);
% validC = valid(201:300,:);
% validD = valid(301:400,:);
% 
% testA = test(1:100,:);
% testB = test(101:200,:);
% testC = test(201:300,:);
% testD = test(301:400,:);
% clear train valid test;


k=1;
max = -1000000;
bestK = -10000;
for k=1 : 5
    suc1 = knn(train, valid, k);
    disp(['succes rate(in fraction) for valid data: ' num2str(suc1) '. K: ' num2str(k)]);
    if suc1 > max
       max = suc1; 
       bestK = k;
       disp(['best K for valid data so far: ' num2str(bestK) ' with success rate of ' num2str(suc1)]);
    end
end

suc2 = knn(train, test, bestK);
disp(['succes rate(in fraction) for test data: ' num2str(suc2) '. with K: ' num2str(bestK)]);

disp('done');

function suc = knn(train, tested, k)
    [testedSize, ~] = size(tested);
    %classes:=  A==1 B==2 C==3 D==4
    trainClasses(1:500,1)  = 1;
    trainClasses(501:1000) = 2;
    trainClasses(1001:1500)= 3;
    trainClasses(1501:2000)= 4;
    
    testedClasses(1:100,1) = 1;
    testedClasses(101:200) = 2;
    testedClasses(201:300) = 3;
    testedClasses(301:400) = 4;
    
    totalCorrect = 0;
    
    for i=1 : testedSize
        testedClass  = testedClasses(i);%need to know what class are we checking now
        ind = knnsearch(train, tested(i,:),'K',k);%look for the nearest k neighbours
        correctClass = trainClasses(ind);%this will get us the class that was selected by the classifier
        if(testedClass == correctClass)
            totalCorrect = totalCorrect + 1;
        end
    end
    
    suc = totalCorrect/testedSize;
end


