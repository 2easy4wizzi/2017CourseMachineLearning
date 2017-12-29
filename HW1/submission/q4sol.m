%authors: 
% Matan Finch, id 300895315
% Gilad Eini , id 034744920
function q4sol()
    clear;
    close all;
    clc;

    load('ABCD_data.mat','train_data','valid_data','test_data');
    train = train_data;
    valid = valid_data;
    test = test_data;
    clear train_data valid_data test_data;
%     %find best K
%     max = -1000000;
%     bestK = -10000;
%     for k=1 : 5
%         suc1 = knn(train, valid, k);
% %         disp(['succes rate(in fraction) for valid data: ' num2str(suc1) '. K: ' num2str(k)]);
%         if suc1 > max
%            max = suc1; 
%            bestK = k;
%            disp(['best K for valid data so far: ' num2str(bestK) ' with success rate of ' num2str(suc1)]);
%         end
%     end
    
    bestK = 1;
    suc2 = knn(train, test, bestK);
    disp(['succes rate(in fraction) for test data: ' num2str(suc2) '. with K: ' num2str(bestK)]);
    disp('done');
end
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
        correctClass  = testedClasses(i);%need to know what class are we checking now
        ind = knnsearch(train, tested(i,:),'K',k);%look for the nearest k neighbours
        [~,indSize] = size(ind);
        votes = zeros(4,1);%check each neighbour's class and fill the votes array
        for j=1 : indSize
            currentNeighbour = ind(j);
            classOfcurrentNeighbour = trainClasses(currentNeighbour);
            votes( classOfcurrentNeighbour ) = votes( classOfcurrentNeighbour ) + 1;
        end
        %this will get us the class that was selected by the classifier
        %notice: in case of a draw we will take the first class.
        [~,clsifiedClass] = max(votes);%the index of the first maximum is the clsifiedClass
        if(clsifiedClass == correctClass)
            totalCorrect = totalCorrect + 1;
        end
    end
    
    suc = totalCorrect/testedSize;
end


