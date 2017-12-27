%authors: 
% Matan Finch, id 
% Gilad Eini , id 034744920
%disp('***********temp*************');
% str1 = 'Mercury';
% str2 = 'Mercury2';
% str3 = 'Mercury3';
% sen = {str1, str2, str3};
% sen = sen';
% all = {sen, sen};
% all = all';
% 
% new =  {cat(1, all{:})};
% clear;
% close all;
% clc;
function suc = ClassifyNB_text(Pw, P)
    %get train data vocabulary so we can ignore words we didn't see
    readTrainData('r8-train-stemmed.txt');
    load('corpus_train.mat','cat', 'lbAll', 'texAll', 'Voc');
    trainVocabulary = Voc;%train Vocabulary
    clear cat lbAll texAll Voc;
    
    %get test data
    readTestData('r8-test-stemmed.txt');
    load('corpus_test.mat','cat', 'lbAll', 'texAll', 'Voc');
    %global vars
    labelAll = lbAll;
    textAll = texAll;
    clear cat lbAll texAll Voc;
    
    %total clasified
    total = 0;
    %total clasified correct
    totalCorrect = 0;
    
    %get number of tests to classify
    [numOfRowsInTestData,~] = size(labelAll);
    
    for row=1 : numOfRowsInTestData
        currentRow = textAll{row,1};
        [numberOfWordsInCurrentRow,~] = size(currentRow);
        correctClass = labelAll{row,1};%to check success rate
        classifierClass = '';%what classifier decided
        maxValue = -1;
        for class=1 : 8
            classificationValue = 1;%P(i); %start with the prior and multiply the class conditional
            for wordInd=1 : numberOfWordsInCurrentRow
                word = currentRow{wordInd, 1};
                wordIndxInVoc = find(strcmp(word, trainVocabulary),1);%we need the word's class condition
                if(~isempty(wordIndxInVoc))%ignore empty
                    classificationValue = classificationValue * Pw(class, wordIndxInVoc);
                end
            end
            if(classificationValue > maxValue)
                classifierClass = correctClass;
                maxValue = classificationValue;
            end
        end
        if(strcmp(correctClass, classifierClass))
            totalCorrect = totalCorrect +1;
        end
        total = total + 1;     
    end
    suc = totalCorrect/total;
    

    clear;
    close all;
    clc;
end