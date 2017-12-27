%authors: 
% Matan Finch, id 300895315
% Gilad Eini , id 034744920
clear;
close all;
clc;
[Pw, P] = learn_NB_text();
suc = ClassifyNB_text(Pw, P);
disp(['success rate on NBc: ' num2str(suc)]);

%Q2 naive bayes alg
function [Pw, P] = learn_NB_text()
    P = zeros(8,1);
    
    %readTrainData('r8-train-stemmed.txt');
    load('corpus_train.mat','cat', 'lbAll', 'texAll', 'Voc');
    %global vars
    categories = cat;
    labelAll = lbAll;
    textAll = texAll;
    vocabulary = Voc;%Vocabulary
    clear cat lbAll texAll Voc;
    %get num of rows (each row is a sample)
    [numOfRowsInTrainData,~] = size(labelAll); %|EXAMPLES|
    [numOfWordsInVoc,~] = size(vocabulary); %|Vocabulary|
    Pw = zeros(8,numOfWordsInVoc);
    
    
    for i=1 : 8 
        %get all of current Ci rows (all samples of class Ci)
        docCi = textAll(strcmp(labelAll, categories{i,1}));%DOCi
        %get the size of Ci's samples
        [numRowOfCi,~] = size(docCi);%|DOCi| 
        %write prior of class Ci on Pi vector(prior of ci:= (ci samples) / (total samples) )
        P(i) = numRowOfCi/numOfRowsInTrainData;
        %cancat all Ci rows
        TextCi = {cat(1, docCi{:})};%Texti
        [TextCiTotalWords,~] = size(TextCi{1,1});%|Texti|
        %calculate P(word|Ci)
        for j=1 : numOfWordsInVoc
            word = vocabulary{j,1};%go over on all the words in Vocabulary
            [wordCountInTextCi,~] = size(find(strcmp(word, TextCi{1,1})));%how much times a word appears TextCi
            Pw(i,j) = (wordCountInTextCi + 1) / (numOfWordsInVoc + TextCiTotalWords);%P(word|Ci)
        end
        disp('done number');
        disp(i);
    end
end

function suc = ClassifyNB_text(Pw, P)

    %get train data vocabulary so we can ignore words we didn't see
    %readTrainData('r8-train-stemmed.txt');
    load('corpus_train.mat','cat', 'lbAll', 'texAll', 'Voc');
    trainVocabulary = Voc;%train Vocabulary
    categories = cat;
    clear cat lbAll texAll Voc;
    
    %get test data
    %readTestData('r8-test-stemmed.txt');
    load('corpus_test.mat','cat', 'lbAll', 'texAll', 'Voc');
    %global vars
    labelAll = lbAll;
    textAll = texAll;
    clear cat lbAll texAll Voc;
    
    %total clasified correct
    totalCorrect = 0;
    %if a test is made from words that NON of them appear in the
    %vocabulary, we will ignore this test
    testDisqualified = 0;
    %get number of tests to classify
    [numOfRowsInTestData,~] = size(labelAll);
    
    for row=1 : numOfRowsInTestData
        currentRow = textAll{row,1};
        [numberOfWordsInCurrentRow,~] = size(currentRow);
        correctClass = labelAll{row,1};%to check success rate
        classifierClass = '';%what classifier decided
        maxValue = -1000000000;
        ignoreThisTest = true;%a flag that indicate that this test has at least one word in the vocabulary
        for class=1 : 8
            classificationValue = log(P(class)); %start with the prior and multiply the class conditional
            for wordInd=1 : numberOfWordsInCurrentRow
                word = currentRow{wordInd, 1};
                wordIndxInVoc = find(strcmp(word, trainVocabulary),1);%we need the word's class condition
                if(~isempty(wordIndxInVoc))%only if found in vocabulary
                    ignoreThisTest = false;
                    classificationValue = classificationValue + log(Pw(class, wordIndxInVoc));
                end
            end
            if(classificationValue > maxValue)
                classifierClass = categories{class,1};
                maxValue = classificationValue;
            end
        end
        if(ignoreThisTest == true)
            testDisqualified = testDisqualified+1;
        end
        if(strcmp(correctClass, classifierClass))
            totalCorrect = totalCorrect +1;
        end
%         disp(['done test number: ' num2str(row) ' .correct so far:' num2str(totalCorrect)]);
    end
    suc = totalCorrect/(numOfRowsInTestData-testDisqualified);
end


