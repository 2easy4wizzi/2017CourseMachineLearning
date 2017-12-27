%authors:
% Matan Finch, id 300895315
% Gilad Eini , id 034744920
clear;
close all;
clc;

load('dataAB.mat','train_dataA','train_dataB','valid_dataA','valid_dataB','test_dataA','test_dataB');

trainA = train_dataA;
trainB = train_dataB;
validA = valid_dataA;
validB = valid_dataB;
testA = test_dataA;
testB = test_dataB;
clear train_dataA train_dataB valid_dataA valid_dataB test_dataA test_dataB;


% windowSize = 2;

% windowSize = 2;
%suc = pnn(trainA, trainB, validA, validB);
%disp(['succes rate(in fraction) for valid data: ' num2str(suc)]);


suc = pnn(trainA, trainB, testA, testB);
disp(['succes rate(in fraction) for test data: ' num2str(suc)]);
disp('done');




