%authors: 
% Matan Finch, id 300895315
% Gilad Eini , id 034744920
function runFaceRec()
clear;
close all;
clc;
load ('FaceData','trainData','testData');
% load FaceData;
FisherFaces(trainData);
suc=recogTest(trainData, testData);
disp(suc);
end