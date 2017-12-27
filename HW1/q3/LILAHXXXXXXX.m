clear;
close all;
clc;

load('dataAB.mat','train_dataA','train_dataB','test_dataA','test_dataB','valid_dataA','valid_dataB');
err1 = VoteLayer(train_dataA,train_dataB,valid_dataA,valid_dataB,2.0);
err2 = VoteLayer(train_dataA,train_dataB,test_dataA,test_dataB,2.0);
disp(err2);


function [ successRate ] = VoteLayer( train_dataA,train_dataB,valid_dataA,valid_dataB,sigma )
[m, ~] = size(train_dataA);
[mv, ~] = size(valid_dataA);
[mvB, ~] = size(valid_dataB);

c = 0;
for i=1:mv
    nextLayerVectorAA = zeros(1, m);
    nextLayerVectorAB = zeros(1, m);
    for j=1:m
     nextLayerVectorAA(1,j) = GaussianEval(valid_dataA(i,:),train_dataA(j,:),sigma);
     nextLayerVectorAB(1,j) = GaussianEval(valid_dataA(i,:),train_dataB(j,:),sigma);
    end
    avg1 = GetAverageGaussians(nextLayerVectorAA,sigma);
    avg2 = GetAverageGaussians(nextLayerVectorAB,sigma);
    if avg1>avg2
        c=c+1;
    end
end

for i=1:mvB
    nextLayerVectorBB = zeros(1, m);
    nextLayerVectorBA = zeros(1, m);
    for j=1:m
     nextLayerVectorBB(1,j) = GaussianEval(valid_dataB(i,:),train_dataB(j,:),sigma);
     nextLayerVectorBA(1,j) = GaussianEval(valid_dataB(i,:),train_dataA(j,:),sigma);
    end
    avg1B = GetAverageGaussians(nextLayerVectorBB,sigma);
    avg2B = GetAverageGaussians(nextLayerVectorBA, sigma);
    if avg1B>avg2B
        c=c+1;
    end
end
successRate = 100*(c/(length(valid_dataA)+length(valid_dataB)));
end

function [ val ] = GetAverageGaussians( vectorOfPreviusLayer,sigma )
%GETAVERAGEGAUSSIANS Summary of this function goes here
%   Detailed explanation goes here

val = (sum(vectorOfPreviusLayer)/length(vectorOfPreviusLayer));

end

function [ val ] = GaussianEval( inputX , trainDataVector, sigma )
%GAUSSIANEVAL Summary of this function goes here
%   Detailed explanation goes herec
val = (1/sqrt(2*pi)*sigma)*exp(-1*power(norm(inputX-trainDataVector),2)/(2*power(sigma,2)));
end