function [trainData, testData] = permAndSplit(R,trainSize)
    %permute R rows
    [Rrows, ~] = size(R);
    idx = randperm(Rrows);
    X = R(idx,:);
    %split 
    trainData = X(1:trainSize, :);
    testData = X((trainSize+1):end, :);
end