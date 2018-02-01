%authors: 
% Matan Finch, id 300895315
% Gilad Eini , id 034744920
function [newX,newY] = sortDataPerClass(samples,labels,classI)
    mat = [labels samples];%combine to one matrix
    ind = (mat(:,1) == classI);%get indices of class i
    newMat = [ mat(ind,:); mat(~ind,:)];%create new matrix where top rows belong to ith class
    newY = newMat(:,1);%splice the matrix to original form - labels and samples
    newY(newY ~= classI) = -1;%binarize - +1 to positive samples and -1 to the rest
    newY(newY == classI) = 1;
    newX = newMat(:,2:end);%splice samples matrix
end
