%authors: 
% Matan Finch, id 300895315
% Gilad Eini , id 034744920
function suc=recogTest(trainData, testData)
% trainData is a Nxd matrix of training data for N=100 images (2 per person). d is the dimension of the features, representing an image.
% testData testData is a nxd matrix of test data for n=50 images (1 per person). d is the dimension of the features, representing an image.
load ('FisherSpace.mat', 'F', 'mn');

c = 50;
[rows,cols] = size(trainData);
normTrainData=normalizeStep(trainData);%normalizing step

models = zeros(c,cols);%calculate 50 models
for i=1 : 2 : rows
    Xk1 = (normTrainData(i,:));%get first sample of person i
    Xk2 = (normTrainData(i+1,:));%get second sample of person i
    modelI = (Xk1 + Xk2)/2;%avrage them
    models((i+1)/2,:) = modelI;%save as model i
end

projectedModels = models* F';%reduce dims using Wopt

[rows,~] = size(testData);
normTestData=normalizeStep(testData);%normalizing step

projectedTestData = normTestData * F';%reduce dims using Wopt
ind = knnsearch(projectedModels, projectedTestData);%classify using Knn(k=1)

suc = 0;%count suc rate - correct:= if ind(i) == i
for i=1 : rows
   if (i == ind(i))
      suc = suc +1; 
   end
end

%suc is the number of correct classifications (when the index of the test image is equal to the predicted index)
suc=suc/size(testData,1);
fprintf('The recognition rate is %2.2f\n',suc);
end