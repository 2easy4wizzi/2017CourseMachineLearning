%authors: 
% Matan Finch, id 300895315
% Gilad Eini , id 034744920
function myRunFaceRec()
clear;
close all;
clc;
load ('FaceData','trainData','testData');
myFisherFaces(trainData);
suc=myRecogTest(trainData, testData);
disp(['success rate:' num2str(suc)]);
end

function myFisherFaces(T)
%T is the Nxd training data for N=100 images (2 per person). d is the dimension of the features, representing an image.
%Implement the code (see instruction in the pdf file)
%F is the low-dimensional basis and mn if the mean of the training set.
normT=myNormalizeStep(T);%normalizing step

%Wpca=AVZ
A = normT';
Stag = A'*A;
[rows,cols] = size(T);
N = rows;% number of samples
c = 50;%number of classes by defenition
[V,Z] = eigs(Stag, N-c);%top N-c eigenValues and eigenVectors
Wpca = A*V*Z;

%Wmda- solve SbV=lamda*SwV
Sb = zeros(cols);%Sbetween classes
miu = (mean(normT))';%mean of all samples
for i=1 : 2 : rows
    miuI = (mean(normT(i:i+1, :)))';%mean of the 2 samples of this person
    Sb = Sb + 2*(miuI-miu)*(miuI-miu)';%distance from the mean of all data
end

SbNew = Wpca'*Sb*Wpca;%Sbetween reduced dims

Sw = zeros(cols);%Swithin classes
for i=1 : 2 : rows
    miuI = (mean(normT(i:i+1, :)))';%mean of the 2 samples of this person
    
    Xk1 = (normT(i,:))';
    Sw = Sw + (Xk1-miuI)*(Xk1-miuI)';%distance of the sample from the mean of the class
    
    Xk2 = (normT(i+1,:))';
    Sw = Sw + (Xk2-miuI)*(Xk2-miuI)';%distance of the sample from the mean of the class
end

SwNew = Wpca'*Sw*Wpca;%Swetween reduced dims
[Vm,~] = eigs(SbNew,SwNew, c-1);%top N-c eigenVectors. dont need eigenValues.

Wmda = Vm';

F = Wmda*Wpca'; % F = Wopt
mn = mean(T); %mean of all samples

save ('FisherSpace.mat', 'F', 'mn');
end

function suc=myRecogTest(trainData, testData)
% trainData is a Nxd matrix of training data for N=100 images (2 per person). d is the dimension of the features, representing an image.
% testData testData is a nxd matrix of test data for n=50 images (1 per person). d is the dimension of the features, representing an image.
load ('FisherSpace.mat', 'F', 'mn');

c = 50;
[rows,cols] = size(trainData);
normTrainData=myNormalizeStep(trainData);%normalizing step

models = zeros(c,cols);%calculate 50 models
for i=1 : 2 : rows
    Xk1 = (normTrainData(i,:));%get first sample of person i
    Xk2 = (normTrainData(i+1,:));%get second sample of person i
    modelI = (Xk1 + Xk2)/2;%avrage them
    models((i+1)/2,:) = modelI;%save as model i
end

projectedModels = models* F';%reduce dims using Wopt

[rows,~] = size(testData);
normTestData=myNormalizeStep(testData);%normalizing step

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

function normData=myNormalizeStep(data)
[rows,cols] = size(data);
normData = zeros(rows,cols);
for i=1 : cols
    colI = data(:,i);
    mI = mean(colI(:));
    colI = colI - mI;
    stdI = std(colI);
    colI = colI/stdI;
    normData(:,i) = colI;
end
end

