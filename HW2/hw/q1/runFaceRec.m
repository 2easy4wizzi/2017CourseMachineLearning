function runFaceRec()
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
[rows,cols] = size(T);
%normalizing step
normT = zeros(rows,cols);
for i=1 : cols
    colI = T(:,i);
    mI = mean(colI(:));
    colI = colI - mI;
    stdI = std(colI);
    colI = colI/stdI;
    normT(:,i) = colI;
end
clear colI mI stdI i;
%Wpca=AVZ
A = normT';
Stag = A'*A;
N = rows;% number of samples
c = 50;%number of classes by defenition
[V,Z] = eigs(Stag, N-c);
Wpca = A*V*Z;
clear N A V Z Stag;

%Wmda- solve SbV=lamda*SwV
Sb = zeros(cols);
miu = (mean(normT))';
for i=1 : 2 : rows
    miuI = (mean(normT(i:i+1, :)))';
    Sb = Sb + 2*(miuI-miu)*(miuI-miu)';
end

SbNew = Wpca'*Sb*Wpca;

Sw = zeros(cols);
for i=1 : 2 : rows
    miuI = (mean(normT(i:i+1, :)))';
    
    Xk1 = (normT(i,:))';
    Sw = Sw + (Xk1-miuI)*(Xk1-miuI)';
    
    Xk2 = (normT(i+1,:))';
    Sw = Sw + (Xk2-miuI)*(Xk2-miuI)';
end

SwNew = Wpca'*Sw*Wpca;
clear Sw Sb;
[Vm,~] = eigs(SbNew,SwNew, c-1);

Wmda = Vm';

F = Wmda*Wpca'; % F = Wopt
mn = mean(T);
% q = reshape(mn,99,99);
% figure,imagesc(q);
save ('FisherSpace.mat', 'F', 'mn');
end

function suc=myRecogTest(trainData, testData)
% trainData is a Nxd matrix of training data for N=100 images (2 per person). d is the dimension of the features, representing an image.
% testData testData is a nxd matrix of test data for n=50 images (1 per person). d is the dimension of the features, representing an image.
load ('FisherSpace.mat', 'F', 'mn');

c = 50;
[rows,cols] = size(trainData);
%normalizing step
normT = zeros(rows,cols);
for i=1 : cols
    colI = trainData(:,i);
    mI = mean(colI(:));
    colI = colI - mI;
    stdI = std(colI);
    colI = colI/stdI;
    normT(:,i) = colI;
end

models = zeros(c,cols);
for i=1 : 2 : rows
    
    Xk1 = (normT(i,:));    
    Xk2 = (normT(i+1,:));
    modelI = (Xk1 + Xk2)/2;
    models((i+1)/2,:) = modelI;
end

projectedModels = models* F';

[rows,cols] = size(testData);
%normalizing step
normT = zeros(rows,cols);
for i=1 : cols
    colI = testData(:,i);
    mI = mean(colI(:));
    colI = colI - mI;
    stdI = std(colI);
    colI = colI/stdI;
    normT(:,i) = colI;
end

projectedTestData = normT * F';
ind = knnsearch(projectedModels, projectedTestData);

suc = 0;
for i=1 : rows
   if (i == ind(i))
      suc = suc +1; 
   end
end


%suc is the number of correct classifications (when the index of the test image is equal to the predicted index)
suc=suc/size(testData,1);
fprintf('The recognition rate is %2.2f\n',suc);
end

