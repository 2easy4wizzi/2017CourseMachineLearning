function runFaceRec()
clear;
close all;
clc;
load ('FaceData','trainData');
myFisherFaces(trainData);
% suc=myRecogTest(trainData, testData);
% disp(['success rate:' suc]);
disp('bi');
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
A = normT;
Stag = A'*A;
N = rows;% number of samples
c = 50;%number of classes by defenition
% [V,Z] = eigs(Stag, N-c);
% save ('VDtop50.mat', 'V', 'Z');
load('VDtop50.mat', 'V', 'Z');
Wpca = A*V*Z;
clear N A V Z Stag;

%Wmda- solve SbV=lamda*SwV
Sb = zeros(cols);
miu = (mean(normT))';
for i=1 : 2 : rows
    miuI = (mean(normT(i:i+1, :)))';
    Sb = Sb + 2*(miuI-miu)*(miuI-miu)';
end

Sw = zeros(cols);
for i=1 : 2 : rows
    miuI = (mean(normT(i:i+1, :)))';
    
    Xk1 = (normT(i,:))';
    Sw = Sw + (Xk1-miuI)*(Xk1-miuI)';
    
    Xk2 = (normT(i+1,:))';
    Sw = Sw + (Xk2-miuI)*(Xk2-miuI)';
end

% save ('SbSw.mat', 'Sb', 'Sw');
% load('SbSw.mat', 'Sb', 'Sw');
[Vm,Zm] = eigs(Sb,Sw, c-1);
% [Vm2,Zm2] = eig(Sb,Sw); test this to see max rank is c-1

Wmda = Vm;
WpcaT49firstRows = Wpca';
WpcaT49firstRows = WpcaT49firstRows(1:49,:);
F = Wmda*(WpcaT49firstRows); % F = Wopt
mn = mean(T);
q = reshape(mn,99,99);
figure,imagesc(q);
save ('FisherSpace.mat', 'F', 'mn');
end

function suc=myRecogTest(trainData, testData)
% trainData is a Nxd matrix of training data for N=100 images (2 per person). d is the dimension of the features, representing an image.
% testData testData is a nxd matrix of test data for n=50 images (1 per person). d is the dimension of the features, representing an image.
load ('FisherSpace.mat', 'F', 'mn');

%suc is the number of correct classifications (when the index of the test image is equal to the predicted index)
suc=suc/size(testData,1);
fprintf('The recognition rate is %2.2f\n',suc);
end

