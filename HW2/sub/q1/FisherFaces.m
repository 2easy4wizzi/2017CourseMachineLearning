%authors: 
% Matan Finch, id 300895315
% Gilad Eini , id 034744920
function FisherFaces(T)
%T is the Nxd training data for N=100 images (2 per person). d is the dimension of the features, representing an image.
%Implement the code (see instruction in the pdf file)
%F is the low-dimensional basis and mn if the mean of the training set.
normT=normalizeStep(T);%normalizing step

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
 