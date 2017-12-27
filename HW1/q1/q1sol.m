%authors: 
% Matan Finch, id 300895315
% Gilad Eini , id 034744920

%Q1
%taks A: create 300 samples for each class (C1, C2)
sampleSize = 300;
%create Data for C1 ~N(-2,1)
miuC1 = -2;
sigC1 = 1;
%dataC1 = normrnd(miuC1, sigC1, [300 1]); 
dataC1 = sigC1*randn(1,sampleSize) + miuC1;

%create Data for C2 ~N(1,1.5)
miuC2 = 1;
sigC2 = 1.5;
%dataC2 = normrnd(miuC2, sigC2, [300 1]);
dataC2 = sigC2*randn(1,sampleSize) + miuC2;

%Task B: estimate paramters for each class and comapare to real values
%estimate miu,sig for C1 and C2
estimatedMiuC1 = (1/sampleSize)*sum(dataC1);
estimatedSigC1 = sqrt((1/sampleSize)*sum((dataC1-estimatedMiuC1).^2));
estimatedMiuC2 = (1/sampleSize)*sum(dataC2);
estimatedSigC2 = sqrt((1/sampleSize)*sum((dataC2-estimatedMiuC2).^2));

%plot C1 real distribution vs C2 real distribution
plotScale = -6:.05:8;
gausRealC1 = normpdf(plotScale,miuC1,sigC1);
gausRealC2 = normpdf(plotScale,miuC2,sigC2);
figure;
plot(plotScale,gausRealC1,plotScale,gausRealC2);
title('Real distributions of C1 and C2');
legend(['C1~N(' num2str(miuC1) ',' num2str(sigC1) ')'], ['C2~N(' num2str(miuC2) ',' num2str(sigC2) ')']);
 

%plot C1 estimated distribution vs C2 estimated distribution
gausEstimateC1 = normpdf(plotScale,estimatedMiuC1,estimatedSigC1);
gausEstimateC2 = normpdf(plotScale,estimatedMiuC2,estimatedSigC2);
figure;
plot(plotScale,gausEstimateC1,plotScale,gausEstimateC2);
title('Estimated distributions of C1 and C2'); 
legend(['C1~N(' num2str(estimatedMiuC1) ',' num2str(estimatedSigC1) ')'], ['C2~N(' num2str(estimatedMiuC2) ',' num2str(estimatedSigC2) ')']);
 

%Task C: make a decision rule
%solve P(X|C1) == P(X|C2):
syms x;
eqnLeft  = (1/(estimatedSigC1*sqrt(2*pi)))*exp(-(x-estimatedMiuC1)^2/(2*estimatedSigC1^2));%P(X|C1)*lamdaC2C1 // lamdaC2C1 == lamdaC1C2 == 1
eqnRight = (1/(estimatedSigC2*sqrt(2*pi)))*exp(-(x-estimatedMiuC2)^2/(2*estimatedSigC2^2));%P(X|C2)*lamdaC1C2
solx = solve( eqnLeft == eqnRight , x);
solx = double(solx);
solx = solx(2,1);
%add to plot the decision rule
figure;
plot(plotScale,gausEstimateC1,plotScale,gausEstimateC2);
hold on
y1=get(gca,'ylim');
plot([solx solx],y1);
hold off
title('Estimated distributions of C1 and C2 with decision rule'); 
legend(['C1~N(' num2str(estimatedMiuC1) ',' num2str(estimatedSigC1) ')'], ['C2~N(' num2str(estimatedMiuC2) ',' num2str(estimatedSigC2) ')'], ['decision: ' num2str(solx)]);
%calculate error
%all above solx(decision) and belongs to C1 is error
misClassifyC1 = sum(dataC1>solx);
%all bellow solx(decision) and belongs to C2 is error
misClassifyC2 = sum(dataC2<solx);
%total error average
errorRate = (misClassifyC1 + misClassifyC2) / (sampleSize*2);
disp(['Error rate (same loss values)is ' num2str(errorRate)]);
 
%Task D: add non-symetric cost
%solve P(X|C1) == P(X|C2):
syms x;
eqnLeft  = (1/(estimatedSigC1*sqrt(2*pi)))*exp(-(x-estimatedMiuC1)^2/(2*estimatedSigC1^2));%P(X|C1)*lamdaC2C1 // lamdaC2C1 4
eqnRight = (1/(estimatedSigC2*sqrt(2*pi)))*exp(-(x-estimatedMiuC2)^2/(2*estimatedSigC2^2));%P(X|C2)*lamdaC1C2 // lamdaC1C2 1
solxBias = solve( eqnLeft == 4*eqnRight , x);
solxBias = double(solxBias);
solxBias = solxBias(2,1);
%add to plot the decision rule
figure;
plot(plotScale,gausEstimateC1,plotScale,gausEstimateC2);
hold on
y1=get(gca,'ylim');
plot([solx solx],y1, [solxBias solxBias],y1);
hold off
title('Estimated distributions of C1 and C2 with 2 decisions rules(old and new)'); 
legend(['C1~N(' num2str(estimatedMiuC1) ',' num2str(estimatedSigC1) ')'], ['C2~N(' num2str(estimatedMiuC2) ',' num2str(estimatedSigC2) ')'], ['decision: ' num2str(solx)], ['new decision: ' num2str(solxBias)]);
%all above solx(decision) and belongs to C1 is error
misClassifyC1 = sum(dataC1>solxBias);
%all bellow solx(decision) and belongs to C2 is error
misClassifyC2 = sum(dataC2<solxBias);
%total error average
errorRate = (misClassifyC1 + misClassifyC2) / (sampleSize*2);
disp(['Error rate (c1 loss is bigger)is ' num2str(errorRate)]);



clear;
close all;
clc;