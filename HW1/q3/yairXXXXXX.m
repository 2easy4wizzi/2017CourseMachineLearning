
function [rate_a, rate_b] = probabilistic_neural_network()
% Function that implement PNN as described in Algorithm 1 for 2-class problem of classifying
%an input sample as letter 'A' or 'B'
load('dataAB.mat');
h = 1; %after a long search we found that this is the value that maximizes the classification success.
[rate_a, ~] = parzen_classifier( train_dataA, train_dataB, test_dataA, h );
[~, rate_b] = parzen_classifier( train_dataA, train_dataB, test_dataB, h );
end

function [ rate_a, rate_b ] = parzen_classifier( train_dataA, train_dataB, test_data, h )
%The function get a test set, training data and a scalar h width, and clasifies
%it via gaussian parzen window

[train_dataA_size, ~] = size(train_dataA);
[train_dataB_size, ~] = size(train_dataB);
[test_data_size, ~] = size(test_data);
counter_a = 0;
counter_b = 0;
for i = 1:test_data_size
    test_point = test_data(i,:);
    p_u_a = 0;
    cov_a = cov(train_dataA);
    for j = 1:train_dataA_size
        train_point_a = train_dataA(j,:);
        u_a = (test_point - train_point_a)/h;
        p_u_a = p_u_a + 1/((det(2*pi*cov_a))^0.5) * exp(-((u_a*inv(cov_a)*u_a')^2)/2);
    end
    p_u_b = 0;
    cov_b = cov(train_dataB);
    for j = 1:train_dataB_size
        train_point_b = train_dataB(j,:);
        u_b = (test_point - train_point_b)/h;
        p_u_b = p_u_b + 1/((det(2*pi*cov_b))^0.5) * exp(-((u_b*inv(cov_b)*u_b')^2)/2);
    end
    if p_u_a > p_u_b
        counter_a = counter_a + 1;
    else
        counter_b = counter_b + 1;
    end
end
rate_a = counter_a / test_data_size;
rate_b = counter_b / test_data_size;
end

