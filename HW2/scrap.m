function main
    clear;
    close all;
    clc;
    
    for i=1 : 2 : 10
    disp(i);
    disp(i+1);
    end
    
    p1 = [ 1 2 5];
    p2 = [ 1 10 15];
    p3 = ((p1+p2)/2);
    
    mat = [ 1 1 1 1; 2 2 2 2; 3 3 3 3];
    disp(mat);
    idx = randperm(3);
    C = mat(idx,:);
    disp(C);
%     disp(['The result is: [' num2str(mat(:).') ']']) ;

end

% firstPic = normT(1,:);
% q = reshape(firstPic,99,99);
% figure,imagesc(q);

% q = reshape(mn,99,99);
% figure,imagesc(q);