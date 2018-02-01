function main
    clear;
    close all;
    clc;
    mat = magic(4);
    mat(1,1) = 1;
    mat(2,1) = 0;
    mat(3,1) = 0;
    mat(4,1) = 1;
    ind = (mat(:,1) == 1);
    newMat = [ mat(ind,:); mat(~ind,:)];

end

% firstPic = normT(1,:);
% q = reshape(firstPic,99,99);
% figure,imagesc(q);

% q = reshape(mn,99,99);
% figure,imagesc(q);

% h=waitbar(0,'Calculating Sb...');
%     waitbar(i/rows);
% close(h);
% 
% for i=1 : 2 : 10
%     disp(i);
%     disp(i+1);
%     end
%     
%     p1 = [ 1 2 5];
%     p2 = [ 1 10 15];
%     p3 = ((p1+p2)/2);
%     
%     mat = [ 1 1 1 1; 2 2 2 2; 3 3 3 3];
%     disp(mat);
%     idx = randperm(3);
%     C = mat(idx,:);
%     disp(C);
% %     disp(['The result is: [' num2str(mat(:).') ']']) ;