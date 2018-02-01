%authors: 
% Matan Finch, id 300895315
% Gilad Eini , id 034744920
function normData=normalizeStep(data)
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