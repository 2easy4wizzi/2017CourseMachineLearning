function y=fxw(x,w)
    y = w(1);
    [~,c] = size(x);
    for i=1 : c
        y = y + w(i)*x(i);
    end
end