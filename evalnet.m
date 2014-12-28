function [accuracy, cost] = evalnet(net, x, y)
    m = size(x, 2);
    b = 1000;
    p = [];
    for i=1:b:m
        j = min(i+b-1, m);
        xij = x(:,i:j);
        for l=1:numel(net)
            xij = net{l}.forw(xij);
        end
        p = [p, xij];
    end
    py = p(sub2ind(size(p), y, 1:numel(y)));
    cost = -mean(log(max(py, realmin(class(py)))));
    [~,h] = max(p);
    accuracy = mean(h==y);
end
