function dx = backward(net, dx)
    for l=numel(net):-1:1
        dx = net{l}.back(dx);
    end
end
