function x = forward(net, x)
    for l=1:numel(net)
        x = net{l}.forw(x);
    end
end
