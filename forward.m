function a = forward(net, x)
    a{1} = x;
    for i=2:numel(net)
        a{i} = bsxfun(@plus, net(i-1).b, net(i-1).w * a{i-1});
        a{i} = a{i} .* (a{i} > 0);
    end
    a{i+1} = bsxfun(@plus, net(i).b, net(i).w * a{i});
end
