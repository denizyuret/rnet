function grad = backward(net, a, dz)
    for i=numel(net):-1:1
        grad(i).w = dz * a{i}';
        grad(i).b = sum(dz, 2);
        if (i==1) break; end;
        dz = (a{i} > 0) .* (net(i).w' * dz);
    end
end
