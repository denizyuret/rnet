function forwback_test(net, x, y, n, epsilon)
    if nargin < 5 epsilon = 1e-4; end
    if nargin < 4 n = 10; end

    % compute the bp gradient in original precision
    grad = forwback(net, x, y);

    % compute the numerical estimate in double precision
    net = dblnet(net);
    x = double(x);
    y = double(y);

    numw = 0;
    for i=1:numel(net)
        numw = numw + numel(net(i).w) + numel(net(i).b);
    end
    bpgrad = [];
    numgrad = [];
    for t=1:n
        r = randi(numw);
        for i=1:numel(net)
            if (r <= numel(net(i).w))
                numgrad(t) = westimate(i, r);
                bpgrad(t) = gather(grad(i).w(r));
                break;
            end
            r = r - numel(net(i).w);
            if (r <= numel(net(i).b))
                numgrad(t) = bestimate(i, r);
                bpgrad(t) = gather(grad(i).b(r));
                break;
            end
            r = r - numel(net(i).b);
        end
    end

    comp = [numgrad', bpgrad'];
    rows = min(n, 10);
    disp(comp(1:rows,:));
    if rows < n disp('...');end;
    diff = norm(numgrad-bpgrad)/norm(numgrad+bpgrad);
    disp(diff);

    function g = westimate(i, r)
        orig = net(i).w(r);
        net(i).w(r) = orig + epsilon;
        cost1 = softmax_cost(forward(net, x), y);
        net(i).w(r) = orig - epsilon;
        cost2 = softmax_cost(forward(net, x), y);
        net(i).w(r) = orig;
        g = gather((cost1 - cost2) / (2 * epsilon));
        % fprintf('net(%d).w(%d)=%g cost1=%.12g cost2=%.12g\n', i, r, orig, cost1, cost2);
    end

    function g = bestimate(i, r)
        orig = net(i).b(r);
        net(i).b(r) = orig + epsilon;
        cost1 = softmax_cost(forward(net, x), y);
        net(i).b(r) = orig - epsilon;
        cost2 = softmax_cost(forward(net, x), y);
        net(i).b(r) = orig;
        g = gather((cost1 - cost2) / (2 * epsilon));
        % fprintf('net(%d).w(%d)=%g cost1=%.12g cost2=%.12g\n', i, r, orig, cost1, cost2);
    end
end

