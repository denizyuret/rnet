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
        numw = numw + numel(net{i});
    end
    bpgrad = [];
    numgrad = [];
    for t=1:n
        r = randi(numw);
        for i=1:numel(net)
            if (r <= numel(net{i}))
                numgrad(t) = estimate(i, r);
                bpgrad(t) = gather(grad{i}(r));
                break;
            end
            r = r - numel(net{i});
        end
    end

    comp = [numgrad', bpgrad'];
    rows = min(n, 10);
    disp(comp(1:rows,:));
    if rows < n disp('...');end;
    diff = norm(numgrad-bpgrad)/norm(numgrad+bpgrad);
    disp(diff);

    function g = estimate(i, r)
        orig = net{i}(r);
        net{i}(r) = orig + epsilon;
        [~,cost1] = softmax_diff(forward(net, x), y);
        net{i}(r) = orig - epsilon;
        [~,cost2] = softmax_diff(forward(net, x), y);
        net{i}(r) = orig;
        g = gather((cost1 - cost2) / (2 * epsilon));
        % fprintf('net(%d).w(%d)=%g cost1=%.12g cost2=%.12g\n', i, r, orig, cost1, cost2);
    end

end

