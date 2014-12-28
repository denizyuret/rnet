function net = adagrad(net, x, y, varargin)
    o = options(net, x, y, varargin{:});
    r = report();
    M = size(x, 2);
    L = numel(net);
    E = o.epochs;
    B = o.batchSize;

    for e = 1:E
        for i = 1:B:M
            j = min(i+B-1, M);
            xij = x(:,i:j);
            yij = y(:,i:j);

            for l=1:L
                xij = net{l}.forw(xij);
            end

            for l=L:-1:1
                yij = net{l}.back(yij);
            end

            for l=1:L
                if (ismethod(net{l}, 'adagrad'))
                    net{l}.adagrad(o.learningRate);
                end
            end

            b = j-i+1;
            p = xij(sub2ind(size(xij), y(:,i:j), 1:b));
            loss = -mean(log(max(p, realmin(class(p)))));
            r = report(loss, net, b, o, r);

        end
    end
end


function r = report(loss, net, m, o, r)
    if nargin < 1
        r.time = tic;
        r.instances = 0;
        r.nextprint = 0;
        r.nexttest = 0;
        r.avgloss = inf;
        fprintf('inst\tavgloss\tspeed\ttime\n');
        return;
    end

    r.instances = r.instances + m;
    if isinf(r.avgloss)
        r.avgloss = loss;
    else
        r.avgloss = 0.01*loss + 0.99*r.avgloss;
    end
    if ~isempty(o.xdev)
        if r.instances >= r.nexttest
            fprintf('Testing dev... ');
            [acc, loss] = evalnet(net, o.xdev, o.ydev);
            fprintf('loss=%g accuracy=%g\n', loss, acc);
            r.nexttest = r.nexttest + o.testStep;
        end
    end
    if r.instances >= r.nextprint
        fprintf('%g\t%.5f\t%.2f\t%.2f\n', r.instances, r.avgloss, r.instances/toc(r.time), toc(r.time));
        r.nextprint = r.nextprint + o.printStep;
    end
end


function o = options(net, x, y, varargin)
    p = inputParser;
    p.addRequired('net', @iscell);
    p.addRequired('x', @isnumeric);
    p.addRequired('y', @isnumeric);
    p.addParamValue('epochs', 1, @isnumeric);
    p.addParamValue('batchSize', 128, @isnumeric);
    p.addParamValue('learningRate', 0.004, @isnumeric);
    p.addParamValue('printStep', 1e4, @isnumeric);
    p.addParamValue('testStep', 1e5, @isnumeric);
    p.addParamValue('xdev', [], @isnumeric);
    p.addParamValue('ydev', [], @isnumeric);
    p.parse(net, x, y, varargin{:});
    o = p.Results;
end


