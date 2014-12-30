function net = train(net, x, y, varargin)
    o = options(net, x, y, varargin{:});
    r = report(net, 0, o, []);
    M = size(x, 2);
    L = numel(net);
    E = o.epochs;
    B = o.batchSize;

    for e = 1:E
        for i = 1:B:M
            j = min(i+B-1, M);

            xij = x(:,i:j);
            for l=1:L
                xij = net{l}.forw(xij);
            end

            yij = y(:,i:j);
            for l=L:-1:2
                yij = net{l}.back(yij);
            end
            net{1}.back(yij);           % last yij is slow and unnecessary

            for l=1:L
                if (ismethod(net{l}, 'update'))
                    net{l}.update(o.learningRate);
                end
            end

            r = report(net, j-i+1, o, r);
        end
    end
end


function r = report(net, m, o, r)
    if isempty(r)
        r.time = tic;
        r.instances = 0;
        r.nexttest = 0;
        fprintf('inst\tloss\tacc...\tspeed\ttime\n');
    end
    r.instances = r.instances + m;
    if r.instances >= r.nexttest
        fprintf('%d', r.instances);
        for i=1:2:numel(o.test)
            [acc, loss] = evalnet(net, o.test{i}, o.test{i+1});
            fprintf('\t%.5f\t%.5f', loss, acc);
        end
        fprintf('\t%.1f\t%.1f\n', r.instances/toc(r.time), toc(r.time));
        r.nexttest = r.nexttest + o.testStep;
    end
end


function o = options(net, x, y, varargin)
    p = inputParser;
    p.addRequired('net', @iscell);
    p.addRequired('x', @isnumeric);
    p.addRequired('y', @isnumeric);
    p.addParamValue('test', {}, @iscell);
    p.addParamValue('epochs', 1, @isnumeric);
    p.addParamValue('batchSize', 128, @isnumeric);
    p.addParamValue('learningRate', 0.004, @isnumeric);
    p.addParamValue('testStep', 1e5, @isnumeric);
    p.parse(net, x, y, varargin{:});
    o = p.Results;
end


