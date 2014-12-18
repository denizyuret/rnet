function w = adagrad(w, x, y, varargin)
    o = options(w, x, y, varargin{:});
    xsize = size(x, 2);
    epsilon = 1e-8;

    for l=1:numel(w)
        G{l} = 0 * w{l};
    end

    r = report();
    for e = 1:o.epochs
        for i = 1:o.batchSize:xsize
            j = min(i+o.batchSize-1, xsize);
            [g, cost] = forwback(w, x(:,i:j), y(:,i:j));
            for l=1:numel(g)
                G{l}(:) = G{l} + g{l} .* g{l};
                dw = o.learningRate * g{l} ./ (epsilon + sqrt(G{l}));
                w{l}(:) = w{l} - dw;
                w{l}(:,1) = w{l}(:,1) - dw(:,1); % caffe blobs_lr:2 hack
                if l < numel(g)         % bias hack
                    w{l}(1,:) = 0;
                    w{l}(1,1) = 1;
                end
            end
            r = report(cost, w, j-i+1, o, r);
        end
    end
end


function o = options(w, x, y, varargin)
    p = inputParser;
    p.addRequired('w', @iscell);
    p.addRequired('x', @isnumeric);
    p.addRequired('y', @isnumeric);
    p.addParamValue('epochs', 1, @isnumeric);
    p.addParamValue('batchSize', 128, @isnumeric);
    p.addParamValue('learningRate', 0.04, @isnumeric);
    p.addParamValue('printStep', 1e4, @isnumeric);
    p.addParamValue('testStep', 1e5, @isnumeric);
    p.addParamValue('xdev', [], @isnumeric);
    p.addParamValue('ydev', [], @isnumeric);
    p.parse(w, x, y, varargin{:});
    o = p.Results;
end


function r = report(cost, w, m, o, r)
    if nargin < 1
        r.time = tic;
        r.instances = 0;
        r.nextprint = 0;
        r.nexttest = 0;
        r.avgcost = inf;
        r.infcost = 0;
        fprintf('inst\tavgcost\tspeed\ttime\n');
        return;
    end
    if isinf(r.avgcost)
        r.avgcost = cost;
    else
        r.avgcost = 0.01*cost + 0.99*r.avgcost;
    end
    if isinf(cost)
        r.infcost = r.infcost + 1;
    end
    r.instances = r.instances + m;
    if ~isempty(o.xdev)
        if r.instances >= r.nexttest
            fprintf('Testing... ');
            [acc, cost] = evalnet(w, o.xdev, o.ydev);
            fprintf('dev-accuracy=%g dev-cost=%g\n', acc, cost);
            r.nexttest = r.nexttest + o.testStep;
        end
    end
    if r.instances >= r.nextprint
        fprintf('%g\t%.5f\t%.2f\t%.2f', r.instances, r.avgcost, r.instances/toc(r.time), toc(r.time));
        if r.infcost > 0 
            fprintf('\tinfcost=%d', r.infcost); 
            r.infcost = 0;
        end
        fprintf('\n');
        r.nextprint = r.nextprint + o.printStep;
    end
end
