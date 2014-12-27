function w = adadelta(w, x, y, varargin)
    o = options(w, x, y, varargin{:});
    xsize = size(x, 2);

    [g, loss] = forwback(w, x(:,1:o.batchSize), y(:,1:o.batchSize));

    r = report();
    r = report(loss, w, 0, o, r);

    for l=1:numel(w)
        G{l} = g{l} .* g{l};
        D{l} = 0 * w{l};
        d{l} = 0 * w{l};
    end

    for e = 1:o.epochs
        for i = 1:o.batchSize:xsize
            j = min(i+o.batchSize-1, xsize);
            [g, loss] = forwback(w, x(:,i:j), y(:,i:j));
            for l=1:numel(g)
                G{l}(:) = o.rho * G{l} + (1 - o.rho) * g{l} .* g{l};
                d{l}(:) = g{l} .* sqrt(D{l} + o.eps) ./ sqrt(G{l} + o.eps);
                D{l}(:) = o.rho * D{l} + (1 - o.rho) * d{l} .* d{l};

                %% hackery to match caffe with bias and local lr
                if (l < numel(g)) d{l}(1,:) = 0; end % bias hack
                d{l}(:,1) = d{l}(:,1) * 2;  % caffe blobs_lr:2 hack
                %% end of hackery

                w{l}(:) = w{l} - d{l};
            end
            r = report(loss, w, j-i+1, o, r);
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
    p.addParamValue('rho', 0.9, @isnumeric);
    p.addParamValue('eps', 1e-8, @isnumeric);
    p.addParamValue('xdev', [], @isnumeric);
    p.addParamValue('ydev', [], @isnumeric);
    p.addParamValue('printStep', 1e4, @isnumeric);
    p.addParamValue('testStep', 1e5, @isnumeric);
    p.parse(w, x, y, varargin{:});
    o = p.Results;
end


function r = report(loss, w, m, o, r)
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
        r.avgloss = o.rho * r.avgloss + (1 - o.rho) * loss;
    end

    if ~isempty(o.xdev)
        if r.instances >= r.nexttest
            fprintf('Testing dev... ');
            [acc, loss] = evalnet(w, o.xdev, o.ydev);
            fprintf('loss=%g accuracy=%g\n', loss, acc);
            r.nexttest = r.nexttest + o.testStep;
        end
    end
    if r.instances >= r.nextprint
        fprintf('%g\t%.5f\t%.2f\t%.2f\n', r.instances, r.avgloss, r.instances/toc(r.time), toc(r.time));
        r.nextprint = r.nextprint + o.printStep;
    end
end
