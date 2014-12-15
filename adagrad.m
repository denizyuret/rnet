function w = adagrad(w, x, y, varargin)
    o = options(w, x, y, varargin{:});
    r = [];
    xsize = size(x, 2);
    gdev = gpuDevice();

    for l=1:numel(w)
        g2{l} = 1e-8 + zeros(size(w{l}), 'like', w{l});
    end

    for e = 1:o.epochs
        for i = 1:o.batchSize:xsize
            j = min(i+o.batchSize-1, xsize);
            [g, cost] = forwback(w, x(:,i:j), y(:,i:j));
            %wait(gdev);
            for l=1:numel(g)
                g2{l}(:) = g2{l} + g{l} .* g{l};
                %wait(gdev);
                w{l}(:) = w{l} - o.learningRate * g{l} ./ sqrt(g2{l});
                %wait(gdev);
            end
            r = report(r, i, cost, o.printStep);
        end
    end
end


function o = options(w, x, y, varargin)
    p = inputParser;
    p.addRequired('w', @iscell);
    p.addRequired('x', @isnumeric);
    p.addRequired('y', @isnumeric);
    p.addParameter('epochs', 1, @isnumeric);
    p.addParameter('batchSize', 128, @isnumeric);
    p.addParameter('learningRate', 0.1, @isnumeric);
    p.addParameter('printStep', 100, @isnumeric);
    p.parse(w, x, y, varargin{:});
    o = p.Results;
end


function r = report(r, i, cost, step)
    if isempty(r)
        r.time = tic;
        r.ncall = 0;
        r.infcost = 0;
        r.mcost = inf;
    end
    if isinf(r.mcost)
        r.mcost = cost;
    else
        r.mcost = (1/step)*cost + (1-(1/step)) * r.mcost;
    end
    if isinf(cost)
        r.infcost = r.infcost + 1;
    end
    if mod(r.ncall, step) == 0
        if mod(r.ncall, 10*step) == 0
            fprintf('inst\tmcost\tspeed\ttime\tinfcost\n');
        end
        fprintf('%d\t%.5f\t%.2f\t%.2f\t%.d\n', i, r.mcost, i/toc(r.time), toc(r.time), r.infcost);
    end
    r.ncall = r.ncall + 1;
end
