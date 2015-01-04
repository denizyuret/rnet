function net = train(net, x, y, varargin)
    o = options(net, x, y, varargin{:});
    r = report(net, o, []);
    M = size(x, 2);
    L = numel(net);
    E = o.epochs;
    B = o.batchSize;

    for e = 1:E
        for i = 1:B:M
            j = min(i+B-1, M);

            a = x(:,i:j);
            for l=1:L
                a = net{l}.forw(a,1);
            end

            d = y(:,i:j);
            for l=L:-1:2
                d = net{l}.back(d);
            end
            % last dx is slow and unnecessary
            net{1}.back(d);

            for l=1:L
                net{l}.update();
            end

            r = report(net, o, r);
        end
    end
    r.flush = 1;
    report(net, o, r);
end


function r = report(net, o, r)
    if ~isempty(r)
        r.instances = r.instances + size(net{1}.x, 2);
    else
        r.time = tic;
        r.instances = 0;
        r.flush = 1;
    end
    if r.flush
        r.nexttest = r.instances;
        r.nextstat = r.instances;
        r.nextsave = r.instances;
        r.flush = 0;
    end
    if o.testStep >= 1 && r.instances >= r.nexttest
        if r.instances == 0
            fprintf('%-13s', 'inst');
            for i=1:2:numel(o.testData)
                fprintf('%-13s%-13s', 'loss', 'acc');
            end
            fprintf('%-13s%-13s\n', 'speed', 'time');
        end
        r.nexttest = r.nexttest + o.testStep;
        fprintf('%-13d', r.instances);
        for i=1:2:numel(o.testData)
            [acc, loss] = evalnet(net, o.testData{i}, o.testData{i+1});
            fprintf('%-13g%-13g', loss, acc);
        end
        fprintf('%-13g%-13g\n', r.instances/toc(r.time), toc(r.time));
    end
    if o.statStep >= 1 && r.instances >= r.nextstat
        r.nextstat = r.nextstat + o.statStep;
        fprintf('\n%-13s%-13s%-13s%-13s%-13s%-13s\n', 'array', 'min', ...
                'rms', 'max', 'nz', 'nzrms');
        for l=1:numel(net)
            summary(net, l, 'x');
            summary(net, l, 'y');
            for i=1:numel(net{l}.w)
                summary(net, l, 'w', i);
            end
            for i=1:numel(net{l}.dw)
                summary(net, l, 'dw', i);
            end
            fprintf('\n');
        end
    end
    if o.saveStep >= 1 && r.instances >= r.nextsave
        fname = sprintf('%s%d.mat', o.saveName, r.instances);
        fprintf('Saving %s...', fname);
        save(fname, 'net', '-v7.3');
        fprintf('done\n');
        if r.nextsave == 0
            r.nextsave = o.saveStep;
        else
            r.nextsave = r.nextsave * 2;
        end
    end
end

function summary(net, l, f, i)
    a = getfield(net{l}, f);
    if nargin > 3
        a = a{i}; 
        nm = sprintf('n%d.%s%d', l, f, i);
    else
        nm = sprintf('n%d.%s', l, f);
    end
    nz = (a(:)~=0);
    fprintf('%-13s%-13g%-13g%-13g%-13g%-13g\n', ...
            nm, min(a(:)), sqrt(mean(a(:).^2)), max(a(:)), ...
            mean(nz), sqrt(mean(a(nz).^2)));
end

function o = options(net, x, y, varargin)
    p = inputParser;
    p.addRequired('net', @iscell);
    p.addRequired('x', @isnumeric);
    p.addRequired('y', @isnumeric);
    p.addParamValue('epochs', 1, @isnumeric);
    p.addParamValue('batchSize', 100, @isnumeric);
    p.addParamValue('testStep', 1e5, @isnumeric);
    p.addParamValue('testData', {}, @iscell);
    p.addParamValue('saveStep', 1e6, @isnumeric);
    p.addParamValue('saveName', 'net', @ischar);
    p.addParamValue('statStep', 1e5, @isnumeric);
    p.parse(net, x, y, varargin{:});
    o = p.Results;
end
