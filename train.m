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
end


function r = report(net, o, r)
    if isempty(r)
        r.time = tic;
        r.instances = 0;
        r.nexttest = 0;
        fprintf('inst');
        for i=1:2:numel(o.test)
            fprintf('\tloss\tacc');
        end
        fprintf('\tspeed\ttime\n');
    else
        r.instances = r.instances + size(net{1}.x, 2);
    end
    if o.verbose >= 1 && r.instances >= r.nexttest
        r.nexttest = r.nexttest + o.testStep;
        fprintf('%d', r.instances);
        for i=1:2:numel(o.test)
            [acc, loss] = evalnet(net, o.test{i}, o.test{i+1});
            fprintf('\t%.5f\t%.5f', loss, acc);
        end
        fprintf('\t%.1f\t%.1f\n', r.instances/toc(r.time), toc(r.time));
        
        if o.verbose >= 2
            fprintf('\n\tmin\trms\tmax\tnz\n');
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
    end
end

function summary(net, l, f, i)
    a = getfield(net{l}, f);
    if nargin > 3
        a = a{i};
        fprintf('n%d.%s%d\t%.4f\t%.4f\t%.4f\t%.4f\n', ...
                l, f, i, min(a(:)), sqrt(mean(a(:).^2)), max(a(:)), mean(a(:)~=0));            
    else
        fprintf('n%d.%s\t%.4f\t%.4f\t%.4f\t%.4f\n', ...
                l, f, min(a(:)), sqrt(mean(a(:).^2)), max(a(:)), mean(a(:)~=0));            
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
    p.addParamValue('testStep', 1e5, @isnumeric);
    p.addParamValue('verbose', 1, @isnumeric);
    p.parse(net, x, y, varargin{:});
    o = p.Results;
end


