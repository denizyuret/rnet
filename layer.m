classdef layer < matlab.mixin.Copyable
    
    properties
        w               % cell array: { weights, biases }
        f               % activation function: {'relu' or 'soft'}

        learningRate    % learning rate
        momentum        % momentum
        L1              % parameter for L1 regularization
        L2              % parameter for L2 regularization
        maxnorm         % parameter for maxnorm regularization

        adagrad         % boolean indicating adagrad trick
        nesterov        % boolean indicating nesterov trick
        dropout         % percentage of inputs to drop

        x,y		% input, output
        dw              % gradient of parameters
        dw1             % moving average of gradients for momentum
        dw2             % sum of squared gradients for adagrad
        mask            % input mask for dropout
    end

    methods
        
        function y = forw(l, x, do_dropout)
            if nargin > 2 && do_dropout && l.dropout
                l.mask = (gpuArray.rand(size(x), 'single') > l.dropout);
                x = x .* l.mask * (1/(1-l.dropout));
            else
                l.mask = [];
            end

            y = l.w{1} * x;

            if numel(l.w) > 1
                y = bsxfun(@plus, y, l.w{2});
            end
            if ~isempty(l.f)
                y = l.([l.f '_forw'])(y);
            end
            l.x = x;
            l.y = y;
        end

        function dx = back(l, dy)
            if ~isempty(l.f)
                dy = l.([l.f '_back'])(dy);
            end

            l.dw{1} = dy * l.x';

            if numel(l.w) > 1
                l.dw{2} = sum(dy, 2);
            end
            if nargout > 0
                dx = l.w{1}' * dy;
                if ~isempty(l.mask)
                    dx = dx .* l.mask * (1/(1-l.dropout));
                end
            end
        end

        function update(l)
            for i=1:numel(l.dw)
                if l.check('L1', i)
                    l.dw{i} = l.dw{i} + l.L1{i} * sign(l.w{i});
                end
                if l.check('L2', i)
                    l.dw{i} = l.dw{i} + l.L2{i} * l.w{i};
                end
                if l.check('adagrad', i)
                    if numel(l.dw2) >= i
                        l.dw2{i} = l.dw{i} .* l.dw{i} + l.dw2{i};
                    else
                        l.dw2{i} = l.dw{i} .* l.dw{i};
                    end
                    l.dw{i} = l.dw{i} ./ (1e-8 + sqrt(l.dw2{i}));
                end
                if l.check('learningRate', i, 1)
                    l.dw{i} = l.learningRate{i} * l.dw{i};
                end
                if l.check('momentum', i)
                    if numel(l.dw1) >= i
                        l.dw1{i} = l.dw{i} + l.momentum{i} * l.dw1{i};
                    else
                        l.dw1{i} = l.dw{i};
                    end
                    if l.check('nesterov', i)
                        l.w{i} = l.w{i} - l.momentum{i} * l.dw1{i};
                    end
                end

                l.w{i} = l.w{i} - l.dw{i};

                if l.check('maxnorm', i, inf)
                    norms = sqrt(sum(l.w{i}.^2, 2));
                    if any(norms > l.maxnorm{i})
                        scale = min(l.maxnorm{i} ./ norms, 1);
                        l.w{i} = bsxfun(@times, l.w{i}, scale);
                    end
                end
            end
        end

        function ok = check(l, p, i, v)
            if isempty(l.(p))
                ok = false;
            else
                if ~iscell(l.(p))
                    lp = l.(p);
                    l.(p) = {};
                    for i=1:numel(l.w)
                        l.(p){i} = lp;
                    end
                end
                if nargin < 4 
                    v=0; 
                end
                ok = (l.(p){i} ~= v);
            end
        end

        function y = relu_forw(l, y)
            y = y .* (y > 0);
        end

        function dy = relu_back(l, dy)
            dy = dy .* (l.y > 0);
        end

        function y = soft_forw(l, y)
            y = softmax(gather(y));
        end

        function dy = soft_back(l, dy)
            m = numel(dy);  % dy is a vector of correct classes
            dy = full(sparse(double(gather(dy)), 1:m, 1));
            dy = (l.y - dy) / m;
        end

        function nll = loss(l, probs, labels)
            py = probs(sub2ind(size(probs), labels, 1:numel(labels)));
            nll = -mean(log(max(py, realmin(class(py)))));
        end

        function acc = accuracy(l, probs, labels)
            [~,maxp] = max(probs);
            acc = mean(maxp == labels);
        end

        function l = layer(varargin)
            for i=1:2:numel(varargin)
                l.(varargin{i}) = varargin{i+1};
            end
        end

    end % methods
end % classdef
