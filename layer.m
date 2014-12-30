classdef layer < matlab.mixin.Copyable
    
    properties
        w               % cell array: { weights, biases }

        f               % activation function: {'relu' or 'soft'}
        fforw,fback     % function handles for activation and its derivative, set f instead of these, private

        learningRate    % learning rate
        momentum        % momentum
        dropout         % percentage of inputs to drop for dropout
        L1              % parameter for L1 regularization
        L2              % parameter for L2 regularization
        maxnorm         % parameter for maxnorm regularization
        adagrad         % boolean indicating adagrad trick
        nesterov        % boolean indicating nesterov trick

        x,y		% input, output
        dw              % gradient of parameters
        dw1             % moving average of gradients for momentum
        dw2             % sum of squared gradients for adagrad
        mask            % input mask for dropout
    end

    methods
        
        function y = forw(l, x, dropout)
            if nargin > 2 && dropout > 0
                l.dropout = dropout;
                l.mask = (gpuArray.rand(size(x), 'single') > l.dropout);
                x = x .* l.mask * (1/(1-l.dropout));
            else
                l.dropout = [];
            end

            y = l.w{1} * x;

            if numel(l.w) > 1
                y = bsxfun(@plus, y, l.w{2});
            end
            if ~isempty(l.fforw)
                y = l.fforw(y);
            end
            l.x = x;
            l.y = y;
        end

        function dx = back(l, dy)
            if ~isempty(l.fback)
                dy = l.fback(dy);
            end

            l.dw{1} = dy * l.x';

            if numel(l.w) > 1
                l.dw{2} = sum(dy, 2);
            end
            if nargout > 0
                dx = l.w{1}' * dy;
                if ~isempty(l.dropout)
                    dx = dx .* l.mask * (1/(1-l.dropout));
                end
            end
        end

        function update(l)
            for i=1:numel(l.dw)
                if ~isempty(l.L1) && l.L1{i} ~= 0
                    l.dw{i} = l.dw{i} + l.L1{i} * sign(l.w{i});
                end
                if ~isempty(l.L2) && l.L2{i} ~= 0
                    l.dw{i} = l.dw{i} + l.L2{i} * l.w{i};
                end
                if ~isempty(l.adagrad) && l.adagrad{i}
                    l.dw2{i} = l.dw{i} .* l.dw{i} + l.dw2{i};
                    l.dw{i} = l.dw{i} ./ (1e-8 + sqrt(l.dw2{i}));
                end
                if ~isempty(l.learningRate) && l.learningRate{i} ~= 1
                    l.dw{i} = l.learningRate{i} * l.dw{i};
                end
                if ~isempty(l.momentum) && l.momentum{i} ~= 0
                    l.dw1{i} = l.dw{i} + l.momentum{i} * l.dw1{i};
                    if ~isempty(l.nesterov) && l.nesterov{i}
                        l.w{i} = l.w{i} - l.momentum{i} * l.dw1{i};
                    end
                end

                l.w{i} = l.w{i} - l.dw{i};

                if ~isempty(l.maxnorm) && isfinite(l.maxnorm{i})
                    norms = sqrt(sum(l.w{i}.^2, 2));
                    if any(norms > l.maxnorm{i})
                        scale = min(l.maxnorm{i} ./ norms, 1);
                        l.w{i} = bsxfun(@times, l.w{i}, scale);
                    end
                end
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
            if ~isempty(l.f)
                l.fforw = @(x)(l.([l.f '_forw'])(x));
                l.fback = @(x)(l.([l.f '_back'])(x));
            end
            if ~isempty(l.adagrad)
                for i=1:numel(l.w)
                    l.dw2{i} = 0 * l.w{i};
                end
            end
            if ~isempty(l.momentum)
                for i=1:numel(l.w)
                    l.dw1{i} = 0 * l.w{i};
                end
            end
        end

    end % methods
end % classdef
