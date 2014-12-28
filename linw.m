classdef linw < handle
    
    properties
        x		% input
        w               % parameters
        dw              % gradient of parameters
        dw2             % sum of squared gradients for adagrad
        eps=1e-8        % numerical stability for adagrad
        bias_hack       % using first row/col of w for bias
    end
    
    methods
        
        function l = linw(w)
            l.w = w;
            l.dw = 0 * w;
            l.dw2 = 0 * w;
            l.bias_hack = (w(1,1)==1 && all(w(1,2:end)==0));
        end

        function y = forw(l, x)
            l.x = x;
            y = l.w * x;
        end

        function dx = back(l, dy)
            l.dw(:) = dy * l.x';
            dx = l.w' * dy;
        end

        function adagrad(l, learningRate)
            l.dw2(:) = l.dw2 + l.dw .* l.dw;
            l.dw(:) = learningRate * l.dw ./ (l.eps + sqrt(l.dw2));

            %% hackery to match caffe with bias and local lr
            if (l.bias_hack)
                l.dw(1,:) = 0; % use first row to create 1's for bias hack
            end 
            l.dw(:,1) = l.dw(:,1) * 2;  % caffe doubles lr for bias
            %% end of hackery

            l.w(:) = l.w - l.dw;
        end

    end % methods
end % classdef
