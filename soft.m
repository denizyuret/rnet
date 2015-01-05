classdef soft < layer
    methods

        function y = fforw(l, y)
            y = softmax(gather(y));
        end

        function dy = fback(l, dy)
        % softmax is the last layer and takes a vector of correct
        % classes as dy to start backprop.
            m = numel(dy);
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

        function l = soft(varargin)
            l = l@layer(varargin{:});
        end

    end
end
