function [accuracy, cost] = evalnet(w, x, y)
z = forward_batch(w, x, 1000);
[~,cost] = softmax_diff(z, y);
[~,h] = max(z);
accuracy = sum(h(:)==y(:))/numel(y);
end
