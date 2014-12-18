function rnet = rndnet(net, std)
if nargin < 2
    std = 0.01;
end
for i=1:numel(net)
    rnet{i} = std * randn(size(net{i}), 'single');
    rnet{i}(:,1) = 0;
    if i < numel(net)
        rnet{i}(1,:)=0;
        rnet{i}(1,1)=1;
    end
end
end
