function xnet = xavnet(net)
    for i=1:numel(net)
        % scale = sqrt(3 / numel(net{i})); % buggy caffe version
        scale = sqrt(6 / (size(net{i}, 1) + size(net{i}, 2)));
        xnet{i} = 2*scale*(rand(size(net{i}), 'like', net{i})-0.5);
        xnet{i}(:,1)=0;
    end
end
