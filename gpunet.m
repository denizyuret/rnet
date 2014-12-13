function gnet = gpunet(net)
    for i=1:numel(net)
        gnet{i} = gpuArray(net{i});
    end
end
