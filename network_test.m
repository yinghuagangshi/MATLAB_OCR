%% network_test.m文件

function [out_idx, an] = network_test(test_data, net)
%% BP神经网络的测试
an = sim(net, test_data); % an 是 10xN 的原始输出矩阵

% 预分配内存
out_idx = zeros(1, size(test_data, 2)); 

for i = 1:size(test_data, 2)
   % 找到最大值的索引 (即预测的数字类别)
   [~, out_idx(i)] = max(an(:, i)); 
end

end