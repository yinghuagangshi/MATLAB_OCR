%% main.m文件

clc;
clear all;
close all;

fprintf('===== 实验开始 =====\n');

%% 1. 读取图像 (使用 MATLAB 内置功能)
disp('（1）正在加载 MATLAB 内置 MNIST 数据集...');
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% 划分训练集 (90%) 和测试集 (10%)
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.9, 'randomized');

ntraindata = numel(imdsTrain.Files);
ntestdata = numel(imdsTest.Files);
fprintf('数据集加载完毕。 训练样本: %d, 测试样本: %d\n', ntraindata, ntestdata);

%% 2. 特征提取 (新方法：标准化的原始灰度像素)
disp('（2）正在提取特征 (使用 28x28=784 灰度像素)...');

% --- 提取训练集特征 ---
train_data = zeros(784, ntraindata); 
for i = 1:ntraindata
    img = readimage(imdsTrain, i);
    
    % --- 修正开始 ---
    % 错误：bw_img = imbinarize(img); (丢弃了灰度信息)
    % 正确：将 0-255 的 uint8 图像转换为 0.0-1.0 的 double 浮点数
    img_double = double(img) / 255.0; 
    % --- 修正结束 ---
    
    train_data(:, i) = reshape(img_double, 784, 1);
end
train_labels_raw = imdsTrain.Labels;

% --- 提取测试集特征 ---
test_data = zeros(784, ntestdata); 
for i = 1:ntestdata
    img = readimage(imdsTest, i);
    
    % --- 修正开始 ---
    img_double = double(img) / 255.0;
    % --- 修正结束 ---
    
    test_data(:, i) = reshape(img_double, 784, 1);
end

test_labels_raw = imdsTest.Labels;
disp('特征提取完毕。');

%% 3. 数据打上标签 (独热编码)
disp('（3）正在转换标签为独热 (one-hot) 编码...');
train_label = dummyvar(train_labels_raw)';
test_label = dummyvar(test_labels_raw)';
disp('标签转换完毕。');

%% 4. 创建并训练BP神经网络
disp('（4）开始训练BP神经网络...(这会打开训练窗口)');
% 注意：我们调用 network_train，它内部的隐藏层神经元数量
% 已经为 784 个输入做了优化 (见下一个文件)
net = network_train(train_data, train_label);
disp('网络训练完成！');

%% 5. 测试BP神经网络 (使用 try...catch 捕获错误)
disp('（5）正在用测试集评估网络性能...');
try
    % 我们尝试执行测试
    [predict_label_idx, test_raw_output] = network_test(test_data, net);
    disp('测试完毕。');

    %% 6. 计算正确率并显示详细报告
    disp('（6）正在计算最终结果...');
    
    % 从独热编码的真实标签中获取索引 (1='0', 2='1', ..., 10='9')
    [~, true_label_idx] = max(test_label, [], 1); 
    
    % 检查维度 (以防万一)
    if ~isequal(size(true_label_idx), size(predict_label_idx))
        disp('!!! 警告: 真实标签和预测标签的维度不匹配。正在尝试修复...');
        predict_label_idx = reshape(predict_label_idx, size(true_label_idx));
    end
    
    % 计算准确率
    error = true_label_idx - predict_label_idx;
    accuracy = size(find(error==0), 2) / ntestdata;
    
    % --- 详细报告 ---
    fprintf('\n===== 最终测试报告 =====\n');
    fprintf('隐藏层神经元: %d\n', net.layers{1}.size);
    fprintf('训练迭代次数 (Epochs): %d\n', net.trainParam.epochs);
    fprintf('----------------------------\n');
    fprintf('>>> 最终测试准确率: %.2f %% <<<\n', accuracy * 100);
    fprintf('============================\n');
    
    % --- 绘制混淆矩阵 (非常重要！) ---
    disp('正在绘制混淆矩阵 (Confusion Matrix)...');
    figure; % 创建一个新窗口
    plotconfusion(test_label, test_raw_output);
    title(sprintf('测试集混淆矩阵 (准确率: %.2f %%)', accuracy * 100));

catch ME
    % 如果 try 块中的任何地方出错，将执行这里
    fprintf('\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
    fprintf('!!!!!!!!!!   脚本在测试中出错  !!!!!!!!!!\n');
    fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
    fprintf('错误信息 (Message): %s\n', ME.message);
    fprintf('\n请检查出错的文件: %s\n', ME.stack(1).file);
    fprintf('出错的行号: %d\n', ME.stack(1).line);
    fprintf('出错的函数: %s\n', ME.stack(1).name);
    fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n');
end

