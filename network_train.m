%% network_train.m文件

function net = network_train(train_data,train_label )
%% BP神经网络的创建、训练
% 优化：输入层为784，隐藏层25个节点太少，增加到100个
n = 100; % 隐藏层神经元

net = patternnet(n);
net.trainParam.epochs = 100; % 保持100次迭代
net.trainParam.lr = 0.1;
net.trainParam.goal = 0.001;
% net.trainFcn = 'trainrp';

% 优化：确保训练窗口总是显示
% net.trainParam.showWindow = true;

% 网络训练
net = train(net, train_data, train_label);
end