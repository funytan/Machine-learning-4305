%  This is a program that attempts to learn and classify inputs from the HAPT dataset
%  to output various types of posture
%% initialization of data into a readable format for net
% import data into matlab

clc;
close all;
clear all;

haptAttr = dlmread('HAPT/haptAttr.txt');
haptPosture = dlmread('HAPT/haptLabel.txt');

% to create a 12 by 8000 matrix which is the format that train function
% takes in
Numofcategories = 12;
t = zeros(Numofcategories, size(haptPosture,1));
% t is a matrix that shows a 1 for the row corresponding to the Posture
% value
for i = 1:size(haptPosture,1)
    t(haptPosture(i), i) = 1;
end                        

%transposing the haptAttr matrix to 561 by 8000
x = haptAttr';

% %create new output matrix t, with 1 in the row corresponding to output
% Numofcategories = 12;
% t = zeros(Numofcategories,8000);
% for i = 1:8000
%     t(haptPosture(i), i) = 1;
% end   
%% declaring of neural network and activation functions
% 100 hidden neurons in a single layer is ideal
net = patternnet([100]); 

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% parameters for trainscg
net.trainParam.epochs = 1000;	
net.trainParam.show = 25;	
net.trainParam.showCommandLine = false;	
net.trainParam.showWindow = true;	
net.trainParam.goal = 0;	
net.trainParam.time = inf;
net.trainParam.min_grad	= 1e-6;	
net.trainParam.max_fail	= 8;	
net.trainParam.sigma = 5.0e-5;	
net.trainParam.lambda = 5.0e-7;

%% This portion of the code sorts the samples by increasing ouput class.
x_combined = [x;haptPosture'];
[B, I] = sort(x_combined(562,:));
% orders the inputs of the x matrix according to increasing outputs
x_sorted = x_combined(:,I);

%% This portion of the code stratifies the samples into having a equal proporton of 1-6, and 7-12 output samples in each set.

no_of_samples = size(x_sorted, 2);

%Find out where 7 starts in the new x matrix
for i=1:no_of_samples
    if x_sorted(562,i) == 7
        break;
    end
end

%% This portion of the code stratifies the samples into having a equal proporton of 1-6, and 7-12 output samples in each set.

[trainInd_1,valInd_1,testInd_1] = dividerand(i-1,0.7,0.15,.15);
[trainInd_minority,valInd_minority,testInd_minority] = dividerand(no_of_samples-i+1,0.7,0.15,.15);
trainInd_2 = trainInd_minority + i - 1;
valInd_2 = valInd_minority + i - 1;
testInd_2 = testInd_minority + i - 1;

%% Add on: This portion of the code generates duplicate samples from the minority classes (7-12)

% n denotes the number of times the minoriy classes are duplicated
n = 10;

x_minority = x_sorted(:,i:no_of_samples);

trainInd_minority = trainInd_2;
valInd_minority = valInd_2;
testInd_minority = testInd_2;

% creates new matrix with duplicated samples from minority classes
for i=1:n
    x_sorted = horzcat(x_sorted, x_minority);
    
    trainInd_minority = trainInd_minority + length(x_minority(1,:));
    valInd_minority = valInd_minority + length(x_minority(1,:));
    testInd_minority = testInd_minority + length(x_minority(1,:));
    
    trainInd_2 = horzcat(trainInd_2,trainInd_minority);
    valInd_2 = horzcat(valInd_2,valInd_minority);
    testInd_2 = horzcat(testInd_2,testInd_minority);
end

% update number of samples
no_of_samples = size(x_sorted, 2);

%% continue: This portion of the code stratifies the samples into having a equal proporton of 1-6, and 7-12 output samples in each set.

trainInd_3 = horzcat(trainInd_1,trainInd_2);
valInd_3 = horzcat(valInd_1,valInd_2);
testInd_3 = horzcat(testInd_1,testInd_2);

% to shuffle the samples in order for the 1-6 and 7-12 outputs be randomly
% fed into net.
trainInd_4 = trainInd_3(randperm(length(trainInd_3)));
valInd_4 = valInd_3(randperm(length(valInd_3)));
testInd_4 = testInd_3(randperm(length(testInd_3)));

%reorders columns in x to be in the order of training, validation and test
%sets.
order_x = horzcat(trainInd_4,valInd_4,testInd_4);
x_sorted = x_sorted(:, order_x);

%takes the top 561 rows to be the new x matrix
x = x_sorted(1:561,:);

%create new output matrix t, with 1 in the row corresponding to output
Numofcategories = 12;
t = zeros(Numofcategories,no_of_samples);
for i = 1:no_of_samples
    t(x_sorted(562,i), i) = 1;
end   

end_train = floor(no_of_samples * 0.7);
start_val = end_train + 1;
end_val = floor(no_of_samples * 0.15) + start_val - 1;
start_test = end_val + 1;

net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:end_train;
net.divideParam.valInd = start_val:end_val;
net.divideParam.testInd = start_test:no_of_samples;
%% training the net
[net,tr] = train(net,x,t);
% view(net);
y = net(x);
%% outputing the confusion matrix of the net
y_test = net(x(:,tr.testInd));
t_test = t(:,tr.testInd);
plotconfusion(t_test,y_test);