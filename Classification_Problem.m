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

%create new output matrix t, with 1 in the row corresponding to output
Numofcategories = 12;
t = zeros(Numofcategories,8000);
for i = 1:8000
    t(haptPosture(i), i) = 1;
end   
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

%% This portion of the code stratifies the samples into having a equal proporton of 1-6, and 7-12 output samples in each set.
x_combined = [x;haptPosture'];
[B, I] = sort(x_combined(562,:));
% orders the inputs of the x matrix according to increasing outputs
x_sorted = x_combined(:,I);

%Find out where 7 starts in the new x matrix
for i=1:8000
    if x_sorted(562,i) == 7
        break;
    end
end
[trainInd_1,valInd_1,testInd_1] = dividerand(i-1,0.7,0.15,.15);
[trainInd_2,valInd_2,testInd_2] = dividerand(8000-i+1,0.7,0.15,.15);
trainInd_2 = trainInd_2 + i - 1;
valInd_2 = valInd_2 + i - 1;
testInd_2 = testInd_2 + i - 1;
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
x_test = x_sorted;
x_sorted = x_sorted(:, order_x);

%takes the top 561 rows to be the new x matrix
x = x_sorted(1:561,:);

%create new output matrix t, with 1 in the row corresponding to output
Numofcategories = 12;
t = zeros(Numofcategories,8000);
for i = 1:8000
    t(x_sorted(562,i), i) = 1;
end   

net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:5600;
net.divideParam.valInd = 5601:6800;
net.divideParam.testInd = 6801:8000;
%% training the net
[net,tr] = train(net,x,t);
% view(net);
y = net(x);
%% outputing the confusion matrix of the net
y_test = net(x(:,tr.testInd));
t_test = t(:,tr.testInd);
plotconfusion(t_test,y_test);