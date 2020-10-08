%% Initialization
clear ; close all; clc

num_labels = 3;      % 3 labels, from 1 to 3 % (note that we will map "0" to label if 10 is there)
                          

%% =========== Part 1: Loading Data =============
% Load Training Data
fprintf('Loading Data ...\n')

moddata = readmatrix('trainDataLabels.dat'); % training data stored in arrays X, y
Input = moddata(:,2:end-1);
Cumulant = cumulant(Input);
size(Cumulant)
X = abs(Input);
X_phase = abs(angle(Input));

%% =========== Part 2: Mapping Data onto Polynoimal Features =============
% Map X onto Polynomial Features
X_poly = [X X_phase X.^2 X_phase.^2 X.^4 X_phase.^4 X.^6 X_phase.^6 X.^8 Cumulant];
m = size(X, 1);
% X_poly=[];
% for i = 1:100
%     X_map = mapFeature(X(:,i), X_phase(:,i));
%     X_poly = [X_poly X_map];
% end

%% =========== Part 3: Feature Normalizing Polynoimal Features ===========
% Feature Normalizing X

[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

X = X_poly;

y = moddata(:,end);

%fprintf('Program paused. Press enter to continue.\n');
%pause;
%% ============ Part 4: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.001;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 5: Predict for One-Vs-All ================

pred_train = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_train == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 6: Testing: Predict for One-Vs-All ================
moddata_test = readmatrix('testData.dat');
Input_test = moddata_test(:,2:end);
Cumulant = cumulant(Input_test);
X_test = (abs(Input_test));
X_test_phase = abs(angle(Input_test));

m_test = size(X_test, 1);
% X_poly_test = [];
% for i = 1:100
% 
%     X_map = mapFeature(X_test(:,i), X_test_phase(:,i));
%     X_poly_test = [X_poly_test X_map];
% end

X_poly_test = [X_test X_test_phase X_test.^2 X_test_phase.^2 X_test.^4 X_test_phase.^4 X_test.^6 X_test_phase.^6 X_test.^8 Cumulant];
[X_poly_test, mu, sigma] = featureNormalize(X_poly_test);  % Normalize
X_poly_test = [ones(m_test, 1), X_poly_test];

pred_test = predictOneVsAll(all_theta, X_poly_test);
id = 1:1:10000;
pred_test = [id' pred_test];

%fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);