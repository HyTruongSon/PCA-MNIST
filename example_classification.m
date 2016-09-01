% Software: PCA for MNIST
% Author: Hy Truong Son
% Position: PhD Student
% Institution: Department of Computer Science, The University of Chicago
% Email: sonpascal93@gmail.com, hytruongson@uchicago.edu
% Website: http://people.inf.elte.hu/hytruongson/
% Copyright 2016 (c) Hy Truong Son. All rights reserved.

function [] = example_classification()
    %% Training
    % Constants
    nRows = 28; % Number of rows
    nCols = 28; % Number of columns
    k_nearest = 1; % For K-Nearest Neighbor algorithm

    % Load the pre-processed training database
    load('training-set.mat', 'X', 'y');
    X_training = X';
    y_training = y;

    % PCA computation
    % Normal PCA computation with computing the covariance matrix
    [X_training, V, d, mean] = PCA(X_training, 'pca.mat'); 
    
    % Load the previous PCA computation
    % load('pca.mat', 'A', 'eigenvectors', 'eigenvalues', 'mean');
    % X_training = A; V = eigenvectors; d = eigenvalues;
    
    % Computation of PCA by Singular Value Decomposition
    % [X, V, d, mean] = PCA_SVD(X, 'pca_svd.mat');
    
    % Sort the eigenvalues in the increasing order
    n = size(d, 1);
    for i = 1 : n / 2
        if d(i) < d(n - i + 1)
            temp = V(:, i);
            V(:, i) = V(:, n - i + 1);
            V(:, n - i + 1) = temp(:);

            temp = d(i);
            d(i) = d(n - i + 1);
            d(n - i + 1) = temp;
        end
    end
    
    figure(1);
    imshow(matrix2image(vector2matrix(mean, nRows, nCols)'));
    title('The mean image');
    
    figure(2);
    k = 40;
    title(['The first ', int2str(k), ' principal components (eigenfaces)']);
    for i = 1 : k
        subplot(4, 10, i);
        imshow(matrix2image(vector2matrix(V(:, i), nRows, nCols)'));
        title(['i = ', int2str(i)]);
    end
    
    % PCA compression
    threshold = 0.90;
    [X_training, V, d] = PCA_compress(X_training, V, d, threshold);
    
    fprintf('Number of principal components to keep: %d\n', size(V, 2));
    
    %% Testing
    % Load the testing database
    load('testing-set.mat', 'X', 'y');
    X_testing = X';
    y_testing = y;
    
    fprintf('Normalize the testing image database\n');
    range = max(max(double(X_testing))) - min(min(double(X_testing)));
    X_testing = double(X_testing) / range;
    
    fprintf('Subtract the whole testing database by the mean sample\n');
    [X_testing, mean] = mean_subtract(double(X_testing));
    
    fprintf('PCA compress on the testing database\n');
    % PCA compress
    X_testing = V' * X_testing;
    
    % Nearest neighbor
    nCorrects = 0;
    nTraining = size(X_training, 2);
    nTesting = size(X_testing, 2);
    
    fprintf('K-Nearest Neighbor classification\n');
    
    % Only keep first 40 principal components
%     X_training = X_training(1:40, :);
%     X_testing = X_testing(1:40, :);
    
    [predict, nCorrects] = k_nearest_neighbor(X_training', y_training, X_testing', y_testing, k_nearest);
    
    % This is only for choosing the closet sample in the training database
    % (k_nearest = 1)
%     for sample = 1 : nTesting
%         fprintf('Sample %d: ', sample);
%         
%         % PCA compress
%         x = X_testing(:, sample);
%         
%         % Nearest-neighbor algorithm
%         predict = 0;
%         min_dist = 0.0;
%         for i = 1 : nTraining
%             if predict == 0
%             	predict = y_training(i);
%                 min_dist = norm(X_training(:, i) - x(:), 2);
%             else
%                 dist = norm(X_training(:, i) - x(:), 2);
%               	if dist < min_dist
%                 	min_dist = dist;
%                  	predict = y_training(i);
%                 end
%             end
%         end
%         
%         % Check the classification
%         if predict == y(sample)
%             nCorrects = nCorrects + 1;
%             fprintf('YES\n');
%         else
%             fprintf('NO\n');
%         end
%     end
    
    fprintf('Classification result: %d/%d = %.2f percent\n', nCorrects, nTesting, double(nCorrects) / nTesting * 100.0);
end