% Software: PCA for MNIST
% Author: Hy Truong Son
% Position: PhD Student
% Institution: Department of Computer Science, The University of Chicago
% Email: sonpascal93@gmail.com, hytruongson@uchicago.edu
% Website: http://people.inf.elte.hu/hytruongson/
% Copyright 2016 (c) Hy Truong Son. All rights reserved.

function [] = example_reconstruction()
    % Constants
    nRows = 28;
    nCols = 28;

    % Load the pre-processed database
    load('training-set.mat', 'X', 'y');
    X = X';
    
    % PCA computation
    % Normal PCA computation with computing the covariance matrix
    [X, V, d, mean] = PCA(X, 'pca.mat'); 
    
    % load('pca.mat', 'A', 'eigenvectors', 'eigenvalues', 'mean');
    % X = A; V = eigenvectors; d = eigenvalues;
    
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
    
    % Choose a random person
    person = randi([1 size(X, 2)]);
    x = X(:, person);
    
    figure(1);
    imshow(matrix2image(vector2matrix(x + mean, nRows, nCols)'));
    title('The original image');
    
    % Compute the coefficients
    c = V' * x;
    recontruct_img = zeros(size(x));

    figure(2);
    title('Reconstruction');
    count = 0;
    for i = 1 : n
        recontruct_img = recontruct_img + c(i) * V(:, i);
        
        if mod(i, 28) == 0
            count = count + 1;
            subplot(4, 7, count);
            imshow(matrix2image(vector2matrix(recontruct_img + mean, nRows, nCols)'));
            title(['k = ', int2str(i)]);
            drawnow;
        end
    end
end