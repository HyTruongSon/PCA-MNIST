% Software: Principal Component Analysis
% Author: Hy Truong Son
% Position: PhD Student
% Institution: Department of Computer Science, The University of Chicago
% Email: sonpascal93@gmail.com, hytruongson@uchicago.edu
% Website: http://people.inf.elte.hu/hytruongson/
% Copyright 2016 (c) Hy Truong Son. All rights reserved.

function [X, eigenvectors, eigenvalues] = PCA_compress(X, eigenvectors, eigenvalues, threshold)
    % Number of eigenvalues
    n = size(eigenvalues, 1);
    
    % Compute the minimum of principal components to keep
    Sum = sum(eigenvalues);
    s = 0.0;
    k = 1;
    for i = 1 : n
        s = s + eigenvalues(i);
        if s / Sum >= threshold
            k = i;
            break;
        end
    end
    
    fprintf('Number of principal components to keep: %d\n', k);
    
    % Compression
    fprintf('Compressing\n');
    eigenvectors = eigenvectors(:, 1 : k);
    eigenvalues = eigenvalues(1 : k);
    X = eigenvectors' * X;
end