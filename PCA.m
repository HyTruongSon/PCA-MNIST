% Software: Principal Component Analysis
% Author: Hy Truong Son
% Position: PhD Student
% Institution: Department of Computer Science, The University of Chicago
% Email: sonpascal93@gmail.com, hytruongson@uchicago.edu
% Website: http://people.inf.elte.hu/hytruongson/
% Copyright 2016 (c) Hy Truong Son. All rights reserved.

function [A, eigenvectors, eigenvalues, mean] = PCA(X, output_fn)
    %% Constants
    % Dimension of each sample
    d = size(X, 1);
    
    % Number of samples
    n = size(X, 2);
    
    %% Normalize the data and subtract the whole database by the mean sample
    fprintf('Normalize the image database\n');
    range = max(max(double(X))) - min(min(double(X)));
    X = double(X) / range;
    
    fprintf('Subtract the whole database by the mean sample\n');
    [A, mean] = mean_subtract(double(X));
    
    %% Case 1: d > n
    if (d > n) 
        fprintf('Case 1: d (%d) > n (%d)\n', d, n);
        
        fprintf('Computing the covariance matrix S\n');
        S = A' * A;
        
        fprintf('Computing the eigenvectors and the eigenvalues of S\n');
        [eigenvectors, D] = eig(S);
        eigenvectors = normalize(A * eigenvectors);
        eigenvalues = diag(D);
        
        % fprintf('Saving the eigenvectors and the eigenvalues to .mat file\n');
        % save(output_fn, 'A', 'eigenvectors', 'eigenvalues', 'mean');
        
        return;
    end
    
    %% Case 2: d <= n
    fprintf('Case 2: d (%d) <= n (%d)\n', d, n);
    
    fprintf('Computing the covariance matrix S\n');
    S = A * A';
    
    fprintf('Computing the eigenvectors and the eigenvalues of S\n');
	[eigenvectors, D] = eig(S);
    eigenvectors = normalize(eigenvectors);
    eigenvalues = diag(D);
    
    % fprintf('Saving the eigenvectors and the eigenvalues to .mat file\n');
	% save(output_fn, 'A', 'eigenvectors', 'eigenvalues', 'mean');
end