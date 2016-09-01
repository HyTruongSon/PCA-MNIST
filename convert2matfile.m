% Software: PCA for MNIST
% Author: Hy Truong Son
% Position: PhD Student
% Institution: Department of Computer Science, The University of Chicago
% Email: sonpascal93@gmail.com, hytruongson@uchicago.edu
% Website: http://people.inf.elte.hu/hytruongson/
% Copyright 2016 (c) Hy Truong Son. All rights reserved.

function [X, y] = convert2matfile()
    % Constants
    nRows = 28;
    nCols = 28;
    N = nRows * nCols;

    % For Training
    % images_fn = 'train-images.idx3-ubyte';
    % labels_fn = 'train-labels.idx1-ubyte';
    % output_fn = 'training-set.mat';
    
    % For Testing
    images_fn = 't10k-images.idx3-ubyte';
    labels_fn = 't10k-labels.idx1-ubyte';
    output_fn = 'testing-set.mat';

    % Processing the images
    images_file = fopen(images_fn, 'rb');
    bytes = fread(images_file);
    bytes = bytes(17 : end);
    nSamples = size(bytes, 1) / N;
    X = zeros(nSamples, N);
    for sample = 1 : nSamples
        X(sample, :) = bytes((sample - 1) * N + 1 : sample * N);
        
        % imshow(matrix2image(vector2matrix(X(sample, :), nRows, nCols)'));
        % waitforbuttonpress;
    end
    fclose(images_file);
    
    % Processing the labels
    labels_file = fopen(labels_fn, 'rb');
    bytes = fread(labels_file);
    bytes = bytes(9:end);
    y = zeros(nSamples, 1);
    y(:) = bytes(:);
    fclose(labels_file);
    
    % Write to the output
    save(output_fn, 'X', 'y');
end