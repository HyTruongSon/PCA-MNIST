// Software: Normalize the whole databsae
// Author: Hy Truong Son
// Position: PhD Student
// Institution: Department of Computer Science, The University of Chicago
// Email: sonpascal93@gmail.com, hytruongson@uchicago.edu
// Website: http://people.inf.elte.hu/hytruongson/
// Copyright 2016 (c) Hy Truong Son. All rights reserved.

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>
#include <ctime>

#include "mex.h"

using namespace std;

void vector2matrix(double *input, int nRows, int nCols, double **output) {
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            output[i][j] = input[j * nRows + i];
        }
    }
}

void matrix2vector(double **input, int nRows, int nCols, double *output) {
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            output[j * nRows + i] = input[i][j];
        }
    }
}

void copy(double *input, int N, double *output) {
    for (int i = 0; i < N; ++i) {
        output[i] = input[i];
    }
}

void mexFunction(int nOutputs, mxArray *output_pointers[], int nInputs, const mxArray *input_pointers[]) {
    if (nInputs != 1) {
        std::cerr << "The number of input parameters must be exactly 1!" << std::endl;
        return;
    }
    
    // Dimension of each sample
    int d = mxGetM(input_pointers[0]);
    
    // Number of samples
    int n = mxGetN(input_pointers[0]);
    
    // Memory allocation
    double **X = new double* [d];
    for (int i = 0; i < d; ++i) {
        X[i] = new double [n];
    }
    
    // Get the data
    vector2matrix(mxGetPr(input_pointers[0]), d, n, X);
    
    // Normalization
    for (int sample = 0; sample < n; ++sample) {
        // Computing the norm L2
        double norm = 0.0;
        for (int i = 0; i < d; ++i) {
            norm += X[i][sample] * X[i][sample];
        }
        norm = sqrt(norm);
        
        // Divide the vector to the norm L2
        for (int i = 0; i < d; ++i) {
            X[i][sample] /= norm;
        }
    }
    
    // Return the results
    output_pointers[0] = mxCreateDoubleMatrix(d, n, mxREAL);
    matrix2vector(X, d, n, mxGetPr(output_pointers[0]));
}