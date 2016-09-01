// Software: K-Nearest Neighbor
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

const double INF = 1e9;

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

double distance(int N, double *x, double *y) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sqrt(sum);
}

void mexFunction(int nOutputs, mxArray *output_pointers[], int nInputs, const mxArray *input_pointers[]) {
    if (nInputs != 5) {
        std::cerr << "There must be exactly 5 input parameters!" << std::endl;
        std::cerr << "[X_training, y_training, X_testing, y_testing, k]" << std::endl;
        return;
    }
    
    int nFeatures = mxGetN(input_pointers[0]);
    int nTraining = mxGetM(input_pointers[0]);
    int nTesting = mxGetM(input_pointers[2]);
    int k = mxGetScalar(input_pointers[4]);
    
    if ((mxGetM(input_pointers[1]) != size_t(1)) && (mxGetN(input_pointers[1]) != size_t(1))) {
        std::cerr << "The y_training must be a vector!" << std::endl;
        return;
    }
    
    if ((mxGetM(input_pointers[1]) != size_t(nTraining)) && (mxGetN(input_pointers[1]) != size_t(nTraining))) {
        std::cerr << "The number of training samples in X_training and y_training must be the same!" << std::endl;
        return;
    }
    
    if (mxGetN(input_pointers[2]) != size_t(nFeatures)) {
        std::cerr << "The number of features of X_testing and X_training must be the same!" << std::endl;
        return;
    }
    
    if ((mxGetM(input_pointers[3]) != size_t(nTesting)) && (mxGetN(input_pointers[3]) != size_t(nTesting))) {
        std::cerr << "The number of testing samples in X_testing and y_testing must be the same!" << std::endl;
        return;
    }
    
    if ((k <= 0) || (k > nTraining)) {
        std::cerr << "Scalar k is out of range!" << std::endl;
        return;
    }
    
    // Input memory
    std::cout << "Number of features: " << nFeatures << std::endl;
    std::cout << "Number of training samples: " << nTraining << std::endl;
    std::cout << "Number of testing samples: " << nTesting << std::endl;
    std::cout << "k = " << k << std::endl;
    
    double **X_training = new double* [nTraining];
    double *y_training = mxGetPr(input_pointers[1]);
    double **X_testing = new double* [nTesting];
    double *y_testing = mxGetPr(input_pointers[3]);
    
    for (int i = 0; i < nTraining; ++i) {
        X_training[i] = new double [nFeatures];
    }
    
    for (int i = 0; i < nTesting; ++i) {
        X_testing[i] = new double [nFeatures];
    }
    
    vector2matrix(mxGetPr(input_pointers[0]), nTraining, nFeatures, X_training);
    vector2matrix(mxGetPr(input_pointers[2]), nTesting, nFeatures, X_testing); 
    
    // Output memory
    output_pointers[0] = mxCreateDoubleMatrix(nTesting, 1, mxREAL);
    double *predict = mxGetPr(output_pointers[0]);
    
    // K-Nearest Neighbor algorithm
    double *min_dists = new double [k];
    int *labels = new int [k];
    
    int nCorrects = 0;        
    for (int i = 0; i < nTesting; ++i) {
        std::cout << "Testing sample " << i << ": ";
        
        for (int v = 0; v < k; ++v) {
            min_dists[v] = INF;
        }
        
        for (int j = 0; j < nTraining; ++j) {
            double dist = distance(nFeatures, X_training[j], X_testing[i]);
            for (int v = 0; v < k; ++v) {
                if (dist < min_dists[v]) {
                    for (int t = v + 1; t < k; ++t) {
                        min_dists[t] = min_dists[t - 1];
                        labels[t] = labels[t - 1];
                    }
                    min_dists[v] = dist;
                    labels[v] = y_training[j];
                    break;
                }
            }
        }
        
        for (int v = 0; v < k - 1; ++v) {
            for (int t = v + 1; t < k; ++t) {
                if (labels[v] > labels[t]) {
                    int temp = labels[v];
                    labels[v] = labels[t];
                    labels[t] = temp;
                }
            }
        }
        
        predict[i] = labels[0];
        int v = 0;
        int longest = 0;
        for (int t = 1; t < k; ++t) {
            if (labels[t] != labels[t - 1]) {
                if (t - v > longest) {
                    longest = t - v;
                    predict[i] = labels[t - 1];
                }
                v = t;
            }
        }
        if (k - v > longest) {
            longest = k - v;
            predict[i] = labels[k - 1];
        }
        
        if (predict[i] == y_testing[i]) {
            ++nCorrects;
            std::cout << "YES" << std::endl;
        } else {
            std::cout << "NO" << std::endl;
        }
    }
    
    // Return the number of correct classification
    output_pointers[1] = mxCreateDoubleScalar(nCorrects);
}