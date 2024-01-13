#include <iostream>
#include <ctime>
#include <Eigen/Dense>
#include "stats/stats.h"

int main() {

    int n_features = 2;

    // dependent variable sample
    std::vector<double> y_sample = { 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1 };

    // input data
    Eigen::VectorXd y = Eigen::Map<Eigen::VectorXd>(y_sample.data(), y_sample.size());

    // independent variable samples
    std::vector<double> sample1 = { 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50 };
    std::vector<double> sample2 = { 5.50, 5.05, 4.75, 4.50, 4.30, 4.05, 3.50, 3.30, 3.05, 2.80, 2.50, 2.30, 2.00, 1.80, 1.80, 1.50, 1.25, 1.05, 0.75, 0.55 };

    // input data
    Eigen::VectorXd x1 = Eigen::Map<Eigen::VectorXd>(sample1.data(), sample1.size());
    Eigen::VectorXd x2 = Eigen::Map<Eigen::VectorXd>(sample2.data(), sample2.size());

    // initialise new feature matrix
    Eigen::MatrixXd X(x1.size(), n_features);

    // add data to feature matrix
    X.col(0) = x1;
    X.col(1) = x2;

    // instantiate logistic regression model
    stats::LogisticRegression model;

    // calculate mean 
    Eigen::VectorXd mean = X.colwise().mean();

    // calculate standard deviation
    Eigen::VectorXd std = ((X.array().transpose().colwise() - mean.array()).square().rowwise().sum() / X.rows()).sqrt();

    // calculate normalised feature matrix/vector
    Eigen::MatrixXd nX = ((X.array().transpose().colwise() - mean.array()).colwise() / std.array()).transpose();
    
    // start timer 
    clock_t start_time = clock();

    // fit logistic model
    model.fit(nX, y);

    // end timer 
    clock_t end_time = clock();

    // new data
    std::vector<double> new_sample1 = { 5 };
    std::vector<double> new_sample2 = { 1.95 };
    // new input data
    Eigen::VectorXd x1new = Eigen::Map<Eigen::VectorXd>(new_sample1.data(), new_sample1.size());
    Eigen::VectorXd x2new = Eigen::Map<Eigen::VectorXd>(new_sample2.data(), new_sample2.size());

    // initialise new feature matrix
    Eigen::MatrixXd Xnew(new_sample1.size(), n_features);

    // add data to feature matrix
    Xnew.col(0) = x1new;
    Xnew.col(1) = x2new;

    // nomalise new feature matrix 
    Eigen::MatrixXd nXnew = ((Xnew.array().transpose().colwise() - mean.array()).colwise() / std.array()).transpose();

    // vector prediction 
    Eigen::VectorXd prediction = model.predict_probability(nXnew);

    // total time in seconds
    double total_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

    Eigen::VectorXd current_leverage(1);

    current_leverage[0] = 0.4;

    // Output results
    std::cout << "Time = " << total_time << "\n\n";
    std::cout << "Converged in " << model.niter << " iterations\n\n";
    std::cout << "beta = \n" << model.beta << "\n\n";
    std::cout << "prediction = \n" << prediction << "\n\n";
    std::cout << "current leverage = \n" << current_leverage << "\n\n";

    return 0;
}