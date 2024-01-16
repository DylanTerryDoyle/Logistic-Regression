#ifndef STATISTICS_H
#define STATISTICS_H

#include <cmath>
#include <LBFGS.h>
#include <Eigen/Dense>

namespace stats {

    /**
      * @class 
      * @brief LogisticRegression class.
      *
      * @param penaulty : boolian, if true L2 penalty is applied, if false no penalty.
      * @param lambda : double, L2 constraint value.
      * 
      */
    class LogisticRegression {
    public:
        // PARAMETERS
        // add L2 penalty if true
        bool penalty;
        // L2 cost constraint size
        double lambda;
        // optimal function output
        double fx;
        // number of iterations
        int niter;


        // DATA
        // beta vector
        Eigen::VectorXd beta;
        // feature matrix
        Eigen::MatrixXd Xf;


        // CONSTRUCTOR
        /**
          * LogisticRegression class constructor.
          *
          * @param[out] penaulty : boolian, if true L2 penalty is applied, if false no penalty.
          * @param[out] lambda : double, L2 constraint value.
          */
        LogisticRegression(bool penalty_ = true, double lambda_ = 0.025) {
            // L2 penalty condition
            penalty = penalty_;
            // L2 constraint value
            lambda = lambda_;
            // initial optimal function value
            fx = 0.0;
            // initial number of iterations
            niter = 0;
        }

        // MEHTODS 

        /**
          * Creates a new matrix of input matrix X with an extra first column of 1's.
          *
          * @param[out] X : matrix.
          * @return out : copy of X with extra column of 1's.
          */
        template<typename Derived>
        Eigen::MatrixXd add_intercept(const Eigen::MatrixBase<Derived>& X) {
            // number of sample data points (rows of X)
            int r = (int)X.rows();

            // number of features including intercept (cols + 1 of X)
            int c = (int)X.cols() + 1;

            // initialise feature matrix with intercept (column of ones)
            Eigen::MatrixXd newX = Eigen::MatrixXd::Zero(r, c);

            // replace first column with 1's
            newX.col(0) = Eigen::VectorXd::Ones(r);

            // add X data to rest of feature matrix Xf
            newX.block(0, 1, r, c - 1) = X;

            // return new matrix 
            return newX;
        }
        
        /**
          * Calculate the logistic function for all elements of the input vector.
          *
          * @param[out] z : vector whose values the logistic function is applied to.
          * @return out : vector of computed logistic values.
          */
        template<typename Derived>
        Eigen::VectorXd logistic(const Eigen::MatrixBase<Derived>& z) {
            return 1.0 / (1.0 + (-z).array().exp());
        }

        /**
          * Calculate the log likelihood error of the logistic functon to use in fit method for the LBFGS optimisation.
          *
          * @param[out] beta : vector of parameter values.
          * @param[out] Xf : feature matrix with intercept (one's in first column).
          * @param[out] y : dependent varaible vector (vector[i] in {0, 1} for all i).
          * @return[out] out : double value of log likelihood error.
          */
        double objective(const Eigen::VectorXd& beta, const Eigen::Ref<const Eigen::MatrixXd>& Xf, const Eigen::Ref<const Eigen::VectorXd>& y) {
            
            // initialise error value
            double error;
            
            // probability model (y_hat)
            Eigen::VectorXd p = logistic(Xf * beta);
            
            // logistic loss error
            // error without penalty 
            if (!penalty) {
                error = (-y.array() * p.array().log() - (1 - y.array()) * (1 - p.array()).log()).sum();
            }
            // error with L2 penalty 
            else {
                // create new beta vector without beta0 (first element) for L2 penalty 
                Eigen::VectorXd betaL2 = beta;
                betaL2(0) = 0.0;
                error = (-y.array() * p.array().log() - (1 - y.array()) * (1 - p.array()).log() + lambda * betaL2.transpose() * betaL2).sum();
            }
            // return error 
            return error;
        }

        /**
          * Update gradient of logistic log likelihood to use in fit method for the LBFGS optimisation.
          *
          * @param[out] beta : vector of parameter values.
          * @param[out] grad : gradient vecror to be updated.
          * @param[out] Xf : feature matrix with intercept (one's in first column).
          * @param[out] y : dependent varaible vector (vector[i] in {0, 1} for all i).
          * @return[out] out : void, no output.
          */
        void gradient(const Eigen::VectorXd& beta, Eigen::VectorXd& grad, const Eigen::Ref<const Eigen::MatrixXd>& Xf, const Eigen::Ref<const Eigen::VectorXd>& y) {

            // probability model (y_hat)
            Eigen::VectorXd p = logistic(Xf * beta);

            // gradient without penalty 
            if (!penalty) {
                grad = Xf.transpose() * (p - y);
            }
            // gradient with L2 penalty 
            else {
                // create new beta vector without beta0 (first element) for L2 penalty 
                Eigen::VectorXd betaL2 = beta;
                betaL2(0) = 0.0;
                grad = Xf.transpose() * (p - y) + (2 * lambda * betaL2) * Xf.rows();
            }
        }
        
        /**
          * Fit logistic regression model to given trainign data using LBFGS optimisation.
          *
          * @param[out] X : feature matrix.
          * @param[out] y : dependent varaible vector (vector[i] in {0, 1} for all i).
          * @return[out] out : void, no output.
          */
        template<typename Derived>
        void fit(const Eigen::MatrixBase<Derived>& X, const Eigen::VectorXd& y) {
            // initialise feature matrix with intercept
            Xf = add_intercept(X);

            // initialise beta vector
            int n = (int)X.cols() + 1;
            beta = Eigen::VectorXd::Zero(n);
            
            // optimisation function
            auto func = [&](const Eigen::VectorXd& beta, Eigen::VectorXd& grad) {
                double result = objective(beta, Xf, y);
                gradient(beta, grad, Xf, y);
                return result;
            };

            // solver parameters
            LBFGSpp::LBFGSParam<double> solver_params;

            // initialise LGBGS solver
            LBFGSpp::LBFGSSolver<double> solver(solver_params);

            // optimisation
            niter = solver.minimize(func, beta, fx);
        }

        /**
          * Vector of probability estimates of matrix/vector X using trained model parameters, only call after fit method on new data.
          *
          * @param[out] X : matrix/vector.
          * @return[out] out : vector of probabiilty estimates.
          */
        template<typename Derived>
        Eigen::VectorXd predict_probability(const Eigen::MatrixBase<Derived>& X) {
            // add intercept 
            Eigen::MatrixXd newXf = add_intercept(X);

            // probability estimate
            Eigen::VectorXd p = logistic(newXf * beta);

            // return probability estimate
            return p;
        }
    };
}

#endif // STATISTICS_H