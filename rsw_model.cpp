// File: rsw_model.cpp

#include "rsw_model.hpp"
#include <omp.h>
#include <cmath>
#include <iostream>

HighPerformanceRSW::HighPerformanceRSW(std::string task, int num_envs, double lambda_reg, 
                                       double alpha, double lr_theta, double lr_mu, 
                                       int epochs, int warmup_epochs, int seed)
    : task(task), num_envs(num_envs), epochs(epochs), warmup_epochs(warmup_epochs), seed(seed),
      lambda_reg(lambda_reg), alpha(alpha), lr_theta(lr_theta), lr_mu(lr_mu) {}

// We keep this function around in case you want to print the loss later, 
// though the analytical gradients no longer require it for training.
double HighPerformanceRSW::compute_loss_internal(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                                  const Eigen::VectorXd& target_theta, const Eigen::VectorXd& target_mu,
                                                  const std::vector<Eigen::VectorXd>& all_env_weights, double current_penalty_weight) const {
    size_t N = X.rows();
    size_t D = X.cols();
    
    Eigen::VectorXd current_mask = (1.0 + (-target_mu.array()).exp()).inverse().matrix();
    Eigen::VectorXd theta_effective = target_theta.cwiseProduct(current_mask);

    Eigen::VectorXd preds = X * theta_effective;
    Eigen::VectorXd base_errors = preds - y;

    double cumulative_base_loss = 0.0;
    Eigen::MatrixXd env_gradients(num_envs, D);

    for (int e = 0; e < num_envs; ++e) {
        const Eigen::VectorXd& w = all_env_weights[e];
        
        if (task == "classification") {
            double ce_loss = 0.0;
            for (size_t i = 0; i < N; ++i) {
                double p = 1.0 / (1.0 + std::exp(-preds(i)));
                p = std::clamp(p, 1e-15, 1.0 - 1e-15);
                ce_loss -= w(i) * (y(i) * std::log(p) + (1.0 - y(i)) * std::log(1.0 - p));
            }
            cumulative_base_loss += ce_loss / N;
        } else {
            cumulative_base_loss += (w.array() * base_errors.array().square()).sum() / N;
        }

        Eigen::VectorXd dynamic_resid = base_errors.cwiseProduct(w);
        env_gradients.row(e) = (X.transpose() * dynamic_resid) * (2.0 / N);
    }

    double avg_base_loss = cumulative_base_loss / num_envs;

    Eigen::VectorXd mean_gradient = env_gradients.colwise().mean();
    Eigen::VectorXd variance_gradient = Eigen::VectorXd::Zero(D);
    
    for (int e = 0; e < num_envs; ++e) {
        variance_gradient += (env_gradients.row(e).transpose() - mean_gradient).cwiseAbs2();
    }
    variance_gradient /= (num_envs - 1.0);

    double mip_penalty = variance_gradient.cwiseProduct(current_mask).squaredNorm();
    double sparsity_penalty = alpha * current_mask.sum();

    return avg_base_loss + current_penalty_weight * (lambda_reg * mip_penalty + sparsity_penalty);
}

void HighPerformanceRSW::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    size_t N = X.rows();
    size_t D = X.cols();

    // Structural Parameter Initialization
    theta = Eigen::VectorXd::Zero(D);
    mu = Eigen::VectorXd::Zero(D);
    mask = Eigen::VectorXd::Constant(D, 0.5);

    // Adam Optimizer State Space Initialization
    Eigen::VectorXd m_t = Eigen::VectorXd::Zero(D), v_t = Eigen::VectorXd::Zero(D);
    Eigen::VectorXd m_m = Eigen::VectorXd::Zero(D), v_m = Eigen::VectorXd::Zero(D);

    const double beta1 = 0.9, beta2 = 0.999, eps_adam = 1e-8;

    std::mt19937 global_gen(seed);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    for (int ep = 1; ep <= epochs; ++ep) {
        
        // --- UPGRADED STEP 1: COVARIATE-DETERMINING ENVIRONMENTS ---
        std::vector<Eigen::VectorXd> all_env_weights(num_envs, Eigen::VectorXd(N));
        for (int e = 0; e < num_envs; ++e) {
            // 1. Generate a random projection vector
            Eigen::VectorXd projection = Eigen::VectorXd::NullaryExpr(D, [&](){ return norm_dist(global_gen); });
            
            for (size_t i = 0; i < N; ++i) {
                // 2. Score the sample based on its actual market features (X)
                double score = X.row(i).dot(projection);
                
                // 3. Apply Sigmoid to create smooth, bounded environment weights (0 to 1)
                all_env_weights[e](i) = 1.0 / (1.0 + std::exp(-score)); 
            }
        }
        
        double penalty_weight = (ep > warmup_epochs) ? 1.0 : 0.0;

        // --- UPGRADED STEP 2: EXACT ANALYTICAL GRADIENTS ---
        // Completely eliminates the O(D) finite difference loops using Matrix Calculus.
        
        Eigen::VectorXd theta_eff = theta.cwiseProduct(mask);
        Eigen::VectorXd err = (X * theta_eff) - y;

        // 1. Calculate Environment-Specific Gradients
        Eigen::MatrixXd env_grads(num_envs, D);
        for(int e = 0; e < num_envs; ++e) {
            env_grads.row(e) = (2.0 / N) * X.transpose() * err.cwiseProduct(all_env_weights[e]);
        }
        Eigen::VectorXd mean_env_grad = env_grads.colwise().mean();

        // 2. Base Empirical Risk Gradient
        Eigen::VectorXd grad_theta_eff = (2.0 / N) * X.transpose() * err;

        // 3. Invariance Penalty Gradient (Hessian-Vector Product Trick)
        if (penalty_weight > 0.0) {
            Eigen::VectorXd penalty_grad = Eigen::VectorXd::Zero(D);
            for(int e = 0; e < num_envs; ++e) {
                Eigen::VectorXd grad_diff = env_grads.row(e).transpose() - mean_env_grad;
                Eigen::VectorXd X_diff = X * grad_diff;
                penalty_grad += (2.0 / N) * X.transpose() * all_env_weights[e].cwiseProduct(X_diff);
            }
            penalty_grad *= (2.0 / (num_envs - 1.0));
            grad_theta_eff += lambda_reg * penalty_grad;
        }

        // 4. Apply Chain Rule to separate Gradients for Theta and Mask (Mu)
        Eigen::VectorXd grad_theta = grad_theta_eff.cwiseProduct(mask);
        
        // Derivative of Sigmoid: dMask/dMu = mask * (1 - mask)
        Eigen::VectorXd mask_deriv = mask.cwiseProduct(Eigen::VectorXd::Ones(D) - mask);
        Eigen::VectorXd grad_mu = grad_theta_eff.cwiseProduct(theta).cwiseProduct(mask_deriv);
        
        if (penalty_weight > 0.0) {
            grad_mu += alpha * mask_deriv; // Sparsity penalty derivative
        }

        // --- STEP 3: Vectorized Momentum and Velocity Updates for Theta Vector ---
        m_t = beta1 * m_t + (1.0 - beta1) * grad_theta;
        v_t = beta2 * v_t + (1.0 - beta2) * grad_theta.cwiseAbs2();
        Eigen::VectorXd m_t_hat = m_t / (1.0 - std::pow(beta1, ep));
        Eigen::VectorXd v_t_hat = v_t / (1.0 - std::pow(beta2, ep));
        theta -= (lr_theta * m_t_hat.array() / (v_t_hat.array().sqrt() + eps_adam)).matrix();

        // --- STEP 4: Parameter Tracking and Momentum Updates for Mask space (Mu) ---
        if (ep > warmup_epochs) {
            int mu_step = ep - warmup_epochs;
            m_m = beta1 * m_m + (1.0 - beta1) * grad_mu;
            v_m = beta2 * v_m + (1.0 - beta2) * grad_mu.cwiseAbs2();
            Eigen::VectorXd m_m_hat = m_m / (1.0 - std::pow(beta1, mu_step));
            Eigen::VectorXd v_m_hat = v_m / (1.0 - std::pow(beta2, mu_step));
            mu -= (lr_mu * m_m_hat.array() / (v_m_hat.array().sqrt() + eps_adam)).matrix();
        }

        // Update active visibility mask values
        mask = (1.0 + (-mu.array()).exp()).inverse().matrix();
    }
}

Eigen::VectorXd HighPerformanceRSW::predict(const Eigen::MatrixXd& X) const {
    // Generate final safe prediction using the calculated sparse invariant mask
    Eigen::VectorXd theta_effective = theta.cwiseProduct(mask);
    return X * theta_effective;
}