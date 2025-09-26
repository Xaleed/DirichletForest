#include <Rcpp.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>

using namespace Rcpp;

// Node structure for the tree
struct Node {
  int feature_index;
  double split_value;
  Node* left;
  Node* right;
  NumericVector alpha_prediction;
  NumericVector mean_prediction;  // Added for mean predictions
  bool is_leaf;
  
  Node() : feature_index(-1), split_value(0.0), left(nullptr), right(nullptr), 
           is_leaf(false) {}
  
  ~Node() {
    if (left) delete left;
    if (right) delete right;
  }
};

// Custom implementation of log gamma function using Lanczos approximation
double custom_lgamma(double x) {
    if (x <= 0) return std::numeric_limits<double>::quiet_NaN();
    
    // For very small values, use recurrence relation
    if (x < 0.5) {
        return custom_lgamma(x + 1) - std::log(x);
    }
    
    // Lanczos approximation with g=7, n=9 (high precision)
    if (x < 12.0) {
        static const double g = 7.0;
        static const double coeff[9] = {
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
        };
        
        x -= 1.0;
        double a = coeff[0];
        for (int i = 1; i < 9; i++) {
            a += coeff[i] / (x + static_cast<double>(i));
        }
        
        double t = x + g + 0.5;
        return 0.5 * std::log(2.0 * M_PI) + (x + 0.5) * std::log(t) - t + std::log(a);
    }
    
    // Stirling's approximation for large values
    const double inv_x = 1.0 / x;
    const double inv_x_sq = inv_x * inv_x;
    
    return (x - 0.5) * std::log(x) - x + 0.91893853320467274178 + // 0.5*log(2*pi)
           inv_x * (0.083333333333333333333 - inv_x_sq * 
           (0.002777777777777777778 - inv_x_sq * 0.0007936507936507937));
}

// Custom implementation of digamma function (psi function)
double custom_digamma(double x) {
    if (x <= 0) return std::numeric_limits<double>::quiet_NaN();
    
    // Use recurrence relation for small values
    double result = 0.0;
    while (x < 6.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    
    // Asymptotic expansion for large x
    const double inv_x = 1.0 / x;
    const double inv_x_sq = inv_x * inv_x;
    
    return result + std::log(x) - 0.5 * inv_x - 
           inv_x_sq * (0.083333333333333333333 - inv_x_sq * 0.0083333333333333333333);
}

// Custom implementation of trigamma function (derivative of digamma)
double custom_trigamma(double x) {
    if (x <= 0) return std::numeric_limits<double>::quiet_NaN();
    
    // Use recurrence relation for small values
    double result = 0.0;
    while (x < 6.0) {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    
    // Asymptotic expansion for large x
    const double inv_x = 1.0 / x;
    const double inv_x_sq = inv_x * inv_x;
    
    return result + inv_x + 0.5 * inv_x_sq + 
           inv_x_sq * inv_x * (0.16666666666666666667 - 0.033333333333333333333 * inv_x_sq);
}

// LU decomposition with partial pivoting for matrix inversion
bool lu_invert(std::vector<std::vector<double>>& A, int n) {
    // Create permutation vector
    std::vector<int> perm(n);
    for (int i = 0; i < n; i++) perm[i] = i;
    
    // LU decomposition
    for (int k = 0; k < n; k++) {
        // Find pivot
        int max_row = k;
        double max_val = std::abs(A[k][k]);
        for (int i = k + 1; i < n; i++) {
            double val = std::abs(A[i][k]);
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }
        
        if (max_val < 1e-14) return false; // Singular matrix
        
        // Swap rows
        if (max_row != k) {
            std::swap(A[k], A[max_row]);
            std::swap(perm[k], perm[max_row]);
        }
        
        // Eliminate
        for (int i = k + 1; i < n; i++) {
            A[i][k] /= A[k][k];
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
    
    // Create identity matrix for inversion
    std::vector<std::vector<double>> inv(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) inv[i][i] = 1.0;
    
    // Apply permutation to identity
    for (int i = 0; i < n; i++) {
        if (perm[i] != i) {
            std::swap(inv[i], inv[perm[i]]);
        }
    }
    
    // Forward substitution
    for (int i = 1; i < n; i++) {
        for (int k = 0; k < i; k++) {
            for (int j = 0; j < n; j++) {
                inv[i][j] -= A[i][k] * inv[k][j];
            }
        }
    }
    
    // Back substitution
    for (int i = n - 1; i >= 0; i--) {
        for (int j = 0; j < n; j++) {
            inv[i][j] /= A[i][i];
        }
        for (int k = 0; k < i; k++) {
            for (int j = 0; j < n; j++) {
                inv[k][j] -= A[k][i] * inv[i][j];
            }
        }
    }
    
    // Copy result back to A
    A = inv;
    return true;
}

// Matrix-vector multiplication
std::vector<double> matvec_multiply(const std::vector<std::vector<double>>& A, 
                                    const std::vector<double>& b) {
    int n = A.size();
    int m = b.size();
    std::vector<double> result(n, 0.0);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result[i] += A[i][j] * b[j];
        }
    }
    
    return result;
}

// Helper function to calculate Dirichlet log-likelihood
double log_likelihood_dirichlet_rcpp(const NumericMatrix& Y, const NumericVector& alpha) {
    int n = Y.nrow();
    int k = Y.ncol();
    double loglik = 0.0;
    double alpha_sum = 0.0;
    
    // Calculate alpha sum
    for (int j = 0; j < k; j++) {
        alpha_sum += alpha[j];
    }
    
    double log_gamma_alpha_sum = custom_lgamma(alpha_sum);
    
    // Pre-compute log_gamma for alpha values
    std::vector<double> log_gamma_alpha(k);
    for (int j = 0; j < k; j++) {
        log_gamma_alpha[j] = custom_lgamma(alpha[j]);
    }
    
    for (int i = 0; i < n; i++) {
        double sum_y = 0.0;
        double row_contrib = 0.0;
        
        // Check validity and calculate contribution
        for (int j = 0; j < k; j++) {
            double y_val = Y(i, j);
            if (y_val <= 0 || y_val >= 1) {
                return -1e18; // Invalid values
            }
            sum_y += y_val;
            row_contrib += (alpha[j] - 1) * std::log(y_val);
        }
        
        if (std::abs(sum_y - 1.0) > 1e-6) {
            return -1e18; // Doesn't sum to 1
        }
        
        loglik += log_gamma_alpha_sum;
        for (int j = 0; j < k; j++) {
            loglik -= log_gamma_alpha[j];
        }
        loglik += row_contrib;
    }
    
    return loglik;
}

// Method of Moments estimation
// Improved Method of Moments estimation to match Julia performance
NumericVector estimate_parameters_mom_rcpp(const NumericMatrix& Y) {
    const int n = Y.nrow();
    const int k = Y.ncol();
    
    // Handle empty matrix case
    if (n == 0) {
        return NumericVector(k, 1.0);
    }
    
    // Handle single sample case
    if (n == 1) {
        NumericVector result(k);
        for (int j = 0; j < k; j++) {
            result[j] = std::max(0.1, std::min(1000.0, Y(0, j)));
        }
        return result;
    }
    
    // Calculate means efficiently
    NumericVector means(k, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            means[j] += Y(i, j);
        }
    }
    
    const double inv_n = 1.0 / n;
    for (int j = 0; j < k; j++) {
        means[j] *= inv_n;
    }
    
    // Calculate sample variances with Bessel's correction (n-1 denominator)
    NumericVector variances(k, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            const double diff = Y(i, j) - means[j];
            variances[j] += diff * diff;
        }
    }
    
    // Use n-1 for sample variance (Bessel's correction)
    const double inv_n_minus_1 = 1.0 / (n - 1);
    for (int j = 0; j < k; j++) {
        variances[j] *= inv_n_minus_1;
    }
    
    // Ensure first variance is not too close to zero (numerical stability)
    const double min_var = 1e-8;
    // --- MODIFICATION HERE: Apply min_var to all variances, not just the first ---
    for (int j = 0; j < k; j++) {
        if (variances[j] < min_var) {
            variances[j] = min_var;
        }
    }
    // -----------------------------------------------------------------------------
    
    // Estimate concentration parameter using first category
    // v = (μ₁(1-μ₁))/σ₁² - 1
    const double numerator = means[0] * (1.0 - means[0]);
    double v_val = numerator / variances[0] - 1.0;
    
    // --- MODIFICATION HERE: Match Julia's max(v, 0.1) for v_val ---
    if (v_val <= 0.0) { // If it's zero or negative
        v_val = 0.1;   // Use 0.1 as the lower bound, matching Julia
    }
    // -----------------------------------------------------------------
    
    // Calculate alpha parameters: αⱼ = v * μⱼ
    NumericVector alpha(k);
    for (int j = 0; j < k; j++) {
        alpha[j] = v_val * means[j];
    }
    
    // Apply bounds to ensure numerical stability
    // Match Julia's clamp.(alpha, 0.1, 1000.0)
    for (int j = 0; j < k; j++) {
        if (alpha[j] < 0.1) {
            alpha[j] = 0.1;
        } else if (alpha[j] > 1000.0) {
            alpha[j] = 1000.0;
        }
    }
    
    return alpha;
}

// MLE estimation with Newton-Raphson
NumericVector estimate_parameters_mle_newton_rcpp(const NumericMatrix& Y, int max_iter = 100, double tol = 1e-6, double lambda = 1e-6) {
    int n = Y.nrow();
    int k = Y.ncol();
    
    if (n == 0) {
        return NumericVector(k, 1.0);
    }
    
    // Initialize with method of moments
    NumericVector alpha = estimate_parameters_mom_rcpp(Y);
    
    // Pre-calculate log Y values
    std::vector<std::vector<double>> log_Y(n, std::vector<double>(k));
    std::vector<std::vector<bool>> valid_log(n, std::vector<bool>(k, false));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if (Y(i, j) > 0) {
                log_Y[i][j] = std::log(Y(i, j));
                valid_log[i][j] = true;
            }
        }
    }
    
    for (int iter = 0; iter < max_iter; iter++) {
        double alpha_sum = 0.0;
        for (int j = 0; j < k; j++) {
            alpha_sum += alpha[j];
        }
        
        double digamma_alpha_sum = custom_digamma(alpha_sum);
        double trigamma_alpha_sum = custom_trigamma(alpha_sum);
        
        // Calculate gradient
        std::vector<double> grad(k, 0.0);
        for (int j = 0; j < k; j++) {
            grad[j] = n * (digamma_alpha_sum - custom_digamma(alpha[j]));
            for (int i = 0; i < n; i++) {
                if (valid_log[i][j]) {
                    grad[j] += log_Y[i][j];
                }
            }
        }
        
        // Calculate Hessian
        std::vector<std::vector<double>> H(k, std::vector<double>(k));
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < k; l++) {
                if (j == l) {
                    H[j][l] = n * (trigamma_alpha_sum - custom_trigamma(alpha[j])) + lambda;
                } else {
                    H[j][l] = n * trigamma_alpha_sum;
                }
            }
        }
        
        // Solve H * delta = -grad
        std::vector<double> delta(k);
        if (!lu_invert(H, k)) {
            // Fallback to diagonal approximation
            for (int j = 0; j < k; j++) {
                double diag_val = n * (trigamma_alpha_sum - custom_trigamma(alpha[j])) + lambda;
                delta[j] = -grad[j] / diag_val;
            }
        } else {
            std::vector<double> neg_grad(k);
            for (int j = 0; j < k; j++) {
                neg_grad[j] = -grad[j];
            }
            delta = matvec_multiply(H, neg_grad);
        }
        
        // Check convergence
        double norm_delta_sq = 0.0;
        for (int j = 0; j < k; j++) {
            norm_delta_sq += delta[j] * delta[j];
        }
        
        if (norm_delta_sq < tol * tol) {
            break;
        }
        
        // Line search
        double step_size = 1.0;
        bool valid_step = false;
        
        for (int ls = 0; ls < 10; ls++) {
            bool all_valid = true;
            for (int j = 0; j < k; j++) {
                double new_alpha = alpha[j] + step_size * delta[j];
                if (new_alpha < 0.1 || new_alpha > 1000.0) {
                    all_valid = false;
                    break;
                }
            }
            
            if (all_valid) {
                for (int j = 0; j < k; j++) {
                    alpha[j] += step_size * delta[j];
                }
                valid_step = true;
                break;
            }
            
            step_size *= 0.5;
        }
        
        if (!valid_step) {
            break;
        }
    }
    
    return alpha;
}

// Calculate mean of observations in leaf
NumericVector calculate_mean_prediction(const NumericMatrix& Y, const IntegerVector& indices) {
  int k = Y.ncol();
  NumericVector means(k, 0.0);
  
  if (indices.size() == 0) {
    // Return uniform distribution if no samples
    for (int j = 0; j < k; j++) {
      means[j] = 1.0 / k;
    }
    return means;
  }
  
  for (int j = 0; j < k; j++) {
    double sum = 0.0;
    for (int i = 0; i < indices.size(); i++) {
      sum += Y(indices[i], j);
    }
    means[j] = sum / indices.size();
  }
  
  return means;
}

// Fit terminal node with both alpha and mean predictions
void FitTerminalNode(Node* node, const NumericMatrix& Y, const IntegerVector& sample_indices,  const std::string& method) {
  if (sample_indices.size() == 0) {
    int k = Y.ncol();
    node->alpha_prediction = NumericVector(k, 1.0);
    node->mean_prediction = NumericVector(k, 1.0/k);
  } else {
    // Create subset of Y for this node
    NumericMatrix Y_subset(sample_indices.size(), Y.ncol());
    for (int i = 0; i < sample_indices.size(); i++) {
      for (int j = 0; j < Y.ncol(); j++) {
        Y_subset(i, j) = Y(sample_indices[i], j);
      }
    }
    
    // Estimate alpha parameters
    if (method == "mle") {
        node->alpha_prediction = estimate_parameters_mle_newton_rcpp(Y_subset);
    } else {
        node->alpha_prediction = estimate_parameters_mom_rcpp(Y_subset);
    }
    
    // Calculate mean predictions
    node->mean_prediction = calculate_mean_prediction(Y, sample_indices);
  }
  
  node->is_leaf = true;
}

// Find best split - completely rewritten to eliminate all warnings
List FindBestSplit(const NumericMatrix& X, const NumericMatrix& Y, 
                   const IntegerVector& sample_indices, 
                   const IntegerVector& feature_subset, 
                   int n_min, 
                   const std::string& method, // --- ADDED: q_threshold parameter ---
                   int q_threshold) {
  
  double best_gain = -std::numeric_limits<double>::infinity();
  int best_feature = -1;
  double best_split_value = 0.0;
  IntegerVector best_left_indices, best_right_indices;
  
  int n_samples = sample_indices.size();
  
  // Calculate parent log-likelihood
  NumericMatrix Y_parent(n_samples, Y.ncol());
  for (int i = 0; i < n_samples; i++) {
    for (int j = 0; j < Y.ncol(); j++) {
      Y_parent(i, j) = Y(sample_indices[i], j);
    }
  }
  
  NumericVector parent_alpha;
  if (method == "mle") {
      parent_alpha = estimate_parameters_mle_newton_rcpp(Y_parent);
  } else {
      parent_alpha = estimate_parameters_mom_rcpp(Y_parent);
  }
  double parent_loglik = log_likelihood_dirichlet_rcpp(Y_parent, parent_alpha);
  
  int n_features = feature_subset.size();
  for (int f = 0; f < n_features; f++) {
    int feature = feature_subset[f];
    
    // Get unique values for this feature
    std::vector<double> values;
    values.reserve(n_samples);
    for (int i = 0; i < n_samples; i++) {
      values.push_back(X(sample_indices[i], feature));
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    
    int n_values = static_cast<int>(values.size());
    if (n_values <= 1) continue;
    
    // Try different split points
    std::vector<double> split_points;
    if (n_values > q_threshold) { // --- MODIFICATION: q_threshold logic ---
        // Calculate quantiles
        for (int k = 1; k <= q_threshold; ++k) {
            double prob = static_cast<double>(k) / (q_threshold + 1);
            // This is a simplified quantile, assuming sorted `values` are unique points.
            // A more robust quantile for general data would be R's `quantile` function.
            // For now, let's use a linear interpolation from the sorted unique values.
            double q_idx = (values.size() - 1) * prob;
            int lower_idx = static_cast<int>(std::floor(q_idx));
            int upper_idx = static_cast<int>(std::ceil(q_idx));

            if (lower_idx == upper_idx) {
                split_points.push_back(values[lower_idx]);
            } else {
                double val1 = values[lower_idx];
                double val2 = values[upper_idx];
                double frac = q_idx - lower_idx;
                split_points.push_back(val1 + frac * (val2 - val1));
            }
        }
         // Ensure split points are distinct
        std::sort(split_points.begin(), split_points.end());
        split_points.erase(std::unique(split_points.begin(), split_points.end()), split_points.end());

        // Ensure split_points are within the range of observed values, 
        // and ideally represent splits between observed unique values.
        // For simplicity, we can also just take midpoints between *selected* unique values
        // that are closest to the calculated quantiles.
        // For now, let's use the raw quantile values directly as split_points.
        // A more exact match to Julia's `quantile` might involve `R::quantile` but requires `RcppArmadillo` or similar.
    } else {
        // Use midpoints between all unique values
        for (int k = 1; k < n_values; k++) {
            split_points.push_back((values[k-1] + values[k]) / 2.0);
        }
    }
    // -----------------------------------------------------------------
    
    for (double split_val : split_points) { // --- MODIFICATION: Iterate over split_points ---
      
      std::vector<int> left_idx, right_idx;
      left_idx.reserve(n_samples);
      right_idx.reserve(n_samples);
      
      for (int i = 0; i < n_samples; i++) {
        int idx = sample_indices[i];
        if (X(idx, feature) <= split_val) {
          left_idx.push_back(idx);
        } else {
          right_idx.push_back(idx);
        }
      }
      
      int n_left = static_cast<int>(left_idx.size());
      int n_right = static_cast<int>(right_idx.size());
      
      if (n_left < n_min || n_right < n_min) {
        continue;
      }
      
      // Calculate log-likelihood for children
      NumericMatrix Y_left(n_left, Y.ncol());
      for (int i = 0; i < n_left; i++) {
        for (int j = 0; j < Y.ncol(); j++) {
          Y_left(i, j) = Y(left_idx[i], j);
        }
      }
      
      NumericMatrix Y_right(n_right, Y.ncol());
      for (int i = 0; i < n_right; i++) {
        for (int j = 0; j < Y.ncol(); j++) {
          Y_right(i, j) = Y(right_idx[i], j);
        }
      }
      
      NumericVector left_alpha;
      if (method == "mle") {
          left_alpha = estimate_parameters_mle_newton_rcpp(Y_left);
      } else {
          left_alpha = estimate_parameters_mom_rcpp(Y_left);
      }
      
      NumericVector right_alpha;
      if (method == "mle") {
         right_alpha = estimate_parameters_mle_newton_rcpp(Y_right);
      } else {
         right_alpha = estimate_parameters_mom_rcpp(Y_right);
      }
      
      double left_loglik = log_likelihood_dirichlet_rcpp(Y_left, left_alpha);
      double right_loglik = log_likelihood_dirichlet_rcpp(Y_right, right_alpha);
      
      double gain = (left_loglik + right_loglik) - parent_loglik;
      
      if (gain > best_gain) {
        best_gain = gain;
        best_feature = feature;
        best_split_value = split_val;
        best_left_indices = IntegerVector(left_idx.begin(), left_idx.end());
        best_right_indices = IntegerVector(right_idx.begin(), right_idx.end());
      }
    }
  }
  
  return List::create(
    Named("gain") = best_gain,
    Named("feature") = best_feature,
    Named("split_value") = best_split_value,
    Named("left_indices") = best_left_indices,
    Named("right_indices") = best_right_indices
  );
}

// Grow tree recursively
Node* GrowTree(const NumericMatrix& X, const NumericMatrix& Y,
               const IntegerVector& sample_indices,
               int current_depth, int d_max, int n_min, int m_try,
               std::mt19937& gen, const std::string& method, // --- ADDED: q_threshold ---
               int q_threshold) { 
  
  Node* node = new Node();
  
  // --- MODIFICATION START: Julia's initial termination condition for splitting ---
  // Julia: if depth >= max_depth || length(node_samples) < min_node_size * 2
  if (current_depth >= d_max || sample_indices.size() < n_min * 2 || sample_indices.size() == 0) {
    FitTerminalNode(node, Y, sample_indices, method);
    return node;
  }
  // --- MODIFICATION END ---
  
  // Feature subset selection
  int n_features = X.ncol();
  IntegerVector all_features = seq(0, n_features - 1);
  std::shuffle(all_features.begin(), all_features.end(), gen);
  IntegerVector feature_subset(all_features.begin(), all_features.begin() + std::min(m_try, n_features));
  
  // Find best split
  // --- MODIFICATION: Pass q_threshold to FindBestSplit ---
  List split_result = FindBestSplit(X, Y, sample_indices, feature_subset, n_min, method, q_threshold);
  double gain = as<double>(split_result["gain"]);
  
  if (gain <= 0 || as<int>(split_result["feature"]) == -1) {
    FitTerminalNode(node, Y, sample_indices, method);
    return node;
  }
  
  // Set node properties
  node->feature_index = as<int>(split_result["feature"]);
  node->split_value = as<double>(split_result["split_value"]);
  node->is_leaf = false;
  
  // Grow children
  IntegerVector left_indices = as<IntegerVector>(split_result["left_indices"]);
  IntegerVector right_indices = as<IntegerVector>(split_result["right_indices"]);
  
  node->left = GrowTree(X, Y, left_indices, current_depth + 1, d_max, n_min, m_try, gen, method, q_threshold); // --- MODIFIED
  node->right = GrowTree(X, Y, right_indices, current_depth + 1, d_max, n_min, m_try, gen, method, q_threshold); // --- MODIFIED
  
  return node;
}

// Build Dirichlet Forest
// [[Rcpp::export]]
List DirichletForest(NumericMatrix X, NumericMatrix Y, int B = 100, 
                     int d_max = 10, int n_min = 5, int m_try = -1, 
                     int seed = 123, std::string method = "mle", // --- ADDED: q_threshold ---
                     int q_threshold = 500000000) { 
  
  int n_samples = X.nrow();
  int n_features = X.ncol();
  
  if (m_try <= 0) {
    m_try = std::max(1, (int)std::sqrt(n_features));
  }
  
  std::mt19937 gen(seed);
  // CORRECTED: Use proper bootstrap sampling with replacement
  std::uniform_int_distribution<int> dis(0, n_samples - 1);
  
  std::vector<Node*> forest(B);
  
  for (int b = 0; b < B; b++) {
    // CORRECTED: Bootstrap sampling WITH replacement (like Julia)
    IntegerVector bootstrap_indices(n_samples);
    for (int i = 0; i < n_samples; i++) {
      bootstrap_indices[i] = dis(gen);
    }
    
    // Grow tree
    forest[b] = GrowTree(X, Y, bootstrap_indices, 0, d_max, n_min, m_try, gen, method, q_threshold); // --- MODIFIED
  }
  
  // Convert to external pointers for R
  List forest_ptrs(B);
  for (int i = 0; i < B; i++) {
    forest_ptrs[i] = XPtr<Node>(forest[i]);
  }
  
  return List::create(
    Named("forest") = forest_ptrs,
    Named("n_trees") = B,
    Named("n_features") = n_features,
    Named("n_classes") = Y.ncol()
  );
}

// Predict single sample through tree
List predict_sample_tree(Node* node, const NumericVector& x) {
  if (node->is_leaf) {
    return List::create(
      Named("alpha_prediction") = node->alpha_prediction,
      Named("mean_prediction") = node->mean_prediction
    );
  }
  
  if (x[node->feature_index] <= node->split_value) {
    return predict_sample_tree(node->left, x);
  } else {
    return predict_sample_tree(node->right, x);
  }
}

// Predict with Dirichlet Forest - returns both alpha and mean predictions
// [[Rcpp::export]]
List PredictDirichletForest(List forest_model, NumericMatrix X_new) {
  
  List forest_ptrs = forest_model["forest"];
  int n_trees = forest_model["n_trees"];
  int n_classes = forest_model["n_classes"];
  int n_samples = X_new.nrow();
  
  NumericMatrix alpha_predictions(n_samples, n_classes);
  NumericMatrix mean_predictions(n_samples, n_classes);
  
  for (int i = 0; i < n_samples; i++) {
    NumericVector sample = X_new(i, _);
    
    NumericVector alpha_sum(n_classes, 0.0);
    NumericVector mean_sum(n_classes, 0.0);
    
    for (int t = 0; t < n_trees; t++) {
      // FIX: Explicit cast to SEXP
      XPtr<Node> tree_ptr(as<SEXP>(forest_ptrs[t]));
      List tree_pred = predict_sample_tree(tree_ptr, sample);
      
      NumericVector alpha_pred = tree_pred["alpha_prediction"];
      NumericVector mean_pred = tree_pred["mean_prediction"];
      
      for (int j = 0; j < n_classes; j++) {
        alpha_sum[j] += alpha_pred[j];
        mean_sum[j] += mean_pred[j];
      }
    }
    
    for (int j = 0; j < n_classes; j++) {
      alpha_predictions(i, j) = alpha_sum[j] / n_trees;
      mean_predictions(i, j) = mean_sum[j] / n_trees;
    }
  }
  
  return List::create(
    Named("alpha_predictions") = alpha_predictions,
    Named("mean_predictions") = mean_predictions
  );
}

// Clean up forest memory
// [[Rcpp::export]]
void delete_dirichlet_forest_rcpp(List forest_model) {
  List forest_ptrs = forest_model["forest"];
  int n_trees = forest_model["n_trees"];
  
  for (int i = 0; i < n_trees; i++) {
    // FIX: Explicit cast to SEXP
    XPtr<Node> tree_ptr(as<SEXP>(forest_ptrs[i]));
    Node* raw_ptr = tree_ptr.get();
    if (raw_ptr != nullptr) {
      delete raw_ptr;
      tree_ptr.release();
    }
  }
}