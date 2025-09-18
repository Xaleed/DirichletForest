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

// Helper function to calculate Dirichlet log-likelihood
double log_likelihood_dirichlet_rcpp(const NumericMatrix& Y, const NumericVector& alpha) {
  int n = Y.nrow();
  int k = Y.ncol();
  double loglik = 0.0;
  double alpha_sum = sum(alpha);
  
  for (int i = 0; i < n; i++) {
    double sum_y = 0.0;
    for (int j = 0; j < k; j++) {
      if (Y(i, j) <= 0 || Y(i, j) >= 1) {
        return -1e18; // Penalize invalid values
      }
      sum_y += Y(i, j);
    }
    
    if (std::abs(sum_y - 1.0) > 1e-6) {
      return -1e18; // Penalize if doesn't sum to 1
    }
    
    loglik += R::lgammafn(alpha_sum);
    for (int j = 0; j < k; j++) {
      loglik -= R::lgammafn(alpha[j]);
      loglik += (alpha[j] - 1) * log(Y(i, j));
    }
  }
  
  return loglik;
}

// Method of Moments estimation
NumericVector estimate_parameters_mom_rcpp(const NumericMatrix& Y) {
  const int n = Y.nrow();
  const int k = Y.ncol();
  
  if (n == 0) {
    return NumericVector(k, 1.0);
  }
  
  NumericVector means(k, 0.0);
  NumericVector variances(k, 0.0);
  
  // Single pass: compute means and sum of squares simultaneously
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      const double val = Y(i, j);
      means[j] += val;
      variances[j] += val * val;
    }
  }
  
  // Finalize calculations
  const double inv_n = 1.0 / n;
  const double inv_n_minus_1 = 1.0 / (n - 1);
  
  for (int j = 0; j < k; ++j) {
    means[j] *= inv_n;
    variances[j] = (variances[j] - n * means[j] * means[j]) * inv_n_minus_1;
  }
  
  // Estimate parameters
  const double min_var = 1e-8;
  variances[0] = std::max(variances[0], min_var);
  
  const double v_val = std::max((means[0] * (1.0 - means[0])) / variances[0] - 1.0, 1.0);
  
  NumericVector alpha(k);
  for (int j = 0; j < k; ++j) {
    alpha[j] = std::max(0.01, std::min(1000.0, v_val * means[j]));
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
  NumericMatrix log_Y(n, k);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < k; j++) {
      if (Y(i, j) <= 0) {
        log_Y(i, j) = -std::numeric_limits<double>::infinity();
      } else {
        log_Y(i, j) = log(Y(i, j));
      }
    }
  }
  
  for (int iter = 0; iter < max_iter; iter++) {
    double alpha_sum = sum(alpha);
    
    // Calculate gradient
    NumericVector grad(k);
    for (int j = 0; j < k; j++) {
      grad[j] = n * (R::digamma(alpha_sum) - R::digamma(alpha[j]));
      for (int i = 0; i < n; i++) {
        if (std::isfinite(log_Y(i, j))) {
          grad[j] += log_Y(i, j);
        }
      }
    }
    
    // Calculate Hessian
    NumericMatrix H(k, k);
    double trigamma_sum = R::trigamma(alpha_sum);
    
    for (int j = 0; j < k; j++) {
      for (int l = 0; l < k; l++) {
        if (j == l) {
          H(j, l) = n * (trigamma_sum - R::trigamma(alpha[j])) + lambda;
        } else {
          H(j, l) = n * trigamma_sum;
        }
      }
    }
    
    // Solve for delta using R's solve function
    NumericMatrix H_mat = clone(H);
    NumericVector neg_grad = -grad;
    
    SEXP solve_call = PROTECT(Rf_lang3(Rf_install("solve"), 
                                       PROTECT(wrap(H_mat)), 
                                       PROTECT(wrap(neg_grad))));
    NumericVector delta = as<NumericVector>(Rf_eval(solve_call, R_GlobalEnv));
    UNPROTECT(3);
    
    // Check convergence
    double norm_delta = sqrt(sum(delta * delta));
    if (norm_delta < tol) {
      break;
    }
    
    // Line search
    double step_size = 1.0;
    NumericVector alpha_new(k);
    bool valid_step = false;
    
    for (int ls = 0; ls < 20; ls++) {
      bool all_valid = true;
      for (int j = 0; j < k; j++) {
        alpha_new[j] = alpha[j] + step_size * delta[j];
        if (alpha_new[j] < 0.1 || alpha_new[j] > 1000.0 || !std::isfinite(alpha_new[j])) {
          all_valid = false;
          break;
        }
      }
      
      if (all_valid) {
        valid_step = true;
        break;
      }
      
      step_size *= 0.5;
    }
    
    if (valid_step) {
      alpha = alpha_new;
    } else {
      break; // No valid step found
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
                   const std::string& method) {
  
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
    for (int k = 1; k < n_values; k++) {
      double split_val = (values[k-1] + values[k]) / 2.0;
      
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
               std::mt19937& gen, const std::string& method) {
  
  Node* node = new Node();
  
  // Check termination conditions
  if (sample_indices.size() < n_min || current_depth >= d_max || sample_indices.size() == 0) {
    FitTerminalNode(node, Y, sample_indices, method);
    return node;
  }
  
  // Feature subset selection
  int n_features = X.ncol();
  IntegerVector all_features = seq(0, n_features - 1);
  std::shuffle(all_features.begin(), all_features.end(), gen);
  IntegerVector feature_subset(all_features.begin(), all_features.begin() + std::min(m_try, n_features));
  
  // Find best split
  List split_result = FindBestSplit(X, Y, sample_indices, feature_subset, n_min, method);
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
  
  node->left = GrowTree(X, Y, left_indices, current_depth + 1, d_max, n_min, m_try, gen, method);
  node->right = GrowTree(X, Y, right_indices, current_depth + 1, d_max, n_min, m_try, gen, method);
  
  return node;
}

// Build Dirichlet Forest
// [[Rcpp::export]]
List DirichletForest(NumericMatrix X, NumericMatrix Y, int B = 100, 
                     int d_max = 10, int n_min = 5, int m_try = -1, 
                     int seed = 123, std::string method = "mle") {
  
  int n_samples = X.nrow();
  int n_features = X.ncol();
  
  if (m_try <= 0) {
    m_try = std::max(1, (int)std::sqrt(n_features));
  }
  
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dis(0, n_samples - 1);
  
  std::vector<Node*> forest(B);
  
  for (int b = 0; b < B; b++) {
    // Bootstrap sampling
    IntegerVector bootstrap_indices(n_samples);
    for (int i = 0; i < n_samples; i++) {
      bootstrap_indices[i] = dis(gen);
    }
    
    // Grow tree
    forest[b] = GrowTree(X, Y, bootstrap_indices, 0, d_max, n_min, m_try, gen, method);
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