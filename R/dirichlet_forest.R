#' Dirichlet Forest
#'
#' Build a Dirichlet Forest for compositional data
#'
#' @param X Numeric matrix of predictors
#' @param Y Numeric matrix of compositional response variables (rows sum to 1)
#' @param B Integer, number of trees in the forest (default: 100)
#' @param d_max Integer, maximum depth of trees (default: 10)
#' @param n_min Integer, minimum number of samples in terminal nodes (default: 5)
#' @param m_try Integer, number of features to try at each split (default: sqrt(ncol(X)))
#' @param seed Integer, random seed (default: 123)
#' @param method String, estimation method ("mle" or "mom", default: "mle")
#'
#' @return A list containing the trained forest model
#' @export
#' @examples
#' \dontrun{
#' # Generate sample data
#' X <- matrix(rnorm(100*5), 100, 5)
#' Y <- matrix(runif(100*3), 100, 3)
#' Y <- Y / rowSums(Y)  # Make compositional
#' 
#' # Train forest
#' forest <- dirichlet_forest(X, Y, B = 50)
#' 
#' # Make predictions
#' predictions <- predict_dirichlet_forest(forest, X[1:10, ])
#' }
dirichlet_forest <- function(X, Y, B = 100, d_max = 10, n_min = 5, 
                             m_try = -1, seed = 123, method = "mom") {
  # Input validation
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.matrix(Y)) Y <- as.matrix(Y)
  
  # Check if Y is compositional (rows sum to 1)
  row_sums <- rowSums(Y)
  if (any(abs(row_sums - 1) > 1e-10)) {
    warning("Y does not appear to be compositional (rows don't sum to 1)")
  }
  
  DirichletForest(X, Y, B, d_max, n_min, m_try, seed, method)
}

#' Predict with Dirichlet Forest
#'
#' Make predictions using a trained Dirichlet Forest
#'
#' @param forest_model A trained forest model from dirichlet_forest()
#' @param X_new Numeric matrix of new predictors
#'
#' @return A list with alpha_predictions and mean_predictions matrices
#' @export
predict_dirichlet_forest <- function(forest_model, X_new) {
  if (!is.matrix(X_new)) X_new <- as.matrix(X_new)
  PredictDirichletForest(forest_model, X_new)
}

#' Clean up Dirichlet Forest
#'
#' Free memory used by the forest model
#'
#' @param forest_model A trained forest model
#' @export
delete_forest <- function(forest_model) {
  delete_dirichlet_forest_rcpp(forest_model)
}