# DirichletForest_distributed  

This repository contains an implementation of a **Dirichlet Random Forest**, designed for modeling **compositional (Dirichlet-distributed) data**.  

⚠️ **Note**: This project is still in progress. For a parallel version, see my [DirichletRandom](https://github.com/Xaleed/DirichletForestParallel.git) repository.  

---

## 📦 Installation  

Clone this repository and install locally in R:  

```r
devtools::install_github("https://github.com/Xaleed/DirichletForest.git")
 
library(DirichletForest)

# Generate predictors
n <- 500
p <- 4
X <- matrix(rnorm(n * p), n, p)

# Generate Dirichlet responses
if (!requireNamespace("MCMCpack", quietly = TRUE)) {
  install.packages("MCMCpack")
}
alpha <- c(2, 3, 4)
Y <- MCMCpack::rdirichlet(n, alpha)

# Fit a distributed Dirichlet Forest with 50 trees using 3 cores
df_par3 <- dirichlet_forest(X, Y, B = 50)

# Predict on new data (here we reuse X for illustration)
pred3 <- predict_dirichlet_forest(df_par3, X)


