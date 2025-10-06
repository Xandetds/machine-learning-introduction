
# Notes — Applied ML 

## Linear Regression

### How does it work
We predict with: $\hat y = b + \mathbf{w}^\top \mathbf{x}$.

The mean error over all points is the MSE:  
$\mathrm{MSE}(b,\mathbf{w}) = \dfrac{1}{m}\sum_i \left(y_i - \hat y_i\right)^2.$

$R^2$ measures variance explained (not accuracy):  
$R^2 = 1 - \dfrac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}.$

---

### Notebook index

- **LinearRegressionMath.ipynb** — *math walk-through*  
  - tiny 4-point dataset  
  - manual formulas: means, Sxy, Sxx → slope & intercept  
  - predictions, MSE, R²  
  - **Goal:** understand the math step by step (no classes, no sklearn)

- **LinearRegression01.ipynb** — *from-scratch implementation*  
  - OLS via pseudoinverse: θ = (X_b)^+ y  
  - methods: `fit` (learn b,w), `predict` (make predictions)  
  - tested on 4 points, prints MSE and R²  
  - **Goal:** turn the math into reusable code

  


