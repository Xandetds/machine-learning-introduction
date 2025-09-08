
# Notes — Applied ML 

## Linear Regression

### How does it work
We predict with: $\hat y = b + \mathbf{w}^\top \mathbf{x}$.

The mean error over all points is the MSE:  
$\mathrm{MSE}(b,\mathbf{w}) = \dfrac{1}{m}\sum_i \left(y_i - \hat y_i\right)^2.$

$R^2$ measures variance explained (not accuracy):  
$R^2 = 1 - \dfrac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}.$

---

### Notebook index (what each one does)

- **LinearRegression>LinearRegressionMath.ipynb** — *math only*.  
  Hand-calculation on a tiny dataset (4 points): means, $S_{xy}$, $S_{xx}$, slope $w = S_{xy}/S_{xx}$, intercept $b = \bar y - w\,\bar x$, predictions, MSE, $R^2$. No classes, no sklearn.

- **LinearRegression>LinearRegression01.ipynb** — *implementation*.  
  From-scratch OLS using the normal equation with pseudoinverse:  
  $\theta=\begin{bmatrix}b\\\mathbf{w}\end{bmatrix}=X_b^{+}\,y$, with $X_b=[\mathbf{1}\ \ X]$.  
  Methods: `fit(X,y)` → learn `intercept_`, `coef_`; `predict(X)` → $\hat y=b+X\mathbf{w}$.  
  Metrics printed: **MSE** and **$R^2$**.  
  


