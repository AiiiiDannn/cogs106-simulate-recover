# Simulate-and-Recover for EZ Diffusion Model

## **Introduction**

This project implements a **simulate-and-recover** process for the **EZ diffusion model**, a simplified version of the drift-diffusion model used to study **binary decision-making** and **response times**.

The goal of this experiment is to evaluate the **consistency** of the EZ diffusion model by:

1. **Simulating synthetic data** based on known parameters.
2. **Recovering those parameters** using inverse equations.
3. **Analyzing estimation accuracy** using **bias** and **mean squared error (MSE)**.

To assess the model’s reliability, **1000 iterations** were conducted for each of three different sample sizes: **\( N = 10, 40, 4000 \)**.

---

## **Methodology**

### **1. Simulating Data**

To generate realistic decision-making data, model parameters were randomly selected within predefined ranges:

- **Boundary separation** $\alpha$ : **[0.5, 2]** → The amount of evidence required for a decision.
- **Drift rate** $\nu$ : **[0.5, 2]** → The speed and direction of evidence accumulation.
- **Non-decision time** $\tau$ : **[0.1, 0.5]** → Time spent on sensory encoding and motor execution, not decision-making.

The **forward equations** of the EZ diffusion model were used to compute **predicted summary statistics**, where:

```math
y = e^{-\alpha \nu}
```

- **Predicted accuracy rate** **$ R^{\text{pred}} $** :

  ```math
  R^{\text{pred}} = \frac{1}{1 + y}
  ```

- **Predicted mean response time** **$ M^{\text{pred}} $** :

  ```math
  M^{\text{pred}} = \tau + \frac{\alpha}{2\nu} \cdot \frac{1 - y}{1 + y}
  ```

- **Predicted variance of response time** **$ V^{\text{pred}} $** :

  ```math
  V^{\text{pred}} = \frac{\alpha}{2 \nu^3} \cdot \frac{1 - 2 \alpha \nu y - y^2}{(1 + y)^2}
  ```

Using these **predicted values**, observed data was simulated as follows:

- **Accuracy** ($R^{\text{obs}}$) was drawn from a binomial distribution.
- **Mean RT** ($M^{\text{obs}}$) was sampled from a normal distribution.
- **Variance RT** ($V^{\text{obs}}$) followed a gamma distribution.

### **2. Recovering Parameters**

The **inverse equations** of the EZ diffusion model were used to estimate parameters from observed data. Given **\( R^{\text{obs}}, M^{\text{obs}}, V^{\text{obs}} \)**, the original parameters were estimated as follows:

- **Estimated drift rate** $\nu^{\text{est}}$ :

  ```math
  v^{\text{est}} = \text{sgn} \left( R^{\text{obs}} - \frac{1}{2} \right)
  \cdot \sqrt[4]{\frac{L \left( R^{\text{obs}^2} L - R^{\text{obs}} L + R^{\text{obs}} - \frac{1}{2} \right)}{V^{\text{obs}}}}
  ```

  where

  ```math
  L = \ln \left( \frac{R^{\text{obs}}}{1 - R^{\text{obs}}} \right)
  ```

- **Estimated boundary separation** $\alpha^{\text{est}}$ :

  ```math
  \alpha^{\text{est}} = \frac{L}{\nu^{\text{est}}}
  ```

- **Estimated non-decision time** $\tau^{\text{est}}$ :

  ```math
  \tau^{\text{est}} = M^{\text{obs}} - \frac{\alpha^{\text{est}}}{2 \nu^{\text{est}}} \cdot \frac{1 - e^{-\nu^{\text{est}} \alpha^{\text{est}}}}{1 + e^{-\nu^{\text{est}} \alpha^{\text{est}}}}
  ```

Each **simulate-and-recover process** was repeated **1000 times** for each sample size (\( N \)):

- **$N = 10$** → Small sample, high variability.
- **$N = 40$** → Moderate sample, better accuracy.
- **$N = 4000$** → Large sample, best accuracy.

For each condition, two key metrics were computed:

- **Bias** ($b$) :

  $$
  b = (\nu, \alpha, \tau) - (\nu^{\text{est}}, \alpha^{\text{est}}, \tau^{\text{est}})
  $$

- **Mean Squared Error (MSE)** :

  ```math
  MSE = \mathbb{E}[b^2]
  ```

---

## **Results**

After running **3000 total iterations**, the following results were obtained:

### **$N = 10$ (Small Sample)**

| Metric                 | Drift Rate ($\nu$) | Boundary Separation ($\alpha$) | Non-decision Time ($\tau$) |
| ---------------------- | ------------------ | ------------------------------ | -------------------------- |
| **Mean Bias**          | -0.3195            | -0.5674                        | 0.0629                     |
| **Mean Squared Error** | 1.1071             | 2.0596                         | 0.0637                     |

**Interpretation:**

- Bias is large, especially for drift rate and boundary separation.
- MSE is high, indicating high variability in parameter recovery.
- Small sample sizes lead to high estimation errors due to increased randomness.

### **$N = 40$ (Moderate Sample)**

| Metric                 | Drift Rate ($\nu$) | Boundary Separation ($\alpha$) | Non-decision Time ($\tau$) |
| ---------------------- | ------------------ | ------------------------------ | -------------------------- |
| **Mean Bias**          | -0.0389            | -0.1099                        | 0.0167                     |
| **Mean Squared Error** | 0.1447             | 0.3288                         | 0.0109                     |

**Interpretation:**

- Bias is much smaller compared to $ N = 10 $.
- MSE is significantly lower, meaning estimates are more reliable.
- More trials reduce randomness, improving parameter recovery accuracy.

### **$ N = 4000 $ (Large Sample)**

| Metric                 | Drift Rate ($\nu$) | Boundary Separation ($\alpha$) | Non-decision Time ($\tau$) |
| ---------------------- | ------------------ | ------------------------------ | -------------------------- |
| **Mean Bias**          | 0.0008             | 0.0004                         | -0.0002                    |
| **Mean Squared Error** | 0.0013             | 0.0001                         | 0.0000                     |

**Interpretation:**

- Bias is nearly zero, confirming that the model is consistent.
- MSE is extremely low, showing excellent parameter recovery.
- With large $ N $, estimates converge to true values, proving the reliability of the EZ model when sample sizes are sufficient.

---

## **How to Run the Experiment**

### Python Environment & Dependencies

Before running the experiment, ensure you have Python 3.7+ installed. The following Python packages are required:

- `Numpy`
- `math`

No external dependencies like `pandas` or `scipy` are used to keep the implementation lightweight.

### Setting Up the Environment

To set up your environment, follow these steps:

1. **Check your Python version:**

```bash
python --version
```

Ensure the output is `Python 3.7` or later.

2. **Install required packages:**

```bash
pip install numpy
```

### Running the Simulation & Testing

To execute the full simulate-and-recover process (3000 iterations for different sample sizes), run:

```bash
./src/main.sh
```

This will output bias and mean squared error (MSE) results for `N = 10, 40, 4000`.

To verify the correctness of the implementation, execute the test suite:

```bash
./test/test.sh
```

This will run all unit tests and ensure the numerical computations are correct.

---

## **Final Thoughts**

This project demonstrates the **importance of sample size** in diffusion modeling. While **small samples** introduce **high error**, larger datasets enable **highly accurate parameter recovery**.

To ensure accurate diffusion model estimates, the following recommendations are made:

- Using at least **\( N = 40 \)** for reasonable accuracy.
- Using **\( N = 4000 \)** for near-perfect parameter recovery.

This highlights a fundamental principle in cognitive modeling:  
**Larger sample sizes lead to more precise and stable estimates.**
