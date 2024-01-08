# D7041E Mini-Project: Regression Model for Housing Price Prediction

## Group Information

- **Group ID:** MINI-PROJECT 14
- **Group Members:**
  - Ahmad Allahham ([ahmall-0@student.ltu.se](mailto:ahmall-0@student.ltu.se)) | 940120-0556
  - Arian Asghari ([ariasg-0@student.ltu.se](mailto:ariasg-0@student.ltu.se)) | 010721-7051
  - Hannes Furhoff ([hanfur-0@student.ltu.se](mailto:hanfur-0@student.ltu.se)) | 010929-4710

---

## Project Overview

This repository contains the implementation and analysis of a Multi-Layer Perceptron (MLP) regression model for predicting housing prices. The project is part of the D7041E course at Luleå University of Technology.

---

## Objectives

The main objectives of the project include:

- Implementing and understanding a publicly available MLP regression model.
- Selecting a synthetic housing dataset for training and evaluation.
- Executing a tutorial for guidance in the implementation process.
- Testing the performance of the model under various configurations.
- Documenting the performance results and analysis.
- Applying data pre-processing techniques for improved model training.
- Systematically choosing hyper-parameters for optimal model performance.
- Utilizing cross-validation techniques during training.
- Recording performance statistics with different random seeds.

---

## Additional Information

- **GitHub Repository:** [https://github.com/afy/d7041e_miniproject](https://github.com/afy/d7041e_miniproject)
- **Dataset Link:** [Housing Price Prediction Data](https://www.kaggle.com/datasets/muhammadbinimran/housing-price-prediction-data)
- **Tutorial Source:** [Building a Regression Model in PyTorch](https://machinelearningmastery.com/building-a-regression-model-in-pytorch/)
- **YouTube Demo Link:** [https://www.youtube.com/watch?v=z8-_-X0Q0zs]

---

## Code Structure

### Dependencies

- NumPy
- PyTorch (torch.nn, torch.optim)
- Collections (OrderedDict)
- Itertools
- Scikit-learn (OneHotEncoder)
- IPython

### Preprocessing

1. **Read Data:** Load data from "housing_data.csv".
2. **Encode Categorical Column:** Utilize one-hot encoding for the "neighborhood" column.
3. **Type Conversion:** Convert all values to float.
4. **Normalization** Normalize data

### Model Training

- Define an MLP regression model using PyTorch.
- Evaluation function: `evaluateModel(settings: ModelSettings, data_train: np.ndarray, data_test: np.ndarray) -> ModelMetrics`

### Cross-validation

- Function: `k_fold_validation(k: int, data: np.ndarray, prices: np.ndarray, settings: ModelSettings) -> ModelMetrics`

### Running Experiments

- Hyperparameter configurations: k_folds, number_of_hidden_layers, learning_rates, batch_sizes, epochs
- Cycling through permutations for training and testing.

---

## How to Run

1. Ensure all dependencies are installed (`numpy`, `torch`, `scikit-learn`).
2. Clone the project repository from [https://github.com/afy/d7041e_miniproject](https://github.com/afy/d7041e_miniproject).
3. Execute the provided Jupyter Notebook or script.
4. Adjust configurations as needed for specific experiments.

---
## Performance Table

| Rank | Score | MSE     | RMSE    | MAE     | MAPE    | Training Time (s) | Settings                                              |
|------|-------|---------|---------|---------|---------|--------------------|--------------------------------------------------------|
| 1    | 84.2% | 0.00907 | 0.09522 | 0.07614 | 0.15833 | 22.29              | Hidden layers=[10, 10], epochs=60, batch_size=1000, ... |
| 2    | 84.1% | 0.00906 | 0.09519 | 0.07609 | 0.15904 | 28.68              | Hidden layers=[10, 10], epochs=20, batch_size=1000, ... |
| 3    | 84.1% | 0.00898 | 0.09476 | 0.07575 | 0.15928 | 36.15              | Hidden layers=[10, 10], epochs=10, batch_size=500, ...  |
| 4    | 84.0% | 0.00899 | 0.09479 | 0.07577 | 0.15955 | 48.81              | Hidden layers=[10, 10], epochs=60, batch_size=500, ...  |
| 5    | 84.0% | 0.00896 | 0.09466 | 0.07569 | 0.15976 | 14.18              | Hidden layers=[10, 10], epochs=3, batch_size=500, ...   |
| 6    | 84.0% | 0.00896 | 0.09467 | 0.07565 | 0.15994 | 19.03              | Hidden layers=[10, 10], epochs=10, batch_size=1000, ... |
| 7    | 84.0% | 0.00902 | 0.09496 | 0.07594 | 0.16003 | 8.22               | Hidden layers=[50, 50], epochs=20, batch_size=1000, ... |
| 8    | 84.0% | 0.00899 | 0.09481 | 0.07584 | 0.16028 | 65.97              | Hidden layers=[10, 10], epochs=20, batch_size=500, ...  |
| 9    | 84.0% | 0.00902 | 0.09499 | 0.07595 | 0.16041 | 18.89              | Hidden layers=[50, 50], epochs=20, batch_size=500, ...  |
| 10   | 84.0% | 0.00894 | 0.09454 | 0.07556 | 0.16045 | 7.54               | Hidden layers=[10, 10], epochs=3, batch_size=1000, ... |

---
### Trends in Performance Metrics

- As the number of training epochs increases, the model tends to achieve better performance, up to a certain point where further epochs may lead to overfitting.
- Increasing the batch size generally results in faster training times but may have a trade-off in terms of model accuracy.
- The choice of learning rate impacts the convergence speed, affecting both training time and final model performance.

### Impact of Hyperparameter Choices

- The number of hidden layers and neurons per layer significantly influences the model's ability to capture complex patterns in the data.
- Cross-validation helps in evaluating the model's generalization across different subsets of the dataset.

---

## Conclusion

In summary, the best-performing configuration based on the highest score (84.2%) includes hidden layers [10, 10], 60 epochs, batch size of 1000, and a learning rate of 0.05. This configuration balances accuracy and training time effectively.

## Consideration of Trade-offs

Considering the trade-offs:

- Increased epochs may improve accuracy but also extend training time.
- Smaller batch sizes may lead to better convergence but may require longer training times.

It's crucial to strike a balance based on project requirements.

## Contact Information

For any inquiries or assistance, feel free to contact the project members:

- Ahmad Allahham: [ahmall-0@student.ltu.se](mailto:ahmall-0@student.ltu.se)
- Arian Asghari: [ariasg-0@student.ltu.se](mailto:ariasg-0@student.ltu.se)
- Hannes Furhoff: [hanfur-0@student.ltu.se](mailto:hanfur-0@student.ltu.se)

---

*Note: This README will be continuously updated as the project progresses.*
