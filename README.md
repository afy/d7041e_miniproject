# D7041E Mini-Project: Regression Model for Housing Price Prediction

## Group Information

- **Group ID:** MINI-PROJECT 14
- **Group Members:**
  - Ahmad Allahham ([ahmall-0@student.ltu.se](mailto:ahmall-0@student.ltu.se)) | 940120-0556
  - Arian Asghari ([ariasg-0@student.ltu.se](mailto:ariasg-0@student.ltu.se)) | 010721-7051
  - Hannes Furhoff ([hanfur-0@student.ltu.se](mailto:hanfur-0@student.ltu.se)) | 010929-4710

---

## Project Overview

This repository contains the implementation and analysis of a Multi-Layer Perceptron (MLP) regression model for predicting housing prices. The project is part of the D7041E course at LuleÃ¥ University of Technology.

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
- **YouTube Demo Link:** [Not Available Yet]

---

## Code Structure

### Dependencies

- NumPy
- PyTorch (torch.nn, torch.optim)
- Collections (OrderedDict)
- Itertools
- Scikit-learn (OneHotEncoder)

### Preprocessing

1. **Read Data:** Load data from "housing_data.csv".
2. **Encode Categorical Column:** Utilize one-hot encoding for the "neighborhood" column.
3. **Type Conversion:** Convert all values to float.

### Model Training

- Define an MLP regression model using PyTorch.
- Evaluation function: `evaluateModel(settings: ModelSettings, data_train: np.ndarray, data_test: np.ndarray) -> ModelMetrics`

### Cross-validation

- Function: `k_fold_validation(k: int, data: np.ndarray, prices: np.ndarray, settings: ModelSettings) -> ModelMetrics`

### Running Experiments

- Hyperparameter configurations: k_folds, layer_length, in_features_sizes, out_features_sizes.
- Cycling through permutations for training and testing.

---

## How to Run

1. Ensure all dependencies are installed (`numpy`, `torch`, `scikit-learn`).
2. Clone the project repository from [https://github.com/afy/d7041e_miniproject](https://github.com/afy/d7041e_miniproject).
3. Execute the provided Jupyter Notebook or script.
4. Adjust configurations as needed for specific experiments.

---

## Contact Information

For any inquiries or assistance, feel free to contact the project members:

- Ahmad Allahham: [ahmall-0@student.ltu.se](mailto:ahmall-0@student.ltu.se)
- Arian Asghari: [ariasg-0@student.ltu.se](mailto:ariasg-0@student.ltu.se)
- Hannes Furhoff: [hanfur-0@student.ltu.se](mailto:hanfur-0@student.ltu.se)

---

*Note: This README will be continuously updated as the project progresses.*
