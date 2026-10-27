# Fruit Classification Using Deep Learning

A deep learning project that classifies fruit images into **10 different fruit categories** using Convolutional Neural Networks (CNNs) and computer vision techniques implemented with PyTorch.

##  Project Overview

The objective of this project is to develop and evaluate deep learning models capable of identifying fruits from images. The project explores image classification using different model architectures and evaluates their performance on a fruit image dataset.

## Fruit Categories

The classifier covers the following 10 fruit classes:

1. Apple
2. Banana
3. Cherry
4. Coconut
5. Grapes
6. Mango
7. Orange
8. Pineapple
9. Strawberry
10. Watermelon

## Models

The `Models/` directory contains the model implementations used in the project.

- `Model_1.py`
- `Model_2.py`
- `Model_3.py`

These implementations can be used to train, evaluate, or run inference with the corresponding architectures. Refer to the source code for the specific architecture and configuration of each model.

## Technologies Used

- **Python** — Core programming language
- **PyTorch** — Deep learning model development and training
- **Torchvision** — Image transformations and computer vision utilities
- **NumPy** — Numerical operations
- **Pandas** — Data handling and analysis
- **Matplotlib** — Data visualization
- **Streamlit** — Web application interface

## Project Structure

```text
FruitClassifier/
├── App/
│   └── app.py
├── Models/
│   ├── Model_1.py
│   ├── Model_2.py
│   └── Model_3.py
├── Documentation
└── README.md
```

*The structure above reflects the main directories; additional files may be present in the repository.*

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/tainwala-ayush/fruit-classification-cnn.git
cd fruit-classification-cnn
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

Install the libraries required by the project:

```bash
pip install torch torchvision numpy pandas matplotlib streamlit pillow scikit-learn torchmetrics
```

Depending on the selected model and application code, additional dependencies may be required.

## Running the Application

Launch the Streamlit application from the project root:

```bash
streamlit run App/app.py
```

Ensure that the trained model checkpoint files and any required configuration or class-label files are available at the paths expected by the application.

## Model Evaluation

The models can be evaluated using classification metrics such as:

- **Accuracy** — Overall proportion of correctly classified images.
- **Precision** — Proportion of positive predictions that are correct.
- **Recall** — Proportion of actual class instances correctly identified.
- **F1-score** — Harmonic mean of precision and recall.
- **Confusion Matrix** — Breakdown of correct predictions and misclassifications across the fruit classes.

These metrics help compare model performance and identify categories that may be difficult to distinguish.

## Future Improvements

Potential improvements include:

- Comparing the performance of different CNN architectures.
- Improving generalization through data augmentation.
- Tuning hyperparameters and training configurations.
- Analysing misclassified images using confusion matrices.
- Improving inference speed and application usability.
- Exploring explainability techniques such as Grad-CAM.

## Author

**Ayush Tainwala**

GitHub: [@tainwala-ayush](https://github.com/tainwala-ayush)

