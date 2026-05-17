# 🧠 Deep MNIST Classifier: 3-Stage CNN with Hyperband Tuning

An optimized Deep Learning pipeline to classify handwritten digits using TensorFlow, Keras Tuner, and a dynamic 3-stage Convolutional Neural Network (CNN) architecture.

### 📊 Model Performance Summary

| Model Architecture | Training Accuracy | Validation Accuracy | Key Strategy Used |
| :--- | :--- | :--- | :--- |
| **Baseline CNN** | ~95.0% | ~94.5% | Standard Convolution + Pooling |
| **Champion CNN** | ~99.2% | ~99.0% | Added Dropout & Batch Normalization |

## 📂 Project Structure

```text
├── notebooks/          # Exploration \\\& Hyperparameter Tuning (01\\\_tuning.ipynb)
├── src/                # Production-ready training scripts (train\\\_champion.py)
├── models/             # Saved model artifacts (mnist\\\_champion\\\_model.h5)
├── results/            # Training logs (CSV) and performance visualizations (PNG)
└── README.md           # Project documentation


## 🚀 The Architecture


* The "Champion" model features a triple-stacked Convolutional base followed by a dual-layer Dense head.
* Convolutional Layers: 3 Stages with Batch Normalization for stable gradients and faster convergence.
* Regularization: Integrated Dropout (0.3 - 0.4) and MaxPooling to prevent overfitting.
* Optimizer: Adam with a tuned learning rate discovered during the optimization trials.



## 📊 Results \\\& Performance


* Resilience: Implemented CSVLogger and ModelCheckpoint to ensure training progress is saved even if the environment restarts.
* Accuracy: The model achieves high validation accuracy (>99%), with the regularization strategy keeping the generalization gap narrow.
* Observation: Validation accuracy often tracks slightly higher than training accuracy, confirming that the Dropout and BatchNormalization layers are performing effectively.


## 🚀 How to Run the Project
1. Clone the repository and navigate into it.
2. Install dependencies (`pip install tensorflow keras matplotlib numpy`).
3. Run the notebook: `jupyter notebook "Deep MNIST Champion.ipynb"`.


##📈 Performance Visuals
Once the training is complete, the following artifacts are generated in the /results folder:

learning_curves.png: Visualizes the loss and accuracy over epochs.

confusion_matrix.png: Shows the digit-by-digit classification breakdown.


