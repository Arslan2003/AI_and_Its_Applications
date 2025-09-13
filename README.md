# Money Laundering Detection with Feedforward Neural Networks
A lightweight AI solution for detecting illicit financial transactions.

<br>

## 📖 Overview

This project implements a Feedforward Neural Network (FNN) to detect potential money laundering transactions using the HI-Small Trans.csv dataset from IBM. While Graph Neural Networks (GNNs) are commonly used for financial graph data, this project demonstrates that a simpler FNN can achieve competitive results with lower computational requirements.

Key features of this project:

- **Binary classification** of transactions as legitimate or illicit.
- **Feature preprocessing**, including one-hot encoding for categorical variables and standardisation.
- **Robust training** with class balancing and early stopping to handle highly imbalanced datasets.
- **Evaluation metrics**, including F1-score, ROC AUC, and confusion matrix for comprehensive model assessment.
- **Visualisations** of model performance: confusion matrix and ROC curve.

<br>

## 🤖 Model Architecture

The FNN is built using **TensorFlow/Keras** and consists of:

- Input layer matching the number of features after preprocessing.
- Two hidden layers with 64 and 32 neurons respectively, using **ReLU activation** and **Dropout** for regularisation.
- Output layer with **sigmoid activation** for binary classification.

Training is performed using **binary cross-entropy loss** and **Adam optimiser**.

<br>

## 📊 Results

**Confusion matrix** of the final trained model:

|                   | Predicted Legitimate | Predicted Illicit |
|-------------------|-------------------|-----------------|
| **Actual Legitimate** | 253641               | 14               |
| **Actual Illicit**    | 177               | 86               |

<br>

**Metrics Summary:**

| Metric       | Value      |
|--------------|-----------|
| Accuracy     | 99.92%    |
| F1-Score     | 0.52      |
| ROC AUC      | 0.87      |

> Note: Due to extreme class imbalance, the F1-score for illicit transactions remains challenging, but the model demonstrates strong practical performance with fast training times and competes with industry-standard models such as Graph Neural Networks.

<br>

## ⚙️ Installation & Usage

1. **Clone the repository:**
```bash
git clone https://github.com/<your-username>/aml-fnn.git
cd aml-fnn
```
2. Set up a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Download the dataset:
The project uses [HI-Small Trans.csv](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml?select=HI-Small_Trans.csv) released by IBM.
5. Run the training and evaluation script:
```
python aml_fnn.py
```

The outputs include a trained FNN model, a confusion matrix, a ROC curve plot, and key evaluation metrics.

6. Experiment!

<br>

## 🤝 Contributing
Contributions, suggestions, and bug reports are welcome! Feel free to open an issue or submit a pull request.

<br>

## 🧑‍💻 Author
[Arslonbek Ishanov](https://github.com/Arslan2003) – First-Class Graduate Data Scientist & AI/ML Enthusiast.

<br>

## ⚖️ License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

<br>

## 🔗 Learn More
Interested in FNN vs GNN for financial fraud detection? Check out my [project report](AML_with_FNN-Report.pdf) for in-depth methodology and analysis.  
<br>
Explore [IBM's Multi-GNN models](https://github.com/IBM/Multi-GNN) for AML.



