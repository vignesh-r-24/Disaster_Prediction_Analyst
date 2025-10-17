#  🌋 Disaster Impact Prediction using Machine Learning

This project uses **Machine Learning (Deep Neural Networks)** to predict **fatalities** caused by disasters based on their **magnitude, economic loss, type, location, and date**.
It also provides rich **visual analytics** using `Matplotlib`, `Seaborn`, and `Plotly` to understand patterns and correlations in historical disaster data.

---

## 📊 Overview

The goal of this project is to analyze global disaster data and build a predictive model that estimates the **number of fatalities** caused by a disaster.
It supports multiple disaster types (earthquakes, floods, hurricanes, etc.) and integrates **temporal features** (year, month, day) to improve prediction accuracy.

---

## 🧠 Key Features

* **Data Preprocessing:**

  * Date parsing & feature extraction (`Year`, `Month`, `Day`, `DayOfYear`)
  * Encoding of categorical variables (`Disaster_Type`, `Location`)
  * Missing value analysis

* **Exploratory Data Analysis (EDA):**

  * Distribution plots (magnitude, fatalities, losses)
  * Correlation heatmap
  * Boxplots, scatter plots, and count visualizations
  * Faceted analysis by disaster type and location

* **Machine Learning Model:**

  * Deep Neural Network built using **TensorFlow / Keras**
  * Features scaled via `StandardScaler`
  * Regression model to predict **fatalities**

* **Model Evaluation:**

  * Metrics: MAE, MSE, RMSE, R² Score
  * Training vs. validation curves
  * Actual vs. Predicted visualization
  * Feature importance estimation (based on neural network weights)

* **Interactive Prediction Function:**

  * Predict fatalities for a **new disaster scenario**
  * Encodes disaster type, location, and computes day of year automatically

---

## 🧰 Technologies Used

| Category        | Tools                                      |
| --------------- | ------------------------------------------ |
| **Language**    | Python 3.x                                 |
| **Libraries**   | Pandas, NumPy, Matplotlib, Seaborn, Plotly |
| **ML / DL**     | Scikit-learn, TensorFlow (Keras)           |
| **Environment** | Google Colab / Jupyter Notebook            |

---

## 📁 Dataset

The project uses a dataset named **`disasters.csv`**, which includes columns such as:

| Column             | Description                                       |
| ------------------ | ------------------------------------------------- |
| `Date`             | Date of the disaster                              |
| `Disaster_Type`    | Type of disaster (e.g., Earthquake, Flood, Storm) |
| `Location`         | Geographic location                               |
| `Magnitude`        | Disaster intensity or magnitude                   |
| `Economic_Loss($)` | Estimated economic loss in USD                    |
| `Fatalities`       | Reported fatalities due to the disaster           |

✅ The notebook automatically converts and processes this data into a cleaned format:
`data.csv`

---

## ⚙️ Installation & Setup

1. **Clone this repository:**

   ```bash
   git clone https://github.com/yourusername/disaster-impact-prediction.git
   cd disaster-impact-prediction
   ```

2. **Install required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook:**

   ```bash
   jupyter notebook disasters.ipynb
   ```

4. **Ensure the dataset (`disasters.csv`) is in the same directory as the notebook.**

---

## 🧩 Model Architecture

```python
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Regression output
])
```

* Optimizer: `Adam`
* Loss: `Mean Squared Error`
* Metrics: `Mean Absolute Error (MAE)`
* Epochs: 100
* Batch size: 32

---

## 📈 Model Evaluation Metrics

After training, the model reports:

| Metric       | Description             |
| ------------ | ----------------------- |
| **MAE**      | Mean Absolute Error     |
| **MSE**      | Mean Squared Error      |
| **RMSE**     | Root Mean Squared Error |
| **R² Score** | Goodness of fit         |

Example output:

```
Mean Absolute Error: 25.74
Mean Squared Error: 1280.51
Root Mean Squared Error: 35.78
R² Score: 0.87
```

---

## 🔍 Visualization Highlights

### 🔸 Distribution of Disaster Magnitudes

Shows how disaster magnitudes vary across types using `Seaborn` histplots.

### 🔸 Correlation Heatmap

Displays feature correlations including `Magnitude`, `Economic_Loss`, and `Fatalities`.

### 🔸 Boxplots and Scatterplots

Visual comparisons of fatalities by disaster type and location.

### 🔸 Training History

Loss and MAE trends across 100 epochs.

---

## 🤖 Example Prediction

```python
example_prediction = predict_disaster_impact(
    magnitude=7.5,
    economic_loss=1000000000,
    disaster_type='Earthquake',
    location='Japan',
    year=2024,
    month=3,
    day=15
)
print(f"Predicted Fatalities: {example_prediction:.0f}")
```

**Output:**

```
Predicted Fatalities: 482
```

---

## 🗂️ Output Files

| File                | Description                       |
| ------------------- | --------------------------------- |
| `disasters.ipynb`   | Main Jupyter Notebook             |
| `disasters.csv`     | Input dataset                     |
| `data.csv`          | Cleaned and processed dataset     |
| `model_summary.txt` | Optional saved model architecture |
| `README.md`         | Project documentation             |

---

## 🧾 Requirements

Create a `requirements.txt` file with:

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
tensorflow
```

Install with:

```bash
pip install -r requirements.txt
```

---



## 📜 License

This project is licensed under the **MIT License** — feel free to use and modify it for educational and research purposes.

---

## 💬 Acknowledgments

* TensorFlow and Keras teams
* Seaborn and Plotly for visualization
* Kaggle & NOAA datasets for disaster records

