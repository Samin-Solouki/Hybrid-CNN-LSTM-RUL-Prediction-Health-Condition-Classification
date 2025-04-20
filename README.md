# Hybrid-CNN-LSTM-RUL-Prediction-Health-Condition-Classification
## **Hybrid CNN-LSTM Deep Learning Architecture for RUL Prediction and Health Condition Classification of Turbofan Engines**

---

### **Project Highlights**
- Developed a dual-purpose deep learning architecture combining CNN and LSTM layers.
- Addressed both **Remaining Useful Life (RUL) prediction** (regression task) and **engine health status classification** (binary classification).
- Leveraged the **C-MAPSS dataset** from NASA's Prognostics Center of Excellence.
- Conducted detailed **data preprocessing**, including sensor selection, normalization, time-window slicing, and data labeling.
- Trained and evaluated models with and without **Early Stopping**, using **Dropout layers** to prevent overfitting.
- Utilized **MSE (Mean Squared Error)** as the loss function and tested both **Adam** and **RMSProp** optimizers.
- Achieved reliable predictions for both degradation forecasting and real-time condition monitoring.

---

### **Project Level**:
Master’s level — Advanced deep learning and data mining project in the field of **Predictive Maintenance (PdM)** and **Cyber-Physical Systems (CPS)**.

---

### **Project Goal**
To design and implement a **multi-task hybrid deep learning model** that utilizes the strengths of CNNs in spatial feature extraction and LSTMs in modeling long-term temporal dependencies, aimed at:
- **Predicting the Remaining Useful Life (RUL)** of aircraft turbofan engines.
- **Classifying health states** of engines into "healthy" and "unhealthy" categories based on time-series sensor data.

---

### **Project Description**
The project focuses on predictive maintenance, a critical element in Industry 4.0, by addressing the challenge of predicting the degradation and failure time of mechanical systems. Specifically, it targets **cyber-physical systems**, where sensors continuously monitor system health.

*You may find the report of the project under the name "Classification of Alzheimer's Disease (AD) and Mild Cognitive Impairment (MCI) using MRI Images_ Solouki.PDF"*

The **C-MAPSS dataset**, collected from a simulation of turbofan engine degradation, includes multivariate time-series sensor readings from multiple engines under different operating conditions and fault modes. The dataset contains:
- Full-run-to-failure engine records (training set)
- Partial operating sequences without failure (test set)
- Ground truth for test set RUL values

**Key preprocessing steps included:**
- **Sensor Selection:** Removed low-variance or uninformative sensors to reduce dimensionality and improve model performance.
- **Min-Max Normalization:** Scaled sensor values to the [0, 1] range.
- **Data Labeling:**
   - **For regression**: Labeled each window with the RUL value of the last timestamp.
   - **For classification**: Labeled as class 0 (healthy) if RUL > window size, else class 1 (unhealthy).
- **Time Windowing:** Created sequential samples using a sliding window approach (length = 30, stride = 1) to capture local temporal dependencies.

**Model Architecture:**
- **CNN Block:** 1D convolution layers extract local and spatial features from sensor sequences.
- **LSTM Block:** Captures long-term temporal dependencies from the sequential feature maps.
- **Fully Connected Layers:** Combine learned features for final output.
   - For regression: Single output node representing predicted RUL.
   - For classification: Sigmoid-activated node for binary classification (healthy/unhealthy).

**Training Strategy:**
- Used both **Adam** and **RMSProp** optimizers.
- **Early Stopping** was applied based on validation loss improvement, with a patience threshold.
- **Dropout layers** were used between dense layers to avoid overfitting.
- Models were trained twice: once with Early Stopping, once without, to evaluate the effect on performance.

---

### **Language and Libraries Used**
- **Language:** Python  
- **Libraries & Tools:**  
   - `NumPy`, `Pandas` – Data manipulation  
   - `Matplotlib`, `Seaborn` – Visualization  
   - `Scikit-learn` – Preprocessing, evaluation  
   - `TensorFlow`, `Keras` – Deep learning model development  
   - `Google Colab` – GPU-accelerated training environment

---

### **Methods and Models**
- **Deep Learning Architecture:**  
   - 1D CNN for spatial feature extraction  
   - LSTM for temporal sequence modeling  
   - Fully Connected Layers for output (regression/classification)
- **Data Preprocessing:**  
   - Sensor selection  
   - Normalization using `MinMaxScaler`  
   - Sliding time window with overlapping sequences  
   - Target labeling for RUL and binary class
- **Optimization & Evaluation:**  
   - Loss: `Mean Squared Error (MSE)`  
   - Optimizers: `Adam`, `RMSProp`  
   - Regularization: `Dropout`, `EarlyStopping`  
   - Metrics: MSE, classification accuracy

---

### **References**
- Maxim Shcherbakov and Cuong Sai. 2022. A Hybrid Deep Learning Framework for Intelligent Predictive Maintenance of Cyber-physical Systems. ACM Trans. Cyber-Phys. Syst. 6, 2, Article 17 (May 2022), 22 pages. https://doi.org/10.1145/3486252
