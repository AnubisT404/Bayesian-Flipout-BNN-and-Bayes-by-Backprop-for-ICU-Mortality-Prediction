# Bayesian-Flipout-BNN-and-Bayes-by-Backprop-for-ICU-Mortality-Prediction

This project applies **Bayesian Neural Networks (BNNs)** and **Bayes by Backprop** for **ICU mortality prediction** using the **MIMIC-III** dataset. It focuses on improving model trustworthiness by capturing **predictive uncertainty**, helping to identify **out-of-domain (OOD) patients** and reducing high-risk predictions.   

---
### **BBBNN (Bayes by Backprop BNN)**
**BBBNN** is based on **Bayes by Backprop**, introduced by **Blundell et al. (2015)**. It approximates the posterior distribution of neural network weights using **variational inference**.  
#### **How It Works:**
1. Instead of deterministic weights, BBBNN assigns a **probability distribution** to each weight (mean and variance).  
2. During training, it **samples weights** from these distributions instead of using fixed values.  
3. It optimizes the **Evidence Lower Bound (ELBO)**, balancing two losses:
   - **Negative Log Likelihood (NLL)**: Measures how well predictions fit the data.  
   - **KL Divergence**: Ensures the weight distributions remain close to the prior.  
4. The final prediction is made by **averaging multiple stochastic forward passes** through the network.

#### **Limitations of BBBBNN:**
- **High variance in weight updates** because each weight is sampled **independently**.
- **Computationally inefficient** for large models due to multiple weight sampling steps.
- **Can lead to poor uncertainty estimates** if not carefully tuned.

---
### **Flipout BNN**
**Flipout BNN** is an improved Bayesian method introduced by **Wen et al. (2018)** to reduce variance in weight sampling while maintaining stochasticity.  
#### **Key Improvements:**
1. **Reparameterization Trick with Perturbations:**
   - Flipout does not independently sample weights for each pass.
   - Instead, it introduces **anti-correlated noise** to simulate different weight perturbations efficiently.
2. **Variance Reduction in Bayesian Inference:**
   - Reduces the **sampling noise** of stochastic gradient updates.
   - Enables **faster convergence** with **better uncertainty estimates**.
3. **Efficient Computation:**
   - Uses a **single set of shared weights** instead of sampling independently for each forward pass.
   - Improves memory efficiency while still enabling Bayesian inference.

---
## Dataset  
The **MIMIC-III** dataset includes **vital signs, lab results, and patient demographics** from **ICU patients**. The preprocessing steps involve:  
- Cleaning and handling missing values  
- Removing newborn ICU stays  
- Aggregating patient history and lab values  
- Splitting the data into training (90%) and testing (10%)
  
---
Run the training script:  
```bash
python train.py --batch-size 128 --test-batch-size 512 --lr 1e-3 --epochs 256
```

---
## Evaluation  
The model is evaluated using:  
- **ROC-AUC Score**  
- **Predictive Uncertainty Analysis**  
- **Out-of-Domain (OOD) Detection**  

---
## Installation  
Install dependencies:  
```bash
pip install -r requirements.txt
```  
Train and evaluate:  
```bash
python train.py --batch-size 128 --test-batch-size 512 --lr 1e-3 --epochs 256
python exp/evaluate.py
```

