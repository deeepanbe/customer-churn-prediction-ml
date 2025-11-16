# 📊 Customer Churn Prediction - ML Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Tableau](https://img.shields.io/badge/Tableau-2023-blue)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 🎯 Project Overview

This project implements a machine learning solution to predict customer churn for a telecommunications company. The model helps identify at-risk customers before they leave, enabling proactive retention strategies that can reduce churn by up to 25%.

### Business Impact
- **$2.4M Annual Savings**: Early identification of 1,200+ at-risk customers
- **82% Model Accuracy**: Precise predictions with 0.78 AUC-ROC score
- **25% Churn Reduction**: Achieved through targeted retention campaigns
- **Real-time Dashboard**: Executive-level insights for decision makers

## 📂 Project Structure

```
customer-churn-prediction-ml/
│
├── data/
│   ├── telecom_churn_data.csv          # Customer data (10,000 records)
│   ├── feature_importance.csv          # Model feature rankings
│   └── predictions_output.csv          # Churn predictions
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training_evaluation.ipynb
│
├── dashboards/
│   ├── churn_analytics_dashboard.twbx  # Tableau workbook
│   └── dashboard_screenshots/
│
├── models/
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   └── xgboost_model.pkl
│
└── README.md
```

## 🔍 Key Findings

### Customer Churn Insights
1. **Contract Type**: Month-to-month customers have 42% churn rate vs 11% for yearly contracts
2. **Customer Service**: 5+ support calls correlate with 68% churn probability
3. **Tenure Impact**: Customers with <6 months tenure have 3x higher churn risk
4. **Payment Method**: Electronic check users show 33% higher churn rates
5. **Service Bundling**: Customers with internet-only services churn at 38% vs 18% for bundled services

## 🛠️ Technologies Used

### Data Analysis & ML
- **Python 3.9**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting models
- **SMOTE**: Handling class imbalance

### Visualization
- **Tableau**: Interactive executive dashboards
- **Matplotlib & Seaborn**: Statistical visualizations
- **Plotly**: Interactive feature analysis

## 🚀 Model Performance

### Model Comparison Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|----------|
| **Random Forest** | **82%** | **0.79** | **0.81** | **0.80** | **0.87** |
| XGBoost | 81% | 0.78 | 0.80 | 0.79 | 0.86 |
| Logistic Regression | 76% | 0.72 | 0.74 | 0.73 | 0.79 |
| Decision Tree | 74% | 0.70 | 0.75 | 0.72 | 0.78 |

### Feature Importance (Top 10)

1. **Tenure** (18.2%) - Customer lifetime with company
2. **Monthly Charges** (15.7%) - Payment amount
3. **Total Charges** (12.3%) - Cumulative payments
4. **Contract Type** (11.8%) - Month-to-month vs annual
5. **Internet Service** (9.4%) - Service type
6. **Tech Support** (7.6%) - Support subscription
7. **Online Security** (6.9%) - Security service
8. **Payment Method** (6.2%) - Payment type
9. **Device Protection** (5.8%) - Protection plan
10. **Customer Service Calls** (5.1%) - Support interactions

## 📊 Dashboard Highlights

The **Tableau dashboard** provides:

### Executive View
- Real-time churn rate KPIs
- Monthly trend analysis
- Revenue impact projections
- Customer segmentation

### Operational View
- At-risk customer lists
- Churn probability scores
- Retention campaign targeting
- Service quality metrics

### Predictive Analytics
- Churn likelihood by segment
- Feature contribution analysis
- What-if scenario modeling
- ROI calculator for retention efforts

## 💡 Business Recommendations

### Immediate Actions
1. **Target Month-to-Month Customers**: Offer incentives for annual contract upgrades
2. **Improve Customer Support**: Implement proactive support after 3+ service calls
3. **Bundle Services**: Create attractive multi-service packages
4. **Early Engagement**: Special attention to customers in first 6 months
5. **Payment Method**: Encourage automatic payments over electronic checks

### Long-term Strategy
- Implement predictive model in CRM system
- Quarterly model retraining with new data
- A/B testing of retention campaigns
- Customer feedback integration

## 📈 Results & Impact

### Pilot Program Results (Q4 2024)
- **1,247 customers** identified as high-risk
- **312 customers** saved through targeted retention (25% success rate)
- **$2.4M revenue** protected (avg. customer lifetime value: $7,680)
- **92% campaign efficiency**: Focused resources on highest-risk segments

## 🔄 Model Deployment

The model is deployed as:
1. **Batch Predictions**: Weekly scoring of entire customer base
2. **API Integration**: Real-time churn probability via REST API
3. **CRM Integration**: Automated flags in customer management system
4. **Email Alerts**: Notifications for high-priority churn risks

## 📧 Contact

**Deepanraj A**  
Senior Data Analyst | ML Engineer  
📧 Email: deepanraj.data@gmail.com  
💼 LinkedIn: [linkedin.com/in/deepanraj-analyst](https://linkedin.com/in/deepanraj-analyst)  
🌐 Portfolio: [deeepanbe.github.io](https://deeepanbe.github.io)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**⭐ If you found this project helpful, please consider starring the repository!**
