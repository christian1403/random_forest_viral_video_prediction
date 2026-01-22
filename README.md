# 🎥 Video Virality Prediction using Random Forest

A machine learning project that predicts whether a video will go viral using Random Forest classification algorithm.

## 📋 Project Overview

This project implements a Random Forest classifier to predict video virality based on various features extracted from video metadata. The model analyzes patterns in video characteristics to determine the likelihood of a video becoming viral.

## 🚀 Features

- **Data Preprocessing**: Clean and prepare video dataset for machine learning
- **Feature Selection**: Identify the most important features for virality prediction
- **Random Forest Model**: Train and optimize a Random Forest classifier
- **Model Evaluation**: Comprehensive evaluation with multiple metrics
- **Visualizations**: Generate plots and charts for model analysis
- **Leak-Free Implementation**: Fixed version prevents data leakage issues

## 📁 Project Structure

```
├── README.md
├── LICENSE
├── dataset/                          # Data files and model outputs
│   ├── processed_viral_dataset.csv   # Preprocessed dataset
│   ├── X_train_fixed.csv            # Training features (fixed)
│   ├── y_train_fixed.csv            # Training labels (fixed)
│   ├── X_test_fixed.csv             # Testing features (fixed)
│   ├── y_test_fixed.csv             # Testing labels (fixed)
│   └── viral_prediction_rf_model_FIXED.pkl  # Trained model
├── visualizations_fixed/             # Generated visualizations
├── data_explorer.py                  # Data exploration utilities
├── fixed_data_preprocessor.py        # Data preprocessing (fixed)
├── fixed_random_forest_trainer.py    # Model training (fixed)
├── fixed_test_model.py              # Model testing (fixed)
└── create_visualizations_fixed.py    # Visualization generation
```

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Required Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

Or install all dependencies at once:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib warnings datetime
```

### Optional Dependencies (for enhanced visualizations)
```bash
pip install plotly jupyter
```

## 🔧 Usage

### 1. Data Preprocessing
```bash
python fixed_data_preprocessor.py
```
This will:
- Load and clean the raw dataset
- Perform feature selection
- Split data into training/testing sets (leak-free)
- Save preprocessed data files

### 2. Train the Model
```bash
python fixed_random_forest_trainer.py
```
This will:
- Train Random Forest classifier with hyperparameter tuning
- Evaluate model performance
- Save the trained model
- Generate evaluation metrics

### 3. Test the Model
```bash
python fixed_test_model.py
```
This will:
- Load the trained model
- Make predictions on test data
- Display detailed performance metrics

### 4. Generate Visualizations
```bash
python create_visualizations_fixed.py
```
This will create various plots:
- Feature importance charts
- Confusion matrix
- ROC curves
- Precision-recall curves

## 📊 Model Performance

The Random Forest model provides:
- **Accuracy**: High accuracy in predicting video virality
- **Feature Importance**: Identifies key factors for viral content
- **Balanced Performance**: Good precision and recall scores
- **Robust Predictions**: Cross-validation ensures model stability

## 🎯 Key Features for Virality Prediction

The model analyzes various video characteristics such as:
- Video engagement metrics
- Content metadata
- Upload timing
- Channel characteristics
- And other relevant features

*(Specific features depend on your dataset)*

## 📈 Results

Results and model evaluation metrics are saved in:
- `dataset/model_evaluation_results_FIXED.csv`
- `dataset/feature_importance_FIXED.csv`
- `dataset/model_summary_FIXED.csv`

## 🐛 Troubleshooting

### Common Issues:
1. **Missing data files**: Run `fixed_data_preprocessor.py` first
2. **Import errors**: Ensure all dependencies are installed
3. **Memory issues**: Consider reducing dataset size for large files

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements.

## 📞 Support

If you encounter any issues or have questions, please create an issue in the project repository.

---

*Built with ❤️ using Python and scikit-learn*