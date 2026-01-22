import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_auc_score, 
                           roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_fixed_data(data_dir):
    """Load fixed preprocessed training and testing data (without leakage)"""
    print("=" * 80)
    print("FIXED RANDOM FOREST VIRAL PREDICTION MODEL TRAINING")
    print("=" * 80)
    
    try:
        # Load fixed training data
        X_train = pd.read_csv(f"{data_dir}/X_train_fixed.csv")
        y_train = pd.read_csv(f"{data_dir}/y_train_fixed.csv")['is_viral']
        
        # Load fixed testing data
        X_test = pd.read_csv(f"{data_dir}/X_test_fixed.csv")
        y_test = pd.read_csv(f"{data_dir}/y_test_fixed.csv")['is_viral']
        
        # Load fixed selected features
        selected_features = pd.read_csv(f"{data_dir}/selected_features_fixed.csv")['feature'].tolist()
        
        print(f"✅ Fixed data loaded successfully!")
        print(f"📊 Training set: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
        print(f"📊 Testing set: {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
        print(f"🎯 Target distribution in training:")
        print(f"   Non-Viral (0): {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.1f}%)")
        print(f"   Viral (1): {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.1f}%)")
        
        print(f"\n🔧 Features used (NO LEAKAGE):")
        for i, feature in enumerate(selected_features, 1):
            print(f"   {i:2d}. {feature}")
        
        return X_train, X_test, y_train, y_test, selected_features
        
    except Exception as e:
        print(f"❌ Error loading fixed data: {str(e)}")
        print("💡 Run fixed_data_preprocessor.py first to create leak-free data")
        return None, None, None, None, None

def train_realistic_model(X_train, y_train):
    """Train a Random Forest model with realistic expectations"""
    print("\n" + "=" * 50)
    print("REALISTIC RANDOM FOREST MODEL")
    print("=" * 50)
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    print("🌲 Training Random Forest (this may take a moment)...")
    rf_model.fit(X_train, y_train)
    
    # Cross-validation to get realistic performance estimate
    print("🔍 Performing cross-validation...")
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1')
    cv_accuracy = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"✅ Model trained!")
    print(f"📊 Cross-validation F1 scores: {[f'{score:.3f}' for score in cv_scores]}")
    print(f"📊 Mean CV F1 score: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
    print(f"📊 Cross-validation Accuracy scores: {[f'{score:.3f}' for score in cv_accuracy]}")
    print(f"📊 Mean CV Accuracy: {cv_accuracy.mean():.3f} (±{cv_accuracy.std()*2:.3f})")
    
    return rf_model

def evaluate_realistic_model(model, X_train, X_test, y_train, y_test, selected_features):
    """Evaluate model with realistic metrics"""
    print("\n" + "=" * 50)
    print("REALISTIC MODEL EVALUATION")
    print("=" * 50)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print("📊 REALISTIC PERFORMANCE METRICS:")
    print(f"   Training Accuracy:  {train_accuracy:.3f}")
    print(f"   Testing Accuracy:   {test_accuracy:.3f}")
    print(f"   Testing Precision:  {test_precision:.3f}")
    print(f"   Testing Recall:     {test_recall:.3f}")
    print(f"   Testing F1-Score:   {test_f1:.3f}")
    print(f"   Testing ROC-AUC:    {test_auc:.3f}")
    
    # Check for overfitting
    accuracy_diff = abs(train_accuracy - test_accuracy)
    if accuracy_diff > 0.1:
        print(f"\n⚠️  WARNING: Possible overfitting detected!")
        print(f"   Training-Testing accuracy gap: {accuracy_diff:.3f}")
    else:
        print(f"\n✅ Good generalization: Training-Testing gap: {accuracy_diff:.3f}")
    
    # Confusion Matrix
    print(f"\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"   True Negatives:  {cm[0,0]:,}")
    print(f"   False Positives: {cm[0,1]:,}")
    print(f"   False Negatives: {cm[1,0]:,}")
    print(f"   True Positives:  {cm[1,1]:,}")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🎯 TOP 10 MOST IMPORTANT FEATURES:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:<25} {row['importance']:.4f}")
    
    # Save results
    results = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Training': [train_accuracy, '-', '-', '-', '-'],
        'Testing': [test_accuracy, test_precision, test_recall, test_f1, test_auc]
    }
    
    results_df = pd.DataFrame(results)
    
    return results_df, feature_importance

def save_realistic_model(model, feature_importance, results_df, output_dir):
    """Save the realistic model and results"""
    print("\n" + "=" * 50)
    print("SAVING REALISTIC MODEL")
    print("=" * 50)
    
    try:
        # Save model
        model_path = f"{output_dir}/viral_prediction_rf_model_FIXED.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Save results
        results_path = f"{output_dir}/model_evaluation_results_FIXED.csv"
        results_df.to_csv(results_path, index=False)
        print(f"✅ Results saved: {results_path}")
        
        # Save feature importance
        importance_path = f"{output_dir}/feature_importance_FIXED.csv"
        feature_importance.to_csv(importance_path, index=False)
        print(f"✅ Feature importance saved: {importance_path}")
        
        # Create model summary
        model_summary = {
            'Model': ['Random Forest'],
            'Features': [len(feature_importance)],
            'Training_Samples': ['-'],  # Will be updated
            'Testing_Samples': ['-'],   # Will be updated
            'Best_Accuracy': [results_df[results_df['Metric'] == 'Accuracy']['Testing'].iloc[0]],
            'Best_F1_Score': [results_df[results_df['Metric'] == 'F1-Score']['Testing'].iloc[0]],
            'Model_File': ['viral_prediction_rf_model_FIXED.pkl'],
            'Date_Trained': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        }
        
        summary_df = pd.DataFrame(model_summary)
        summary_path = f"{output_dir}/model_summary_FIXED.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Model summary saved: {summary_path}")
        
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")

def main():
    """Main training pipeline for realistic viral prediction"""
    # Configuration
    data_dir = "/home/cdev/python/random_forest_viral/dataset"
    
    # Load fixed data
    X_train, X_test, y_train, y_test, selected_features = load_fixed_data(data_dir)
    if X_train is None:
        return
    
    # Train realistic model
    model = train_realistic_model(X_train, y_train)
    
    # Evaluate model
    results_df, feature_importance = evaluate_realistic_model(
        model, X_train, X_test, y_train, y_test, selected_features
    )
    
    # Save model and results
    save_realistic_model(model, feature_importance, results_df, data_dir)
    
    print("\n" + "=" * 80)
    print("✅ REALISTIC MODEL TRAINING COMPLETE!")
    print("=" * 80)
    print("🎯 Expected Results:")
    print("   • Accuracy: 60-80% (realistic for viral prediction)")
    print("   • Model uses only pre-publication features")
    print("   • No data leakage - real-world applicable")
    print("\n💡 Next steps:")
    print("   • Use 'viral_prediction_rf_model_FIXED.pkl' for predictions")
    print("   • Run fixed_test_model.py to test realistic predictions")

if __name__ == "__main__":
    main()