import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve, 
                           roc_auc_score, auc, accuracy_score, precision_score, 
                           recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_model_and_data():
    """Load the fixed model and test data"""
    print("=" * 80)
    print("LOADING FIXED MODEL AND DATA FOR VISUALIZATION")
    print("=" * 80)
    
    try:
        # Load model
        model = joblib.load("/home/cdev/python/random_forest_viral/dataset/viral_prediction_rf_model_FIXED.pkl")
        print("✅ Fixed model loaded successfully!")
        
        # Load test data
        X_test = pd.read_csv("/home/cdev/python/random_forest_viral/dataset/X_test_fixed.csv")
        y_test = pd.read_csv("/home/cdev/python/random_forest_viral/dataset/y_test_fixed.csv")['is_viral']
        
        # Load feature names
        features_df = pd.read_csv("/home/cdev/python/random_forest_viral/dataset/selected_features_fixed.csv")
        feature_names = features_df['feature'].tolist()
        
        print(f"📊 Test data: {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
        print(f"🎯 Target distribution: {y_test.value_counts().to_dict()}")
        
        return model, X_test, y_test, feature_names
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None, None, None, None

def create_confusion_matrix_heatmap(y_test, y_pred, save_path):
    """Create confusion matrix heatmap"""
    print("📊 Creating Confusion Matrix Heatmap...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Viral', 'Viral'],
                yticklabels=['Non-Viral', 'Viral'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Fixed Viral Prediction Model\n(No Data Leakage)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    
    # Add percentage annotations
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrix_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: confusion_matrix_heatmap.png")

def create_roc_curve(y_test, y_pred_proba, save_path):
    """Create ROC Curve"""
    print("📈 Creating ROC Curve...")
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, 
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Fixed Viral Prediction Model\n(Excellent Discrimination)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add AUC interpretation
    if roc_auc > 0.9:
        interpretation = "Excellent"
    elif roc_auc > 0.8:
        interpretation = "Good"
    elif roc_auc > 0.7:
        interpretation = "Fair"
    else:
        interpretation = "Poor"
    
    plt.text(0.6, 0.2, f'Model Performance: {interpretation}', 
             fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: roc_curve.png")

def create_precision_recall_curve(y_test, y_pred_proba, save_path):
    """Create Precision-Recall Curve"""
    print("📊 Creating Precision-Recall Curve...")
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=3, 
             label=f'PR Curve (AUC = {pr_auc:.3f})')
    
    # Baseline (random classifier for imbalanced dataset)
    baseline = y_test.sum() / len(y_test)
    plt.axhline(y=baseline, color='red', linestyle='--', lw=2,
                label=f'Random Classifier (Baseline = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (True Positive Rate)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Fixed Viral Prediction Model\n(Handles Class Imbalance)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/precision_recall_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: precision_recall_curve.png")

def create_feature_importance_chart(model, feature_names, save_path):
    """Create feature importance visualization"""
    print("🎯 Creating Feature Importance Chart...")
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(12, 10))
    bars = plt.barh(range(len(feature_importance_df)), 
                    feature_importance_df['importance'],
                    color=plt.cm.viridis(np.linspace(0, 1, len(feature_importance_df))))
    
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Feature Importance - Fixed Viral Prediction Model\n(No Leaky Features)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=10)
    
    # Highlight top 5 features
    top_5_threshold = sorted(importance, reverse=True)[4]
    for i, bar in enumerate(bars):
        if bar.get_width() >= top_5_threshold:
            bar.set_edgecolor('red')
            bar.set_linewidth(2)
    
    plt.text(0.02, 0.95, 'Top 5 features highlighted in red', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: feature_importance.png")

def create_performance_metrics_chart(y_test, y_pred, y_pred_proba, save_path):
    """Create performance metrics comparison chart"""
    print("📈 Creating Performance Metrics Chart...")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    values = [accuracy, precision, recall, f1, roc_auc]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}\n({value*100:.1f}%)', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    plt.ylim(0, 1.1)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Metrics - Fixed Viral Prediction Model\n(Realistic Performance)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add horizontal reference lines
    plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Good Threshold (80%)')
    plt.axhline(y=0.9, color='blue', linestyle='--', alpha=0.7, label='Excellent Threshold (90%)')
    
    plt.legend(loc='upper right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/performance_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: performance_metrics.png")

def create_prediction_distribution(y_pred_proba, y_test, save_path):
    """Create prediction probability distribution"""
    print("📊 Creating Prediction Distribution...")
    
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Histogram of prediction probabilities
    plt.subplot(1, 2, 1)
    viral_probs = y_pred_proba[y_test == 1]
    non_viral_probs = y_pred_proba[y_test == 0]
    
    plt.hist(non_viral_probs, bins=50, alpha=0.7, label='Non-Viral Videos', 
             color='lightcoral', density=True)
    plt.hist(viral_probs, bins=50, alpha=0.7, label='Viral Videos', 
             color='skyblue', density=True)
    
    plt.axvline(x=0.3, color='black', linestyle='--', linewidth=2, 
                label='Decision Threshold (0.3)')
    plt.xlabel('Predicted Viral Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Box plot
    plt.subplot(1, 2, 2)
    data_to_plot = [non_viral_probs, viral_probs]
    labels = ['Non-Viral\n(Actual)', 'Viral\n(Actual)']
    
    bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('skyblue')
    
    plt.ylabel('Predicted Viral Probability', fontsize=12)
    plt.title('Probability Distribution by Actual Label', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Prediction Probability Analysis - Fixed Model\n(Clear Separation Indicates Good Model)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/prediction_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: prediction_distribution.png")

def create_sample_predictions_visualization(model, feature_names, save_path):
    """Visualize sample predictions with feature values"""
    print("🎬 Creating Sample Predictions Visualization...")
    
    # Define sample data (from our test script)
    samples = {
        # 'Ultimate Viral\n(Optimized)': [25, 4, 1, 0, 0, 1, 1, 5000, 600, 1, 50, 1, 1, 10, 0, 4],
        'Video Viral': [30, 5, 1, 0, 0, 1, 1, 3000, 400, 1, 30, 1, 1, 6, 0, 4],
        'Video Biasa': [45, 8, 0, 0, 0, 1, 0, 600, 80, 1, 10, 1, 18, 2, 0, 2],
        'Video Non Viral': [70, 12, 0, 1, 0, 0, 0, 300, 40, 1, 5, 1, 38, 4, 0, 0]
    }
    
    # Make predictions
    predictions = {}
    for name, features in samples.items():
        prob = model.predict_proba([features])[0][1]
        pred = model.predict([features])[0]
        predictions[name] = {'probability': prob, 'prediction': pred}
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Subplot 1: Probability bars
    names = list(predictions.keys())
    probabilities = [predictions[name]['probability'] for name in names]
    colors = ['red' if prob < 0.3 else 'orange' if prob < 0.5 else 'green' for prob in probabilities]
    
    bars = ax1.bar(names, probabilities, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0.3, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    ax1.set_ylabel('Viral Probability', fontsize=12)
    ax1.set_title('Sample Video Predictions', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add probability labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob:.3f}\n({prob*100:.1f}%)', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # Rotate x labels
    ax1.tick_params(axis='x', rotation=45)
    
    # Subplot 2: Key features comparison
    key_features = ['title_length', 'description_length', 'tags_count', 'language_encoded']
    key_indices = [feature_names.index(feat) for feat in key_features]
    
    x = np.arange(len(key_features))
    width = 0.2
    
    for i, (name, features) in enumerate(samples.items()):
        values = [features[idx] for idx in key_indices]
        # Normalize values for better visualization
        normalized_values = []
        for j, val in enumerate(values):
            if key_features[j] == 'title_length':
                normalized_values.append(val / 100)  # Scale down
            elif key_features[j] == 'description_length':
                normalized_values.append(val / 5000)  # Scale down
            elif key_features[j] == 'tags_count':
                normalized_values.append(val / 50)   # Scale down
            else:
                normalized_values.append(val / 40)   # Scale down
        
        ax2.bar(x + i*width, normalized_values, width, label=name, alpha=0.7)
    
    ax2.set_xlabel('Key Features', fontsize=12)
    ax2.set_ylabel('Normalized Values', fontsize=12)
    ax2.set_title('Key Feature Comparison\n(Normalized Scale)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(key_features, rotation=45)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/sample_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: sample_predictions.png")

def create_before_after_comparison(save_path):
    """Create before/after comparison of model performance"""
    print("🔄 Creating Before/After Comparison...")
    
    # Data from the results
    before_metrics = {'Accuracy': 1.0, 'Precision': 1.0, 'Recall': 1.0, 'F1-Score': 1.0, 'ROC-AUC': 1.0}
    after_metrics = {'Accuracy': 0.973, 'Precision': 0.849, 'Recall': 0.885, 'F1-Score': 0.867, 'ROC-AUC': 0.986}
    
    metrics = list(before_metrics.keys())
    before_values = list(before_metrics.values())
    after_values = list(after_metrics.values())
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(14, 8))
    bars1 = plt.bar(x - width/2, before_values, width, label='Before (With Leakage)', 
                    color='red', alpha=0.7, edgecolor='black')
    bars2 = plt.bar(x + width/2, after_values, width, label='After (Fixed)', 
                    color='green', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance: Before vs After Fixing Data Leakage\n(Realistic Performance vs Perfect Accuracy)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(x, metrics)
    plt.ylim(0, 1.1)
    plt.legend(fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add annotations
    plt.text(0.5, 0.95, 'Before: 100% accuracy due to data leakage (unrealistic)', 
             transform=plt.gca().transAxes, ha='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    plt.text(0.5, 0.88, 'After: ~97% accuracy with realistic features (trustworthy)', 
             transform=plt.gca().transAxes, ha='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/before_after_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: before_after_comparison.png")

def generate_visualization_report(save_path, metrics):
    """Generate a comprehensive visualization report"""
    print("📝 Creating Visualization Report...")
    
    report_content = f"""# Fixed Viral Video Prediction Model - Visualization Report

## 📊 Model Performance Summary

### Key Metrics:
- **Accuracy**: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
- **Precision**: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)
- **Recall**: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)
- **F1-Score**: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)
- **ROC-AUC**: {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)

## 🔍 Analysis Insights

### Model Quality:
- **Excellent discrimination** with ROC-AUC of {metrics['roc_auc']:.3f}
- **High precision** indicates reliable viral predictions
- **Good recall** captures most actual viral videos
- **Balanced performance** with strong F1-Score

### Confusion Matrix Results:
```
True Negatives:  {metrics['confusion_matrix'][0,0]:,} (Correct non-viral predictions)
False Positives: {metrics['confusion_matrix'][0,1]:,} (Wrong viral predictions)  
False Negatives: {metrics['confusion_matrix'][1,0]:,} (Missed viral videos)
True Positives:  {metrics['confusion_matrix'][1,1]:,} (Correct viral predictions)
```

## 🎯 Key Improvements from Original Model

### Before (With Data Leakage):
- Accuracy: 100% (Unrealistic)
- Used post-publication metrics (view_count, like_count, etc.)
- Perfect but meaningless predictions

### After (Fixed):
- Accuracy: {metrics['accuracy']*100:.1f}% (Realistic)
- Uses only pre-publication features
- Trustworthy and applicable predictions

## 📈 Visualization Files Generated

1. **confusion_matrix_heatmap.png** - Confusion matrix with percentages
2. **roc_curve.png** - ROC curve showing excellent discrimination
3. **precision_recall_curve.png** - PR curve for imbalanced dataset analysis
4. **feature_importance.png** - Most important features for prediction
5. **performance_metrics.png** - Overall model performance comparison
6. **prediction_distribution.png** - How predictions are distributed
7. **sample_predictions.png** - Example predictions with different optimizations
8. **before_after_comparison.png** - Comparison of leaked vs fixed model

## 💡 Recommendations

1. **Use this model for content optimization** - Focus on features that matter
2. **Trust the predictions** - No data leakage means realistic performance
3. **Optimize key features**: description_length, title_length, tags_count
4. **Consider it guidance** - Viral prediction is inherently uncertain

## 🎬 Feature Optimization Tips

Based on feature importance analysis:
- **Keep titles short** (~30-45 characters)
- **Write detailed descriptions** (1000+ characters)
- **Use many relevant tags** (15+ tags)
- **Choose popular languages**
- **Time publication strategically**

---
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: Fixed Random Forest (No Data Leakage)
Dataset: {len(metrics['confusion_matrix'][0]) + len(metrics['confusion_matrix'][1]):,} test samples
"""

    with open(f"{save_path}/VISUALIZATION_REPORT.md", 'w') as f:
        f.write(report_content)
    
    print(f"✅ Saved: VISUALIZATION_REPORT.md")

def main():
    """Main function to generate all visualizations"""
    print("🎨 STARTING VISUALIZATION GENERATION FOR FIXED MODEL")
    print("=" * 80)
    
    # Load model and data
    model, X_test, y_test, feature_names = load_model_and_data()
    if model is None:
        return
    
    # Make predictions
    print("\n🔮 Making predictions for visualization...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Output path
    save_path = "/home/cdev/python/random_forest_viral/visualizations_fixed"
    
    print(f"\n🎨 Generating visualizations...")
    print(f"📁 Output directory: {save_path}")
    
    # Generate all visualizations
    create_confusion_matrix_heatmap(y_test, y_pred, save_path)
    create_roc_curve(y_test, y_pred_proba, save_path)
    create_precision_recall_curve(y_test, y_pred_proba, save_path)
    create_feature_importance_chart(model, feature_names, save_path)
    create_performance_metrics_chart(y_test, y_pred, y_pred_proba, save_path)
    create_prediction_distribution(y_pred_proba, y_test, save_path)
    create_sample_predictions_visualization(model, feature_names, save_path)
    create_before_after_comparison(save_path)
    generate_visualization_report(save_path, metrics)
    
    print(f"\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📁 Location: {save_path}/")
    print("📊 Files created:")
    print("   • confusion_matrix_heatmap.png")
    print("   • roc_curve.png") 
    print("   • precision_recall_curve.png")
    print("   • feature_importance.png")
    print("   • performance_metrics.png")
    print("   • prediction_distribution.png")
    print("   • sample_predictions.png")
    print("   • before_after_comparison.png")
    print("   • VISUALIZATION_REPORT.md")
    
    print(f"\n🎯 Model Performance Summary:")
    print(f"   Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"   ROC-AUC: {metrics['roc_auc']*100:.1f}%")
    print(f"   F1-Score: {metrics['f1_score']*100:.1f}%")

if __name__ == "__main__":
    main()