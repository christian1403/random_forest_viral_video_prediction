import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_processed_dataset(file_path):
    """Load the processed dataset with viral target variable"""
    print("=" * 80)
    print("DATA PREPROCESSING & FEATURE SELECTION")
    print("=" * 80)
    
    try:
        print(f"📁 Loading processed dataset from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Check target variable
        target_dist = df['is_viral'].value_counts()
        print(f"🎯 Target distribution:")
        print(f"   Non-Viral (0): {target_dist[0]:,} ({target_dist[0]/len(df)*100:.1f}%)")
        print(f"   Viral (1): {target_dist[1]:,} ({target_dist[1]/len(df)*100:.1f}%)")
        
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    print("\n" + "=" * 50)
    print("HANDLING MISSING VALUES")
    print("=" * 50)
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    missing_columns = missing_counts[missing_counts > 0]
    
    if len(missing_columns) > 0:
        print("📊 Missing values found:")
        for col, count in missing_columns.items():
            percentage = (count / len(df)) * 100
            print(f"   {col}: {count:,} ({percentage:.1f}%)")
        
        # Handle specific columns
        if 'description' in missing_columns:
            df['description'].fillna('', inplace=True)
            print(f"✅ Filled missing descriptions with empty string")
        
        if 'video_tags' in missing_columns:
            df['video_tags'].fillna('', inplace=True)
            print(f"✅ Filled missing video_tags with empty string")
        
        if 'langauge' in missing_columns:  # Note: there's a typo in original data
            df['langauge'].fillna('unknown', inplace=True)
            print(f"✅ Filled missing language with 'unknown'")
        
        # Fill any remaining numeric missing values with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
                print(f"✅ Filled missing {col} with median value")
                
    else:
        print("✅ No missing values found!")
    
    return df

def create_text_features(df):
    """Create features from text columns (title, description, video_tags)"""
    print("\n" + "=" * 50)
    print("CREATING TEXT-BASED FEATURES")
    print("=" * 50)
    
    # Title features
    df['title_word_count'] = df['title'].str.split().str.len()
    df['title_has_caps'] = df['title'].str.contains(r'[A-Z]{2,}').astype(int)
    df['title_has_numbers'] = df['title'].str.contains(r'\d').astype(int)
    df['title_has_exclamation'] = df['title'].str.contains('!').astype(int)
    df['title_has_question'] = df['title'].str.contains('\?').astype(int)
    
    # Description features
    df['description_word_count'] = df['description'].str.split().str.len().fillna(0)
    df['has_description'] = (df['description'].str.len() > 0).astype(int)
    
    # Video tags features
    df['tags_count'] = df['video_tags'].str.split(',').str.len().fillna(0)
    df['has_tags'] = (df['video_tags'].str.len() > 0).astype(int)
    
    print("✅ Created text-based features:")
    text_features = ['title_word_count', 'title_has_caps', 'title_has_numbers', 
                    'title_has_exclamation', 'title_has_question', 'description_word_count',
                    'has_description', 'tags_count', 'has_tags']
    
    for i, feature in enumerate(text_features, 1):
        print(f"   {i:2d}. {feature}")
    
    return df, text_features

def encode_categorical_features(df):
    """Encode categorical features for machine learning"""
    print("\n" + "=" * 50)
    print("ENCODING CATEGORICAL FEATURES")
    print("=" * 50)
    
    categorical_encodings = {}
    
    # Country encoding
    le_country = LabelEncoder()
    df['country_encoded'] = le_country.fit_transform(df['country'])
    categorical_encodings['country'] = le_country
    print(f"✅ Encoded country: {len(le_country.classes_)} unique values")
    
    # Language encoding
    le_language = LabelEncoder()
    df['language_encoded'] = le_language.fit_transform(df['langauge'])
    categorical_encodings['language'] = le_language
    print(f"✅ Encoded language: {len(le_language.classes_)} unique values")
    
    # Channel encoding (top channels only, others as 'other')
    top_channels = df['channel_name'].value_counts().head(50).index
    df['channel_category'] = df['channel_name'].apply(
        lambda x: x if x in top_channels else 'other'
    )
    le_channel = LabelEncoder()
    df['channel_encoded'] = le_channel.fit_transform(df['channel_category'])
    categorical_encodings['channel'] = le_channel
    print(f"✅ Encoded top 50 channels + 'other': {len(le_channel.classes_)} categories")
    
    # Day of week is already numeric (0-6)
    print(f"✅ Day of week already numeric (0-6)")
    
    encoded_features = ['country_encoded', 'language_encoded', 'channel_encoded']
    
    return df, categorical_encodings, encoded_features

def select_features_for_model(df):
    """Select the best features for Random Forest model"""
    print("\n" + "=" * 50)
    print("FEATURE SELECTION")
    print("=" * 50)
    
    # Define potential features (excluding target and non-predictive columns)
    exclude_columns = [
        'title', 'channel_name', 'description', 'thumbnail_url', 'video_id', 
        'channel_id', 'video_tags', 'kind', 'publish_date', 'snapshot_date',
        'langauge', 'channel_category', 'is_viral'  # target variable
    ]
    
    # Get all numeric and encoded features
    potential_features = [col for col in df.columns if col not in exclude_columns]
    
    print(f"📊 Potential features for model: {len(potential_features)}")
    
    # Separate features and target
    X = df[potential_features]
    y = df['is_viral']
    
    # Handle any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    print(f"📊 Final numeric features: {len(X.columns)}")
    print("\n🔍 Selected features:")
    for i, feature in enumerate(X.columns, 1):
        print(f"   {i:2d}. {feature}")
    
    return X, y, list(X.columns)

def analyze_feature_importance(X, y, feature_names):
    """Analyze feature importance using Random Forest"""
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    # Quick Random Forest to get feature importance
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_temp.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("🌲 Top 15 Most Important Features (Random Forest):")
    for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:<25} | Importance: {row['importance']:.4f}")
    
    return feature_importance

def perform_statistical_feature_selection(X, y, k=20):
    """Perform statistical feature selection"""
    print(f"\n" + "=" * 50)
    print(f"STATISTICAL FEATURE SELECTION (TOP {k})")
    print("=" * 50)
    
    # Use mutual information for feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'score': selector.scores_,
        'selected': selector.get_support()
    }).sort_values('score', ascending=False)
    
    print(f"🎯 Selected top {k} features by mutual information:")
    for i, feature in enumerate(selected_features, 1):
        score = feature_scores[feature_scores['feature'] == feature]['score'].iloc[0]
        print(f"   {i:2d}. {feature:<25} | Score: {score:.4f}")
    
    return X_selected, selected_features, selector

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    print(f"\n" + "=" * 50)
    print("DATA SPLITTING")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"📊 Data split completed:")
    print(f"   Training set: {X_train.shape[0]:,} samples ({(1-test_size)*100:.0f}%)")
    print(f"   Testing set: {X_test.shape[0]:,} samples ({test_size*100:.0f}%)")
    
    print(f"\n🎯 Target distribution in training set:")
    train_dist = y_train.value_counts()
    print(f"   Non-Viral (0): {train_dist[0]:,} ({train_dist[0]/len(y_train)*100:.1f}%)")
    print(f"   Viral (1): {train_dist[1]:,} ({train_dist[1]/len(y_train)*100:.1f}%)")
    
    print(f"\n🎯 Target distribution in testing set:")
    test_dist = y_test.value_counts()
    print(f"   Non-Viral (0): {test_dist[0]:,} ({test_dist[0]/len(y_test)*100:.1f}%)")
    print(f"   Viral (1): {test_dist[1]:,} ({test_dist[1]/len(y_test)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

def save_preprocessed_data(X_train, X_test, y_train, y_test, selected_features, 
                          categorical_encodings, output_dir):
    """Save preprocessed data for model training"""
    print(f"\n" + "=" * 50)
    print("SAVING PREPROCESSED DATA")
    print("=" * 50)
    
    try:
        # Save training and testing data
        pd.DataFrame(X_train, columns=selected_features).to_csv(
            f"{output_dir}/X_train.csv", index=False)
        pd.DataFrame(X_test, columns=selected_features).to_csv(
            f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False, header=['is_viral'])
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False, header=['is_viral'])
        
        # Save feature names
        pd.DataFrame({'feature': selected_features}).to_csv(
            f"{output_dir}/selected_features.csv", index=False)
        
        print(f"✅ Preprocessed data saved:")
        print(f"   📁 X_train.csv: {X_train.shape}")
        print(f"   📁 X_test.csv: {X_test.shape}")
        print(f"   📁 y_train.csv: {y_train.shape}")
        print(f"   📁 y_test.csv: {y_test.shape}")
        print(f"   📁 selected_features.csv: {len(selected_features)} features")
        
    except Exception as e:
        print(f"❌ Error saving preprocessed data: {str(e)}")

def main():
    """Main function to run data preprocessing and feature selection"""
    # File paths
    input_path = "/home/cdev/python/random_forest/dataset/processed_viral_dataset.csv"
    output_dir = "/home/cdev/python/random_forest/dataset"
    
    # Load processed dataset
    df = load_processed_dataset(input_path)
    if df is None:
        return None
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Create text-based features
    df, text_features = create_text_features(df)
    
    # Encode categorical features
    df, categorical_encodings, encoded_features = encode_categorical_features(df)
    
    # Select features for model
    X, y, potential_features = select_features_for_model(df)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(X, y, potential_features)
    
    # Statistical feature selection (top 20 features)
    X_selected, selected_features, selector = perform_statistical_feature_selection(X, y, k=20)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X_selected, y)
    
    # Save preprocessed data
    save_preprocessed_data(X_train, X_test, y_train, y_test, selected_features, 
                          categorical_encodings, output_dir)
    
    print("\n" + "=" * 80)
    print("🎯 DATA PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"✅ Missing values handled")
    print(f"✅ {len(text_features)} text features created")
    print(f"✅ {len(encoded_features)} categorical features encoded")
    print(f"✅ Top {len(selected_features)} features selected")
    print(f"✅ Data split into train/test sets")
    print(f"✅ Preprocessed data saved and ready for training")
    print("\n🔄 NEXT STEPS:")
    print("   4. ✅ Data preprocessing completed")
    print("   5. 🌲 Train Random Forest model")
    print("   6. 📊 Model evaluation & optimization")
    print("=" * 80)
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'selected_features': selected_features,
        'feature_importance': feature_importance,
        'categorical_encodings': categorical_encodings
    }

if __name__ == "__main__":
    # Run data preprocessing
    preprocessing_results = main()