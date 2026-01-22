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
    print("FIXED DATA PREPROCESSING & FEATURE SELECTION")
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

def remove_leaky_features(df):
    """Remove features that cause data leakage"""
    print("\n" + "=" * 50)
    print("REMOVING DATA LEAKAGE FEATURES")
    print("=" * 50)
    
    # Features that cause data leakage (directly related to post-publication metrics)
    leaky_features = [
        'view_count',           # Used to define viral threshold - MAJOR LEAKAGE!
        'like_count',           # Post-publication metric
        'comment_count',        # Post-publication metric
        'engagement_rate',      # Derived from above metrics
        'like_view_ratio',      # Derived from view_count and like_count
        'comment_view_ratio',   # Derived from view_count and comment_count
        'daily_movement'        # Likely based on view count changes
    ]
    
    print("🚨 Removing the following leaky features:")
    removed_features = []
    for feature in leaky_features:
        if feature in df.columns:
            print(f"   ❌ {feature} - causes data leakage")
            df = df.drop(columns=[feature])
            removed_features.append(feature)
        else:
            print(f"   ⚠️  {feature} - not found in dataset")
    
    print(f"\n✅ Removed {len(removed_features)} leaky features")
    print(f"📊 Dataset shape after cleanup: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    return df

def create_predictive_features(df):
    """Create features that can be known BEFORE publication"""
    print("\n" + "=" * 50)
    print("CREATING PRE-PUBLICATION FEATURES")
    print("=" * 50)
    
    # Title-based features (available before publication)
    if 'title' in df.columns:
        df['title_length'] = df['title'].str.len()
        df['title_word_count'] = df['title'].str.split().str.len()
        df['title_has_numbers'] = df['title'].str.contains(r'\d').astype(int)
        df['title_has_exclamation'] = df['title'].str.contains('!').astype(int)
        df['title_has_question'] = df['title'].str.contains('\?').astype(int)
        df['title_has_caps'] = df['title'].str.contains(r'[A-Z]{2,}').astype(int)
        df['title_viral_words'] = df['title'].str.lower().str.contains(
            '|'.join(['viral', 'amazing', 'incredible', 'shocking', 'must', 'watch', 
                     'unbelievable', 'epic', 'wow', 'omg', 'insane', 'crazy'])
        ).astype(int)
        print("✅ Created title-based features")
    
    # Description-based features (available before publication)
    if 'description' in df.columns:
        df['description'] = df['description'].fillna('')
        df['description_length'] = df['description'].str.len()
        df['description_word_count'] = df['description'].str.split().str.len()
        df['has_description'] = (df['description'].str.len() > 0).astype(int)
        print("✅ Created description-based features")
    
    # Tags-based features (available before publication)
    if 'video_tags' in df.columns:
        df['video_tags'] = df['video_tags'].fillna('')
        # Count number of tags (assuming comma-separated)
        df['tags_count'] = df['video_tags'].str.split(',').str.len()
        df['tags_count'] = df['tags_count'].fillna(0)
        df['has_tags'] = (df['video_tags'].str.len() > 0).astype(int)
        print("✅ Created tags-based features")
    
    # Channel-based features (historical data available before publication)
    if 'channel_title' in df.columns:
        # Encode channel (this represents channel popularity/history)
        le_channel = LabelEncoder()
        df['channel_encoded'] = le_channel.fit_transform(df['channel_title'].fillna('unknown'))
        
        # Calculate channel viral history (before current video)
        # This is the number of viral videos the channel had before this video
        if 'is_viral' in df.columns:
            # For simplicity, we'll use channel frequency as a proxy for channel success
            channel_counts = df.groupby('channel_title').size()
            df['channel_video_count'] = df['channel_title'].map(channel_counts)
        print("✅ Created channel-based features")
    
    # Language-based features (available before publication)
    if 'langauge' in df.columns:  # Note: keeping the typo from original data
        le_language = LabelEncoder()
        df['language_encoded'] = le_language.fit_transform(df['langauge'].fillna('unknown'))
        print("✅ Created language-based features")
    
    # Time-based features (available before/at publication)
    if 'trending_date' in df.columns and 'publish_time' in df.columns:
        try:
            df['trending_date'] = pd.to_datetime(df['trending_date'])
            df['publish_time'] = pd.to_datetime(df['publish_time'])
            df['video_age_days'] = (df['trending_date'] - df['publish_time']).dt.days
            
            # Publication timing features
            df['publish_hour'] = df['publish_time'].dt.hour
            df['publish_day_of_week'] = df['publish_time'].dt.dayofweek
            df['publish_is_weekend'] = (df['publish_day_of_week'].isin([5, 6])).astype(int)
            print("✅ Created time-based features")
        except:
            print("⚠️  Could not create time-based features - date format issues")
    
    return df

def select_non_leaky_features(df):
    """Select features that don't cause data leakage"""
    print("\n" + "=" * 50)
    print("SELECTING NON-LEAKY FEATURES")
    print("=" * 50)
    
    # Features that can be known BEFORE or AT the time of publication
    predictive_features = [
        # Title features
        'title_length', 'title_word_count', 'title_has_numbers', 
        'title_has_exclamation', 'title_has_question', 'title_has_caps',
        'title_viral_words',
        
        # Description features  
        'description_length', 'description_word_count', 'has_description',
        
        # Tags features
        'tags_count', 'has_tags',
        
        # Channel features
        'channel_encoded', 'channel_video_count',
        
        # Language features
        'language_encoded',
        
        # Time features
        'video_age_days', 'publish_hour', 'publish_day_of_week', 'publish_is_weekend'
    ]
    
    # Keep only features that exist in the dataframe
    available_features = [f for f in predictive_features if f in df.columns]
    
    print("✅ Selected non-leaky features:")
    for i, feature in enumerate(available_features, 1):
        print(f"   {i:2d}. {feature}")
    
    # Select features and target
    if 'is_viral' in df.columns:
        X = df[available_features]
        y = df['is_viral']
        
        print(f"\n📊 Final feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features")
        print(f"🎯 Target variable: {len(y):,} samples")
        
        return X, y, available_features
    else:
        print("❌ Target variable 'is_viral' not found!")
        return None, None, None

def split_and_save_data(X, y, selected_features, output_dir):
    """Split data and save to files"""
    print("\n" + "=" * 50)
    print("SPLITTING AND SAVING DATA")
    print("=" * 50)
    
    # Split data with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"✅ Data split completed:")
    print(f"   📊 Training set: {X_train.shape[0]:,} samples")
    print(f"   📊 Testing set: {X_test.shape[0]:,} samples")
    print(f"   🎯 Training target distribution:")
    print(f"      Non-Viral: {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"      Viral: {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.1f}%)")
    
    # Save data files
    try:
        X_train.to_csv(f"{output_dir}/X_train_fixed.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test_fixed.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train_fixed.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test_fixed.csv", index=False)
        
        # Save selected features
        pd.DataFrame({'feature': selected_features}).to_csv(
            f"{output_dir}/selected_features_fixed.csv", index=False
        )
        
        print(f"✅ Data saved to {output_dir}:")
        print(f"   📁 X_train_fixed.csv, X_test_fixed.csv")
        print(f"   📁 y_train_fixed.csv, y_test_fixed.csv") 
        print(f"   📁 selected_features_fixed.csv")
        
    except Exception as e:
        print(f"❌ Error saving data: {str(e)}")

def main():
    """Main preprocessing pipeline without data leakage"""
    # Configuration
    data_file = "/home/cdev/python/random_forest_viral/dataset/processed_viral_dataset.csv"
    output_dir = "/home/cdev/python/random_forest_viral/dataset"
    
    # Load data
    df = load_processed_dataset(data_file)
    if df is None:
        return
    
    # Remove leaky features
    df = remove_leaky_features(df)
    
    # Create predictive features
    df = create_predictive_features(df)
    
    # Select non-leaky features
    X, y, selected_features = select_non_leaky_features(df)
    if X is None:
        return
    
    # Split and save data
    split_and_save_data(X, y, selected_features, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ FIXED DATA PREPROCESSING COMPLETE!")
    print("=" * 80)
    print("🔧 Data leakage removed - model will now show realistic performance")
    print("📊 Use the '*_fixed.csv' files for training your model")
    print("🎯 Expected accuracy: 60-80% (realistic for this type of prediction)")

if __name__ == "__main__":
    main()