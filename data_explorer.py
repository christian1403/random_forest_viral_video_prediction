import pandas as pd
import numpy as np
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_explore_dataset(file_path):
    """
    Load CSV dataset and perform initial data exploration
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print("=" * 80)
    print("VIDEO VIRAL PREDICTION - DATASET EXPLORER")
    print("=" * 80)
    
    try:
        # Load dataset with pandas
        print(f"📁 Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None

def display_basic_info(df):
    """Display basic dataset information"""
    print("\n" + "=" * 50)
    print("BASIC DATASET INFORMATION")
    print("=" * 50)
    
    print(f"Total records: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n📝 Column Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

def display_data_types(df):
    """Display data types and missing values"""
    print("\n" + "=" * 50)
    print("DATA TYPES & MISSING VALUES")
    print("=" * 50)
    
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    
    print(info_df.to_string(index=False))

def display_sample_data(df):
    """Display sample data from the dataset"""
    print("\n" + "=" * 50)
    print("SAMPLE DATA - FIRST 5 ROWS")
    print("=" * 50)
    
    # Display first 5 rows
    print(df.head().to_string())
    
    print("\n" + "=" * 50)
    print("SAMPLE DATA - LAST 5 ROWS")
    print("=" * 50)
    
    # Display last 5 rows
    print(df.tail().to_string())

def analyze_numeric_columns(df):
    """Analyze numeric columns for viral prediction insights"""
    print("\n" + "=" * 50)
    print("NUMERIC COLUMNS ANALYSIS")
    print("=" * 50)
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("No numeric columns found.")
        return
    
    print(f"Numeric columns: {numeric_cols}")
    print("\n📈 Statistical Summary:")
    print(df[numeric_cols].describe())
    
    # Display correlation matrix for key metrics
    key_metrics = ['view_count', 'like_count', 'comment_count', 'daily_rank']
    available_metrics = [col for col in key_metrics if col in numeric_cols]
    
    if len(available_metrics) > 1:
        print(f"\n🔗 Correlation Matrix for Key Metrics: {available_metrics}")
        correlation_matrix = df[available_metrics].corr()
        print(correlation_matrix.round(3))

def analyze_categorical_columns(df):
    """Analyze categorical columns"""
    print("\n" + "=" * 50)
    print("CATEGORICAL COLUMNS ANALYSIS")
    print("=" * 50)
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        print("No categorical columns found.")
        return
    
    print(f"Categorical columns: {categorical_cols}")
    
    for col in categorical_cols[:3]:  # Show first 3 categorical columns
        print(f"\n📊 Unique values in '{col}': {df[col].nunique():,}")
        if df[col].nunique() < 20:  # Only show value counts for columns with few unique values
            print(f"Value counts for '{col}':")
            print(df[col].value_counts().head(10))

def identify_potential_targets(df):
    """Identify potential target variables for viral prediction"""
    print("\n" + "=" * 50)
    print("VIRAL PREDICTION TARGET ANALYSIS")
    print("=" * 50)
    
    # Look for potential indicators of viral content
    potential_indicators = []
    
    if 'view_count' in df.columns:
        view_stats = df['view_count'].describe()
        print(f"📹 View Count Statistics:")
        print(f"   Mean: {view_stats['mean']:,.0f}")
        print(f"   Median: {view_stats['50%']:,.0f}")
        print(f"   Top 75%: {view_stats['75%']:,.0f}")
        potential_indicators.append('view_count')
    
    if 'like_count' in df.columns:
        like_stats = df['like_count'].describe()
        print(f"\n👍 Like Count Statistics:")
        print(f"   Mean: {like_stats['mean']:,.0f}")
        print(f"   Median: {like_stats['50%']:,.0f}")
        potential_indicators.append('like_count')
    
    if 'daily_rank' in df.columns:
        rank_stats = df['daily_rank'].describe()
        print(f"\n🏆 Daily Rank Statistics:")
        print(f"   Mean: {rank_stats['mean']:,.0f}")
        print(f"   Best (lowest): {rank_stats['min']:,.0f}")
        potential_indicators.append('daily_rank')
    
    print(f"\n💡 Suggested approach for viral prediction:")
    print("   - High view_count could indicate viral content")
    print("   - Low daily_rank (closer to 1) indicates trending content")
    print("   - High like_count relative to view_count shows engagement")
    print("   - Consider creating binary target: viral (1) vs non-viral (0)")

def main():
    """Main function to run the data exploration"""
    # Dataset file path
    dataset_path = "/home/cdev/python/random_forest/dataset/output_ID.csv"
    
    # Load the dataset
    df = load_and_explore_dataset(dataset_path)
    
    if df is not None:
        # Perform comprehensive data exploration
        display_basic_info(df)
        display_data_types(df)
        display_sample_data(df)
        analyze_numeric_columns(df)
        analyze_categorical_columns(df)
        identify_potential_targets(df)
        
        print("\n" + "=" * 80)
        print("🎯 NEXT STEPS FOR RANDOM FOREST MODEL:")
        print("=" * 80)
        print("1. ✅ Data exploration completed")
        print("2. 🎯 Define viral threshold (e.g., top 10% by views)")
        print("3. 🧹 Clean and preprocess data")
        print("4. 🔧 Feature engineering")
        print("5. 🌲 Train Random Forest model")
        print("6. 📊 Evaluate model performance")
        print("=" * 80)
        
        return df
    else:
        print("Failed to load dataset. Please check the file path and try again.")
        return None

if __name__ == "__main__":
    # Run the data exploration
    dataset = main()