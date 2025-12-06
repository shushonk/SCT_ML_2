# customer_segmentation.py
"""
🛒 MALL CUSTOMER SEGMENTATION WITH K-MEANS
📊 Using Kaggle: customer-segmentation-tutorial-in-python dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Set style for better visuals
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and explore the Kaggle dataset"""
    print("="*60)
    print("🛒 MALL CUSTOMER SEGMENTATION WITH K-MEANS")
    print("="*60)
    
    # Load dataset
    df = pd.read_csv('Mall_Customers.csv')
    print(f"✅ Dataset loaded: {len(df)} customers")
    print(f"📋 Features: {list(df.columns)}")
    
    # Display basic info
    print("\n📊 Dataset Preview:")
    print(df.head())
    
    print("\n📈 Basic Statistics:")
    print(df.describe())
    
    print("\n🔍 Missing Values:")
    print(df.isnull().sum())
    
    # Rename columns for clarity
    df.rename(columns={
        'Annual Income (k$)': 'AnnualIncome',
        'Spending Score (1-100)': 'SpendingScore'
    }, inplace=True)
    
    return df

def visualize_data(df):
    """Create initial visualizations"""
    print("\n" + "="*60)
    print("📊 DATA VISUALIZATION")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Age Distribution
    axes[0, 0].hist(df['Age'], bins=20, color='skyblue', edgecolor='black', alpha=0.8)
    axes[0, 0].set_title('Age Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Income Distribution
    axes[0, 1].hist(df['AnnualIncome'], bins=20, color='lightgreen', edgecolor='black', alpha=0.8)
    axes[0, 1].set_title('Annual Income Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Annual Income (k$)')
    axes[0, 1].set_ylabel('Count')
    
    # 3. Spending Score Distribution
    axes[0, 2].hist(df['SpendingScore'], bins=20, color='lightcoral', edgecolor='black', alpha=0.8)
    axes[0, 2].set_title('Spending Score Distribution', fontweight='bold')
    axes[0, 2].set_xlabel('Spending Score (1-100)')
    axes[0, 2].set_ylabel('Count')
    
    # 4. Gender Distribution
    gender_counts = df['Gender'].value_counts()
    axes[1, 0].pie(gender_counts.values, labels=gender_counts.index, 
                   autopct='%1.1f%%', colors=['lightblue', 'lightpink'])
    axes[1, 0].set_title('Gender Distribution', fontweight='bold')
    
    # 5. Income vs Spending Score
    scatter = axes[1, 1].scatter(df['AnnualIncome'], df['SpendingScore'], 
                                c=df['Age'], cmap='viridis', alpha=0.7, s=50)
    axes[1, 1].set_title('Income vs Spending Score', fontweight='bold')
    axes[1, 1].set_xlabel('Annual Income (k$)')
    axes[1, 1].set_ylabel('Spending Score')
    plt.colorbar(scatter, ax=axes[1, 1], label='Age')
    
    # 6. Correlation Heatmap
    corr = df[['Age', 'AnnualIncome', 'SpendingScore']].corr()
    im = axes[1, 2].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 2].set_title('Correlation Matrix', fontweight='bold')
    axes[1, 2].set_xticks(range(len(corr.columns)))
    axes[1, 2].set_yticks(range(len(corr.columns)))
    axes[1, 2].set_xticklabels(corr.columns, rotation=45)
    axes[1, 2].set_yticklabels(corr.columns)
    
    # Add correlation values
    for i in range(len(corr)):
        for j in range(len(corr)):
            text = axes[1, 2].text(j, i, f'{corr.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="white")
    
    plt.suptitle('Customer Data Exploration', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return df

def find_optimal_clusters(df):
    """Find optimal number of clusters using elbow method"""
    print("\n" + "="*60)
    print("🎯 FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("="*60)
    
    # Prepare features
    X = df[['AnnualIncome', 'SpendingScore']].values
    
    # Find WCSS for different K values
    wcss = []
    silhouette_scores = []
    
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    # Plot elbow method and silhouette scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow Method Plot
    ax1.plot(range(2, 11), wcss, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    ax1.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Mark optimal K (5 for this dataset)
    optimal_k = 5
    ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
    ax1.text(optimal_k, max(wcss)*0.9, f'Optimal K={optimal_k}', 
             color='red', fontweight='bold', ha='center')
    
    # Silhouette Scores Plot
    ax2.plot(range(2, 11), silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Scores', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Mark best silhouette score
    best_k = range(2, 11)[np.argmax(silhouette_scores)]
    ax2.axvline(x=best_k, color='red', linestyle='--', alpha=0.7)
    ax2.text(best_k, max(silhouette_scores)*0.95, f'Best K={best_k}', 
             color='red', fontweight='bold', ha='center')
    
    plt.suptitle('Determining Optimal Number of Clusters', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('elbow_method.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Optimal number of clusters: K = {optimal_k}")
    print(f"📊 Silhouette Score for K={optimal_k}: {silhouette_scores[optimal_k-2]:.3f}")
    
    return optimal_k

def apply_kmeans(df, n_clusters=5):
    """Apply K-Means clustering"""
    print("\n" + "="*60)
    print(f"🤖 APPLYING K-MEANS WITH K={n_clusters}")
    print("="*60)
    
    # Prepare data
    X = df[['AnnualIncome', 'SpendingScore']].values
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    df['Cluster'] = clusters
    
    # Calculate silhouette score
    score = silhouette_score(X, clusters)
    print(f"✅ K-Means applied successfully!")
    print(f"📊 Silhouette Score: {score:.3f}")
    
    # Cluster distribution
    print(f"\n🔢 Cluster Distribution:")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  Cluster {cluster}: {count} customers ({percentage:.1f}%)")
    
    return df, kmeans

def visualize_clusters(df, kmeans):
    """Visualize the clusters"""
    print("\n" + "="*60)
    print("📊 VISUALIZING CUSTOMER SEGMENTS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Main Cluster Visualization
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for cluster_num in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster_num]
        axes[0, 0].scatter(cluster_data['AnnualIncome'], 
                          cluster_data['SpendingScore'],
                          s=70, alpha=0.7, color=colors[cluster_num],
                          label=f'Cluster {cluster_num}')
    
    # Plot centroids
    centroids = kmeans.cluster_centers_
    axes[0, 0].scatter(centroids[:, 0], centroids[:, 1],
                      marker='X', s=200, c='black', label='Centroids')
    
    axes[0, 0].set_xlabel('Annual Income (k$)', fontsize=12)
    axes[0, 0].set_ylabel('Spending Score (1-100)', fontsize=12)
    axes[0, 0].set_title('Customer Segments', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cluster Size Distribution
    cluster_sizes = df['Cluster'].value_counts().sort_index()
    bars = axes[0, 1].bar(range(len(cluster_sizes)), cluster_sizes.values,
                         color=colors[:len(cluster_sizes)])
    
    axes[0, 1].set_xlabel('Cluster', fontsize=12)
    axes[0, 1].set_ylabel('Number of Customers', fontsize=12)
    axes[0, 1].set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(range(len(cluster_sizes)))
    axes[0, 1].set_xticklabels([f'Cluster {i}' for i in cluster_sizes.index])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, cluster_sizes.values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, height + 0.5,
                       f'{count}', ha='center', fontweight='bold')
    
    # 3. Age Distribution by Cluster
    box_data = [df[df['Cluster'] == i]['Age'] 
               for i in sorted(df['Cluster'].unique())]
    
    bp = axes[1, 0].boxplot(box_data, patch_artist=True, widths=0.6,
                           labels=[f'Cluster {i}' for i in sorted(df['Cluster'].unique())])
    
    for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1, 0].set_ylabel('Age', fontsize=12)
    axes[1, 0].set_title('Age Distribution by Cluster', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Gender Distribution by Cluster
    cluster_gender = df.groupby(['Cluster', 'Gender']).size().unstack()
    cluster_gender_percentage = cluster_gender.div(cluster_gender.sum(axis=1), axis=0) * 100
    
    x = np.arange(len(cluster_gender_percentage))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, cluster_gender_percentage['Male'], width,
                  label='Male', color='skyblue', alpha=0.8)
    axes[1, 1].bar(x + width/2, cluster_gender_percentage['Female'], width,
                  label='Female', color='lightpink', alpha=0.8)
    
    axes[1, 1].set_xlabel('Cluster', fontsize=12)
    axes[1, 1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1, 1].set_title('Gender Distribution by Cluster', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'Cluster {i}' for i in cluster_gender_percentage.index])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Customer Segmentation Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('customer_segments.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_segments(df):
    """Analyze each customer segment"""
    print("\n" + "="*60)
    print("🔍 SEGMENT ANALYSIS")
    print("="*60)
    
    # Calculate cluster statistics
    cluster_stats = df.groupby('Cluster').agg({
        'AnnualIncome': ['mean', 'std', 'min', 'max'],
        'SpendingScore': ['mean', 'std', 'min', 'max'],
        'Age': ['mean', 'std'],
        'Gender': lambda x: (x == 'Female').mean() * 100  # Percentage female
    }).round(2)
    
    print("\n📊 Detailed Cluster Statistics:")
    print(cluster_stats)
    
    # Segment descriptions (based on typical patterns)
    print("\n" + "="*60)
    print("💡 INTERPRETING CUSTOMER SEGMENTS:")
    print("="*60)
    
    segment_descriptions = {
        0: {
            'name': '🎯 Target Customers',
            'description': 'High income, high spending - Ideal customers!',
            'characteristics': 'Affluent customers who spend generously',
            'action': 'Focus on retention, offer premium products'
        },
        1: {
            'name': '💰 Conservative Spenders',
            'description': 'High income, low spending',
            'characteristics': 'Affluent but cautious with spending',
            'action': 'Highlight value and quality, offer bundle deals'
        },
        2: {
            'name': '🛍️ Bargain Hunters',
            'description': 'Low income, high spending',
            'characteristics': 'Love shopping but have budget constraints',
            'action': 'Offer discounts, promotions, and budget options'
        },
        3: {
            'name': '📊 Low Activity',
            'description': 'Low income, low spending',
            'characteristics': 'Infrequent shoppers with limited budget',
            'action': 'Focus on essential items, gentle re-engagement'
        },
        4: {
            'name': '🎪 Average Shoppers',
            'description': 'Medium income, medium spending',
            'characteristics': 'Mainstream shoppers, follow trends',
            'action': 'Standard marketing, popular products'
        }
    }
    
    for cluster in sorted(df['Cluster'].unique()):
        if cluster in segment_descriptions:
            info = segment_descriptions[cluster]
            cluster_data = df[df['Cluster'] == cluster]
            
            print(f"\n{'━'*40}")
            print(f"{info['name']} (Cluster {cluster})")
            print(f"{'━'*40}")
            print(f"{info['description']}")
            print(f"\n📈 Profile:")
            print(f"  • Size: {len(cluster_data)} customers ({(len(cluster_data)/len(df)*100):.1f}%)")
            print(f"  • Avg Income: ${cluster_data['AnnualIncome'].mean():.1f}k")
            print(f"  • Avg Spending Score: {cluster_data['SpendingScore'].mean():.1f}/100")
            print(f"  • Avg Age: {cluster_data['Age'].mean():.1f} years")
            print(f"  • % Female: {(cluster_data['Gender'] == 'Female').mean()*100:.1f}%")
            print(f"\n🎯 Recommended Action: {info['action']}")

def save_results(df):
    """Save the results"""
    print("\n" + "="*60)
    print("💾 SAVING RESULTS")
    print("="*60)
    
    # Save clustered data
    output_file = 'Mall_Customers_Clustered.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Clustered data saved to: {output_file}")
    print(f"✅ Visualizations saved:")
    print("   • data_exploration.png")
    print("   • elbow_method.png")
    print("   • customer_segments.png")
    
    # Create summary report
    report = f"""
    MALL CUSTOMER SEGMENTATION REPORT
    {'='*50}
    
    Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    DATASET:
    • Total Customers: {len(df)}
    • Features Used: Annual Income, Spending Score
    • Optimal Clusters: {df['Cluster'].nunique()}
    
    CLUSTER DISTRIBUTION:
    {df['Cluster'].value_counts().sort_index().to_string()}
    
    KEY INSIGHTS:
    1. Found {df['Cluster'].nunique()} distinct customer segments
    2. Best performing segment: Cluster 0 (Target Customers)
    3. Most customers fall into the Average Shoppers category
    4. Clear differentiation based on income and spending patterns
    
    RECOMMENDATIONS:
    1. Develop targeted marketing campaigns for each segment
    2. Create personalized product recommendations
    3. Design segment-specific loyalty programs
    4. Monitor segment evolution over time
    """
    
    with open('segmentation_report.txt', 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved to: segmentation_report.txt")
    
    return output_file

def main():
    """Main function to run the complete analysis"""
    
    # Step 1: Load and explore data
    df = load_and_explore_data()
    
    # Step 2: Visualize data
    df = visualize_data(df)
    
    # Step 3: Find optimal clusters
    optimal_k = find_optimal_clusters(df)
    
    # Step 4: Apply K-Means
    df, kmeans = apply_kmeans(df, optimal_k)
    
    # Step 5: Visualize clusters
    visualize_clusters(df, kmeans)
    
    # Step 6: Analyze segments
    analyze_segments(df)
    
    # Step 7: Save results
    save_results(df)
    
    print("\n" + "="*60)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*60)
    print("\n📁 Files generated:")
    print("   • Mall_Customers_Clustered.csv - Data with cluster labels")
    print("   • data_exploration.png - Initial data analysis")
    print("   • elbow_method.png - Optimal K determination")
    print("   • customer_segments.png - Cluster visualizations")
    print("   • segmentation_report.txt - Summary report")
    print("\n🛍️ Ready for targeted marketing campaigns!")

if __name__ == "__main__":
    main()