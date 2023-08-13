import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('/mnt/data/all_metrics.csv')

# Drop 'Unnamed: 0' column
df = df.drop(columns='Unnamed: 0')

# Replace NaN values with 0
df = df.fillna(0)

# Create a pivot table with Models as rows, Datasets as columns, and mean Accuracy as values
heatmap_data = df.pivot_table(values='Accuracy', index='Model', columns='Dataset')

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5)
plt.title('Heatmap of Model Performance (Accuracy)')
plt.show()

# Select the columns to plot
cols_to_plot = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-score (macro)', 'AUC-ROC (macro)']

# Melt the dataframe to have one row per model per metric
df_melted = df.melt(id_vars='Model', value_vars=cols_to_plot, var_name='Metric', value_name='Score')

# Plot the boxplots
plt.figure(figsize=(15, 10))
sns.boxplot(data=df_melted, x='Score', y='Metric', hue='Model')
plt.title('Distribution of Performance Metrics for Each Model')
plt.legend(loc='lower right')
plt.show()

# Calculate the correlation matrix for the selected metrics
correlation_matrix = df[cols_to_plot].corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
plt.title('Correlation Matrix of Performance Metrics')
plt.show()

# Perform PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(df[cols_to_include])

# Create a dataframe with the PCA results
df_pca = pd.DataFrame(data=pca_results, columns=['PC1', 'PC2'])
df_pca['Model'] = df['Model']

# Plot the PCA results
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Model')
plt.title('PCA of Model Performance')
plt.show()

# Calculate the mean ranking for each model across all metrics
rankings = df[cols_to_plot].rank(method='min').mean(axis=1)
df['Ranking'] = rankings

# Group by model and calculate the mean ranking for each model
model_rankings = df.groupby('Model')['Ranking'].mean()

# Sort by ranking
model_rankings = model_rankings.sort_values()

# Plot the rankings
model_rankings.plot(kind='barh', figsize=(10, 6))
plt.title('Overall Ranking of Models')
plt.xlabel('Mean Ranking')
plt.ylabel('Model')
plt.show()

# Perform ANOVA
anova_results = stats.f_oneway(*(df[df['Model'] == model]['Accuracy'] for model in df['Model'].unique()))

print(anova_results)

def plot_model_performance(model, metric):
    """
    Plot the performance of a selected model on a selected metric.
    
    Parameters:
    model (str): The name of the model.
    metric (str): The name of the metric.
    """
    # Select the data for the model and metric
    data = df[df['Model'] == model][metric]
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    sns.barplot(x=data.index, y=data, color='skyblue')
    plt.title(f'{model} Performance on {metric}')
    plt.xlabel('Run')
    plt.ylabel(metric)
    plt.show()

# Test the function with the 'RandomForest' model and 'Accuracy' metric
plot_model_performance('RandomForest', 'Accuracy')