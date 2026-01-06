/**
 * Data Analysis Tools - Sidebar components for Data Analysis mode
 */

// ===========================
// Dataset Definitions
// ===========================

const DATASETS = {
    kaggle: [
        { id: 'titanic', name: '🚢 Titanic', size: '891 rows', url: 'https://www.kaggle.com/c/titanic' },
        { id: 'housing', name: '🏠 Housing Prices', size: '1,460 rows', url: 'https://www.kaggle.com/c/house-prices-advanced-regression-techniques' },
        { id: 'creditcard', name: '💳 Credit Card Fraud', size: '284k rows', url: 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud' },
        { id: 'heart', name: '❤️ Heart Disease', size: '303 rows', url: 'https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset' },
        { id: 'mnist', name: '✍️ MNIST Digits', size: '70k images', url: 'https://www.kaggle.com/c/digit-recognizer' },
        { id: 'stock', name: '📈 Stock Market', size: '1M+ rows', url: 'https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset' }
    ],
    uci: [
        { id: 'iris', name: '🌸 Iris', size: '150 rows', url: 'https://archive.ics.uci.edu/ml/datasets/iris' },
        { id: 'wine', name: '🍷 Wine Quality', size: '4,898 rows', url: 'https://archive.ics.uci.edu/ml/datasets/wine+quality' },
        { id: 'adult', name: '👤 Adult Census', size: '48,842 rows', url: 'https://archive.ics.uci.edu/ml/datasets/adult' },
        { id: 'car', name: '🚗 Car Evaluation', size: '1,728 rows', url: 'https://archive.ics.uci.edu/ml/datasets/car+evaluation' },
        { id: 'mushroom', name: '🍄 Mushroom', size: '8,124 rows', url: 'https://archive.ics.uci.edu/ml/datasets/mushroom' }
    ],
    sklearn: [
        { id: 'iris', name: '🌸 Iris', size: '150 rows', code: 'from sklearn.datasets import load_iris' },
        { id: 'digits', name: '🔢 Digits', size: '1,797 images', code: 'from sklearn.datasets import load_digits' },
        { id: 'diabetes', name: '🩺 Diabetes', size: '442 rows', code: 'from sklearn.datasets import load_diabetes' },
        { id: 'wine', name: '🍷 Wine', size: '178 rows', code: 'from sklearn.datasets import load_wine' },
        { id: 'breast_cancer', name: '🎗️ Breast Cancer', size: '569 rows', code: 'from sklearn.datasets import load_breast_cancer' }
    ],
    huggingface: [
        { id: 'imdb', name: '🎬 IMDB Reviews', size: '50k rows', url: 'https://huggingface.co/datasets/imdb' },
        { id: 'squad', name: '❓ SQuAD', size: '100k QA pairs', url: 'https://huggingface.co/datasets/squad' },
        { id: 'mnist', name: '✍️ MNIST', size: '70k images', url: 'https://huggingface.co/datasets/mnist' },
        { id: 'glue', name: '📚 GLUE', size: 'Multiple', url: 'https://huggingface.co/datasets/glue' },
        { id: 'wikipedia', name: '📖 Wikipedia', size: '~6M articles', url: 'https://huggingface.co/datasets/wikipedia' }
    ]
};

// ===========================
// Code Snippets for Data Analysis
// ===========================

const DATA_SNIPPETS = {
    import_pandas: `import pandas as pd
import numpy as np

# Pandas is the go-to library for data analysis
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)

# Quick demo - create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['NYC', 'LA', 'Chicago', 'Houston'],
    'Salary': [50000, 60000, 70000, 55000]
}
df = pd.DataFrame(data)
print("\\nSample DataFrame:")
print(df)`,

    load_kaggle: `# Load dataset from Kaggle
# First, install kagglehub: pip install kagglehub

import kagglehub

# Download a dataset
# path = kagglehub.dataset_download("username/dataset-name")

# For Titanic dataset:
# path = kagglehub.dataset_download("heptapod/titanic")

# Alternative: Use sklearn for common datasets
from sklearn.datasets import load_iris, load_wine, load_diabetes

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
print(f"Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {iris.target_names}")`,

    eda_basic: `import pandas as pd
import numpy as np

# Basic Exploratory Data Analysis template

def perform_eda(df):
    """Comprehensive EDA on a DataFrame"""
    
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    print("\\n" + "=" * 50)
    print("DATA TYPES")
    print("=" * 50)
    print(df.dtypes)
    
    print("\\n" + "=" * 50)
    print("MISSING VALUES")
    print("=" * 50)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing'] > 0])
    
    print("\\n" + "=" * 50)
    print("STATISTICAL SUMMARY")
    print("=" * 50)
    print(df.describe())
    
    print("\\n" + "=" * 50)
    print("UNIQUE VALUES PER COLUMN")
    print("=" * 50)
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

# Example usage with sample data
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': ['x', 'y', 'x', 'z', 'y'],
    'C': [1.1, 2.2, 3.3, 4.4, 5.5]
}
df = pd.DataFrame(data)
perform_eda(df)`,

    data_cleaning: `import pandas as pd
import numpy as np

# Data Cleaning Pipeline

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.log = []
    
    def handle_missing(self, strategy='mean', columns=None):
        """Handle missing values"""
        cols = columns or self.df.select_dtypes(include=[np.number]).columns
        
        for col in cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    self.log.append(f"Dropped rows with missing {col}")
                    continue
                else:
                    fill_value = strategy  # Custom value
                
                self.df[col].fillna(fill_value, inplace=True)
                self.log.append(f"Filled {col} with {strategy}: {fill_value:.2f}")
        
        return self
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        initial = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial - len(self.df)
        self.log.append(f"Removed {removed} duplicate rows")
        return self
    
    def remove_outliers(self, columns=None, method='iqr', threshold=1.5):
        """Remove outliers using IQR method"""
        cols = columns or self.df.select_dtypes(include=[np.number]).columns
        
        for col in cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            
            initial = len(self.df)
            self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
            removed = initial - len(self.df)
            if removed > 0:
                self.log.append(f"Removed {removed} outliers from {col}")
        
        return self
    
    def get_cleaned_data(self):
        print("Cleaning Log:")
        for entry in self.log:
            print(f"  • {entry}")
        return self.df

# Example usage
data = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 100, 6],
    'B': [10, 20, 30, 40, 50, 60]
})
print("Original data:")
print(data)

cleaner = DataCleaner(data)
cleaned = cleaner.handle_missing('mean').remove_outliers().get_cleaned_data()
print("\\nCleaned data:")
print(cleaned)`,

    visualization: `import numpy as np

# Data Visualization with Matplotlib (browser-compatible)
# Note: Full matplotlib may not be available in browser
# Using text-based visualization as fallback

def text_histogram(data, bins=10, width=40):
    """Create a text-based histogram"""
    hist, edges = np.histogram(data, bins=bins)
    max_count = max(hist)
    
    print("\\nHistogram:")
    print("-" * (width + 15))
    
    for i, count in enumerate(hist):
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = "█" * bar_len
        label = f"{edges[i]:.1f}-{edges[i+1]:.1f}"
        print(f"{label:>10} | {bar} ({count})")
    
    print("-" * (width + 15))

def text_correlation(df):
    """Display correlation matrix"""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    print("\\nCorrelation Matrix:")
    print(corr.round(2).to_string())

# Example
import pandas as pd

np.random.seed(42)
data = {
    'feature1': np.random.normal(50, 10, 100),
    'feature2': np.random.normal(30, 5, 100),
    'target': np.random.normal(100, 20, 100)
}
df = pd.DataFrame(data)

print("Data Statistics:")
print(df.describe())

text_histogram(df['feature1'])
text_correlation(df)`,

    clustering: `import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# K-Means Clustering Example

# Generate sample data
np.random.seed(42)
n_samples = 300

# Create 3 clusters
cluster1 = np.random.randn(n_samples//3, 2) + [2, 2]
cluster2 = np.random.randn(n_samples//3, 2) + [-2, -2]
cluster3 = np.random.randn(n_samples//3, 2) + [2, -2]

X = np.vstack([cluster1, cluster2, cluster3])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k using elbow method
print("Finding optimal number of clusters...")
inertias = []
silhouettes = []

for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.3f}")

# Fit final model with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

print(f"\\nFinal clustering with k=3:")
print(f"Cluster sizes: {np.bincount(labels)}")
print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.3f}")
print(f"Cluster Centers:\\n{kmeans.cluster_centers_}")`
};

// ===========================
// Dataset Browser
// ===========================

const DatasetBrowser = {
    currentSource: 'kaggle',

    init() {
        this.setupSourceButtons();
        this.setupSearch();
        this.setupDatasetItems();
        this.render();
    },

    setupSourceButtons() {
        document.querySelectorAll('.dataset-source-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.dataset-source-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentSource = btn.dataset.source;
                this.render();
            });
        });
    },

    setupSearch() {
        const searchInput = document.getElementById('datasetSearch');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.render(e.target.value);
            });
        }
    },

    setupDatasetItems() {
        document.getElementById('datasetList')?.addEventListener('click', (e) => {
            const item = e.target.closest('.dataset-item');
            if (item) {
                const datasetId = item.dataset.dataset;
                this.loadDataset(datasetId);
            }
        });
    },

    render(filter = '') {
        const container = document.getElementById('datasetList');
        if (!container) return;

        const datasets = DATASETS[this.currentSource] || [];
        const filtered = datasets.filter(d =>
            d.name.toLowerCase().includes(filter.toLowerCase()) ||
            d.id.toLowerCase().includes(filter.toLowerCase())
        );

        container.innerHTML = filtered.map(d => `
            <div class="dataset-item" data-dataset="${d.id}">
                <span class="dataset-name">${d.name}</span>
                <span class="dataset-size">${d.size}</span>
            </div>
        `).join('');
    },

    loadDataset(datasetId) {
        const datasets = DATASETS[this.currentSource];
        const dataset = datasets?.find(d => d.id === datasetId);

        if (!dataset) return;

        let code = '';

        if (this.currentSource === 'sklearn') {
            code = `# Load ${dataset.name} from sklearn
${dataset.code}

data = ${dataset.code.split(' ')?.pop()?.replace('load_', '')}()
X, y = data.data, data.target

print(f"Dataset: {dataset.name}")
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {len(set(y))}")`;
        } else if (this.currentSource === 'kaggle') {
            code = `# Load ${dataset.name} from Kaggle
# Install kagglehub: pip install kagglehub

import kagglehub
import pandas as pd

# Dataset URL: ${dataset.url}
# Download and load:
# path = kagglehub.dataset_download("dataset-path")
# df = pd.read_csv(path + "/train.csv")

print("Dataset: ${dataset.name}")
print("Info: ${dataset.size}")`;
        } else if (this.currentSource === 'huggingface') {
            code = `# Load ${dataset.name} from HuggingFace
# Install datasets: pip install datasets

from datasets import load_dataset

# Dataset URL: ${dataset.url}
dataset = load_dataset("${dataset.id}")

print("Dataset: ${dataset.name}")
print(dataset)`;
        } else {
            code = `# Load ${dataset.name} from UCI ML Repository
# URL: ${dataset.url}

import pandas as pd

# Download from: ${dataset.url}
# df = pd.read_csv("path/to/data.csv")

print("Dataset: ${dataset.name}")
print("Size: ${dataset.size}")`;
        }

        if (window.QMLApp) {
            window.QMLApp.insertSnippet(code);
            window.QMLApp.logToConsole(`Loaded dataset template: ${dataset.name}`, 'info');
        }
    }
};

// ===========================
// Mining Tools
// ===========================

const MiningTools = {
    init() {
        this.setupToolButtons();
    },

    setupToolButtons() {
        document.querySelectorAll('#dataSidebar .tool-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tool = btn.dataset.tool;
                this.insertToolCode(tool);
            });
        });
    },

    insertToolCode(tool) {
        const toolCodes = {
            missing: `# Handle Missing Values
import pandas as pd
import numpy as np

def handle_missing(df, strategy='mean'):
    """
    Handle missing values in DataFrame
    Strategies: 'mean', 'median', 'mode', 'drop', 'ffill', 'bfill'
    """
    df = df.copy()
    
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                if strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
            else:
                if strategy == 'mode':
                    df[col].fillna(df[col].mode()[0], inplace=True)
            
            if strategy == 'drop':
                df = df.dropna(subset=[col])
            elif strategy in ['ffill', 'bfill']:
                df[col].fillna(method=strategy, inplace=True)
    
    return df

# Example
print("Missing values handler ready!")`,

            normalize: `# Data Normalization
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_data(X, method='minmax'):
    """
    Normalize data using different methods
    Methods: 'minmax' (0-1), 'zscore' (mean=0, std=1)
    """
    if method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    return scaler.fit_transform(X)

# Example
data = np.array([[1, 2], [3, 4], [5, 6], [100, 200]])
print("Original:", data.flatten())
print("Normalized:", normalize_data(data).flatten())`,

            encode: `# Categorical Encoding
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def encode_categorical(df, columns=None, method='label'):
    """
    Encode categorical variables
    Methods: 'label', 'onehot', 'ordinal'
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    
    for col in columns:
        if method == 'label':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        elif method == 'onehot':
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
    
    return df

# Example
data = pd.DataFrame({'color': ['red', 'blue', 'green', 'red']})
print("Original:", data['color'].tolist())
print("Encoded:", encode_categorical(data)['color'].tolist())`,

            scale: `# Feature Scaling
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np

# StandardScaler: mean=0, std=1
# RobustScaler: uses median and IQR (better for outliers)

data = np.array([[1, 2], [3, 4], [5, 6], [1000, 2000]])

standard = StandardScaler().fit_transform(data)
robust = RobustScaler().fit_transform(data)

print("Original:", data.flatten())
print("Standard Scaled:", standard.flatten().round(2))
print("Robust Scaled:", robust.flatten().round(2))`,

            pca: `# Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate sample high-dimensional data
np.random.seed(42)
X = np.random.randn(100, 10)  # 100 samples, 10 features

# Standardize first
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)

print(f"Original dimensions: {X.shape}")
print(f"Reduced dimensions: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.round(3)}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.1%}")`,

            selection: `# Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 20)  # 20 features
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Only first 2 are relevant

# Method 1: SelectKBest
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)
print(f"SelectKBest indices: {np.argsort(selector.scores_)[-5:]}")

# Method 2: Feature importance from Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
top_features = np.argsort(importances)[-5:]
print(f"RF top features: {top_features}")`,

            poly: `# Polynomial Features
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Create polynomial and interaction features
X = np.array([[1, 2], [3, 4], [5, 6]])

# Degree 2: includes x^2, xy, y^2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print("Original features:", X.shape[1])
print("Polynomial features:", X_poly.shape[1])
print("Feature names:", poly.get_feature_names_out())`,

            kmeans: `# K-Means Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.vstack([
    np.random.randn(100, 2) + [2, 2],
    np.random.randn(100, 2) + [-2, -2],
    np.random.randn(100, 2) + [2, -2]
])

# Fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

print(f"Cluster sizes: {np.bincount(labels)}")
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X, labels):.3f}")`,

            dbscan: `# DBSCAN Clustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate sample data with noise
np.random.seed(42)
X = np.vstack([
    np.random.randn(100, 2) + [2, 2],
    np.random.randn(100, 2) + [-2, -2],
    np.random.randn(20, 2) * 5  # Noise
])

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise}")`,

            hierarchical: `# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

np.random.seed(42)
X = np.random.randn(20, 2)

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)

print(f"Cluster assignments: {labels}")
print(f"Cluster sizes: {np.bincount(labels)}")

# Linkage matrix for dendrogram
Z = linkage(X, method='ward')
print(f"\\nLinkage matrix shape: {Z.shape}")
print("(Use scipy dendrogram to visualize)")`,

            apriori: `# Association Rule Mining - Apriori
# Requires: pip install mlxtend

from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Sample transaction data
transactions = pd.DataFrame({
    'bread': [1, 1, 0, 1, 1],
    'butter': [1, 0, 1, 1, 0],
    'milk': [1, 1, 1, 0, 1],
    'eggs': [0, 1, 1, 1, 0],
    'cheese': [1, 0, 0, 1, 1]
})

# Find frequent itemsets
frequent = apriori(transactions, min_support=0.4, use_colnames=True)
print("Frequent Itemsets:")
print(frequent)

# Generate association rules
rules = association_rules(frequent, metric='lift', min_threshold=1.0)
print("\\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])`,

            fpgrowth: `# FP-Growth Algorithm
# Requires: pip install mlxtend

from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd

# Sample transaction data
transactions = pd.DataFrame({
    'bread': [1, 1, 0, 1, 1, 0, 1],
    'butter': [1, 0, 1, 1, 0, 1, 1],
    'milk': [1, 1, 1, 0, 1, 1, 0],
    'eggs': [0, 1, 1, 1, 0, 1, 1]
})

# FP-Growth is faster than Apriori for large datasets
frequent = fpgrowth(transactions, min_support=0.3, use_colnames=True)
print("Frequent Itemsets (FP-Growth):")
print(frequent)

# Generate rules
rules = association_rules(frequent, metric='confidence', min_threshold=0.6)
print("\\nStrong Association Rules:")
print(rules[['antecedents', 'consequents', 'confidence', 'lift']].head())`
        };

        const code = toolCodes[tool];
        if (code && window.QMLApp) {
            window.QMLApp.insertSnippet(code);
            window.QMLApp.logToConsole(`Inserted ${tool} code template`, 'info');
        }
    }
};

// ===========================
// EDA Preview Chart
// ===========================

const EDAPreview = {
    canvas: null,
    ctx: null,

    init() {
        this.canvas = document.getElementById('edaChart');
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.renderSampleChart();
    },

    renderSampleChart() {
        if (!this.ctx) return;

        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear
        this.ctx.fillStyle = '#1a1d24';
        this.ctx.fillRect(0, 0, width, height);

        // Draw sample histogram
        const data = [15, 35, 55, 45, 30, 20, 10, 25, 40, 50];
        const maxVal = Math.max(...data);
        const barWidth = (width - 40) / data.length;
        const padding = 20;

        // Gradient
        const gradient = this.ctx.createLinearGradient(0, height, 0, 0);
        gradient.addColorStop(0, '#10b981');
        gradient.addColorStop(1, '#3b82f6');

        this.ctx.fillStyle = gradient;

        data.forEach((val, i) => {
            const barHeight = (val / maxVal) * (height - padding * 2);
            const x = padding + i * barWidth;
            const y = height - padding - barHeight;

            this.ctx.beginPath();
            this.ctx.roundRect(x + 2, y, barWidth - 4, barHeight, 2);
            this.ctx.fill();
        });

        // Axis
        this.ctx.strokeStyle = '#4b5563';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.moveTo(padding, padding);
        this.ctx.lineTo(padding, height - padding);
        this.ctx.lineTo(width - padding, height - padding);
        this.ctx.stroke();
    },

    updateStats(rows, cols, missing) {
        const rowsEl = document.getElementById('statRows');
        const colsEl = document.getElementById('statCols');
        const missingEl = document.getElementById('statMissing');

        if (rowsEl) rowsEl.textContent = rows?.toLocaleString() || '-';
        if (colsEl) colsEl.textContent = cols || '-';
        if (missingEl) missingEl.textContent = missing || '-';
    }
};

// ===========================
// Initialize
// ===========================

function initializeDataTools() {
    console.log('Initializing Data Analysis Tools...');

    DatasetBrowser.init();
    MiningTools.init();
    EDAPreview.init();

    // Add data snippets to global snippets
    if (window.CODE_SNIPPETS) {
        Object.assign(window.CODE_SNIPPETS, DATA_SNIPPETS);
    }

    console.log('Data Analysis Tools initialized!');
}

// Export
window.DataTools = {
    DatasetBrowser,
    MiningTools,
    EDAPreview,
    DATASETS,
    DATA_SNIPPETS,
    init: initializeDataTools
};
