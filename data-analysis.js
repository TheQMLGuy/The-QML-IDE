/**
 * Data Analysis Module - Deep Data Mining & Analysis
 * Comprehensive EDA and data mining inspired by best practices
 */

// ===========================
// Analysis State
// ===========================

const AnalysisState = {
    currentData: null,
    currentColumns: [],
    results: null,
    isAnalyzing: false
};

// ===========================
// Statistical Analysis
// ===========================

const Statistics = {
    mean(arr) {
        if (!arr || arr.length === 0) return null;
        const nums = arr.filter(x => typeof x === 'number' && !isNaN(x));
        return nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
    },

    median(arr) {
        const nums = arr.filter(x => typeof x === 'number' && !isNaN(x)).sort((a, b) => a - b);
        if (nums.length === 0) return null;
        const mid = Math.floor(nums.length / 2);
        return nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
    },

    mode(arr) {
        const counts = {};
        arr.forEach(x => counts[x] = (counts[x] || 0) + 1);
        let maxCount = 0, mode = null;
        Object.entries(counts).forEach(([val, count]) => {
            if (count > maxCount) { maxCount = count; mode = val; }
        });
        return mode;
    },

    std(arr) {
        const nums = arr.filter(x => typeof x === 'number' && !isNaN(x));
        if (nums.length < 2) return 0;
        const m = this.mean(nums);
        const variance = nums.reduce((acc, x) => acc + (x - m) ** 2, 0) / (nums.length - 1);
        return Math.sqrt(variance);
    },

    variance(arr) {
        const s = this.std(arr);
        return s * s;
    },

    skewness(arr) {
        const nums = arr.filter(x => typeof x === 'number' && !isNaN(x));
        if (nums.length < 3) return 0;
        const m = this.mean(nums);
        const s = this.std(nums);
        if (s === 0) return 0;
        const n = nums.length;
        const sum = nums.reduce((acc, x) => acc + ((x - m) / s) ** 3, 0);
        return (n / ((n - 1) * (n - 2))) * sum;
    },

    kurtosis(arr) {
        const nums = arr.filter(x => typeof x === 'number' && !isNaN(x));
        if (nums.length < 4) return 0;
        const m = this.mean(nums);
        const s = this.std(nums);
        if (s === 0) return 0;
        const n = nums.length;
        const sum = nums.reduce((acc, x) => acc + ((x - m) / s) ** 4, 0);
        return ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * sum -
            (3 * (n - 1) ** 2) / ((n - 2) * (n - 3));
    },

    percentile(arr, p) {
        const nums = arr.filter(x => typeof x === 'number' && !isNaN(x)).sort((a, b) => a - b);
        if (nums.length === 0) return null;
        const idx = (p / 100) * (nums.length - 1);
        const lower = Math.floor(idx);
        const upper = Math.ceil(idx);
        if (lower === upper) return nums[lower];
        return nums[lower] + (nums[upper] - nums[lower]) * (idx - lower);
    },

    quartiles(arr) {
        return {
            q1: this.percentile(arr, 25),
            q2: this.percentile(arr, 50),
            q3: this.percentile(arr, 75)
        };
    },

    min(arr) {
        const nums = arr.filter(x => typeof x === 'number' && !isNaN(x));
        return nums.length > 0 ? Math.min(...nums) : null;
    },

    max(arr) {
        const nums = arr.filter(x => typeof x === 'number' && !isNaN(x));
        return nums.length > 0 ? Math.max(...nums) : null;
    },

    range(arr) {
        const minVal = this.min(arr);
        const maxVal = this.max(arr);
        return minVal !== null && maxVal !== null ? maxVal - minVal : null;
    },

    iqr(arr) {
        const q = this.quartiles(arr);
        return q.q3 !== null && q.q1 !== null ? q.q3 - q.q1 : null;
    },

    // Correlation coefficients
    pearsonCorrelation(x, y) {
        if (x.length !== y.length || x.length < 2) return 0;
        const n = x.length;
        const meanX = this.mean(x);
        const meanY = this.mean(y);
        let num = 0, denX = 0, denY = 0;
        for (let i = 0; i < n; i++) {
            const dx = x[i] - meanX;
            const dy = y[i] - meanY;
            num += dx * dy;
            denX += dx * dx;
            denY += dy * dy;
        }
        const den = Math.sqrt(denX * denY);
        return den === 0 ? 0 : num / den;
    }
};

// ===========================
// Data Quality Assessment
// ===========================

const DataQuality = {
    assessMissingValues(data, columns) {
        const results = {};
        columns.forEach(col => {
            const colData = data.map(row => row[col]);
            const missing = colData.filter(x => x === null || x === undefined || x === '' ||
                (typeof x === 'number' && isNaN(x))).length;
            results[col] = {
                missing: missing,
                total: colData.length,
                percentage: ((missing / colData.length) * 100).toFixed(2)
            };
        });
        return results;
    },

    detectOutliersIQR(arr) {
        const nums = arr.filter(x => typeof x === 'number' && !isNaN(x));
        const q = Statistics.quartiles(nums);
        if (q.q1 === null || q.q3 === null) return { outliers: [], indices: [] };

        const iqr = q.q3 - q.q1;
        const lowerBound = q.q1 - 1.5 * iqr;
        const upperBound = q.q3 + 1.5 * iqr;

        const outliers = [];
        const indices = [];
        nums.forEach((val, idx) => {
            if (val < lowerBound || val > upperBound) {
                outliers.push(val);
                indices.push(idx);
            }
        });

        return { outliers, indices, lowerBound, upperBound };
    },

    detectOutliersZScore(arr, threshold = 3) {
        const nums = arr.filter(x => typeof x === 'number' && !isNaN(x));
        const mean = Statistics.mean(nums);
        const std = Statistics.std(nums);
        if (std === 0) return { outliers: [], indices: [] };

        const outliers = [];
        const indices = [];
        nums.forEach((val, idx) => {
            const zScore = Math.abs((val - mean) / std);
            if (zScore > threshold) {
                outliers.push(val);
                indices.push(idx);
            }
        });

        return { outliers, indices };
    },

    detectDuplicates(data) {
        const seen = new Map();
        const duplicates = [];

        data.forEach((row, idx) => {
            const key = JSON.stringify(row);
            if (seen.has(key)) {
                duplicates.push({ index: idx, duplicateOf: seen.get(key) });
            } else {
                seen.set(key, idx);
            }
        });

        return {
            count: duplicates.length,
            percentage: ((duplicates.length / data.length) * 100).toFixed(2),
            indices: duplicates
        };
    },

    inferDataType(arr) {
        const nonNull = arr.filter(x => x !== null && x !== undefined && x !== '');
        if (nonNull.length === 0) return 'unknown';

        const nums = nonNull.filter(x => typeof x === 'number' || !isNaN(parseFloat(x)));
        if (nums.length === nonNull.length) {
            const hasDecimals = nums.some(x => String(x).includes('.'));
            return hasDecimals ? 'float' : 'integer';
        }

        // Check for datetime
        const datePatterns = nonNull.filter(x => {
            const d = new Date(x);
            return !isNaN(d.getTime());
        });
        if (datePatterns.length > nonNull.length * 0.8) return 'datetime';

        // Check for categorical
        const unique = new Set(nonNull);
        if (unique.size < nonNull.length * 0.5 && unique.size < 50) return 'categorical';

        return 'string';
    }
};

// ===========================
// Correlation Analysis
// ===========================

const CorrelationAnalysis = {
    computeMatrix(data, numericColumns) {
        const matrix = {};

        numericColumns.forEach(col1 => {
            matrix[col1] = {};
            const arr1 = data.map(row => parseFloat(row[col1])).filter(x => !isNaN(x));

            numericColumns.forEach(col2 => {
                const arr2 = data.map(row => parseFloat(row[col2])).filter(x => !isNaN(x));
                // Align arrays (simple approach - use min length)
                const minLen = Math.min(arr1.length, arr2.length);
                const x = arr1.slice(0, minLen);
                const y = arr2.slice(0, minLen);
                matrix[col1][col2] = Statistics.pearsonCorrelation(x, y);
            });
        });

        return matrix;
    },

    findStrongCorrelations(matrix, threshold = 0.7) {
        const strong = [];
        const cols = Object.keys(matrix);

        for (let i = 0; i < cols.length; i++) {
            for (let j = i + 1; j < cols.length; j++) {
                const corr = matrix[cols[i]][cols[j]];
                if (Math.abs(corr) >= threshold) {
                    strong.push({
                        column1: cols[i],
                        column2: cols[j],
                        correlation: corr.toFixed(4),
                        strength: Math.abs(corr) >= 0.9 ? 'very strong' : 'strong'
                    });
                }
            }
        }

        return strong;
    }
};

// ===========================
// Feature Analysis
// ===========================

const FeatureAnalysis = {
    computeBasicImportance(data, columns, targetColumn) {
        if (!targetColumn || !columns.includes(targetColumn)) return {};

        const importance = {};
        const targetData = data.map(row => row[targetColumn]);

        columns.forEach(col => {
            if (col === targetColumn) return;
            const colData = data.map(row => row[col]);

            // For numeric columns, use correlation
            if (DataQuality.inferDataType(colData) === 'float' ||
                DataQuality.inferDataType(colData) === 'integer') {
                const nums = colData.map(x => parseFloat(x)).filter(x => !isNaN(x));
                const targets = targetData.slice(0, nums.length).map(x => parseFloat(x)).filter(x => !isNaN(x));
                const minLen = Math.min(nums.length, targets.length);
                importance[col] = Math.abs(Statistics.pearsonCorrelation(
                    nums.slice(0, minLen),
                    targets.slice(0, minLen)
                ));
            } else {
                // For categorical, use simple variance ratio
                importance[col] = 0.5; // Placeholder
            }
        });

        return importance;
    },

    computeInformationGain(data, column, targetColumn) {
        // Simplified information gain calculation
        const uniqueValues = [...new Set(data.map(row => row[column]))];
        const targetValues = [...new Set(data.map(row => row[targetColumn]))];

        // Calculate base entropy
        const baseEntropy = this.entropy(data.map(row => row[targetColumn]));

        // Calculate conditional entropy
        let condEntropy = 0;
        uniqueValues.forEach(val => {
            const subset = data.filter(row => row[column] === val);
            const weight = subset.length / data.length;
            condEntropy += weight * this.entropy(subset.map(row => row[targetColumn]));
        });

        return baseEntropy - condEntropy;
    },

    entropy(arr) {
        const counts = {};
        arr.forEach(x => counts[x] = (counts[x] || 0) + 1);
        let entropy = 0;
        const n = arr.length;
        Object.values(counts).forEach(count => {
            const p = count / n;
            if (p > 0) entropy -= p * Math.log2(p);
        });
        return entropy;
    }
};

// ===========================
// Pattern Discovery (Association Rules)
// ===========================

const PatternDiscovery = {
    getFrequentItemsets(transactions, minSupport) {
        const itemCounts = {};
        const n = transactions.length;

        // Count single items
        transactions.forEach(transaction => {
            transaction.forEach(item => {
                itemCounts[item] = (itemCounts[item] || 0) + 1;
            });
        });

        // Filter by min support
        const frequentItems = Object.entries(itemCounts)
            .filter(([_, count]) => count / n >= minSupport)
            .map(([item, count]) => ({
                items: [item],
                support: (count / n).toFixed(4),
                count: count
            }));

        return frequentItems;
    },

    generateAssociationRules(frequentItemsets, minConfidence) {
        const rules = [];

        // Simple rule generation for pairs
        frequentItemsets.forEach(itemset => {
            if (itemset.items.length >= 2) {
                const [antecedent, consequent] = itemset.items;
                const antecedentSupport = frequentItemsets.find(
                    i => i.items.length === 1 && i.items[0] === antecedent
                )?.support || 0;

                const confidence = itemset.support / antecedentSupport;
                if (confidence >= minConfidence) {
                    rules.push({
                        antecedent: [antecedent],
                        consequent: [consequent],
                        support: itemset.support,
                        confidence: confidence.toFixed(4)
                    });
                }
            }
        });

        return rules;
    }
};

// ===========================
// Clustering Exploration
// ===========================

const ClusteringExploration = {
    computeDistanceMatrix(data, columns) {
        const n = data.length;
        const matrix = Array(n).fill(null).map(() => Array(n).fill(0));

        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                let dist = 0;
                columns.forEach(col => {
                    const v1 = parseFloat(data[i][col]) || 0;
                    const v2 = parseFloat(data[j][col]) || 0;
                    dist += (v1 - v2) ** 2;
                });
                dist = Math.sqrt(dist);
                matrix[i][j] = dist;
                matrix[j][i] = dist;
            }
        }

        return matrix;
    },

    suggestOptimalK(data, columns, maxK = 10) {
        // Simple elbow method approximation
        const suggestions = [];
        // In practice, would compute within-cluster sum of squares for each k
        // This is a placeholder
        for (let k = 2; k <= Math.min(maxK, Math.floor(data.length / 2)); k++) {
            suggestions.push({ k, score: 1 / k }); // Simplified
        }
        return suggestions;
    }
};

// ===========================
// Distribution Analysis
// ===========================

const DistributionAnalysis = {
    computeHistogram(arr, bins = 10) {
        const nums = arr.filter(x => typeof x === 'number' && !isNaN(x));
        if (nums.length === 0) return { bins: [], counts: [] };

        const min = Math.min(...nums);
        const max = Math.max(...nums);
        const binWidth = (max - min) / bins;

        const binEdges = [];
        const counts = Array(bins).fill(0);

        for (let i = 0; i <= bins; i++) {
            binEdges.push(min + i * binWidth);
        }

        nums.forEach(val => {
            let binIdx = Math.floor((val - min) / binWidth);
            if (binIdx >= bins) binIdx = bins - 1;
            if (binIdx < 0) binIdx = 0;
            counts[binIdx]++;
        });

        return { binEdges, counts, binWidth };
    },

    computeValueCounts(arr, topN = 10) {
        const counts = {};
        arr.forEach(x => counts[x] = (counts[x] || 0) + 1);

        return Object.entries(counts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, topN)
            .map(([value, count]) => ({
                value,
                count,
                percentage: ((count / arr.length) * 100).toFixed(2)
            }));
    }
};

// ===========================
// Main Data Analysis Module
// ===========================

const DataAnalysis = {
    analyze(data, columns, options = {}) {
        console.log('Starting comprehensive data analysis...');
        AnalysisState.isAnalyzing = true;
        AnalysisState.currentData = data;
        AnalysisState.currentColumns = columns;

        const results = {
            overview: this.computeOverview(data, columns),
            descriptiveStats: this.computeDescriptiveStats(data, columns),
            dataQuality: this.computeDataQuality(data, columns),
            distributions: this.computeDistributions(data, columns),
            correlations: null,
            patterns: null
        };

        // Compute correlations only for numeric columns
        const numericCols = columns.filter(col => {
            const dtype = DataQuality.inferDataType(data.map(row => row[col]));
            return dtype === 'float' || dtype === 'integer';
        });

        if (numericCols.length >= 2) {
            results.correlations = {
                matrix: CorrelationAnalysis.computeMatrix(data, numericCols),
                strongCorrelations: CorrelationAnalysis.findStrongCorrelations(
                    CorrelationAnalysis.computeMatrix(data, numericCols)
                )
            };
        }

        AnalysisState.results = results;
        AnalysisState.isAnalyzing = false;

        // Update active tab with results
        if (typeof TabManager !== 'undefined') {
            const tab = TabManager.getActiveTab();
            if (tab) {
                tab.analysisResults = results;
            }
        }

        console.log('Data analysis complete!');
        return results;
    },

    computeOverview(data, columns) {
        const dtypes = {};
        columns.forEach(col => {
            dtypes[col] = DataQuality.inferDataType(data.map(row => row[col]));
        });

        return {
            rows: data.length,
            columns: columns.length,
            dtypes: dtypes,
            numericColumns: Object.entries(dtypes).filter(([_, t]) => t === 'float' || t === 'integer').map(([c]) => c),
            categoricalColumns: Object.entries(dtypes).filter(([_, t]) => t === 'categorical' || t === 'string').map(([c]) => c),
            memoryEstimate: `~${((JSON.stringify(data).length / 1024)).toFixed(2)} KB`
        };
    },

    computeDescriptiveStats(data, columns) {
        const stats = {};

        columns.forEach(col => {
            const colData = data.map(row => row[col]);
            const dtype = DataQuality.inferDataType(colData);

            if (dtype === 'float' || dtype === 'integer') {
                const nums = colData.map(x => parseFloat(x)).filter(x => !isNaN(x));
                const quartiles = Statistics.quartiles(nums);
                stats[col] = {
                    type: 'numeric',
                    count: nums.length,
                    mean: Statistics.mean(nums)?.toFixed(4),
                    std: Statistics.std(nums)?.toFixed(4),
                    min: Statistics.min(nums),
                    q1: quartiles.q1?.toFixed(4),
                    median: quartiles.q2?.toFixed(4),
                    q3: quartiles.q3?.toFixed(4),
                    max: Statistics.max(nums),
                    skewness: Statistics.skewness(nums)?.toFixed(4),
                    kurtosis: Statistics.kurtosis(nums)?.toFixed(4)
                };
            } else {
                const unique = [...new Set(colData)];
                stats[col] = {
                    type: 'categorical',
                    count: colData.length,
                    unique: unique.length,
                    top: Statistics.mode(colData),
                    topFreq: colData.filter(x => x === Statistics.mode(colData)).length,
                    valueDistribution: DistributionAnalysis.computeValueCounts(colData, 5)
                };
            }
        });

        return stats;
    },

    computeDataQuality(data, columns) {
        return {
            missingValues: DataQuality.assessMissingValues(data, columns),
            duplicates: DataQuality.detectDuplicates(data),
            outliers: this.detectAllOutliers(data, columns)
        };
    },

    detectAllOutliers(data, columns) {
        const outliers = {};

        columns.forEach(col => {
            const colData = data.map(row => row[col]);
            const dtype = DataQuality.inferDataType(colData);

            if (dtype === 'float' || dtype === 'integer') {
                const nums = colData.map(x => parseFloat(x)).filter(x => !isNaN(x));
                const iqrOutliers = DataQuality.detectOutliersIQR(nums);
                outliers[col] = {
                    method: 'IQR',
                    count: iqrOutliers.outliers.length,
                    percentage: ((iqrOutliers.outliers.length / nums.length) * 100).toFixed(2),
                    bounds: { lower: iqrOutliers.lowerBound?.toFixed(4), upper: iqrOutliers.upperBound?.toFixed(4) }
                };
            }
        });

        return outliers;
    },

    computeDistributions(data, columns) {
        const distributions = {};

        columns.forEach(col => {
            const colData = data.map(row => row[col]);
            const dtype = DataQuality.inferDataType(colData);

            if (dtype === 'float' || dtype === 'integer') {
                const nums = colData.map(x => parseFloat(x)).filter(x => !isNaN(x));
                distributions[col] = {
                    type: 'histogram',
                    data: DistributionAnalysis.computeHistogram(nums)
                };
            } else {
                distributions[col] = {
                    type: 'valueCounts',
                    data: DistributionAnalysis.computeValueCounts(colData)
                };
            }
        });

        return distributions;
    },

    // Advanced analysis (on-demand)
    runAdvancedAnalysis(data, columns, options = {}) {
        console.log('Running advanced analysis...');

        const results = {
            featureImportance: null,
            patterns: null,
            clusteringSuggestions: null
        };

        if (options.targetColumn) {
            results.featureImportance = FeatureAnalysis.computeBasicImportance(
                data, columns, options.targetColumn
            );
        }

        if (options.transactionColumn) {
            const transactions = data.map(row =>
                Object.values(row).filter(x => x !== null && x !== '')
            );
            results.patterns = {
                frequentItems: PatternDiscovery.getFrequentItemsets(transactions, options.minSupport || 0.1)
            };
        }

        const numericCols = columns.filter(col => {
            const dtype = DataQuality.inferDataType(data.map(row => row[col]));
            return dtype === 'float' || dtype === 'integer';
        });

        if (numericCols.length >= 2) {
            results.clusteringSuggestions = ClusteringExploration.suggestOptimalK(data, numericCols);
        }

        return results;
    },

    // Display results in sidebar
    displayResults(results) {
        if (!results) return;

        // Update EDA stats in sidebar
        const statRows = document.getElementById('statRows');
        const statCols = document.getElementById('statCols');
        const statMissing = document.getElementById('statMissing');

        if (statRows) statRows.textContent = results.overview?.rows || '-';
        if (statCols) statCols.textContent = results.overview?.columns || '-';

        if (statMissing && results.dataQuality?.missingValues) {
            const totalMissing = Object.values(results.dataQuality.missingValues)
                .reduce((sum, col) => sum + col.missing, 0);
            statMissing.textContent = totalMissing;
        }

        console.log('Analysis results displayed');
    },

    // Generate code for analysis
    generateAnalysisCode(dataframeName = 'df') {
        return `import pandas as pd
import numpy as np

# Comprehensive EDA
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {${dataframeName}.shape}")
print(f"\\nColumn Types:\\n{${dataframeName}.dtypes}")

print("\\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(${dataframeName}.describe(include='all'))

print("\\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
missing = ${dataframeName}.isnull().sum()
missing_pct = (missing / len(${dataframeName}) * 100).round(2)
print(pd.DataFrame({'Missing': missing, 'Percentage': missing_pct}))

print("\\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)
print(f"Duplicate rows: {${dataframeName}.duplicated().sum()}")

# Correlation matrix for numeric columns
numeric_cols = ${dataframeName}.select_dtypes(include=[np.number]).columns
if len(numeric_cols) >= 2:
    print("\\n" + "=" * 60)
    print("CORRELATION MATRIX")
    print("=" * 60)
    print(${dataframeName}[numeric_cols].corr().round(3))`;
    }
};

// Export for global access
window.DataAnalysis = DataAnalysis;
window.Statistics = Statistics;
window.DataQuality = DataQuality;
window.CorrelationAnalysis = CorrelationAnalysis;
window.FeatureAnalysis = FeatureAnalysis;
window.PatternDiscovery = PatternDiscovery;
window.DistributionAnalysis = DistributionAnalysis;
