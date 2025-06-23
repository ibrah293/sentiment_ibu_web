# Model Visualization Guide

This guide explains how to generate AUC-ROC plots and other visualizations for your sentiment analysis models.

## 🎨 Overview

The visualization module provides comprehensive plotting capabilities for analyzing model performance:

- **AUC-ROC Curves** - Evaluate model discrimination ability
- **Confusion Matrices** - Visualize classification performance
- **Model Comparison Charts** - Compare different algorithms
- **Feature Importance Plots** - Understand feature contributions
- **Learning Curves** - Analyze model learning behavior

## 📋 Prerequisites

Make sure you have the required dependencies:

```bash
pip install matplotlib seaborn scikit-learn numpy pandas
```

## 🚀 Quick Start

### 1. Simple Visualization Test

Start with the simple test to verify everything works:

```bash
python test_visualization.py
```

This will generate:
- `sample_auc_roc.png` - AUC-ROC curves for sample models
- `sample_confusion_matrices.png` - Confusion matrices
- `sample_model_comparison.png` - Model performance comparison

### 2. Full Project Visualization

Generate comprehensive visualizations for your sentiment analysis models:

```bash
python generate_visualizations.py
```

This creates a `model_plots/` directory with:
- `sentiment/` - Sentiment analysis visualizations
- `emotion/` - Emotion detection visualizations

## 📊 Visualization Types

### 1. AUC-ROC Curves

**What it shows:** Model discrimination ability across different thresholds

**How to interpret:**
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.5**: Random classifier
- **AUC > 0.8**: Good classifier
- **AUC > 0.9**: Excellent classifier

**Example:**
```python
from model_visualization import ModelVisualizer

visualizer = ModelVisualizer()
visualizer.plot_auc_roc_curves(models, X_test, y_test, 'sentiment')
```

### 2. Confusion Matrices

**What it shows:** Detailed classification performance for each class

**How to interpret:**
- **Diagonal elements**: Correct predictions
- **Off-diagonal elements**: Misclassifications
- **Higher diagonal values**: Better performance

**Example:**
```python
visualizer.plot_confusion_matrices(models, X_test, y_test, 'sentiment')
```

### 3. Model Comparison

**What it shows:** Side-by-side comparison of different metrics

**Metrics included:**
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

**Example:**
```python
visualizer.plot_model_comparison(models, X_test, y_test, 'sentiment')
```

### 4. Feature Importance

**What it shows:** Which features contribute most to predictions

**Available for:**
- Random Forest
- Other tree-based models

**Example:**
```python
visualizer.plot_feature_importance(models, feature_names)
```

### 5. Learning Curves

**What it shows:** How model performance changes with training data size

**How to interpret:**
- **Converging curves**: Model is learning well
- **Large gap**: Potential overfitting
- **Low scores**: Underfitting

**Example:**
```python
visualizer.plot_learning_curves(models, X_train, y_train, 'sentiment')
```

## 🔧 Integration with Your Models

### Using with Existing Models

```python
# Import your existing models
from app import sentiment_models, emotion_models
from model_visualization import ModelVisualizer

# Create visualizer
visualizer = ModelVisualizer()

# Generate comprehensive report
visualizer.create_comprehensive_report(
    models=sentiment_models,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    task_type='sentiment',
    save_dir='my_plots'
)
```

### Custom Data Preparation

```python
# Prepare your data
from data_preprocessing import TextPreprocessor
from text_vectorization import TextVectorizer

preprocessor = TextPreprocessor()
vectorizer = TextVectorizer('tfidf')

# Process your data
processed_text = preprocessor.preprocess_dataset(your_data, 'text', 'label')
X = vectorizer.fit_transform(processed_text['processed_text'])

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

## 📈 Understanding the Results

### AUC-ROC Analysis

1. **Compare curves**: Higher curves indicate better performance
2. **Check AUC values**: Look for values above 0.8
3. **Identify best model**: Choose the model with highest AUC

### Confusion Matrix Analysis

1. **Check diagonal dominance**: More values on diagonal = better performance
2. **Identify problem classes**: Look for classes with many misclassifications
3. **Understand errors**: See which classes are confused with each other

### Model Comparison Analysis

1. **Overall winner**: Look for model with highest scores across metrics
2. **Trade-offs**: Some models may excel in precision vs recall
3. **Consistency**: Check if performance is consistent across metrics

## 🎯 Best Practices

### 1. Data Quality
- Ensure balanced classes for fair comparison
- Use sufficient test data for reliable metrics
- Preprocess data consistently across models

### 2. Model Selection
- Compare multiple algorithms
- Consider computational requirements
- Balance performance vs interpretability

### 3. Visualization
- Use consistent color schemes
- Include proper labels and titles
- Save high-resolution plots for presentations

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install matplotlib seaborn scikit-learn
   ```

2. **Memory Issues**
   - Reduce dataset size for testing
   - Use smaller models initially

3. **Plot Not Showing**
   ```python
   import matplotlib.pyplot as plt
   plt.show()  # Add this if plots don't display
   ```

4. **Model Compatibility**
   - Ensure models have `predict_proba()` method for ROC curves
   - Check for `feature_importances_` for importance plots

### Error Messages

- **"No models with feature importance found"**: Only tree-based models support this
- **"Error plotting model"**: Check if model has required methods
- **"Memory error"**: Reduce dataset size or use simpler models

## 📁 File Structure

After running visualizations, you'll have:

```
project/
├── model_plots/
│   ├── sentiment/
│   │   ├── sentiment_auc_roc.png
│   │   ├── sentiment_confusion_matrices.png
│   │   ├── sentiment_model_comparison.png
│   │   ├── sentiment_feature_importance.png
│   │   └── sentiment_learning_curves.png
│   └── emotion/
│       ├── emotion_auc_roc.png
│       ├── emotion_confusion_matrices.png
│       ├── emotion_model_comparison.png
│       ├── emotion_feature_importance.png
│       └── emotion_learning_curves.png
├── sample_auc_roc.png
├── sample_confusion_matrices.png
└── sample_model_comparison.png
```

## 🎨 Customization

### Changing Colors
```python
visualizer = ModelVisualizer()
visualizer.colors['my_model'] = '#FF0000'  # Red color
```

### Custom Plot Sizes
```python
plt.figure(figsize=(15, 10))  # Larger plot
```

### Saving Options
```python
plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')
```

## 📚 Additional Resources

- [Scikit-learn ROC Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
- [Matplotlib Tutorial](https://matplotlib.org/stable/tutorials/index.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)

## 🎉 Conclusion

The visualization module provides powerful tools for understanding and comparing your sentiment analysis models. Use these plots to:

- **Evaluate model performance** objectively
- **Compare different algorithms** systematically
- **Identify areas for improvement**
- **Communicate results** effectively

Start with the simple test script to verify everything works, then generate comprehensive visualizations for your specific models! 