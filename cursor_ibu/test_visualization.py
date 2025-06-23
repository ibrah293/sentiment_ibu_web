#!/usr/bin/env python3
"""
Simple Test Script for Model Visualization
This script demonstrates basic visualization functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

def create_sample_data():
    """Create sample data for testing"""
    print("📊 Creating sample dataset...")
    
    # Create synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                             n_redundant=5, n_classes=3, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_sample_models(X_train, y_train):
    """Train sample models for testing"""
    print("🤖 Training sample models...")
    
    models = {}
    
    # Logistic Regression
    print("   Training Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    models['logistic_regression'] = lr
    
    # Random Forest
    print("   Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    
    # Naive Bayes (Gaussian for continuous data)
    print("   Training Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    models['naive_bayes'] = nb
    
    return models

def plot_auc_roc_simple(models, X_test, y_test):
    """Simple AUC-ROC plot"""
    print("📈 Generating AUC-ROC plot...")
    
    plt.figure(figsize=(10, 8))
    
    # Binarize the output for multi-class ROC
    classes = [0, 1, 2]
    y_test_bin = label_binarize(y_test, classes=classes)
    
    colors = {
        'logistic_regression': '#FF6B6B',
        'random_forest': '#4ECDC4',
        'naive_bayes': '#45B7D1'
    }
    
    model_names = {
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'naive_bayes': 'Naive Bayes'
    }
    
    for model_name, model in models.items():
        try:
            # Get prediction probabilities
            y_pred_proba = model.predict_proba(X_test)
            
            # Compute micro-average ROC curve and ROC area
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color=colors[model_name], lw=2,
                    label=f'{model_names[model_name]} (AUC = {roc_auc:.3f})')
            
        except Exception as e:
            print(f"Error plotting {model_name}: {e}")
            continue
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('AUC-ROC Curves for Sample Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sample_auc_roc.png', dpi=300, bbox_inches='tight')
    print("✅ AUC-ROC plot saved as 'sample_auc_roc.png'")
    plt.show()

def plot_confusion_matrices_simple(models, X_test, y_test):
    """Simple confusion matrix plot"""
    print("📊 Generating confusion matrices...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    model_names = {
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'naive_bayes': 'Naive Bayes'
    }
    
    for idx, (model_name, model) in enumerate(models.items()):
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot confusion matrix
            im = axes[idx].imshow(cm, interpolation='nearest', cmap='Blues')
            axes[idx].set_title(f'{model_names[model_name]}\nConfusion Matrix', fontweight='bold')
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[idx].text(j, i, format(cm[i, j], 'd'),
                                 ha="center", va="center",
                                 color="white" if cm[i, j] > thresh else "black")
            
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            
        except Exception as e:
            print(f"Error plotting confusion matrix for {model_name}: {e}")
            axes[idx].text(0.5, 0.5, f'Error plotting\n{model_name}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{model_name} - Error')
    
    plt.suptitle('Confusion Matrices for Sample Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✅ Confusion matrices saved as 'sample_confusion_matrices.png'")
    plt.show()

def plot_model_comparison_simple(models, X_test, y_test):
    """Simple model comparison plot"""
    print("📋 Generating model comparison...")
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Calculate metrics for each model
    metrics = {}
    
    for model_name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            metrics[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        except Exception as e:
            print(f"Error calculating metrics for {model_name}: {e}")
            continue
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    model_names = list(metrics.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        values = [metrics[model][metric] for model in model_names]
        
        ax = [ax1, ax2, ax3, ax4][idx]
        bars = ax.bar(model_names, values, color=colors[:len(model_names)], alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'{label} Comparison', fontweight='bold')
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Model comparison saved as 'sample_model_comparison.png'")
    plt.show()
    
    # Print metrics table
    print(f"\n📊 Detailed Metrics:")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    
    for model_name in model_names:
        m = metrics[model_name]
        print(f"{model_name.replace('_', ' ').title():<20} "
              f"{m['accuracy']:<10.3f} {m['precision']:<10.3f} "
              f"{m['recall']:<10.3f} {m['f1_score']:<10.3f}")

def main():
    """Main function to run the visualization test"""
    print("🎨 Model Visualization Test")
    print("=" * 40)
    
    try:
        # Step 1: Create sample data
        X_train, X_test, y_train, y_test = create_sample_data()
        
        # Step 2: Train models
        models = train_sample_models(X_train, y_train)
        
        # Step 3: Generate visualizations
        plot_auc_roc_simple(models, X_test, y_test)
        plot_confusion_matrices_simple(models, X_test, y_test)
        plot_model_comparison_simple(models, X_test, y_test)
        
        print("\n✅ All visualizations generated successfully!")
        print("📁 Generated files:")
        print("   - sample_auc_roc.png")
        print("   - sample_confusion_matrices.png")
        print("   - sample_model_comparison.png")
        
    except Exception as e:
        print(f"\n❌ Error during visualization test: {e}")
        print("🔧 Please check your matplotlib installation.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Visualization test completed successfully!")
    else:
        print("\n💥 Visualization test failed.") 