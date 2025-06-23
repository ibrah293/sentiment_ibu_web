#!/usr/bin/env python3
"""
Model Visualization Module for Sentiment Analysis Project
Generates AUC-ROC plots, confusion matrices, and other model performance visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelVisualizer:
    def __init__(self):
        """Initialize the model visualizer"""
        self.colors = {
            'logistic_regression': '#FF6B6B',
            'random_forest': '#4ECDC4', 
            'naive_bayes': '#45B7D1',
            'svm': '#96CEB4',
            'neural_network': '#FFEAA7'
        }
        
        self.model_names = {
            'logistic_regression': 'Logistic Regression',
            'random_forest': 'Random Forest',
            'naive_bayes': 'Naive Bayes',
            'svm': 'Support Vector Machine',
            'neural_network': 'Neural Network'
        }
    
    def plot_auc_roc_curves(self, models, X_test, y_test, task_type='sentiment', save_path=None):
        """
        Plot AUC-ROC curves for multiple models
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            task_type: 'sentiment' or 'emotion'
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Define classes based on task type
        if task_type == 'sentiment':
            classes = ['negative', 'neutral', 'positive']
            n_classes = 3
        else:  # emotion
            classes = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
            n_classes = 7
        
        # Binarize the output for multi-class ROC
        y_test_bin = label_binarize(y_test, classes=classes)
        
        for model_name, model in models.items():
            if model_name in self.colors:
                color = self.colors[model_name]
                display_name = self.model_names[model_name]
                
                # Get prediction probabilities
                try:
                    y_pred_proba = model.predict_proba(X_test)
                    
                    # Compute ROC curve and ROC area for each class
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    # Compute micro-average ROC curve and ROC area
                    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    
                    # Plot micro-average ROC curve
                    plt.plot(fpr["micro"], tpr["micro"],
                            color=color, lw=2,
                            label=f'{display_name} (AUC = {roc_auc["micro"]:.3f})')
                    
                except Exception as e:
                    print(f"Error plotting {model_name}: {e}")
                    continue
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'AUC-ROC Curves for {task_type.title()} Analysis Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add text box with task information
        plt.text(0.02, 0.98, f'Task: {task_type.title()} Analysis\nClasses: {n_classes}', 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ AUC-ROC plot saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrices(self, models, X_test, y_test, task_type='sentiment', save_path=None):
        """
        Plot confusion matrices for all models
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            task_type: 'sentiment' or 'emotion'
            save_path: Path to save the plot
        """
        n_models = len(models)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        # Define classes based on task type
        if task_type == 'sentiment':
            classes = ['negative', 'neutral', 'positive']
        else:  # emotion
            classes = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
        
        for idx, (model_name, model) in enumerate(models.items()):
            if idx >= 4:  # Limit to 4 plots
                break
                
            try:
                # Get predictions
                y_pred = model.predict(X_test)
                
                # Create confusion matrix
                cm = confusion_matrix(y_test, y_pred, labels=classes)
                
                # Plot confusion matrix
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=classes, yticklabels=classes,
                           ax=axes[idx], cbar_kws={'shrink': 0.8})
                
                axes[idx].set_title(f'{self.model_names.get(model_name, model_name)}\nConfusion Matrix', 
                                  fontweight='bold')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
                
            except Exception as e:
                print(f"Error plotting confusion matrix for {model_name}: {e}")
                axes[idx].text(0.5, 0.5, f'Error plotting\n{model_name}', 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{model_name} - Error')
        
        # Hide unused subplots
        for idx in range(n_models, 4):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Confusion Matrices for {task_type.title()} Analysis Models', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrices saved to: {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, models, X_test, y_test, task_type='sentiment', save_path=None):
        """
        Create a comprehensive model comparison plot
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            task_type: 'sentiment' or 'emotion'
            save_path: Path to save the plot
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        # Calculate metrics for each model
        metrics = {}
        
        for model_name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                # Calculate AUC for ROC
                from sklearn.metrics import roc_auc_score
                if task_type == 'sentiment':
                    classes = ['negative', 'neutral', 'positive']
                else:
                    classes = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
                
                y_test_bin = label_binarize(y_test, classes=classes)
                auc_score = roc_auc_score(y_test_bin, y_pred_proba, average='micro')
                
                metrics[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc_score
                }
                
            except Exception as e:
                print(f"Error calculating metrics for {model_name}: {e}")
                continue
        
        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data for plotting
        model_names = list(metrics.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
            values = [metrics[model][metric] for model in model_names]
            colors = [self.colors.get(model, '#666666') for model in model_names]
            
            ax = [ax1, ax2, ax3, ax4][idx]
            bars = ax.bar(model_names, values, color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'{label} Comparison', fontweight='bold')
            ax.set_ylabel(label)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Model Performance Comparison - {task_type.title()} Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Model comparison plot saved to: {save_path}")
        
        plt.show()
        
        # Print detailed metrics table
        print(f"\n📊 Detailed Metrics for {task_type.title()} Analysis:")
        print("=" * 80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
        print("-" * 80)
        
        for model_name in model_names:
            m = metrics[model_name]
            print(f"{self.model_names.get(model_name, model_name):<20} "
                  f"{m['accuracy']:<10.3f} {m['precision']:<10.3f} "
                  f"{m['recall']:<10.3f} {m['f1_score']:<10.3f} {m['auc']:<10.3f}")
    
    def plot_feature_importance(self, models, feature_names=None, save_path=None):
        """
        Plot feature importance for models that support it (Random Forest, etc.)
        
        Args:
            models: Dictionary of trained models
            feature_names: List of feature names
            save_path: Path to save the plot
        """
        # Filter models that have feature_importances_
        importance_models = {}
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance_models[name] = model
        
        if not importance_models:
            print("❌ No models with feature importance found")
            return
        
        n_models = len(importance_models)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, model) in enumerate(importance_models.items()):
            importances = model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'Feature {i}' for i in range(len(importances))]
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            # Plot top 20 features
            top_n = min(20, len(importances))
            top_indices = indices[:top_n]
            
            ax = axes[idx]
            bars = ax.bar(range(top_n), importances[top_indices], 
                         color=self.colors.get(model_name, '#666666'), alpha=0.8)
            
            ax.set_title(f'{self.model_names.get(model_name, model_name)}\nFeature Importance', 
                        fontweight='bold')
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            ax.set_xticks(range(top_n))
            ax.set_xticklabels([feature_names[i] for i in top_indices], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Feature importance plot saved to: {save_path}")
        
        plt.show()
    
    def plot_learning_curves(self, models, X_train, y_train, task_type='sentiment', save_path=None):
        """
        Plot learning curves for models (if supported)
        
        Args:
            models: Dictionary of trained models
            X_train: Training features
            y_train: Training labels
            task_type: 'sentiment' or 'emotion'
            save_path: Path to save the plot
        """
        from sklearn.model_selection import learning_curve
        
        # Filter models that work well with learning curves
        curve_models = {}
        for name, model in models.items():
            if name in ['logistic_regression', 'random_forest', 'naive_bayes']:
                curve_models[name] = model
        
        if not curve_models:
            print("❌ No suitable models for learning curves found")
            return
        
        n_models = len(curve_models)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, model) in enumerate(curve_models.items()):
            try:
                # Calculate learning curves
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X_train, y_train, cv=5, n_jobs=-1, 
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring='accuracy'
                )
                
                # Calculate mean and std
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                ax = axes[idx]
                ax.plot(train_sizes, train_mean, 'o-', color=self.colors.get(model_name, '#666666'),
                       label='Training score', linewidth=2, markersize=6)
                ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1,
                              color=self.colors.get(model_name, '#666666'))
                
                ax.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-validation score',
                       linewidth=2, markersize=6)
                ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
                
                ax.set_title(f'{self.model_names.get(model_name, model_name)}\nLearning Curve', 
                           fontweight='bold')
                ax.set_xlabel('Training Examples')
                ax.set_ylabel('Score')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Error plotting learning curve for {model_name}: {e}")
                axes[idx].text(0.5, 0.5, f'Error plotting\n{model_name}', 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{model_name} - Error')
        
        plt.suptitle(f'Learning Curves - {task_type.title()} Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Learning curves saved to: {save_path}")
        
        plt.show()
    
    def create_comprehensive_report(self, models, X_train, X_test, y_train, y_test, 
                                  task_type='sentiment', feature_names=None, save_dir='plots'):
        """
        Create a comprehensive visualization report
        
        Args:
            models: Dictionary of trained models
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            task_type: 'sentiment' or 'emotion'
            feature_names: List of feature names
            save_dir: Directory to save plots
        """
        import os
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"🎨 Creating comprehensive visualization report for {task_type} analysis...")
        print("=" * 60)
        
        # 1. AUC-ROC Curves
        print("📈 Generating AUC-ROC curves...")
        self.plot_auc_roc_curves(models, X_test, y_test, task_type, 
                                os.path.join(save_dir, f'{task_type}_auc_roc.png'))
        
        # 2. Confusion Matrices
        print("📊 Generating confusion matrices...")
        self.plot_confusion_matrices(models, X_test, y_test, task_type,
                                   os.path.join(save_dir, f'{task_type}_confusion_matrices.png'))
        
        # 3. Model Comparison
        print("📋 Generating model comparison...")
        self.plot_model_comparison(models, X_test, y_test, task_type,
                                 os.path.join(save_dir, f'{task_type}_model_comparison.png'))
        
        # 4. Feature Importance (if applicable)
        print("🔍 Generating feature importance analysis...")
        self.plot_feature_importance(models, feature_names,
                                   os.path.join(save_dir, f'{task_type}_feature_importance.png'))
        
        # 5. Learning Curves
        print("📚 Generating learning curves...")
        self.plot_learning_curves(models, X_train, y_train, task_type,
                                os.path.join(save_dir, f'{task_type}_learning_curves.png'))
        
        print(f"\n✅ Comprehensive report saved to '{save_dir}' directory!")
        print("📁 Generated files:")
        for file in os.listdir(save_dir):
            if file.startswith(task_type):
                print(f"   - {file}")

def demo_visualization():
    """Demo function to show how to use the visualizer"""
    print("🎨 Model Visualization Demo")
    print("=" * 40)
    
    # This would be used with your actual models
    print("To use the visualizer with your models:")
    print("1. Import the visualizer: from model_visualization import ModelVisualizer")
    print("2. Create instance: visualizer = ModelVisualizer()")
    print("3. Use with your models:")
    print("   visualizer.create_comprehensive_report(sentiment_models, X_train, X_test, y_train, y_test, 'sentiment')")
    print("   visualizer.create_comprehensive_report(emotion_models, X_train, X_test, y_train, y_test, 'emotion')")

if __name__ == "__main__":
    demo_visualization() 