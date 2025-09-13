"""
Final CNN Evaluation Module for 5x5 Cross-Validation
Implements comprehensive test evaluation following final_eval.py pattern
Evaluates all 25 trained models with variance-based selection and ensemble evaluation
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings
from datetime import datetime
import glob
import random
import traceback
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, matthews_corrcoef,
    precision_recall_curve, average_precision_score, roc_curve, confusion_matrix
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from model import SMILESCNNModel
from data_loader import SMILESDataModule
from utils import (
    set_all_seeds, setup_logging, calculate_metrics_summary,
    save_results, save_dataframe_with_timestamp
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Configuration constants for variance-based model selection
VARIANCE_THRESHOLD = 0.1  # 10% coefficient of variation
MIN_MODELS_PER_CV = 1      # Minimum models from each CV run
TOTAL_MODELS_TO_SELECT = 5 # Total models for ensemble

class FinalCNNEvaluator:
    """Final comprehensive evaluation of CNN models from 5x5 cross-validation"""
    
    def __init__(
        self,
        features: np.ndarray,
        labels_df: pd.DataFrame,
        cv_results_dir: str,
        best_hyperparameters: Dict[str, Any],
        output_dir: str = None
    ):
        """
        Initialize final CNN evaluator
        
        Args:
            features: Full feature array
            labels_df: Full labels dataframe
            cv_results_dir: Path to cross-validation results directory
            best_hyperparameters: Hyperparameters used for training
            output_dir: Output directory for evaluation results
        """
        self.features = features
        self.labels_df = labels_df
        self.cv_results_dir = cv_results_dir
        self.best_hyperparameters = best_hyperparameters
        
        # Setup paths
        self.eval_dir = output_dir or os.path.join(
            Config.OUTPUT_DIR, "final_cnn_evaluation"
        )
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.eval_dir, "final_cnn_evaluation.log")
        self._setup_logging(log_file)
        
        # Get data dimensions
        self.vocab_size = features.shape[-1]
        self.sequence_length = features.shape[1]
        
        # Use existing SMILESDataModule for test data
        self.data_module = SMILESDataModule(
            features=features,
            labels_df=labels_df,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            use_class_weights=False  # No class weights for test evaluation
        )
        self.data_module.setup(stage="test")
        
        # Verify we have test data
        if self.data_module.test_dataset is None:
            raise ValueError("No test data found")
            
        # Load best model information from CV results
        self.best_models_info = self._load_best_model_info()
        
        # Results storage
        self.individual_results = []
        self.ensemble_results = None
        
        # Get test data info
        test_dataset = self.data_module.test_dataset
        test_labels = test_dataset.labels.numpy()
        
        logger.info(f"Final CNN Evaluation Setup:")
        logger.info(f"  Test samples: {len(test_dataset)}")
        logger.info(f"  Test class distribution: {dict(zip(*np.unique(test_labels, return_counts=True)))}")
        logger.info(f"  Models to evaluate: {len(self.best_models_info)}")
        logger.info(f"  Data shape: {test_dataset.features.shape}")
        logger.info(f"  Evaluation directory: {self.eval_dir}")
        logger.info("  ✓ Setup completed successfully")
    
    def _setup_logging(self, log_file: str):
        """Setup logging configuration"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configure logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    def _load_best_model_info(self) -> Dict[str, str]:
        """Load and identify the 25 best models from cross-validation results"""
        try:
            # Look for model checkpoints JSON file
            checkpoints_file = os.path.join(self.cv_results_dir, 'model_checkpoints.json')
            
            if not os.path.exists(checkpoints_file):
                raise FileNotFoundError(f"Model checkpoints file not found: {checkpoints_file}")
                
            with open(checkpoints_file, 'r') as f:
                checkpoints = json.load(f)
                
            logger.info(f"Loaded {len(checkpoints)} model checkpoints from CV results")
            
            # Verify checkpoint files exist
            valid_checkpoints = {}
            for fold_id, checkpoint_path in checkpoints.items():
                if os.path.exists(checkpoint_path):
                    valid_checkpoints[fold_id] = checkpoint_path
                else:
                    logger.warning(f"Checkpoint not found: {checkpoint_path}")
                    
            logger.info(f"Found {len(valid_checkpoints)} valid checkpoints")
            
            if len(valid_checkpoints) == 0:
                raise ValueError("No valid model checkpoints found")
                
            return valid_checkpoints
            
        except Exception as e:
            logger.error(f"Error loading best model info: {str(e)}")
            raise
            
    def _get_test_loader(self) -> DataLoader:
        """Get DataLoader for test set evaluation"""
        return self.data_module.test_dataloader()
    
    def _evaluate_on_test(self, model, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on test set with comprehensive metrics"""
        model.eval()
        all_preds = []
        all_labels = []
        
        try:
            with torch.no_grad():
                for batch in test_loader:
                    features, labels = batch['features'], batch['labels']
                    outputs = model(features)
                    preds = torch.sigmoid(outputs)  # CNN outputs logits
                    
                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
                    
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            # Calculate comprehensive metrics
            metrics = {
                'raw_predictions': all_preds.tolist(),
                'labels': all_labels.tolist()
            }
            
            # Threshold-independent metrics
            auc_score = float(roc_auc_score(all_labels, all_preds))
            metrics.update({
                'auc': auc_score,
                'roc_auc': auc_score,
                'average_precision': float(average_precision_score(all_labels, all_preds)),
            })
            
            # Calculate optimal thresholds
            thresholds, threshold_metrics = self._find_optimal_thresholds(all_labels, all_preds)
            
            # Use optimal F1 threshold for MCC calculation
            optimal_f1_threshold = thresholds.get('f1', 0.5)
            optimal_binary_preds = (all_preds >= optimal_f1_threshold).astype(int)
            metrics['matthews_corrcoef_raw'] = float(matthews_corrcoef(all_labels, optimal_binary_preds))
            
            # Store thresholds and metrics
            metrics['thresholds'] = {k: float(v) for k, v in thresholds.items()}
            
            for threshold_name, threshold_data in threshold_metrics.items():
                metrics[threshold_name] = threshold_data
                if 'confusion_matrix' in threshold_data:
                    metrics[f"{threshold_name}_confusion_matrix"] = threshold_data.pop('confusion_matrix')
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise
            
    def _evaluate_single_model(
        self,
        fold_id: str,
        checkpoint_path: str,
        test_loader: DataLoader
    ) -> Dict[str, Any]:
        """Evaluate a single model with comprehensive metrics"""
        logger.info(f"Evaluating {fold_id} on test set...")
        
        try:
            # Set seed for reproducible evaluation
            set_all_seeds(Config.RANDOM_SEED + hash(fold_id) % 10000)
            
            # Load model from checkpoint
            hyperparams = self.best_hyperparameters.copy()
            hyperparams.update({
                'vocab_size': self.vocab_size,
                'sequence_length': self.sequence_length,
                'class_weights': None
            })
            
            model = SMILESCNNModel.load_from_checkpoint(
                checkpoint_path,
                strict=False,
                **hyperparams
            )
            model.eval()
            
            # Get comprehensive metrics
            metrics = self._evaluate_on_test(model, test_loader)
            
            # Parse fold info from fold_id (e.g., "cv1_fold2")
            cv_fold_parts = fold_id.replace('cv', '').split('_fold')
            cv = int(cv_fold_parts[0])
            fold = int(cv_fold_parts[1])
            
            # Add metadata
            metrics.update({
                'fold_id': fold_id,
                'cv': cv,
                'fold': fold,
                'checkpoint_path': checkpoint_path,
                'test_samples': len(self.data_module.test_dataset),
                'status': 'success'
            })
            
            logger.info(f"  {fold_id}: AUC = {metrics.get('auc', 'N/A'):.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate model {fold_id}: {e}")
            return {
                'fold_id': fold_id,
                'cv': 0,
                'fold': 0,
                'checkpoint_path': checkpoint_path,
                'status': 'failed',
                'error': str(e)
            }
    
    def _evaluate_individual_models(self):
        """Evaluate all individual models from cross-validation"""
        logger.info("Starting individual model evaluation...")
        
        test_loader = self._get_test_loader()
        successful_models = 0
        failed_models = []
        
        for fold_id, checkpoint_path in self.best_models_info.items():
            try:
                metrics = self._evaluate_single_model(fold_id, checkpoint_path, test_loader)
                
                if metrics.get('status') == 'success':
                    # Process metrics to ensure they're comparison-safe
                    processed_metrics = {}
                    for key, value in metrics.items():
                        if isinstance(value, dict):
                            for nested_key, nested_value in value.items():
                                processed_metrics[f"{key}_{nested_key}"] = nested_value
                        else:
                            processed_metrics[key] = value
                    
                    self.individual_results.append(processed_metrics)
                    successful_models += 1
                    
                    logger.info(
                        f"CV{metrics['cv']} Fold{metrics['fold']}: "
                        f"AUC: {metrics.get('auc', 0.0):.4f}"
                    )
                else:
                    failed_models.append((metrics.get('cv', 0), metrics.get('fold', 0)))
                    
            except Exception as e:
                logger.error(f"Failed processing {fold_id}: {str(e)}")
                failed_models.append(fold_id)
                continue
                
        # Log summary
        total_models = len(self.best_models_info)
        logger.info(f"Model Loading Summary:")
        logger.info(f"Successfully loaded {successful_models}/{total_models} models")
        if failed_models:
            logger.warning(f"Failed models: {failed_models}")
            
        if successful_models == 0:
            raise RuntimeError("No models were successfully loaded")
    
    def _find_optimal_thresholds(self, labels: np.ndarray, preds: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Dict]]:
        """Find optimal thresholds based on different criteria"""
        def find_threshold(y_true, y_pred, criterion='f1'):
            if criterion == 'f1':
                precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
                f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
                optimal_idx = np.argmax(f1_scores)
                return thresholds[optimal_idx], f1_scores[optimal_idx], None
            elif criterion == 'youdens':
                fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                youdens_j = tpr - fpr
                optimal_idx = np.argmax(youdens_j)
                return thresholds[optimal_idx], youdens_j[optimal_idx], None
            elif criterion == 'eer':
                fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                fnr = 1 - tpr
                optimal_idx = np.argmin(np.abs(fpr - fnr))
                return thresholds[optimal_idx], fpr[optimal_idx], fnr[optimal_idx]
            elif criterion == 'pr_break_even':
                precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
                pr_be = np.abs(precision - recall)
                optimal_idx = np.argmin(pr_be)
                return thresholds[optimal_idx], precision[optimal_idx], recall[optimal_idx]
                
        try:
            thresholds = {'default': 0.5}
            metrics_at_thresholds = {}
            
            criteria = ['f1', 'youdens', 'eer', 'pr_break_even']
            for criterion in criteria:
                threshold, _, _ = find_threshold(labels, preds, criterion)
                thresholds[criterion] = float(threshold)
                
                # Calculate metrics at this threshold
                binary_preds = (preds >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()
                
                specificity = float(tn / (tn + fp + 1e-10))
                npv = float(tn / (tn + fn + 1e-10))
                balanced_accuracy = float((recall_score(labels, binary_preds, zero_division=0) + specificity) / 2)
                
                threshold_metrics = {
                    'threshold_value': float(threshold),
                    'accuracy': float(accuracy_score(labels, binary_preds)),
                    'precision': float(precision_score(labels, binary_preds, zero_division=0)),
                    'recall': float(recall_score(labels, binary_preds, zero_division=0)),
                    'f1': float(f1_score(labels, binary_preds, zero_division=0)),
                    'cohen_kappa': float(cohen_kappa_score(labels, binary_preds)),
                    'matthews_corrcoef': float(matthews_corrcoef(labels, binary_preds)),
                    'confusion_matrix': {
                        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
                    },
                    'specificity': specificity,
                    'npv': npv,
                    'balanced_accuracy': balanced_accuracy
                }
                
                metrics_at_thresholds[f'metrics_at_{criterion}'] = threshold_metrics
                
            return thresholds, metrics_at_thresholds
            
        except Exception as e:
            logger.error(f"Error in threshold optimization: {str(e)}")
            raise
    
    def _calculate_variance_statistics(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate variance statistics for model selection.
        
        Variance threshold justification:
        - CV < 10%: Literature standard for "excellent reproducibility"
        - Statistical rationale: When CV < 10%, model differences are within measurement noise
        - Practical benefit: Diversity-based selection maximizes ensemble error decorrelation
        
        Args:
            values: Metric values from all models
            
        Returns:
            Statistical measures including coefficient of variation
        """
        try:
            if not values or len(values) < 2:
                return {'mean': 0.0, 'std': 0.0, 'cv': float('inf')}
                
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)  # Sample standard deviation
            cv = std_val / mean_val if mean_val != 0 else float('inf')
            
            return {
                'mean': float(mean_val),
                'std': float(std_val),
                'cv': float(cv)
            }
            
        except Exception as e:
            logger.error(f"Error calculating variance statistics: {str(e)}")
            return {'mean': 0.0, 'std': 0.0, 'cv': float('inf')}
            
    def _validate_model_availability(self, cv: int, fold: int) -> bool:
        """Verify that model checkpoint exists"""
        try:
            fold_id = f"cv{cv}_fold{fold}"
            checkpoint_path = self.best_models_info.get(fold_id)
            return checkpoint_path is not None and os.path.exists(checkpoint_path)
        except Exception as e:
            logger.error(f"Error validating model availability for CV{cv} Fold{fold}: {str(e)}")
            return False
            
    def _select_diverse_models(self, primary_metric: str) -> List[Dict]:
        """
        Select models using diversity-based strategy when variance is low.
        
        Algorithm:
        1. Select best model from each CV run (based on primary metric)
        2. Fill remaining slots with maximum fold diversity
        
        Args:
            primary_metric: Primary metric for selection ('auc' for classification)
            
        Returns:
            Selected models ensuring CV representation
        """
        try:
            # Group results by CV run
            cv_groups = {}
            for result in self.individual_results:
                cv = result['cv']
                if cv not in cv_groups:
                    cv_groups[cv] = []
                cv_groups[cv].append(result)
                
            selected_models = []
            
            # Step 1: Select best model from each CV run
            for cv in sorted(cv_groups.keys()):
                cv_models = cv_groups[cv]
                # Filter models with valid checkpoints
                valid_cv_models = [
                    model for model in cv_models
                    if self._validate_model_availability(model['cv'], model['fold'])
                ]
                
                if not valid_cv_models:
                    logger.warning(f"No valid checkpoints found for CV{cv}")
                    continue
                    
                # Higher is better for AUC
                best_model = max(valid_cv_models, key=lambda x: x.get(primary_metric, 0.0))
                selected_models.append(best_model)
                logger.info(f"Selected best model from CV{cv}: {primary_metric.upper()}={best_model[primary_metric]:.4f}")
                
            # Step 2: Fill remaining slots with maximum fold diversity
            remaining_slots = max(0, TOTAL_MODELS_TO_SELECT - len(selected_models))
            selected_identifiers = {(m['cv'], m['fold']) for m in selected_models}
            
            # Get all remaining models with valid checkpoints
            remaining_models = [
                result for result in self.individual_results
                if (result['cv'], result['fold']) not in selected_identifiers
                and self._validate_model_availability(result['cv'], result['fold'])
            ]
            
            logger.info(f"Found {len(remaining_models)} valid remaining models for diversity selection")
            
            # Sort remaining models by performance
            remaining_models.sort(key=lambda x: x.get(primary_metric, 0.0), reverse=True)
            
            # Select remaining models, avoiding duplicates
            for candidate in remaining_models:
                if len(selected_models) >= TOTAL_MODELS_TO_SELECT:
                    break
                    
                candidate_id = (candidate['cv'], candidate['fold'])
                if candidate_id not in selected_identifiers:
                    selected_models.append(candidate)
                    selected_identifiers.add(candidate_id)
                    logger.info(f"Selected additional model CV{candidate['cv']} Fold{candidate['fold']}: "
                              f"{primary_metric.upper()}={candidate[primary_metric]:.4f}")
                              
            return selected_models[:TOTAL_MODELS_TO_SELECT]
            
        except Exception as e:
            logger.error(f"Error in diversity-based model selection: {str(e)}")
            return []
            
    def _select_best_models(self) -> List[Dict]:
        """
        Select best models using variance-based strategy.
        
        Uses AUC as primary metric (threshold-independent) to ensure consistency
        with optimal threshold approach used in cross-validation.
        
        Returns:
            Selected models if variance is low, empty list if high variance
        """
        try:
            primary_metric = 'auc'  # Use AUC for CNN classification
            
            # Extract primary metric values for variance analysis
            primary_values = [result.get(primary_metric, 0.0) for result in self.individual_results]
            
            # Calculate variance statistics
            variance_stats = self._calculate_variance_statistics(primary_values)
            cv_value = variance_stats['cv']
            
            # Log variance analysis
            logger.info(f"Variance analysis: CV={cv_value:.4f} (threshold={VARIANCE_THRESHOLD:.4f})")
            
            # Decision based on variance
            if cv_value < VARIANCE_THRESHOLD:
                logger.info(f"Low variance detected - using diversity-based selection")
                logger.info(f"Selected models ensure representation from all CV runs")
                
                # Use diversity-based selection
                selected_models = self._select_diverse_models(primary_metric)
                
                if not selected_models:
                    logger.warning("Diversity-based selection failed, returning empty list")
                    return []
                    
                # Format results with proper metrics
                best_models = []
                for model in selected_models:
                    model_info = {
                        'cv': model['cv'],
                        'fold': model['fold'],
                        'metrics': {
                            'auc': float(model['auc']),
                            'average_precision': float(model['average_precision']),
                            'matthews_corrcoef': float(model.get('matthews_corrcoef_raw', 0.0)),
                            # Use optimal threshold metrics
                            'tpr': float(model.get('metrics_at_f1', {}).get('recall', 0.0)),
                            'tnr': float(model.get('metrics_at_f1', {}).get('specificity', 0.0))
                        }
                    }
                    best_models.append(model_info)
                    
                # Log final selection
                logger.info(f"\nSelected {len(best_models)} models using diversity-based strategy:")
                for i, model in enumerate(best_models, 1):
                    metrics = model['metrics']
                    logger.info(f"\nModel {i} - CV{model['cv']} Fold{model['fold']}:")
                    logger.info(f"  AUC: {metrics['auc']:.4f}")
                    logger.info(f"  Average Precision: {metrics['average_precision']:.4f}")
                    logger.info(f"  MCC: {metrics['matthews_corrcoef']:.4f}")
                    
                return best_models
                
            else:
                # High variance - skip automated selection
                logger.warning(f"High variance detected: CV={cv_value:.4f} exceeds threshold={VARIANCE_THRESHOLD:.4f}")
                logger.warning(f"Skipping automated model selection - manual review recommended")
                logger.info(f"No best_models.json will be created due to high variance")
                return []
                
        except Exception as e:
            logger.error(f"Error selecting best models: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def _save_best_models(self):
        """Save the selected best models to JSON with detailed metrics"""
        try:
            best_models = self._select_best_models()
            
            if not best_models:
                logger.warning("No best models were selected")
                return
                
            # Create best_models directory
            best_models_dir = os.path.join(self.eval_dir, "best_models")
            os.makedirs(best_models_dir, exist_ok=True)
            
            best_models_file = os.path.join(best_models_dir, 'best_models.json')
            
            logger.info(f"\nPreparing to save best models to: {best_models_file}")
            
            # Format model data
            model_data = {"models": []}
            
            for model in best_models:
                model_info = {
                    'cv': int(model['cv']),
                    'fold': int(model['fold']),
                    'init': 1,  # Default init for CNN models
                    'metrics': {}
                }
                
                # Add CNN-specific metrics
                model_info['metrics'].update({
                    'auc': float(model['metrics']['auc']),
                    'tpr': float(model['metrics']['tpr']),
                    'tnr': float(model['metrics']['tnr']),
                    'matthews_corrcoef': float(model['metrics']['matthews_corrcoef']),
                    'average_precision': float(model['metrics']['average_precision'])
                })
                
                model_data["models"].append(model_info)
                
            # Save to JSON file
            with open(best_models_file, 'w') as f:
                json.dump(model_data, f, indent=4)
                
            # Log results
            folds_used = [model['fold'] for model in model_data["models"]]
            unique_folds = set(folds_used)
            
            logger.info(f"\n✓ Selected and saved best {len(model_data['models'])} models to: {best_models_file}")
            logger.info(f"Fold diversity achieved: {len(unique_folds)} unique folds out of {len(model_data['models'])} models")
            logger.info(f"Folds used: {sorted(unique_folds)}")
            logger.info("Best models saved (ordered by performance):")
            
            for i, model in enumerate(model_data["models"], 1):
                metrics = model['metrics']
                logger.info(f"\nModel {i} - CV{model['cv']} Fold{model['fold']}:")
                logger.info(f"  AUC: {metrics['auc']:.4f}")
                logger.info(f"  Average Precision: {metrics['average_precision']:.4f}")
                logger.info(f"  MCC: {metrics['matthews_corrcoef']:.4f}")
                
        except Exception as e:
            logger.error(f"Error saving best models: {str(e)}")
            logger.error(traceback.format_exc())

    def _evaluate_ensemble(self):
        """Create and evaluate ensemble of all models"""
        try:
            test_loader = self._get_test_loader()
            all_predictions = []
            failed_models = []
            successful_models = 0
            
            total_models = len(self.individual_results)
            logger.info(f"\nStarting ensemble evaluation of {total_models} models...")
            
            # Iterate through individual results to get ensemble predictions
            for result in self.individual_results:
                try:
                    fold_id = result['fold_id']
                    checkpoint_path = result['checkpoint_path']
                    cv = result['cv']
                    fold = result['fold']
                    auc = result['auc']
                    
                    logger.info(f"Loading model {fold_id} (AUC={auc:.4f})...")
                    
                    # Load model
                    hyperparams = self.best_hyperparameters.copy()
                    hyperparams.update({
                        'vocab_size': self.vocab_size,
                        'sequence_length': self.sequence_length,
                        'class_weights': None
                    })
                    
                    model = SMILESCNNModel.load_from_checkpoint(
                        checkpoint_path,
                        strict=False,
                        **hyperparams
                    )
                    model.eval()
                    
                    # Get predictions
                    preds = self._get_predictions(model, test_loader)
                    all_predictions.append(preds)
                    successful_models += 1
                    
                except Exception as e:
                    failed_models.append((cv, fold))
                    logger.error(f"Failed processing {fold_id}: {str(e)}")
                    continue
                    
                finally:
                    if 'model' in locals():
                        del model
                        torch.cuda.empty_cache()
                        
            if not all_predictions:
                raise ValueError("No models were successfully evaluated. Cannot create ensemble.")
                
            # Log summary
            logger.info(f"\nEnsemble Summary:")
            logger.info(f"Successfully processed {successful_models}/{total_models} models")
            if failed_models:
                logger.warning(f"Failed models: {failed_models}")
                
            # Stack predictions and average them
            all_predictions = np.stack(all_predictions, axis=0)
            ensemble_preds = np.mean(all_predictions, axis=0)
            
            # Get true labels
            labels = []
            for batch in test_loader:
                labels.extend(batch['labels'].numpy())
            labels = np.array(labels)
            
            # Calculate metrics
            auc_score = float(roc_auc_score(labels, ensemble_preds))
            avg_precision = float(average_precision_score(labels, ensemble_preds))
            
            # Find optimal thresholds for ensemble predictions
            thresholds, threshold_metrics = self._find_optimal_thresholds(labels, ensemble_preds)
            
            self.ensemble_results = {
                'auc': auc_score,
                'average_precision': avg_precision,
                'predictions': ensemble_preds.tolist(),
                'labels': labels.tolist(),
                'num_models': int(successful_models),
                'failed_models': failed_models,
                'thresholds': {k: float(v) for k, v in thresholds.items()}
            }
            
            # Add threshold metrics
            for threshold_name, threshold_data in threshold_metrics.items():
                self.ensemble_results[threshold_name] = threshold_data
                
            logger.info(f"Ensemble AUC: {self.ensemble_results['auc']:.4f}")
            
        except Exception as e:
            logger.error(f"Error in ensemble evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def _get_predictions(self, model, loader: DataLoader) -> np.ndarray:
        """Get model predictions"""
        model.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch in loader:
                features = batch['features']
                outputs = model(features)
                preds = torch.sigmoid(outputs)  # CNN outputs logits
                all_preds.extend(preds.cpu().numpy().flatten())
                
        return np.array(all_preds)
        
    def _generate_plots(self):
        """Generate comprehensive visualization plots for model performance"""
        plot_dir = os.path.join(self.eval_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        if not hasattr(self, 'ensemble_results') or self.ensemble_results is None:
            logger.warning("No ensemble results available for plotting")
            return
            
        preds = np.array(self.ensemble_results['predictions'])
        labels = np.array(self.ensemble_results['labels'])
        thresholds = self.ensemble_results['thresholds']
        
        try:
            # 1. ROC Curve with threshold points
            plt.figure(figsize=(10, 6))
            fpr, tpr, roc_thresholds = roc_curve(labels, preds)
            plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {self.ensemble_results["auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            
            # Add threshold points
            threshold_colors = {'default': 'red', 'f1': 'green', 
                              'youdens': 'purple', 'eer': 'orange',
                              'pr_break_even': 'brown'}
            
            for name, threshold in thresholds.items():
                if len(roc_thresholds) > 0:
                    threshold_idx = np.abs(roc_thresholds - threshold).argmin()
                    plt.plot(fpr[threshold_idx], tpr[threshold_idx], 'o', 
                            color=threshold_colors.get(name, 'black'), 
                            label=f'{name.replace("_", " ").title()} ({threshold:.3f})')
            
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve with Optimal Thresholds')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'roc_curve.png'), bbox_inches='tight', dpi=300)
            plt.close()
            
            # 2. Precision-Recall Curve
            plt.figure(figsize=(10, 6))
            precision, recall, pr_thresholds = precision_recall_curve(labels, preds)
            plt.plot(recall, precision, 'b-', 
                    label=f'PR (AP = {self.ensemble_results["average_precision"]:.3f})')
            
            # Add baseline
            plt.axhline(y=sum(labels)/len(labels), color='k', linestyle='--', 
                       label='Random')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'pr_curve.png'), bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f"Plots saved to: {plot_dir}")
            
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            logger.error(traceback.format_exc())
            
    def _save_results(self):
        """Save comprehensive evaluation results"""
        # Prepare summary DataFrame for individual models
        summary_data = []
        for result in self.individual_results:
            base_metrics = {
                'cv': result.get('cv'),
                'fold': result.get('fold'),
                'fold_id': result.get('fold_id')
            }
            
            # Add threshold-independent metrics
            base_metrics.update({
                'roc_auc': result.get('auc', 'N/A'),
                'average_precision': result.get('average_precision', 'N/A'),
                'matthews_corrcoef_raw': result.get('matthews_corrcoef_raw', 'N/A')
            })
            
            # Add metrics at each threshold
            for threshold_name in ['f1', 'youdens', 'eer', 'pr_break_even']:
                metrics_key = f'metrics_at_{threshold_name}'
                if metrics_key in result:
                    threshold_metrics = result[metrics_key]
                    for metric_name, value in threshold_metrics.items():
                        if not isinstance(value, dict):  # Skip nested structures like confusion matrix
                            base_metrics[f'{threshold_name}_{metric_name}'] = value
            
            summary_data.append(base_metrics)
            
        # Create and save summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.eval_dir, 'individual_model_metrics.csv'), index=False)
        
        # Save ensemble results
        if self.ensemble_results:
            ensemble_df = pd.DataFrame([self.ensemble_results])
            ensemble_df.to_csv(os.path.join(self.eval_dir, 'ensemble_metrics.csv'), index=False)
            
        # Save full results including statistical analyses
        results = {
            'config': {
                'task_name': 'CNN',
                'task_type': 'classification',
                'classification': True
            },
            'individual_results': self.individual_results,
            'ensemble_results': self.ensemble_results,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(self.eval_dir, 'full_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
    def _generate_summary_report(self):
        """Generate human-readable summary report"""
        try:
            auc_values = [r['auc'] for r in self.individual_results]
            
            # Initialize the report list
            report = [
                "Final CNN Evaluation Summary",
                "=" * 50,
            ]
            
            # Add basic statistics
            if auc_values:
                report.extend([
                    f"\nIndividual Models ({len(auc_values)} total):",
                    f"Mean AUC: {np.mean(auc_values):.4f} +/- {np.std(auc_values):.4f}",
                    f"Median AUC: {np.median(auc_values):.4f}",
                    f"Min AUC: {np.min(auc_values):.4f}",
                    f"Max AUC: {np.max(auc_values):.4f}"
                ])
                
            # Add detailed individual model metrics
            report.extend(["\nDetailed Individual Model Metrics:"])
            
            # Sort models by AUC for better readability
            sorted_models = sorted(self.individual_results, 
                                 key=lambda x: x.get('auc', 0.0),
                                 reverse=True)
            
            for model in sorted_models:
                cv = model['cv']
                fold = model['fold']
                auc = model['auc']
                avg_precision = model['average_precision']
                mcc = model.get('matthews_corrcoef_raw', 0.0)
                
                report.extend([
                    f"\nCV{cv} Fold{fold} Metrics:",
                    f"  AUC: {auc:.4f}",
                    f"  MCC: {mcc:.4f}",
                    f"  Average Precision: {avg_precision:.4f}"
                ])
                
            # Add ensemble results
            if hasattr(self, 'ensemble_results') and self.ensemble_results:
                report.extend(["\nEnsemble Model Performance:"])
                report.extend([
                    f"AUC: {self.ensemble_results.get('auc', 'N/A'):.4f}"
                ])
                
                optimal_metrics = self.ensemble_results.get('metrics_at_f1', {})
                if optimal_metrics:
                    optimal_threshold = optimal_metrics.get('threshold_value', 0.5)
                    report.extend([
                        f"\nAt optimal threshold {optimal_threshold:.4f}:",
                        f"  Precision: {optimal_metrics.get('precision', 'N/A'):.4f}",
                        f"  Recall: {optimal_metrics.get('recall', 'N/A'):.4f}",
                        f"  F1: {optimal_metrics.get('f1', 'N/A'):.4f}",
                        f"  MCC: {optimal_metrics.get('matthews_corrcoef', 'N/A'):.4f}",
                        f"  Balanced Accuracy: {optimal_metrics.get('balanced_accuracy', 'N/A'):.4f}"
                    ])
                    
            # Add best models section if available
            try:
                best_models = self._select_best_models()
                if best_models:
                    report.extend([
                        f"\nSelected Best 5 Models (Diversity-based Selection):",
                        "-" * 50
                    ])
                    
                    for i, model in enumerate(best_models, 1):
                        metrics = model['metrics']
                        report.append(f"\nModel {i}:")
                        report.append(f"CV: {model['cv']}, Fold: {model['fold']}")
                        report.extend([
                            f"AUC: {metrics['auc']:.4f}",
                            f"MCC: {metrics['matthews_corrcoef']:.4f}",
                            f"Average Precision: {metrics['average_precision']:.4f}"
                        ])
            except Exception as e:
                logger.warning(f"Could not include best models section: {str(e)}")
                
            # Write report to file
            with open(os.path.join(self.eval_dir, 'summary_report.txt'), 'w') as f:
                f.write('\n'.join(report))
                
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            logger.error(traceback.format_exc())
            
    def evaluate_all(self):
        """Run complete evaluation pipeline"""
        try:
            # Core evaluation steps
            self._evaluate_individual_models()
            self._evaluate_ensemble()
            
            # Calculate ensemble metrics with thresholds
            if hasattr(self, 'ensemble_results'):
                ensemble_preds = np.array(self.ensemble_results.get('predictions', []))
                ensemble_labels = np.array(self.ensemble_results.get('labels', []))
                
                if len(ensemble_preds) > 0 and len(ensemble_labels) > 0:
                    # Calculate thresholds and metrics
                    thresholds, threshold_metrics = self._find_optimal_thresholds(
                        ensemble_labels, ensemble_preds
                    )
                    
                    # Store results
                    self.ensemble_results['thresholds'] = thresholds
                    for key, value in threshold_metrics.items():
                        self.ensemble_results[key] = value
                        
                    # Generate plots
                    self._generate_plots()
                    
            # Save best models
            self._save_best_models()
            
            # Save all results
            self._save_results()
            
            # Generate summary report
            self._generate_summary_report()
            
            logger.info("\n🎯 Evaluation completed successfully!")
            
        except Exception as e:
            logger.error(f"Critical error during evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

def main():
    """Main function for final CNN evaluation"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Final comprehensive CNN evaluation')
    parser.add_argument('--cv_results_dir', type=str, 
                       default='outputs',
                       help='Path to cross-validation results directory')
    parser.add_argument('--output_dir', type=str,
                       default=None,
                       help='Output directory for evaluation results')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Set seeds
    set_all_seeds()
    
    # Load data
    try:
        features_path = os.path.join(Config.OUTPUT_DIR, 'features.npy')
        labels_path = os.path.join(Config.OUTPUT_DIR, 'labels.csv')
        
        features = np.load(features_path)
        labels_df = pd.read_csv(labels_path)
        
        # Use optimized hyperparameters
        best_hyperparameters = {
            'layers': 3,
            'filters': 256,
            'kernel_size': 7,
            'dropout': 0.1992056469820396,
            'learning_rate': 0.00019566708363815042
        }
        
        logger.info(f"Loaded data: {features.shape} features, {len(labels_df)} labels")
        logger.info(f"Using hyperparameters: {best_hyperparameters}")
        
        # Create evaluator
        evaluator = FinalCNNEvaluator(
            features=features,
            labels_df=labels_df,
            cv_results_dir=args.cv_results_dir,
            best_hyperparameters=best_hyperparameters,
            output_dir=args.output_dir
        )
        
        # Run complete evaluation
        evaluator.evaluate_all()
        
        logger.info("Final CNN evaluation completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"Required files not found: {e}")
        logger.info("Please ensure cross-validation has been completed first.")
    except Exception as e:
        logger.error(f"Final evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()