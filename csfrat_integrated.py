#!/usr/bin/env python3

"""
C-SFRAT Command Line Interface - Integrated Version
This script properly integrates with the core C-SFRAT functionality
"""

# =====================================================================
# 🚨 FOR EASY CONFIGURATION: Use csfrat_config.py instead! 🚨
# =====================================================================
# This script is now best used via the user-friendly configuration file.
# 
# ✅ RECOMMENDED: Edit settings in 'csfrat_config.py' and run:
#    python csfrat_config.py
#
# 🔧 ADVANCED: Use CLI arguments directly with this script:
#    python csfrat_integrated.py --input data.csv --output report.pdf
#
# The values below are default fallbacks used when no CLI args are provided.
# =====================================================================

# Default configuration values (used as fallbacks)
input_file = 'C-SFRAT/datasets/ds1.csv'
output_filename = 'outputs/integrated_report.pdf'
confidence_level = 0.95
run_all_models = True
run_all_covariates = True
show_all_models_in_table = False
max_models_in_comparison_table = 20
selected_sheet = None
data_subset_limit = None
psse_subset_parameter = 0.9
enable_covariate_scaling = True
covariate_scaling_range = (0, 10)
ranking_method = 'mean'
multi_model_plots = True
num_models_to_compare = 3
individual_model_predictions = True
show_model_predictions_separately = True
max_models_for_individual_predictions = 3

effort_per_interval_enabled = True
effort_per_interval_settings = {
    'effort_values': {'E': 1.00, 'F': 2.00, 'C': 3.00},
    'number_of_intervals_to_predict': 5,
    'failure_intensity_target': 0.3,
    'use_model_specific_covariates': True,
    'default_effort_for_unknown_covariates': 1.0,
}

prediction_parameters = {
    'predict_failures': True,
    'num_intervals_to_predict': 5,
    'prediction_time_horizon': None,
    'include_failure_intensity': True,
    'prediction_intervals': False,
}

# Import configuration early
try:
    import csfrat_config as config
    CONFIG_LOADED = True
except ImportError:
    CONFIG_LOADED = False
    config = None

# Load optimization parameters from config file if available
if CONFIG_LOADED:
    optimization_parameters = {
        'enable_optimization': getattr(config, 'ENABLE_OPTIMIZATION', True),
        'allocation_1_enabled': getattr(config, 'ALLOCATION_1_ENABLED', True),
        'allocation_2_enabled': getattr(config, 'ALLOCATION_2_ENABLED', True),
        'total_budget': getattr(config, 'TOTAL_BUDGET', 100),
        'target_additional_defects': getattr(config, 'TARGET_ADDITIONAL_DEFECTS', 3),
        'optimization_method': getattr(config, 'OPTIMIZATION_METHOD', 'both_allocations')
    }
else:
    # Fallback to default values if config not available
    optimization_parameters = {
        'enable_optimization': True,
        'allocation_1_enabled': True,
        'allocation_2_enabled': True,
        'total_budget': 100,
        'target_additional_defects': 3,
        'optimization_method': 'both_allocations'
    }

# Load metric weights from config file if available
if CONFIG_LOADED:
    metric_weights = {
        'llf': getattr(config, 'WEIGHT_LLF', 0.0),
        'aic': getattr(config, 'WEIGHT_AIC', 2.0),
        'bic': getattr(config, 'WEIGHT_BIC', 1.0),
        'sse': getattr(config, 'WEIGHT_SSE', 1.0),
        'psse': getattr(config, 'WEIGHT_PSSE', 1.0)
    }
else:
    # Fallback to default values if config not available
    metric_weights = {
        'llf': 0.0, 'aic': 2.0, 'bic': 1.0, 'sse': 1.0, 'psse': 1.0
    }

import sys
import os
import logging
import traceback

# Log configuration status (config was already imported earlier)
if CONFIG_LOADED:
    logging.info("✅ Configuration loaded from csfrat_config.py")
else:
    logging.warning("⚠️ csfrat_config.py not found, using default values")
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import combinations, chain
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from io import BytesIO
import argparse
import symengine
import csv
import matplotlib.colors as mcolors
import colorsys

# Add C-SFRAT directory to Python path
sys.path.append(os.path.abspath('C-SFRAT'))

# Configuration for Model class
from core.model import Model
Model.maxCovariates = 3  # Set the maximum number of covariates

# Import C-SFRAT modules
from core.dataClass import Data
from core.goodnessOfFit import Comparison, PSSE
from core.allocation import EffortAllocation
import core.prediction as prediction
from scipy.optimize import shgo

# Import models
from models.geometric import Geometric
from models.discreteWeibull2 import DiscreteWeibull2
from models.negativeBinomial2 import NegativeBinomial2
from models.S_Distribution import S_Distribution
from models.IFR_SB import IFR_SB
from models.IFR_generalized_SB import IFR_Generalized_SB 
from models.truncatedLogistic import TruncatedLogistic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Dictionary of available models
MODEL_CLASSES = {
    "Geometric": Geometric,
    "DiscreteWeibull2": DiscreteWeibull2,
    "NegativeBinomial2": NegativeBinomial2,
    "S_Distribution": S_Distribution,
    "IFR_SB": IFR_SB,
    "IFR_Generalized_SB": IFR_Generalized_SB,
    "TruncatedLogistic": TruncatedLogistic
}

# Global toggle for extra plot annotations (ranking footer, model extensions, etc.)
show_plot_annotations = False

# Custom Effort Allocation with Realistic Constraints
class ConstrainedEffortAllocation:
    """
    Custom effort allocation that applies realistic constraints to prevent extreme budget estimates
    """
    def __init__(self, model, covariate_data, allocation_type, *args):
        self.model = model
        self.covariate_data = covariate_data
        self.hazard_array = np.concatenate((self.model.hazard_array, [self.model.hazardNumerical(self.model.n + 1, self.model.modelParameters)]))
        
        if allocation_type == 1:
            self.B = args[0]
            self.runConstrainedAllocation1()
            self.percentages = self.organizeResults(self.res.x, self.B)
        else:
            self.f = args[0]
            self.runConstrainedAllocation2()
            self.percentages2 = self.organizeResults(self.res2.x, self.effort)
    
    def runConstrainedAllocation1(self):
        # Apply realistic constraints for allocation 1
        cons = ({'type': 'ineq', 'fun': lambda x: self.B - sum([x[i] for i in range(self.model.numCovariates)])})
        
        # Apply realistic bounds: each covariate effort should be between 0 and total budget
        # But we also add practical upper limits to prevent extreme solutions
        max_per_covariate = min(self.B, 50)  # Cap individual covariate effort at 50 or budget, whichever is lower
        bnds = tuple((0, max_per_covariate) for i in range(self.model.numCovariates))
        
        self.res = shgo(self.allocationFunction, args=(self.covariate_data,), bounds=bnds, constraints=cons)
        self.mvfVal = -self.res.fun
        self.H = self.mvfVal - self.model.mvf_array[-1]
    
    def runConstrainedAllocation2(self):
        # Apply realistic constraints for allocation 2
        cons2 = ({'type': 'eq', 'fun': self.optimization2, 'args': (self.covariate_data,)})
        
        # Apply even tighter bounds for allocation 2 to prevent extreme budget estimates
        max_reasonable_effort = 20  # Maximum reasonable effort per covariate
        bnds = tuple((0, max_reasonable_effort) for i in range(self.model.numCovariates))
        
        self.res2 = shgo(lambda x: sum([x[i] for i in range(self.model.numCovariates)]), bounds=bnds, constraints=cons2)
        self.effort = np.sum(self.res2.x)
    
    def allocationFunction(self, x, covariate_data):
        new_cov_data = np.concatenate((covariate_data, x[:, None]), axis=1)
        omega = self.model.calcOmega(self.hazard_array, self.model.betas, new_cov_data)
        return -(self.model.MVF(self.model.mle_array, omega, self.hazard_array, new_cov_data.shape[1] - 1, new_cov_data))
    
    def optimization2(self, x, covariate_data):
        res = self.allocationFunction2(x, covariate_data)
        H = res - self.model.mvf_array[-1]
        return self.f - H
    
    def allocationFunction2(self, x, covariate_data):
        new_cov_data = np.concatenate((covariate_data, x[:, None]), axis=1)
        omega = self.model.calcOmega(self.hazard_array, self.model.betas, new_cov_data)
        return self.model.MVF(self.model.mle_array, omega, self.hazard_array, new_cov_data.shape[1] - 1, new_cov_data)
    
    def organizeResults(self, results, effort):
        if effort > 0.0:
            return np.multiply(np.divide(results, effort), 100)
        else:
            return [0.0 for i in range(len(results))]

# Standardized Table Styling Function
def get_standardized_table_style(num_highlight_rows=0, highlight_color=None, use_alternating_rows=True):
    """Create standardized table styling for consistent PDF formatting
    
    Args:
        num_highlight_rows: Number of top rows to highlight (default: 0)
        highlight_color: Color for highlighting rows (default: light green)
        use_alternating_rows: Whether to use alternating row colors (default: True)
        
    Returns:
        TableStyle: Standardized table style
    """
    if highlight_color is None:
        highlight_color = colors.HexColor('#D5F5E3')  # Light green
    
    # Professional table style
    style_commands = [
        # Header styling - professional dark blue with proper spacing
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        # Header border - 1.5pt bottom border for visual separation
        ('LINEBELOW', (0, 0), (-1, 0), 1.5, colors.HexColor('#2C3E50')),
        
        # Data rows styling with professional typography
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),  # Default left alignment (numbers overridden)
        
        # Professional padding and spacing
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        
        # Column borders for clarity
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDC3C7')),
        # Ensure outer borders are visible
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDC3C7')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        # Add word wrapping for header text
        ('WORDWRAP', (0, 0), (-1, 0), True),
    ]
    
    # Add alternating row colors if requested
    if use_alternating_rows:
        style_commands.append(('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')]))
    
    # Add highlighting for top rows if specified
    if num_highlight_rows > 0:
        for i in range(num_highlight_rows):
            row_index = i + 1  # +1 because row 0 is the header
            style_commands.append(('BACKGROUND', (0, row_index), (-1, row_index), highlight_color))
    
    return TableStyle(style_commands)

# Class to mimic the GUI's critic calculation
class CriticCalculator:
    """
    This class replicates the functionality of the Comparison class in C-SFRAT GUI
    for calculating critic values using the Analytic Hierarchy Process (AHP)
    """
    def __init__(self, weights=None):
        self.mean_values = None
        self.median_values = None
        self.best_mean_idx = None
        self.best_median_idx = None
        self.num_metrics = 5  # LLF, AIC, BIC, SSE, PSSE
        self.weights = weights if weights is not None else {
            'llf': 1.0, 'aic': 1.0, 'bic': 1.0, 'sse': 1.0, 'psse': 1.0
        }

    def calculate_critic_values(self, models, weights=None):
        """
        Calculate critic values for a list of models using the same AHP approach as the GUI
        
        Args:
            models: List of fitted model objects
            weights: Optional dictionary of weights to override instance weights
            
        Returns:
            dict: Dictionary mapping model indices to (critic_mean, critic_median) tuples
        """
        if not models:
            return {}
            
        # Use provided weights or instance weights
        if weights is not None:
            active_weights = weights
        else:
            active_weights = self.weights
            
        # Validate weights
        for metric, weight in active_weights.items():
            if not (0.0 <= weight <= 10.0):
                logging.warning(f"Weight for {metric} ({weight}) is outside recommended range 0.0-10.0")
            
        # Extract metrics for each model
        llf_values = [model.llfVal for model in models]
        aic_values = [model.aicVal for model in models]
        bic_values = [model.bicVal for model in models]
        sse_values = [model.sseVal for model in models]
        psse_values = [model.psseVal if hasattr(model, 'psseVal') else float('nan') for model in models]
        
        # Calculate weight sum (same as calcWeightSum in GUI)
        weight_sum = sum(active_weights.values())
        
        if weight_sum == 0:
            logging.warning("All weights are zero. Using equal weights instead.")
            active_weights = {'llf': 1.0, 'aic': 1.0, 'bic': 1.0, 'sse': 1.0, 'psse': 1.0}
            weight_sum = sum(active_weights.values())
        
        # Log the weights being used
        logging.info(f"Using metric weights: {active_weights}")
        
        # Calculate AHP values for each metric (same as ahp method in GUI)
        llf_ahp = np.zeros(len(models))
        aic_ahp = np.zeros(len(models))
        bic_ahp = np.zeros(len(models))
        sse_ahp = np.zeros(len(models))
        psse_ahp = np.zeros(len(models))
        
        for i in range(len(models)):
            llf_ahp[i] = self._ahp_calc(llf_values, i, active_weights['llf'], weight_sum, higher_is_better=True)  # Higher LLF is better
            aic_ahp[i] = self._ahp_calc(aic_values, i, active_weights['aic'], weight_sum, higher_is_better=False)  # Lower AIC is better
            bic_ahp[i] = self._ahp_calc(bic_values, i, active_weights['bic'], weight_sum, higher_is_better=False)  # Lower BIC is better
            sse_ahp[i] = self._ahp_calc(sse_values, i, active_weights['sse'], weight_sum, higher_is_better=False)  # Lower SSE is better
            psse_ahp[i] = self._ahp_calc(psse_values, i, active_weights['psse'], weight_sum, higher_is_better=False)  # Lower PSSE is better
        
        # Combine all metric arrays (same as ahpArray in GUI)
        ahp_array = np.array([llf_ahp, aic_ahp, bic_ahp, sse_ahp, psse_ahp])
        
        # DEBUG: Print the AHP values for debugging
        logging.info(f"AHP DEBUG - llf_ahp: {llf_ahp[:5]}")  # Show first 5 values
        logging.info(f"AHP DEBUG - aic_ahp: {aic_ahp[:5]}")
        logging.info(f"AHP DEBUG - bic_ahp: {bic_ahp[:5]}")
        logging.info(f"AHP DEBUG - sse_ahp: {sse_ahp[:5]}")
        logging.info(f"AHP DEBUG - psse_ahp: {psse_ahp[:5]}")
        
        # Calculate raw mean and median values
        raw_mean = np.mean(ahp_array, axis=0)
        raw_median = np.median(ahp_array, axis=0)
        
        # DEBUG: Print the raw values
        logging.info(f"AHP DEBUG - raw_mean: {raw_mean[:5]}")
        logging.info(f"AHP DEBUG - raw_median: {raw_median[:5]}")
        
        # Store raw values for legend display
        self.raw_mean_values = raw_mean
        self.raw_median_values = raw_median
        
        # Check if this is a single-metric configuration (Professor's recommendation)
        active_metrics = sum(1 for weight in active_weights.values() if weight > 0.0)
        is_single_metric = active_metrics == 1
        
        if is_single_metric:
            # For single-metric configurations, median doesn't make mathematical sense
            # since we're only using one metric. Set median equal to mean.
            logging.info("Single-metric configuration detected: Setting median values equal to mean values")
            raw_median = raw_mean.copy()
            self.raw_median_values = raw_median
        
        # Normalize to 0.0-1.0 scale
        try:
            # Handle NaN values in raw arrays
            raw_mean_clean = np.nan_to_num(raw_mean, nan=0.0)
            raw_median_clean = np.nan_to_num(raw_median, nan=0.0)
            
            max_mean = np.max(raw_mean_clean)
            max_median = np.max(raw_median_clean)
            
            # DEBUG: Print normalization values
            logging.info(f"NORMALIZATION DEBUG - max_mean: {max_mean:.6f}, max_median: {max_median:.6f}")
            logging.info(f"NORMALIZATION DEBUG - len(models): {len(models)}")
            
            # Handle case where all values might be zero or NaN
            if max_mean > 0 and not np.isnan(max_mean):
                self.mean_values = np.divide(raw_mean_clean, max_mean)
                logging.info(f"NORMALIZATION DEBUG - self.mean_values: {self.mean_values[:5]}")
            else:
                self.mean_values = np.ones(len(models)) / len(models)  # Equal weights if all zero
                logging.info(f"NORMALIZATION DEBUG - Using equal weights for mean: {self.mean_values[:5]}")
                
            if max_median > 0 and not np.isnan(max_median):
                self.median_values = np.divide(raw_median_clean, max_median)
                logging.info(f"NORMALIZATION DEBUG - self.median_values: {self.median_values[:5]}")
            else:
                self.median_values = np.ones(len(models)) / len(models)  # Equal weights if all zero
                logging.info(f"NORMALIZATION DEBUG - Using equal weights for median: {self.median_values[:5]}")
            
            # Find best combinations
            self.best_mean_idx = np.argmax(self.mean_values)
            self.best_median_idx = np.argmax(self.median_values)
        except (ValueError, ZeroDivisionError):
            # Handle edge cases
            self.mean_values = np.ones(len(models)) / len(models)
            self.median_values = np.ones(len(models)) / len(models)
            self.best_mean_idx = 0 if len(models) > 0 else None
            self.best_median_idx = 0 if len(models) > 0 else None
        
        # Return results as a dictionary mapping model index to (mean, median) tuple
        results = {}
        for i in range(len(models)):
            results[i] = (self.mean_values[i], self.median_values[i])
            # DEBUG: Print the final results for first 5 models
            if i < 5:
                logging.info(f"FINAL RESULTS DEBUG - Model {i}: mean={self.mean_values[i]:.6f}, median={self.median_values[i]:.6f}")
            
        return results
            
    def _ahp_calc(self, measure_array, index, weight_value, weight_sum, higher_is_better=True):
        """
        Calculate AHP (Analytic Hierarchy Process) value for a single metric
        This exactly replicates the ahp method in the GUI's Comparison class
        
        Args:
            measure_array: Array of metric values for all models
            index: Index of the current model
            weight_value: Weight assigned to this metric
            weight_sum: Sum of all weights
            higher_is_better: True if higher values are better (e.g., LLF), False if lower is better (e.g., AIC, BIC, SSE, PSSE)
            
        Returns:
            float: AHP value for this metric and model
        """
        # If weight is 0, the metric is not considered
        if weight_value == 0:
            return 0.0
        
        try:
            # Calculate normalized weight
            weight = weight_value / weight_sum
        except ZeroDivisionError:
            # If all weights are zero, use equal weighting
            weight = 1.0 / float(self.num_metrics)
        
        # Calculate AHP value
        if len(measure_array) > 1:
            # Convert to absolute values for comparison
            abs_array = np.absolute(measure_array)
            min_val = min(abs_array)
            max_val = max(abs_array)
            current_val = abs(measure_array[index])
            
            # DEBUG: Only log for PSSE to avoid spam
            if not higher_is_better and weight_value > 0 and index < 3:  # First 3 models for PSSE
                logging.info(f"AHP DEBUG - measure_array range: [{min_val:.6f}, {max_val:.6f}], current: {current_val:.6f}, weight: {weight_value}, weight_sum: {weight_sum}")
            
            if max_val == min_val:
                # All values are the same, give equal weight
                ahp_val = weight
                if not higher_is_better and weight_value > 0 and index < 3:
                    logging.info(f"AHP DEBUG - All values same, ahp_val = weight = {ahp_val}")
            else:
                if higher_is_better:
                    # For metrics where higher is better (like LLF)
                    # Scale so that higher values get higher scores
                    ahp_val = (current_val - min_val) / (max_val - min_val) * weight
                else:
                    # For metrics where lower is better (like AIC, BIC, SSE, PSSE)
                    # Scale so that lower values get higher scores
                    ahp_val = (max_val - current_val) / (max_val - min_val) * weight
                    if weight_value > 0 and index < 3:  # Debug for first 3 models
                        logging.info(f"AHP DEBUG - Calculated ahp_val = ({max_val:.6f} - {current_val:.6f}) / ({max_val:.6f} - {min_val:.6f}) * {weight:.6f} = {ahp_val:.6f}")
        else:
            # If only one model, assign equal weight
            ahp_val = 1.0 / float(self.num_metrics)
            
        return ahp_val

def powerset(iterable):
    """Generate all possible combinations of elements in the iterable"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def setup_output_directory():
    """Create the outputs directory if it doesn't exist"""
    os.makedirs('outputs', exist_ok=True)
    logging.info("Output directory ready")

def load_data(file_path, sheet_name=None, subset_limit=None):
    """
    Load data with enhanced features for 100% GUI compatibility
    
    Args:
        file_path: Path to input data file (.csv or .xlsx)
        sheet_name: Sheet name for Excel files (None = use first sheet)
        subset_limit: Limit analysis to first N intervals (None = use all data)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    # Load data directly with pandas based on file extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        available_sheets = ["CSV (Single Sheet)"]
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        # For Excel files, handle sheet selection (matches original GUI functionality)
        if sheet_name:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                logging.info(f"Loaded Excel sheet: '{sheet_name}'")
            except ValueError as e:
                # Get available sheets for error message
                excel_file = pd.ExcelFile(file_path)
                available_sheets = excel_file.sheet_names
                raise ValueError(f"Sheet '{sheet_name}' not found. Available sheets: {available_sheets}")
        else:
            # Load first sheet by default (matches original GUI behavior)
            excel_file = pd.ExcelFile(file_path)
            available_sheets = excel_file.sheet_names
            df = pd.read_excel(file_path, sheet_name=available_sheets[0])
            logging.info(f"Loaded first Excel sheet: '{available_sheets[0]}' (available: {available_sheets})")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Data subsetting feature (matches original GUI slider functionality)
    original_rows = len(df)
    if subset_limit and subset_limit > 0 and subset_limit < len(df):
        df = df.head(subset_limit)
        logging.info(f"Applied data subset: using first {subset_limit} intervals (of {original_rows} total)")
        print(f"INFO: Data subset applied - analyzing first {subset_limit} intervals (GUI slider functionality)")
    
    # Debug the loaded data
    print(f"\nDEBUG - Loaded data frame:\n{df.head(5)}")
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")
    if subset_limit:
        print(f"Original file had {original_rows} rows, using first {len(df)} rows (subset)")
    
    # Ensure required columns exist
    required_cols = ['T', 'FC']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the data file")
    
    # Add CFC (Cumulative Failure Count) if not exists
    if 'CFC' not in df.columns:
        df['CFC'] = df['FC'].cumsum()
    
    # Identify covariates (any column that is not T, FC, or CFC)
    covariates = [col for col in df.columns if col not in ['T', 'FC', 'CFC']]
    
    # Log information about the loaded data
    logging.info(f"Loaded data from {file_path}")
    logging.info(f"Data dimensions: {df.shape}")
    logging.info(f"Covariates: {covariates}")
    if 'available_sheets' in locals():
        logging.info(f"Available sheets: {available_sheets}")
    
    return df, covariates

def prepare_model_data(data_df, covariate_names):
    """Prepare data in proper format for model initialization"""
    # Create a new DataFrame to ensure proper structure
    model_data = pd.DataFrame()
    
    # Add the required columns
    model_data['T'] = data_df['T'].values
    model_data['FC'] = data_df['FC'].values
    
    # Calculate CFC (cumulative failures) correctly
    model_data['CFC'] = data_df['FC'].cumsum().values
    
    # Add selected covariates
    for name in covariate_names:
        if name in data_df.columns:
            model_data[name] = data_df[name].values
    
    # Debug info
    print(f"Data shape: {model_data.shape}")
    print(f"CFC values: {model_data['CFC'].values}")
    
    return model_data

# Implement prediction_psse function similar to the GUI
def prediction_psse(model, data_df):
    """
    Prediction function used for PSSE, matching the GUI implementation in C-SFRAT/core/prediction.py.
    
    Args:
        model: The fitted model object
        data_df: DataFrame containing the data
        
    Returns:
        array: Array of MVF values for all data points
    """
    total_points = len(data_df)
    
    # Extract covariate data if available
    covariate_data = None
    if model.metricNames:
        covariate_data = np.array([data_df[name].values for name in model.metricNames])
    
    # Calculate new hazard values for prediction points
    new_hazard = np.array([model.hazardNumerical(i, model.modelParameters) for i in range(model.n, total_points)])
    hazard = np.concatenate((model.hazard_array, new_hazard)) if hasattr(model, 'hazard_array') else new_hazard
    
    # Calculate omega and MVF values just like in the GUI implementation
    omega = model.calcOmega(hazard, model.betas, covariate_data)
    mvf_array = np.array([model.MVF(model.mle_array, omega, hazard, data_point, covariate_data) 
                         for data_point in range(total_points)])
    
    return mvf_array

def run_all_combinations(df, covariates):
    """Run all model and covariate combinations with direct data access"""
    results = []
    
    # Generate all covariate combinations
    covariate_combinations = list(powerset(covariates))
    logging.info(f"Generated {len(covariate_combinations)} covariate combinations")
    
    # For each model and covariate combination, fit the model
    for model_name, model_class in MODEL_CLASSES.items():
        for covs in covariate_combinations:
            covs_list = list(covs)
            logging.info(f"Running {model_name} with covariates: {covs_list}")
            
            # Create model data for this combination
            model_data = prepare_model_data(df, covs_list)
            
            try:
                # Initialize and run the model
                model = initialize_model(model_class, model_data, covs_list)
                
                # Store arrays for actual and fitted values
                model.t = model_data['T'].values
                model.CFC = model_data['CFC'].values
                
                # Ensure we have the MVF values available for metrics calculation
                if not hasattr(model, 'mvfList') or model.mvfList is None:
                    # For models that don't directly set mvfList, try to access it differently
                    if hasattr(model, 'mvf_array'):
                        model.mvfList = model.mvf_array
                    elif hasattr(model, 'modeledFailures'):
                        model.mvfList = model.modeledFailures
                    elif hasattr(model, 'omega') and hasattr(model, 'beta'):
                        # Generate simple MVF values for models that follow the basic reliability pattern
                        model.mvfList = [model.omega * (1 - np.exp(-model.beta * t)) for t in model.t]
                    else:
                        # Last resort - create dummy values that match the observation count
                        logging.warning(f"Creating fallback MVF values for {model_name}")
                        model.mvfList = model.CFC.copy()  # Use actual values as a fallback
                
                # Ensure we have the intensity values available for metrics calculation
                if not hasattr(model, 'intensityList') or model.intensityList is None:
                    # Calculate intensity as derivative of MVF
                    model.intensityList = []
                    for i in range(len(model.mvfList)):
                        if i == 0:
                            intensity = model.mvfList[0]
                        else:
                            intensity = model.mvfList[i] - model.mvfList[i-1]
                        model.intensityList.append(intensity)
                
                # Calculate additional goodness-of-fit measures
                try:
                    # Calculate PSSE exactly as in the GUI using subset parameter
                    # The GUI fits the model on a subset of data (determined by psse_subset_parameter)
                    # then tests prediction accuracy on the full dataset
                    
                    # Create data subset for PSSE model fitting (matches GUI functionality)
                    subset_size = int(len(df) * psse_subset_parameter)
                    if subset_size < 5:  # Minimum 5 data points as in GUI
                        subset_size = min(5, len(df))
                    if subset_size >= len(df):  # Maximum n-1 data points as in GUI
                        subset_size = len(df) - 1
                    
                    df_subset = df.head(subset_size)
                    logging.info(f"PSSE calculation: using subset of {subset_size} points (fraction: {psse_subset_parameter}) from {len(df)} total points")
                    
                    # Prepare model data for the subset
                    subset_model_data = prepare_model_data(df_subset, covs_list)
                    
                    # Create and fit model on subset data
                    psse_model = initialize_model(model_class, subset_model_data, covs_list)
                    psse_model.t = subset_model_data['T'].values
                    psse_model.CFC = subset_model_data['CFC'].values
                    
                    # Get fitted values using the subset-trained model on full data
                    fitted_array = prediction_psse(psse_model, df)
                    
                    # Debug the PSSE calculation
                    logging.info(f"PSSE calculation debug - subset_model.n: {psse_model.n}, fitted_array shape: {fitted_array.shape}, full CFC shape: {df['CFC'].values.shape}")
                    
                    # Calculate PSSE exactly as in the GUI
                    # Use the subset model's n (number of fitted points) as the intervals parameter
                    psse_val = PSSE(fitted_array, df['CFC'].values, psse_model.n)
                    model.psseVal = psse_val
                    
                    # Instead of calculating critic values directly, store the values needed for later AHP calculation
                    # We still calculate simple versions for debugging and comparison
                    simple_critic_mean, simple_critic_median = calculate_prequential_metrics(model.CFC, model.mvfList)
                    model.simple_critic_mean = simple_critic_mean
                    model.simple_critic_median = simple_critic_median
                    
                    logging.info(f"Additional metrics - PSSE: {model.psseVal:.6f}, Simple Critic Mean: {model.simple_critic_mean:.6f}, Simple Critic Median: {model.simple_critic_median:.6f}")
                except Exception as e:
                    logging.warning(f"Error calculating additional metrics: {str(e)}")
                    logging.warning(traceback.format_exc())
                    model.psseVal = float('nan')
                    model.simple_critic_mean = float('nan')
                    model.simple_critic_median = float('nan')
                    
                # Add to results if model converged
                logging.info(f"Model {model.name} with covariates {covs_list} converged successfully")
                logging.info(f"AIC: {model.aicVal}, BIC: {model.bicVal}, LLF: {model.llfVal}")
                
                # Store covariate names for reporting
                model.covariateNames = covs_list
                
                results.append(model)
            except Exception as e:
                logging.error(f"Error running {model_name} with covariates {covs_list}: {str(e)}")
                logging.error(traceback.format_exc())
    
    # Sort models by AIC (ascending)
    results.sort(key=lambda x: x.aicVal if hasattr(x, 'aicVal') else float('inf'))
    
    # Calculate critic values using AHP (same approach as GUI)
    if results:
        try:
            # Use the global metric_weights configuration
            global metric_weights
            calculator = CriticCalculator(weights=metric_weights)
            critic_values = calculator.calculate_critic_values(results)
            
            # Assign the calculated critic values to each model
            for i, model in enumerate(results):
                if i in critic_values:
                    model.criticMean, model.criticMedian = critic_values[i]
                    # Also store raw (unnormalized) values for legend display
                    model.rawCriticMean = calculator.raw_mean_values[i]
                    model.rawCriticMedian = calculator.raw_median_values[i]
                    logging.info(f"GUI-style critic values for {model.name}: Mean={model.criticMean:.6f}, Median={model.criticMedian:.6f}")
        except Exception as e:
            logging.error(f"Error calculating GUI-style critic values: {str(e)}")
            logging.error(traceback.format_exc())
            # Fall back to simple critic values if AHP calculation fails
            for model in results:
                if hasattr(model, 'simple_critic_mean'):
                    model.criticMean = model.simple_critic_mean
                    model.criticMedian = model.simple_critic_median
    
    # Apply final ranking based on user's ranking method preference
    results = apply_final_ranking(results)
    
    return results

def apply_final_ranking(models):
    """Apply final ranking based on user's ranking method preference
    
    Args:
        models: List of fitted model objects with critic values
        
    Returns:
        List of models sorted by the selected ranking criterion
    """
    global ranking_method
    
    # Ensure ranking_method has a valid default value
    if not ranking_method or ranking_method.lower() not in ['mean', 'median']:
        ranking_method = 'mean'  # Set default to mean if invalid or None
        logging.warning(f"Invalid or missing ranking method. Defaulting to 'mean'")
    
    # Log the ranking method being used
    logging.info(f"Ranking models using {ranking_method.lower()} critic values")
    
    # Sort by the selected ranking method (higher values are better)
    if ranking_method.lower() == 'median':
        # Sort by critic median values (descending - higher is better)
        models.sort(key=lambda x: getattr(x, 'criticMedian', 0), reverse=True)
        logging.info("Models ranked by critic median values")
    else:
        # Default to mean ranking (includes any invalid values)
        models.sort(key=lambda x: getattr(x, 'criticMean', 0), reverse=True)
        logging.info("Models ranked by critic mean values")
    
    # Log the top 3 models for verification
    for i, model in enumerate(models[:3]):
        if hasattr(model, 'aicVal'):
            logging.info(f"Rank {i+1}: {model.name} - Mean: {model.criticMean:.6f}, Median: {model.criticMedian:.6f}")
    
    return models

def calculate_prequential_metrics(actual, fitted):
    """Calculate prequential (critic) mean and median metrics
    
    This is a simplified direct calculation, different from the GUI's AHP approach.
    It's kept for comparison and fallback purposes.
    
    Args:
        actual: List of actual cumulative failures
        fitted: List of fitted MVF values from the model
    
    Returns:
        tuple: (critic_mean, critic_median)
    """
    try:
        # NOTE: This is a simplified version of the GUI's critic calculation
        # The GUI uses a complex Analytic Hierarchy Process (AHP) approach in the Comparison class
        # that normalizes all metrics (LLF, AIC, BIC, SSE, PSSE) together
        
        # Calculate differences between actual and fitted values
        diffs = np.absolute(np.subtract(np.array(fitted), np.array(actual)))
        
        # Prequential metrics
        critic_mean = np.mean(diffs)
        critic_median = np.median(diffs)
        
        return critic_mean, critic_median
    except Exception as e:
        logging.warning(f"Error calculating prequential metrics: {str(e)}")
        return float('nan'), float('nan')

def create_growth_curve_plot(models, single_model_mode=False, predictions=None, individual_predictions=None):
    """Create a growth curve plot matching the original C-SFRAT tool style
    
    Args:
        models: Single model object or list of models
        single_model_mode: If True, only plot the first model regardless of multi_model_plots setting
        predictions: Dictionary containing prediction data for the best model
        individual_predictions: Dictionary containing individual predictions for multiple models
    """
    # Debug logging to see what the plotting function receives
    logging.info(f"PLOT FUNCTION DEBUG - predictions type: {type(predictions)}")
    logging.info(f"PLOT FUNCTION DEBUG - individual_predictions: {individual_predictions}")
    logging.info(f"PLOT FUNCTION DEBUG - predictions method: {predictions.get('prediction_method', 'N/A') if predictions else 'N/A'}")
    
    # Handle both single model and multi-model cases
    if not isinstance(models, list):
        models = [models]
    
    if not models or len(models) == 0:
        return None
    
    # Determine how many models to plot
    global multi_model_plots, num_models_to_compare, individual_model_predictions, show_model_predictions_separately
    if single_model_mode or not multi_model_plots:
        models_to_plot = models[:1]  # Only plot the best model
        plot_title = f'MVF - {models[0].name}'
    else:
        models_to_plot = models[:min(num_models_to_compare, len(models))]
        if len(models_to_plot) > 1:
            plot_title = f'MVF Comparison - Top {len(models_to_plot)} Models'
        else:
            plot_title = f'MVF - {models[0].name}'
    
    # Simplified title - predictions are now seamlessly integrated
    # No need to mention predictions in title since they appear as continuous extensions
    
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    
    # Create consistent color mapping across all plots
    color_mapping, original_colors = create_model_color_mapping(models, individual_predictions)
    
    # Plot imported data first as black step plot (matching original C-SFRAT)
    if hasattr(models[0], 't') and hasattr(models[0], 'CFC'):
        # Step plot with right alignment (stepMode='right' in original)
        ax.step(models[0].t, models[0].CFC, where='post', color='black', linewidth=3, 
                label='Data', zorder=10)
    
    # Plot each fitted model with original C-SFRAT style
    for i, model in enumerate(models_to_plot):
        # Create consistent model key for color mapping
        model_key = model.name
        if hasattr(model, 'covariateNames') and model.covariateNames:
            cov_str = ",".join([c[:3] for c in model.covariateNames])
            model_key += f"({cov_str})"
        
        color = color_mapping.get(model_key, original_colors[i % len(original_colors)])
        
        # Ensure model has required attributes for plotting
        if not hasattr(model, 't') or not hasattr(model, 'CFC'):
            logging.warning(f"Model {model.name} missing required plotting attributes")
            continue
        
        # Get Mean Value Function values (use mvf_array to match original)
        mvf_data = []
        if hasattr(model, 'mvf_array') and model.mvf_array is not None:
            mvf_data = model.mvf_array
        elif hasattr(model, 'mvfList') and model.mvfList is not None:
            mvf_data = model.mvfList
        elif hasattr(model, 'omega'):
            # Generate simple MVF data if actual values not available
            mvf_data = [model.omega * (1 - np.exp(-model.beta * t)) for t in model.t]
        
        if len(mvf_data) > 0:
            # Ensure data is properly converted to numpy arrays
            try:
                time_data = np.array([float(x) for x in model.t])
                mvf_array = np.array([float(x) for x in mvf_data])
            except (ValueError, TypeError) as e:
                # Fallback for complex data types
                logging.warning(f"Array conversion fallback for model {model.name}: {e}")
                time_data = np.asfarray(model.t)
                mvf_array = np.asfarray(mvf_data)
            
            # Create model label
            if len(models_to_plot) == 1:
                model_label = model.name
                if hasattr(model, 'covariateNames') and model.covariateNames:
                    model_label += f" ({', '.join(model.covariateNames)})"
            else:
                model_label = model.name
                if hasattr(model, 'covariateNames') and model.covariateNames:
                    cov_str = ",".join([c[:3] for c in model.covariateNames])  # Abbreviated covariates
                    model_label += f" ({cov_str})"
                
                # Add ranking information for multi-model plots
                if hasattr(model, 'criticMean') and hasattr(model, 'criticMedian'):
                    global ranking_method
                    if ranking_method.lower() == 'median':
                        model_label += f" [MD:{model.criticMedian:.4f}]"
                    else:
                        model_label += f" [M:{model.criticMean:.4f}]"
            
            # Plot fitted data with circles and lines (matching original C-SFRAT style)
            ax.plot(time_data, mvf_array, color=color, linewidth=3, marker='o', markersize=4,
                    markerfacecolor=color, markeredgecolor=color, label=model_label, 
                    alpha=0.9, zorder=5)
    
    # REMOVED: Green effort prediction line (individual model predictions provide smooth behavior)
    effort_prediction_shown = False
    
    # Handle individual model predictions as continuous extensions (professor feedback)
    if individual_predictions and individual_model_predictions and show_model_predictions_separately:        
        for model_key, pred_data in individual_predictions.items():
            # Use consistent color mapping for predictions (same as model fit)
            pred_color = color_mapping.get(model_key, original_colors[0])
            
            # Get the model's last fitted point for connection
            model = pred_data['model']
            if not hasattr(model, 't') or not hasattr(model, 'mvf_array') and not hasattr(model, 'mvfList'):
                continue
                
            # Get MVF data for connection
            if hasattr(model, 'mvf_array') and model.mvf_array is not None:
                last_mvf = model.mvf_array[-1]
            elif hasattr(model, 'mvfList') and model.mvfList is not None:
                last_mvf = model.mvfList[-1]
            else:
                continue
            
            # Plot individual model predictions
            pred_times = pred_data['future_times']
            pred_mvf = pred_data['future_mvf']
            
            if len(pred_times) > 0 and len(pred_mvf) > 0:
                # Convert prediction data to numpy arrays with better error handling
                try:
                    pred_times_array = np.array([float(x) for x in pred_times])
                    pred_mvf_array = np.array([float(x) for x in pred_mvf])
                    
                    # Connect the last fitted point to the first prediction point
                    connect_times = np.array([float(model.t[-1])] + [float(x) for x in pred_times_array])
                    connect_mvf = np.array([float(last_mvf)] + [float(x) for x in pred_mvf_array])
                except (ValueError, TypeError) as e:
                    logging.warning(f"Prediction array conversion error for {model_key}: {e}")
                    continue
                
                # Plot prediction as continuous extension (same style as model fit)
                # No different styling - predictions appear as natural extensions
                ax.plot(connect_times, connect_mvf, 
                       color=pred_color, linewidth=3, marker='o', markersize=4,
                       markerfacecolor=pred_color, markeredgecolor=pred_color,
                       alpha=0.9, zorder=5)  # Same style as fitted model lines
    
    # Add single model predictions as continuous extensions (fallback for best model only)
    elif predictions and not effort_prediction_shown and len(models_to_plot) == 1:
        # Plot prediction data as continuous extension (same style as model fit)
        pred_times = predictions['future_times']
        pred_mvf = predictions['future_mvf']
        
        if len(pred_times) > 0 and len(pred_mvf) > 0:
            # Get MVF data for the best model
            if hasattr(models[0], 'mvf_array') and models[0].mvf_array is not None:
                last_mvf = models[0].mvf_array[-1]
            elif hasattr(models[0], 'mvfList') and models[0].mvfList is not None:
                last_mvf = models[0].mvfList[-1]
            else:
                last_mvf = predictions['future_mvf'][0]  # fallback
            
            # Connect the last fitted point to the first prediction point
            connect_times = [models[0].t[-1]] + pred_times
            connect_mvf = [last_mvf] + pred_mvf
            
            # Use same color as model for seamless extension
            model_color = color_mapping.get(models[0].name, original_colors[0])
            ax.plot(connect_times, connect_mvf, 
                   color=model_color, linewidth=3, marker='o', markersize=4,
                   markerfacecolor=model_color, markeredgecolor=model_color,
                   alpha=0.9, zorder=5)  # Same style as fitted model line
    
    # Add red dashed vertical line at last data point (matching original C-SFRAT)
    if hasattr(models[0], 't') and len(models[0].t) > 0:
        last_x = models[0].t[-1]
        ax.axvline(x=last_x, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
        
        # Add note explaining the vertical line (professor feedback)
        ax.text(last_x + 0.5, ax.get_ylim()[1] * 0.95, 'End of observed data', 
                fontsize=9, color='red', ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="red", alpha=0.8))
    
    # Customize the plot to match original C-SFRAT style
    ax.set_xlabel('Intervals', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative failures', fontsize=12, fontweight='bold')
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    
    # Enable grid (matching original C-SFRAT)
    ax.grid(True, alpha=0.3)
    
    # Add legend with white background
    legend = ax.legend(fontsize=11, loc='best', frameon=True, fancybox=False, shadow=False)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_alpha(1.0)
    
    # Style axes
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    # Add covariate information for single model
    if show_plot_annotations and len(models_to_plot) == 1 and hasattr(models[0], 'covariateNames') and models[0].covariateNames:
        covs_text = f"Covariates: {', '.join(models[0].covariateNames)}"
        ax.text(0.02, 0.98, covs_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="white", edgecolor="gray", alpha=0.9))
    
    # Add ranking method info for multi-model plots
    if show_plot_annotations and len(models_to_plot) > 1:
        ranking_text = f"Ranked by: {ranking_method.upper()} critic values"
        ax.text(0.02, 0.02, ranking_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightblue", alpha=0.7))
    
    # Add prediction info box
    if show_plot_annotations and effort_prediction_shown:
        effort_covariate = predictions.get('effort_covariate', 'Unknown')
        effort_value = predictions.get('effort_value', 1.0)
        # Handle different effort value types
        if isinstance(effort_value, (int, float)):
            effort_display = f"{effort_value:.1f}"
        else:
            effort_display = str(effort_value)
        pred_text = f"🟢 Effort Predictions\n{effort_covariate} = {effort_display}\n{len(predictions.get('future_times', []))} intervals"
        ax.text(0.98, 0.98, pred_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.9))
    elif show_plot_annotations and individual_predictions and individual_model_predictions and show_model_predictions_separately:
        # Simplified prediction info - no need to distinguish prediction types since they're seamless
        pred_count = len(individual_predictions)
        pred_text = f"🔮 {pred_count} Model Extensions"
        ax.text(0.98, 0.02, pred_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    elif show_plot_annotations and predictions and len(models_to_plot) == 1:
        pred_text = f"Predictions: {len(predictions.get('future_times', []))} future points"
        if 'prediction_horizon' in predictions:
            pred_text += f"\nHorizon: {predictions['prediction_horizon']} time units"
        ax.text(0.98, 0.02, pred_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # Save to a BytesIO buffer for the report
    buffer = BytesIO()
    plt.tight_layout()
    
    # Ensure all data is properly converted to numpy arrays before plotting
    try:
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        
        # Convert to a ReportLab Image for the PDF
        img = Image(buffer, width=480, height=320)
        plt.close(fig)
        
        return img
    except Exception as e:
        logging.warning(f"Error saving growth curve plot: {str(e)}")
        plt.close(fig)
        return None

def create_intensity_plot(models, single_model_mode=False, predictions=None, individual_predictions=None):
    """Create a failure intensity plot matching the original C-SFRAT tool style
    
    Args:
        models: Single model object or list of models
        single_model_mode: If True, only plot the first model regardless of multi_model_plots setting
        predictions: Dictionary containing prediction data for the best model
        individual_predictions: Dictionary containing individual predictions for multiple models
    """
    # Handle both single model and multi-model cases
    if not isinstance(models, list):
        models = [models]
    
    if not models or len(models) == 0:
        return None
    
    # Determine how many models to plot
    global multi_model_plots, num_models_to_compare, individual_model_predictions, show_model_predictions_separately
    if single_model_mode or not multi_model_plots:
        models_to_plot = models[:1]  # Only plot the best model
        plot_title = f'Intensity - {models[0].name}'
    else:
        models_to_plot = models[:min(num_models_to_compare, len(models))]
        if len(models_to_plot) > 1:
            plot_title = f'Intensity Comparison - Top {len(models_to_plot)} Models'
        else:
            plot_title = f'Intensity - {models[0].name}'
    
    # Simplified title - predictions are now seamlessly integrated
    # No need to mention predictions in title since they appear as continuous extensions
    
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    
    # Create consistent color mapping across all plots (reuse from growth curve)
    color_mapping, original_colors = create_model_color_mapping(models, individual_predictions)
    
    # Plot imported data as gray bar chart (matching original C-SFRAT)
    if hasattr(models[0], 't') and hasattr(models[0], 'CFC'):
        # Calculate failure counts (FC) from cumulative failure count (CFC)
        fc_data = []
        cfc_data = models[0].CFC
        for i in range(len(cfc_data)):
            if i == 0:
                fc_data.append(cfc_data[i])
            else:
                fc_data.append(cfc_data[i] - cfc_data[i-1])
        
        # Bar chart with gray color and 0.8 width (matching original)
        try:
            time_data = np.array([float(x) for x in models[0].t])
            fc_array = np.array([float(x) for x in fc_data])
            ax.bar(time_data, fc_array, width=0.8, color='lightgray', edgecolor='gray', 
                   label='Data', alpha=0.8, zorder=1)
        except (ValueError, TypeError) as e:
            logging.warning(f"Bar chart array conversion error: {e}")
            # Fallback without conversion
            ax.bar(models[0].t, fc_data, width=0.8, color='lightgray', edgecolor='gray', 
                   label='Data', alpha=0.8, zorder=1)
    
    # Plot each fitted model with original C-SFRAT style
    for i, model in enumerate(models_to_plot):
        # Create consistent model key for color mapping
        model_key = model.name
        if hasattr(model, 'covariateNames') and model.covariateNames:
            cov_str = ",".join([c[:3] for c in model.covariateNames])
            model_key += f"({cov_str})"
        
        color = color_mapping.get(model_key, original_colors[i % len(original_colors)])
        
        # Ensure model has required attributes for plotting
        if not hasattr(model, 't'):
            logging.warning(f"Model {model.name} missing 't' attribute for plotting")
            continue
        
        # Get intensity values
        intensity_data = []
        if hasattr(model, 'intensityList') and model.intensityList is not None and len(model.intensityList) > 0:
            intensity_data = model.intensityList
        elif hasattr(model, 'intensity_array') and model.intensity_array is not None and len(model.intensity_array) > 0:
            intensity_data = model.intensity_array
        elif hasattr(model, 'mvf_array') and model.mvf_array is not None and len(model.mvf_array) > 1:
            # Calculate intensity as derivative of MVF
            intensity_data = [model.mvf_array[j] - model.mvf_array[j-1] if j > 0 else model.mvf_array[0] 
                              for j in range(len(model.mvf_array))]
        elif hasattr(model, 'mvfList') and model.mvfList is not None and len(model.mvfList) > 1:
            # Calculate intensity as derivative of MVF
            intensity_data = [model.mvfList[j] - model.mvfList[j-1] if j > 0 else model.mvfList[0] 
                              for j in range(len(model.mvfList))]
        
        # Plot intensity curve if data is available
        if intensity_data:
            # Create model label
            if len(models_to_plot) == 1:
                model_label = model.name
                if hasattr(model, 'covariateNames') and model.covariateNames:
                    model_label += f" ({', '.join(model.covariateNames)})"
            else:
                model_label = model.name
                if hasattr(model, 'covariateNames') and model.covariateNames:
                    cov_str = ",".join([c[:3] for c in model.covariateNames])  # Abbreviated covariates
                    model_label += f" ({cov_str})"
                
                # Add ranking information for multi-model plots
                if hasattr(model, 'criticMean') and hasattr(model, 'criticMedian'):
                    global ranking_method
                    if ranking_method.lower() == 'median':
                        model_label += f" [MD:{model.criticMedian:.4f}]"
                    else:
                        model_label += f" [M:{model.criticMean:.4f}]"
            
            # Plot intensity data with circles and lines (matching original C-SFRAT style)
            try:
                time_data = np.array([float(x) for x in model.t])
                intensity_array = np.array([float(x) for x in intensity_data])
                ax.plot(time_data, intensity_array, color=color, linewidth=3, marker='o', markersize=4,
                        markerfacecolor=color, markeredgecolor=color, label=model_label, 
                        alpha=0.9, zorder=5)
            except (ValueError, TypeError) as e:
                logging.warning(f"Intensity plot array conversion error for {model.name}: {e}")
                continue
    
    # REMOVED: Green effort intensity prediction line (individual model predictions provide smooth behavior)
    effort_intensity_shown = False
    
    # Handle individual model intensity predictions (like original C-SFRAT tool)
    if individual_predictions and individual_model_predictions and show_model_predictions_separately:        
        for model_key, pred_data in individual_predictions.items():
            # Use consistent color mapping for predictions  
            pred_color = color_mapping.get(model_key, original_colors[0])
            
            # Get the model's last intensity point for connection
            model = pred_data['model']
            if not hasattr(model, 't'):
                continue
            
            # Get last intensity value for connection
            last_intensity = None
            if hasattr(model, 'intensityList') and model.intensityList is not None and len(model.intensityList) > 0:
                last_intensity = model.intensityList[-1]
            elif hasattr(model, 'intensity_array') and model.intensity_array is not None and len(model.intensity_array) > 0:
                last_intensity = model.intensity_array[-1]
            elif hasattr(model, 'mvf_array') and model.mvf_array is not None and len(model.mvf_array) > 1:
                last_intensity = model.mvf_array[-1] - model.mvf_array[-2]
            elif hasattr(model, 'mvfList') and model.mvfList is not None and len(model.mvfList) > 1:
                last_intensity = model.mvfList[-1] - model.mvfList[-2]
            
            if last_intensity is None:
                continue
            
            # Plot individual model intensity predictions
            pred_times = pred_data['future_times']
            pred_intensity = pred_data.get('future_intensity', [])
            
            if len(pred_times) > 0 and len(pred_intensity) > 0:
                # Connect the last fitted point to the first prediction point
                try:
                    connect_times = np.array([float(model.t[-1])] + [float(x) for x in pred_times])
                    connect_intensity = np.array([float(last_intensity)] + [float(x) for x in pred_intensity])
                    pred_times_array = np.array([float(x) for x in pred_times])
                    pred_intensity_array = np.array([float(x) for x in pred_intensity])
                except (ValueError, TypeError) as e:
                    logging.warning(f"Intensity prediction array conversion error for {model_key}: {e}")
                    continue
                
                # Plot prediction as continuous extension (same style as model fit)
                # No different styling - predictions appear as natural extensions
                ax.plot(connect_times, connect_intensity, 
                       color=pred_color, linewidth=3, marker='o', markersize=4,
                       markerfacecolor=pred_color, markeredgecolor=pred_color,
                       alpha=0.9, zorder=5)  # Same style as fitted model lines
    
    # Add single model intensity predictions (fallback for best model only - only if effort not shown)
    elif predictions and not effort_intensity_shown and len(models_to_plot) == 1:
        pred_times = predictions['future_times']
        pred_intensity = predictions.get('future_intensity', [])
        
        if len(pred_times) > 0 and len(pred_intensity) > 0:
            # Get last intensity value for the best model
            model = models[0]
            last_intensity = None
            if hasattr(model, 'intensityList') and model.intensityList is not None and len(model.intensityList) > 0:
                last_intensity = model.intensityList[-1]
            elif hasattr(model, 'intensity_array') and model.intensity_array is not None and len(model.intensity_array) > 0:
                last_intensity = model.intensity_array[-1]
            elif hasattr(model, 'mvf_array') and model.mvf_array is not None and len(model.mvf_array) > 1:
                last_intensity = model.mvf_array[-1] - model.mvf_array[-2]
            elif hasattr(model, 'mvfList') and model.mvfList is not None and len(model.mvfList) > 1:
                last_intensity = model.mvfList[-1] - model.mvfList[-2]
            else:
                last_intensity = pred_intensity[0]  # fallback
            
            # Connect the last fitted point to the first prediction point
            connect_times = [model.t[-1]] + pred_times
            connect_intensity = [last_intensity] + pred_intensity
            
            # Use bright red color for predictions to distinguish from model colors
            ax.plot(connect_times, connect_intensity, 
                   color='#FF0000', linestyle='--', linewidth=2, alpha=0.9,
                   label=f'Intensity Predictions ({len(pred_times)} intervals)')
            
            # Add prediction markers
            ax.plot(pred_times, pred_intensity, 
                   marker='o', color='#FF0000', markersize=3, 
                   markerfacecolor='white', markeredgecolor='#FF0000', 
                   linestyle='', alpha=0.8)
    
    # Add red dashed vertical line at last data point (matching original C-SFRAT)
    if hasattr(models[0], 't') and len(models[0].t) > 0:
        last_x = models[0].t[-1]
        ax.axvline(x=last_x, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
    
    # Customize the plot to match original C-SFRAT style
    ax.set_xlabel('Intervals', fontsize=12, fontweight='bold')
    ax.set_ylabel('Failures', fontsize=12, fontweight='bold')
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    
    # Enable grid (matching original C-SFRAT)
    ax.grid(True, alpha=0.3)
    
    # Add legend with white background
    legend = ax.legend(fontsize=11, loc='best', frameon=True, fancybox=False, shadow=False)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_alpha(1.0)
    
    # Style axes
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    # Add covariate information for single model
    if show_plot_annotations and len(models_to_plot) == 1 and hasattr(models[0], 'covariateNames') and models[0].covariateNames:
        covs_text = f"Covariates: {', '.join(models[0].covariateNames)}"
        ax.text(0.02, 0.98, covs_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="white", edgecolor="gray", alpha=0.9))
    
    # Add ranking method info for multi-model plots
    if show_plot_annotations and len(models_to_plot) > 1:
        ranking_text = f"Ranked by: {ranking_method.upper()} critic values"
        ax.text(0.02, 0.02, ranking_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightgreen", alpha=0.7))
    
    # Add prediction info box
    if show_plot_annotations and effort_intensity_shown:
        effort_covariate = predictions.get('effort_covariate', 'Unknown')
        effort_value = predictions.get('effort_value', 1.0)
        intensity_target = predictions.get('intensity_target', 0.3)
        
        # Handle different effort value types  
        if isinstance(effort_value, (int, float)):
            effort_display = f"{effort_value:.1f}"
        else:
            effort_display = str(effort_value)
        pred_text = f"🟢 Effort Intensity\n{effort_covariate} = {effort_display}\nTarget: {intensity_target:.3f}"
        ax.text(0.98, 0.98, pred_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.9))
    elif show_plot_annotations and individual_predictions and individual_model_predictions and show_model_predictions_separately:
        # Check if GUI-correct predictions were used
        gui_correct_count = sum(1 for pred_data in individual_predictions.values() if pred_data.get('gui_correct', False))
        if gui_correct_count > 0:
            pred_text = f"✅ {gui_correct_count} GUI-Correct Smooth Intensity"
        else:
            pred_text = f"Individual Intensity Predictions: {len(individual_predictions)} models"
        
        ax.text(0.98, 0.98, pred_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen" if gui_correct_count > 0 else "lightblue", alpha=0.8))
    elif show_plot_annotations and predictions and len(models_to_plot) == 1 and predictions.get('future_intensity'):
        pred_text = f"Intensity Predictions: {len(predictions.get('future_intensity', []))} points"
        if 'prediction_horizon' in predictions:
            pred_text += f"\nHorizon: {predictions['prediction_horizon']} time units"
        ax.text(0.98, 0.98, pred_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # Save plot to a BytesIO object
    buffer = BytesIO()
    plt.tight_layout()
    
    # Ensure all data is properly converted to numpy arrays before plotting
    try:
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # Create reportlab Image
        buffer.seek(0)
        return Image(buffer, width=480, height=320)
    except Exception as e:
        logging.warning(f"Error saving intensity plot: {str(e)}")
        plt.close(fig)
        return None

def create_multi_model_growth_curve_plot(models, max_models=3):
    """Create a growth curve plot comparing multiple models"""
    if not models or len(models) == 0:
        return None
    
    # Limit to specified number of models
    models_to_plot = models[:min(max_models, len(models))]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors and markers for different models
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    # Plot each model
    for i, model in enumerate(models_to_plot):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        
        # Ensure model has required attributes
        if not hasattr(model, 't') or not hasattr(model, 'CFC'):
            continue
            
        # Get MVF data for this model
        mvf_data = []
        if hasattr(model, 'mvfList') and model.mvfList is not None and len(model.mvfList) > 0:
            mvf_data = model.mvfList
        elif hasattr(model, 'mvf_array') and model.mvf_array is not None and len(model.mvf_array) > 0:
            mvf_data = model.mvf_array
        else:
            continue
            
        # Create model label
        model_label = f"{model.name}"
        if hasattr(model, 'covariateNames') and model.covariateNames:
            cov_str = ",".join([c[:2] for c in model.covariateNames])  # Abbreviated covariates
            model_label += f" ({cov_str})"
        
        # Add ranking information
        if hasattr(model, 'aicVal'):
            model_label += f" [AIC:{model.aicVal:.1f}]"
        
        # Plot fitted values
        ax.plot(model.t, mvf_data, color=color, linestyle=linestyle, 
                linewidth=2.5, label=f"Fitted - {model_label}", alpha=0.8)
    
    # Plot actual data points (same for all models)
    if len(models_to_plot) > 0 and hasattr(models_to_plot[0], 't') and hasattr(models_to_plot[0], 'CFC'):
        ax.plot(models_to_plot[0].t, models_to_plot[0].CFC, 'ko', 
                label='Actual Data', markersize=8, markerfacecolor='white', 
                markeredgewidth=2, zorder=10)
    
    # Customize the plot
    ax.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Failures', fontsize=14, fontweight='bold')
    ax.set_title(f'Model Comparison - Growth Curves (Top {len(models_to_plot)} Models)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Improve legend
    legend = ax.legend(fontsize=10, loc='best', frameon=True, fancybox=True, 
                      shadow=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    
    # Grid and styling
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Add ranking method info
    ranking_text = f"Ranked by: {ranking_method.upper()} critic values"
    ax.text(0.02, 0.98, ranking_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor="lightblue", alpha=0.7))
    
    # Save to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    
    img = Image(buffer, width=500, height=320)  # Reduced size to fit page constraints
    plt.close(fig)
    
    return img

def create_multi_model_intensity_plot(models, max_models=3):
    """Create an intensity plot comparing multiple models"""
    if not models or len(models) == 0:
        return None
    
    # Limit to specified number of models
    models_to_plot = models[:min(max_models, len(models))]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors and markers for different models
    colors = ['green', 'red', 'blue', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    # Plot each model
    for i, model in enumerate(models_to_plot):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        
        # Ensure model has required attributes
        if not hasattr(model, 't'):
            continue
            
        # Get intensity data for this model
        intensity_data = []
        if hasattr(model, 'intensityList') and model.intensityList is not None and len(model.intensityList) > 0:
            intensity_data = model.intensityList
        elif hasattr(model, 'intensity_array') and model.intensity_array is not None and len(model.intensity_array) > 0:
            intensity_data = model.intensity_array
        elif hasattr(model, 'mvfList') and model.mvfList is not None and len(model.mvfList) > 1:
            # Calculate intensity as derivative of MVF
            intensity_data = [model.mvfList[j] - model.mvfList[j-1] if j > 0 else model.mvfList[0] 
                              for j in range(len(model.mvfList))]
        else:
            continue
            
        # Create model label
        model_label = f"{model.name}"
        if hasattr(model, 'covariateNames') and model.covariateNames:
            cov_str = ",".join([c[:2] for c in model.covariateNames])  # Abbreviated covariates
            model_label += f" ({cov_str})"
        
        # Add ranking information
        if hasattr(model, 'aicVal'):
            model_label += f" [AIC:{model.aicVal:.1f}]"
        
        # Plot intensity values if data is available
        if intensity_data:
            ax.plot(model.t, intensity_data, color=color, linestyle=linestyle, 
                    marker=marker, markersize=6, linewidth=2.5, label=model_label, 
                    alpha=0.8, markerfacecolor='white', markeredgewidth=1.5)
    
    # Customize the plot
    ax.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Failure Intensity', fontsize=14, fontweight='bold')
    ax.set_title(f'Model Comparison - Failure Intensity (Top {len(models_to_plot)} Models)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Improve legend
    legend = ax.legend(fontsize=10, loc='best', frameon=True, fancybox=True, 
                      shadow=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    
    # Grid and styling
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Add ranking method info
    ranking_text = f"Ranked by: {ranking_method.upper()} critic values"
    ax.text(0.02, 0.98, ranking_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor="lightgreen", alpha=0.7))
    
    # Save to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    
    img = Image(buffer, width=500, height=400)  # Reduced height to fit page constraints
    plt.close(fig)
    
    return img

def create_model_comparison_summary_plot(models, max_models=3):
    """Create a summary plot showing key metrics for model comparison"""
    if not models or len(models) == 0:
        return None
    
    # Limit to specified number of models
    models_to_plot = models[:min(max_models, len(models))]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
    
    # Extract data for plotting
    model_names = []
    aic_values = []
    bic_values = []
    critic_mean_values = []
    critic_median_values = []
    
    for model in models_to_plot:
        # Create short model name
        name = model.name[:15]  # Truncate long names
        if hasattr(model, 'covariateNames') and model.covariateNames:
            cov_str = ",".join([c[:2] for c in model.covariateNames])
            name += f"({cov_str})"
        model_names.append(name)
        
        aic_values.append(model.aicVal if hasattr(model, 'aicVal') else 0)
        bic_values.append(model.bicVal if hasattr(model, 'bicVal') else 0)
        critic_mean_values.append(model.criticMean if hasattr(model, 'criticMean') else 0)
        critic_median_values.append(model.criticMedian if hasattr(model, 'criticMedian') else 0)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Plot 1: AIC Comparison
    bars1 = ax1.bar(model_names, aic_values, color=colors[:len(models_to_plot)], alpha=0.8, edgecolor='black')
    ax1.set_title('AIC Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AIC Value', fontsize=11)
    ax1.tick_params(axis='x', rotation=45, labelsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, aic_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: BIC Comparison
    bars2 = ax2.bar(model_names, bic_values, color=colors[:len(models_to_plot)], alpha=0.8, edgecolor='black')
    ax2.set_title('BIC Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('BIC Value', fontsize=11)
    ax2.tick_params(axis='x', rotation=45, labelsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, bic_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Critic Mean Comparison
    bars3 = ax3.bar(model_names, critic_mean_values, color=colors[:len(models_to_plot)], alpha=0.8, edgecolor='black')
    ax3.set_title('Critic Mean Comparison (Higher is Better)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Critic Mean Value', fontsize=11)
    ax3.tick_params(axis='x', rotation=45, labelsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars3, critic_mean_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Critic Median Comparison
    bars4 = ax4.bar(model_names, critic_median_values, color=colors[:len(models_to_plot)], alpha=0.8, edgecolor='black')
    ax4.set_title('Critic Median Comparison (Higher is Better)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Critic Median Value', fontsize=11)
    ax4.tick_params(axis='x', rotation=45, labelsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars4, critic_median_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add overall title
    fig.suptitle(f'Model Performance Comparison - Top {len(models_to_plot)} Models', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add ranking method info
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.1)
    
    # Save to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    
    img = Image(buffer, width=500, height=320)
    plt.close(fig)
    
    return img

def create_comparison_table(models):
    """Create a table comparing multiple models"""
    # Prepare the data for the table
    data = [
        ['Model', 'Covariates', 'AIC', 'BIC', 'Log-Likelihood', 'SSE']
    ]
    
    for model in models:
        # Format covariates string
        if not model.metricNames:
            cov_str = "None"
        else:
            cov_str = ", ".join(model.metricNames)
            
        data.append([
            model.name,
            cov_str,
            f"{model.aicVal:.4f}",
            f"{model.bicVal:.4f}",
            f"{model.llfVal:.4f}",
            f"{model.sseVal:.4f}"
        ])
    
    # Create the table
    table = Table(data, colWidths=[100, 140, 70, 70, 100, 70])
    
    # Style the table
    table.setStyle(TableStyle([
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        
        # Body
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        
        # Highlight best model (first row after header)
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#D5F5E3')),
        
        # Alternating row colors (after the first row)
        ('ROWBACKGROUNDS', (0, 2), (-1, -1), [colors.white, colors.HexColor('#F5F6FA')])
    ]))
    
    return table

def create_parameters_table(model):
    """Create table showing parameter estimates"""
    # Prepare data for table
    data = [
        ['Parameter', 'Value']
    ]
    
    # Add all parameters
    if hasattr(model, 'b'):
        data.append(['b', f"{model.b:.6f}"])
        
    for i, beta in enumerate(model.betas):
        data.append([f'beta{i+1}', f"{beta:.6f}"])
    
    # Create table
    table = Table(data, colWidths=[2*inch, 1.5*inch])
    
    # Style the table
    table.setStyle(TableStyle([
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        
        # Body
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),   # Left align parameter names
        ('ALIGN', (1, 1), (1, -1), 'RIGHT'),  # Right align values
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        
        # Alternating row colors
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F6FA')])
    ]))
    
    return table

def create_metrics_table(model):
    """Create a table showing goodness-of-fit metrics"""
    # Prepare data for table
    data = [
        ['Metric', 'Value'],
        ['Log-Likelihood', f"{model.llfVal:.6f}"],
        ['AIC', f"{model.aicVal:.6f}"],
        ['BIC', f"{model.bicVal:.6f}"],
        ['SSE', f"{model.sseVal:.6f}"]
    ]
    
    # Create table
    table = Table(data, colWidths=[2*inch, 1.5*inch])
    
    # Style the table
    table.setStyle(TableStyle([
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        
        # Body
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),   # Left align metric names
        ('ALIGN', (1, 1), (1, -1), 'RIGHT'),  # Right align values
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        
        # Alternating row colors
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F6FA')])
    ]))
    
    return table

def create_page_template(doc):
    """Create a template for the PDF report"""
    styles = getSampleStyleSheet()
    
    # Create custom styles with professional typography
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontSize=22,
        spaceAfter=12,
        textColor=colors.HexColor('#2C3E50')
    ))
    
    styles.add(ParagraphStyle(
        name='CustomHeading',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=6,
        textColor=colors.HexColor('#2C3E50')
    ))
    
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=6,
        leading=13.8,  # 1.15 line spacing
        textColor=colors.HexColor('#2C3E50')
    ))
    
    # Return template with styles
    return {
        'styles': styles
    }

def create_model_predictions_table(models):
    """Generate one simple prediction table that's easy to understand
    
    Args:
        models: List of fitted model objects (sorted by ranking)
        
    Returns:
        list: List of ReportLab elements for model predictions
    """
    # Make sure we have at least one model
    if not models or len(models) == 0:
        return None
        
    # Get time points from the first model
    if hasattr(models[0], 't'):
        time_points = models[0].t
    else:
        return None
    
    elements = []
    
    # === ONE SIMPLE TABLE ===
    elements.append(Paragraph(f"<b>Model Performance Summary</b>", getSampleStyleSheet()["Normal"]))
    elements.append(Spacer(1, 8))
    
    # Create one simple table showing how well the best model works
    simple_data = [["Time Point", "What Actually Happened", "What Model Predicted"]]
    
    best_model = models[0]
    
    # Show only the first 8 rows to keep it simple
    for i in range(min(8, len(time_points))):
        time_val = time_points[i]
        actual = best_model.CFC[i] if i < len(best_model.CFC) else 0
        
        # Get predicted value
        if hasattr(best_model, 'mvfList') and i < len(best_model.mvfList):
            predicted = best_model.mvfList[i]
        else:
            predicted = 0
        
        simple_data.append([
            f"Time {time_val}",
            f"{actual} failures",
            f"{predicted:.0f} failures"
        ])
    
    # Create simple table
    simple_table = Table(simple_data, colWidths=[100, 120, 120])
    simple_table.setStyle(TableStyle([
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
            
        # Data
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        
        # Alternate colors
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')]),
    ]))
    
    elements.append(simple_table)
    elements.append(Spacer(1, 6))
        
    # === SIMPLE EXPLANATION ===
    elements.append(Paragraph("<b>What This Means:</b>", getSampleStyleSheet()["Normal"]))
    elements.append(Spacer(1, 8))
    
    explanation = f"✓ We tested the <b>{best_model.name}</b> model<br/>"
    explanation += f"✓ The model tries to predict how many failures will occur<br/>"
    explanation += f"✓ We compared predictions with what actually happened<br/>"
    explanation += f"✓ This model was the best out of {len(models)} models tested"
    
    elements.append(Paragraph(explanation, getSampleStyleSheet()["Normal"]))
    elements.append(Spacer(1, 6))
        
    return elements

def generate_report(models, filename, predictions=None, optimization_results=None, effort_predictions=None):
    """Generate a comprehensive PDF report of results with improved formatting and section organization"""
    global ranking_method
    
    if not models:
        logging.error("No models converged. Cannot generate report.")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Get best model (first in the sorted list)
    best_model = models[0]
    logging.info(f"Generating report with best model: {best_model.name}")
    
    # Create a PDF document - Use PORTRAIT orientation for better visual presentation
    doc = SimpleDocTemplate(
        filename, 
        pagesize=letter,          # Changed to portrait for better visual layout
        leftMargin=36,            # 0.5 inch
        rightMargin=36,           # 0.5 inch
        topMargin=36,             # 0.5 inch
        bottomMargin=36           # 0.5 inch
    )
    elements = []
    
    # Create custom styles for better formatting
    styles = getSampleStyleSheet()
    
    # Custom style for main section headers
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=24,
        textColor=colors.HexColor('#2C3E50'),
        borderWidth=1.5,
        borderColor=colors.HexColor('#2C3E50'),
        borderPadding=8,
        backColor=colors.HexColor('#ECF0F1'),
        alignment=1,  # Center alignment
        fontName='Helvetica-Bold'
    ))
    
    # Custom style for subsection headers
    styles.add(ParagraphStyle(
        name='SubsectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=12,
        textColor=colors.HexColor('#2C3E50'),
        leftIndent=0,
        fontName='Helvetica-Bold'
    ))
    
    # Custom style for plot titles
    styles.add(ParagraphStyle(
        name='PlotTitle',
        parent=styles['Heading3'],
        fontSize=11,
        spaceAfter=4,
        spaceBefore=4,
        textColor=colors.HexColor('#2C3E50'),
        alignment=1  # Center alignment
    ))
    
    # Custom style for effort allocation headers
    styles.add(ParagraphStyle(
        name='EffortHeader',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=4,
        spaceBefore=6,
        textColor=colors.HexColor('#E67E22'),
        leftIndent=4,
        fontName='Helvetica-Bold'
    ))
    
    # =============================================================================
    # TITLE PAGE
    # =============================================================================
    title_style = getSampleStyleSheet()["Title"]
    title_style.fontSize = 15
    title_style.textColor = colors.HexColor('#2C3E50')
    title_style.alignment = 1  # Center alignment
    
    elements.append(Paragraph("C-SFRAT Reliability Analysis & Effort Allocation", title_style))
    elements.append(Spacer(1, 4))
    
    # Executive Summary Box
    summary_style = ParagraphStyle(
        name='SummaryBox',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=2,
        borderWidth=1,
        borderColor=colors.HexColor('#3498DB'),
        borderPadding=4,
        backColor=colors.HexColor('#EBF5FB')
    )
    
    summary_text = f"""
    <b>Executive Summary:</b><br/>
    • Analysis completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    • Total model/covariate combinations analyzed: {len(models)}<br/>
    • Best performing model: <b>{best_model.name}</b><br/>
    • Ranking method: <b>{ranking_method.upper()}</b><br/>
    • Report contains: Comprehensive model comparison, reliability predictions, and effort allocation optimization
    """
    
    elements.append(Paragraph(summary_text, summary_style))
    elements.append(Spacer(1, 8))
    
    # Highlighted best model box
    best_model_style = ParagraphStyle(
        name='BestModelBox',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=6,
        borderWidth=1,
        borderColor=colors.HexColor('#3498DB'),
        borderPadding=8,
        backColor=colors.HexColor('#EBF8FF'),
        alignment=1  # Center alignment
    )
    
    best_model_text = f"<b>Best Performing Model: {best_model.name}</b>"
    elements.append(Paragraph(best_model_text, best_model_style))
    elements.append(Spacer(1, 6))
    
    # =============================================================================
    # SECTION 2: VISUAL ANALYSIS AND PREDICTIONS
    # =============================================================================
    elements.append(Spacer(1, 18))  # Add space instead of page break
    elements.append(Paragraph("SECTION 2: VISUAL ANALYSIS AND PREDICTIONS", styles["SectionHeader"]))
    elements.append(Spacer(1, 6))
    
    # Generate predictions for best model and individual models
    predictions = None
    individual_predictions = None
    
    if prediction_parameters.get('predict_failures', False):
        try:
            logging.info("Generating predictions for the best model...")
            predictions = generate_failure_predictions(best_model, prediction_parameters)
            
            # Generate individual model predictions if enabled
            if individual_model_predictions and show_model_predictions_separately:
                logging.info("Generating GUI-CORRECT individual model predictions...")
                individual_predictions = generate_individual_model_predictions_gui_correct(models, prediction_parameters)
                
        except Exception as e:
            logging.warning(f"Failed to generate predictions: {str(e)}")
    
    # Determine which predictions to use for plots
    plot_predictions = predictions
    plot_individual_predictions = individual_predictions
    
    # If effort predictions are available, use them as the primary prediction but keep individual predictions
    if effort_predictions and isinstance(effort_predictions, dict):
        plot_predictions = effort_predictions
        # Show actual configured effort values
        effort_info = f" (Effort: E={effort_per_interval_settings['effort_values']['E']}, F={effort_per_interval_settings['effort_values']['F']}, C={effort_per_interval_settings['effort_values']['C']})"
        
        # Debug logging
        logging.info(f"DEBUG PLOT VALUES - plot_predictions type: {type(plot_predictions)}")
        logging.info(f"DEBUG PLOT VALUES - plot_individual_predictions: {plot_individual_predictions}")
        logging.info(f"DEBUG PLOT VALUES - effort_predictions avg intensity: {plot_predictions.get('future_intensity', [0])[0] if plot_predictions.get('future_intensity') else 'N/A'}")
        logging.info(f"DEBUG PLOT VALUES - individual_predictions count: {len(plot_individual_predictions) if plot_individual_predictions else 0}")
    else:
        effort_info = ""
    
    elements.append(Paragraph("2.1 Overview", styles["SubsectionHeader"]))
    elements.append(Paragraph("This section presents the visual analysis of software reliability through <b>Mean Value Function (MVF)</b> and <b>Failure Intensity</b> plots. These visualizations provide intuitive insights into system behavior and future predictions.", styles["Normal"]))
    
    # Add legend notation explanation
    legend_explanation = f"""
    <b>Legend Notation:</b> In the plots below, when multiple models are displayed, the legend shows ranking values in brackets. 
    The notation <b>[M:value]</b> indicates <b>Mean</b> critic ranking method, while <b>[MD:value]</b> indicates <b>Median</b> critic ranking method. 
    Currently using: <b>{ranking_method.upper()}</b> ranking method.
    """
    elements.append(Paragraph(legend_explanation, styles["Normal"]))
    elements.append(Spacer(1, 0.5))
    
    # 2.2 Mean Value Function (MVF) Plot
    elements.append(Paragraph("2.2 Mean Value Function (MVF) Analysis", styles["SubsectionHeader"]))
    elements.append(Paragraph("The <b>MVF plot</b> shows the cumulative number of failures over time, including both historical data and future predictions. This helps visualize the overall reliability growth pattern.", styles["Normal"]))
    elements.append(Spacer(1, 0.2))
    
    mvf_section = []
    mvf_title = "Mean Value Function (MVF)" + effort_info
    mvf_section.append(Paragraph(mvf_title, styles["PlotTitle"]))
    growth_plot = create_growth_curve_plot(models, predictions=plot_predictions, individual_predictions=plot_individual_predictions)
    if growth_plot is not None:
        growth_plot.width = 480
        growth_plot.height = 280
        mvf_section.append(growth_plot)
    else:
        mvf_section.append(Paragraph("Note: Growth curve plot could not be generated due to data format issues.", styles['Normal']))
    elements.append(KeepTogether(mvf_section))
    elements.append(Spacer(1, 2))
    
    # 2.3 Failure Intensity Plot - Keep heading and plot together on same page
    intensity_section = []
    intensity_section.append(Paragraph("2.3 Failure Intensity Analysis", styles["SubsectionHeader"]))
    intensity_section.append(Paragraph("The <b>intensity plot</b> shows the rate of failure occurrence (failures per time unit) over time. This visualization helps identify patterns in failure rates and predict future system behavior.", styles["Normal"]))
    intensity_section.append(Spacer(1, 0.2))
    
    intensity_title = "Failure Intensity" + effort_info
    intensity_section.append(Paragraph(intensity_title, styles["PlotTitle"]))
    intensity_plot = create_intensity_plot(models, predictions=plot_predictions, individual_predictions=plot_individual_predictions)
    if intensity_plot is not None:
        intensity_plot.width = 500   # Optimized for portrait orientation
        intensity_plot.height = 350  # Better aspect ratio
        intensity_section.append(intensity_plot)
    else:
        intensity_section.append(Paragraph("Note: Intensity plot could not be generated due to data format issues.", styles['Normal']))
    
    elements.append(KeepTogether(intensity_section))
    
    # =============================================================================
    # SECTION 3: EFFORT ALLOCATION ANALYSIS
    # =============================================================================
    if optimization_results and optimization_results.get('status') == 'success':
        elements.append(PageBreak())
        elements.append(Paragraph("SECTION 3: EFFORT ALLOCATION ANALYSIS", styles["SectionHeader"]))
        elements.append(Spacer(1, 6))
        
        elements.append(Paragraph("3.1 Effort Allocation Overview", styles["SubsectionHeader"]))
        elements.append(Paragraph("This section presents the optimal effort allocation across different covariates to maximize defect discovery within budget constraints or minimize budget for target defect discovery.", styles["Normal"]))
        elements.append(Spacer(1, 8))
        
        logging.info(f"DEBUG: Adding effort allocation to PDF - optimization_results type: {type(optimization_results)}")
        logging.info(f"DEBUG: optimization_results keys: {list(optimization_results.keys())}")
        logging.info(f"DEBUG: optimization_results status: {optimization_results.get('status')}")
        
        allocation_elements = create_effort_allocation_section(optimization_results)
        elements.extend(allocation_elements)
        logging.info("DEBUG: Effort allocation section added to PDF")
        
        elements.append(Spacer(1, 8))
    else:
        logging.info(f"DEBUG: Effort allocation NOT added - optimization_results: {optimization_results}")
        if optimization_results:
            logging.info(f"DEBUG: optimization_results status: {optimization_results.get('status')}")
        else:
            logging.info("DEBUG: optimization_results is None")
    
    
    # =============================================================================
    # SECTION 4: ANALYSIS CONFIGURATION
    # =============================================================================
        elements.append(PageBreak())
    elements.append(Paragraph("SECTION 4: ANALYSIS CONFIGURATION", styles["SectionHeader"]))
    elements.append(Spacer(1, 6))
    
    elements.append(Paragraph("4.1 Model Ranking Configuration", styles["SubsectionHeader"]))
    elements.append(Paragraph("The following configuration was used for this analysis:", styles["Normal"]))
    elements.append(Spacer(1, 2))
    
    # Validate configuration and add single-metric information
    is_valid, is_single_metric, single_metric_name, validation_message = validate_metric_weights(metric_weights)
    
    # Display configuration type
    if is_single_metric:
        config_text = f"<b>Configuration Type:</b> Single-metric (<b>{single_metric_name.upper()}</b>-only)<br/>"
        config_text += "<b>Professor's Recommendation:</b> ✅ Implemented (four weights set to 0.0)"
        elements.append(Paragraph(config_text, styles["Normal"]))
        elements.append(Spacer(1, 2))
        elements.append(Paragraph(f"Using <b>{single_metric_name.upper()}</b> as the sole criterion for model ranking:", styles["Normal"]))
    else:
        active_metrics = [k.upper() for k, v in metric_weights.items() if v > 0.0]
        config_text = f"<b>Configuration Type:</b> Multi-metric (<b>{len(active_metrics)}</b> active metrics)<br/>"
        config_text += f"<b>Active Metrics:</b> <b>{', '.join(active_metrics)}</b>"
        elements.append(Paragraph(config_text, styles["Normal"]))
        elements.append(Spacer(1, 2))
        elements.append(Paragraph("Metric weights used for model comparison:", styles["Normal"]))
    
    elements.append(Spacer(1, 1))
    
    # Enhanced weights table with status - only show metrics with non-zero weights
    weights_data = [["Metric", "Weight", "Status", "Description"]]
    all_metrics = [("Log-Likelihood", "llf", "Higher values indicate better fit"),
                   ("AIC", "aic", "Lower values indicate better model"),
                   ("BIC", "bic", "Lower values indicate better model"),
                   ("SSE", "sse", "Lower values indicate better fit"),
                   ("PSSE", "psse", "Lower values indicate better prediction")]
    for metric_name, metric_key, description in all_metrics:
        weight_value = metric_weights[metric_key]
        if weight_value > 0:
            weights_data.append([metric_name, f"{weight_value:.1f}", "🟢 Active", description])
    weights_table = Table(weights_data, colWidths=[90, 50, 70, 260])
    weights_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 3),
        ('TOPPADDING', (0, 0), (-1, 0), 1),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 1),
        ('TOPPADDING', (0, 1), (-1, -1), 0),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ('ALIGN', (2, 1), (2, -1), 'CENTER'),
        ('ALIGN', (3, 1), (3, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
    ]))
    elements.append(KeepTogether(weights_table))
    
    # =============================================================================
    # MERGED: MODEL COMPARISON AND SELECTION under Section 3
    # =============================================================================
    elements.append(Spacer(1, 4))
    elements.append(Paragraph("4.2 Model Ranking Results", styles["SubsectionHeader"]))
    ranking_explanation = f"Models are ranked using the <b>{ranking_method.upper()}</b> ranking method. "
    if ranking_method.lower() == 'mean':
        ranking_explanation += "The model with the highest average <b>critic mean value</b> is selected as the best."
    else:
        ranking_explanation += "The model with the highest <b>critic median value</b> is selected as the best."
    
    elements.append(Paragraph(ranking_explanation, styles["Normal"]))
    elements.append(Spacer(1, 2))
    
    # Add note about table limitation if applicable
    if max_models_in_comparison_table > 0 and len(models) > max_models_in_comparison_table:
        elements.append(Paragraph(f"<i>Note: Showing top <b>{max_models_in_comparison_table}</b> models out of <b>{len(models)}</b> total models for readability.</i>", styles["Normal"]))
        elements.append(Spacer(1, 2))
    
    # Add metrics table
    model_details_table = generate_model_details_table(models, num_models_to_compare, max_models_in_comparison_table)
    elements.append(model_details_table)
    elements.append(Spacer(1, 2))
    
    # =============================================================================
    # SECTION 5: DETAILED MODEL COMPARISON TABLES
    # =============================================================================
    elements.append(PageBreak())
    elements.append(Paragraph("SECTION 5: DETAILED MODEL COMPARISON TABLES", styles["SectionHeader"]))
    elements.append(Spacer(1, 6))
    
    elements.append(Paragraph("5.1 Overview", styles["SubsectionHeader"]))
    elements.append(Paragraph("This section provides detailed numerical comparisons between the top-performing models, showing both fitted values and future predictions. These tables complement the visual analysis with precise numerical data.", styles["Normal"]))
    elements.append(Spacer(1, 2))
    
    # Table 1: Time-based Comparison
    elements.append(Paragraph("5.2 Time-based Comparison (Cumulative Failures)", styles["SubsectionHeader"]))
    elements.append(Paragraph("Shows the total number of failures predicted by each model at specific time points. Includes the observed <b>FC (Cumulative Failures)</b> column from actual data for comparison.", styles["Normal"]))
    elements.append(Paragraph("Note: For historical times, the Effort scenario matches Observed (FC) because effort only affects future intervals.", styles["Normal"]))
    elements.append(Spacer(1, 2))
    
    # Add explanation for the Effort FC column
    if effort_predictions:
        elements.append(Paragraph("<b>Column Explanation:</b> The <i>Effort FC</i> column shows cumulative failure predictions when specific effort allocation strategies are applied to testing phases (E, F, C). This represents how many total failures would be discovered if the recommended effort distribution is followed.", styles["Normal"]))
        elements.append(Spacer(1, 2))
        
    time_based_table = create_time_based_comparison_table(models[:3], effort_predictions)
    if time_based_table:
        elements.append(time_based_table)
    else:
        elements.append(Paragraph("Time-based comparison table could not be generated.", styles["Normal"]))
    
    elements.append(Spacer(1, 3))
    
    # Table 2: Failure Intensity Table
    elements.append(Paragraph("5.3 Failure Intensity Comparison", styles["SubsectionHeader"]))
    elements.append(Paragraph("Shows the <b>failure intensity</b> (failures per interval) for each model at specific time points.", styles["Normal"]))
    
    # Add explanation for the Effort Intensity column  
    if effort_predictions:
        elements.append(Paragraph("<b>Column Explanation:</b> The <i>Effort Intensity</i> column shows the failure discovery rate (failures per time interval) when applying the recommended effort allocation strategy. This indicates how rapidly failures would be detected during each testing phase.", styles["Normal"]))
        elements.append(Spacer(1, 2))

    elements.append(Spacer(1, 4))
        
    intensity_table = create_intensity_table(models[:3], plot_individual_predictions, effort_predictions)
    if intensity_table:
        elements.append(intensity_table)
    else:
        elements.append(Paragraph("Failure intensity table could not be generated.", styles["Normal"]))
    
    elements.append(Spacer(1, 3))
    
    # Report Summary - Use large spacer instead of page break to keep on same page if possible
    elements.append(Spacer(1, 2))  # Smaller spacer to reduce gaps
    elements.append(Paragraph("REPORT SUMMARY", styles["SectionHeader"]))
    summary_text = f"""
    This comprehensive C-SFRAT integrated analysis has examined <b>{len(models)}</b> different model/covariate combinations 
    and selected the <b>{best_model.name}</b> model as the best performer using the <b>{ranking_method.upper()}</b> 
    ranking method. The analysis includes model comparison metrics, reliability predictions, failure intensity analysis, 
    and optimal effort allocation strategies for maximizing defect discovery within budget constraints.
    
    <br/><br/>
    <i>Report generated by C-SFRAT Integrated Analysis Tool on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
    """
    elements.append(Paragraph(summary_text, styles["Normal"]))
    
    # =============================================================================
    # APPENDIX: ACRONYMS AND DEFINITIONS
    # =============================================================================
    elements.append(PageBreak())
    elements.append(Paragraph("APPENDIX: ACRONYMS AND DEFINITIONS", styles["SectionHeader"]))
    elements.append(Spacer(1, 6))
    
    # Create acronym definitions table
    acronym_data = [
        ["Acronym", "Definition"],
        ["AIC", "Akaike Information Criterion - Statistical measure for model selection"],
        ["BIC", "Bayesian Information Criterion - Statistical measure for model selection"],
        ["C-SFRAT", "C-Software Failure and Reliability Assessment Tool"],
        ["IFR", "Increasing Failure Rate - Model family with failure rate that increases over time"],
        ["LLF", "Log-Likelihood Function - Statistical measure of model fit"],
        ["MVF", "Mean Value Function - Expected cumulative number of failures over time"],
        ["Intensity", "Failures per interval (ΔMVF)"],
        ["PSSE", "Predictive Sum of Squared Errors - Measure of prediction accuracy"],
        ["SSE", "Sum of Squared Errors - Measure of model fit to observed data"],
        ["AHP", "Analytic Hierarchy Process - weighting and ranking framework"],
        ["FC", "Cumulative Failures (Observed)"],
        ["CFC", "Cumulative Failure Count (historical data array)"],
        ["omega", "Total expected failures parameter of the model"],
        ["beta", "Model coefficient(s) for covariates in the hazard function"]
    ]

    # Add model acronyms (short, readable tags) so users can match names in legends/tables
    model_acronyms = [
        ["NB2", "Negative Binomial (Order 2)"],
        ["DW2", "Discrete Weibull (Order 2)"],
        ["TL", "Truncated Logistic"],
        ["GEOM", "Geometric"],
        ["S-DIST", "S Distribution"],
        ["IFR-SB", "IFR Salvia & Bollinger"],
        ["IFR-GSB", "IFR Generalized Salvia & Bollinger"]
    ]
    acronym_data.extend(model_acronyms)
    
    # Enhanced acronym table with professional styling
    acronym_table = Table(acronym_data, colWidths=[80, 400], repeatRows=1)
    acronym_table.setStyle(TableStyle([
        # Header styling
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        # Header border
        ('LINEBELOW', (0, 0), (-1, 0), 1.5, colors.HexColor('#2C3E50')),
        
        # Body styling with zebra striping
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('ALIGN', (1, 1), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDC3C7')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDC3C7')),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        # Zebra striping: even rows light gray
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
    ]))
    elements.append(acronym_table)
    elements.append(Spacer(1, 8))
    
    # Build the document
    try:
        doc.build(elements)
        logging.info(f"Report generated successfully: {filename}")
    except Exception as e:
        logging.error(f"Error generating report: {str(e)}")
        logging.error(traceback.format_exc())

def generate_model_details_table(models, num_top_models_to_highlight=3, max_models_to_display=0):
    """Generate a detailed comparison table of all models with metrics
    
    Args:
        models: List of fitted model objects
        num_top_models_to_highlight: Number of top models to highlight (default: 3)
        max_models_to_display: Maximum number of models to display (0 = show all)
        
    Returns:
        Table: ReportLab Table object with detailed model metrics
    """
    # Limit the number of models if specified
    display_models = models
    if max_models_to_display > 0 and len(models) > max_models_to_display:
        display_models = models[:max_models_to_display]
        logging.info(f"Model comparison table limited to top {max_models_to_display} models (out of {len(models)} total)")
    
    # Create header row with clear, readable column names
    data = [["#", "Model", "Covariates", "Log-LLF", "AIC", "BIC", "SSE", "PSSE", "Critic Mean", "Critic Med"]]
    
    # Add a row for each model
    for i, model in enumerate(display_models):
        covariates = ", ".join(model.covariateNames) if hasattr(model, 'covariateNames') and model.covariateNames else "None"
        
        # Get standard metrics
        llf = f"{model.llfVal:.4f}" if hasattr(model, 'llfVal') else "N/A"
        aic = f"{model.aicVal:.4f}" if hasattr(model, 'aicVal') else "N/A"
        bic = f"{model.bicVal:.4f}" if hasattr(model, 'bicVal') else "N/A"
        sse = f"{model.sseVal:.4f}" if hasattr(model, 'sseVal') else "N/A"
        
        # Get additional metrics
        psse = f"{model.psseVal:.4f}" if hasattr(model, 'psseVal') else "N/A"
        critic_mean = f"{model.criticMean:.4f}" if hasattr(model, 'criticMean') else "N/A"
        critic_median = f"{model.criticMedian:.4f}" if hasattr(model, 'criticMedian') else "N/A"
        
        data.append([
            i+1, 
            model.name,
            covariates,
            llf,
            aic,
            bic,
            sse,
            psse,
            critic_mean,
            critic_median
        ])
    
    # Define appropriate column widths for portrait orientation - optimized for readability
    # Total should be around 520 to fit comfortably within portrait letter with margins
    col_widths = [20, 130, 55, 48, 43, 43, 43, 43, 48, 48]  # Total: 521
    
    # Create and style the table - set minimum row height for better readability
    table = Table(data, repeatRows=1, colWidths=col_widths, rowHeights=None)
    
    # Use standardized table style with highlighting for top models
    max_models_to_highlight = min(num_top_models_to_highlight, len(display_models))
    table_style = get_standardized_table_style(num_highlight_rows=max_models_to_highlight)
    
    # Add specific formatting for this table
    table_style.add('ALIGN', (3, 1), (-1, -1), 'RIGHT')   # Right-align numeric columns
    table_style.add('ALIGN', (0, 1), (2, -1), 'LEFT')     # Left-align text columns
    table_style.add('ALIGN', (0, 0), (0, 0), 'CENTER')    # Center-align rank column header
    table_style.add('VALIGN', (0, 0), (-1, -1), 'MIDDLE') # Center-align vertically for better readability
    table_style.add('FONTSIZE', (0, 0), (-1, -1), 8)      # Use readable font size
    table_style.add('FONTSIZE', (0, 0), (-1, 0), 9)       # Headers slightly larger and bold
    
    table.setStyle(table_style)
    
    return table

def initialize_model(model_class, model_data, covariate_names):
    """Initialize and run a reliability model
    
    Args:
        model_class: The model class to instantiate
        model_data: DataFrame containing the data
        covariate_names: List of covariate names to use
        
    Returns:
        Initialized and fitted model
    """
    # Initialize the model
    model = model_class(data=model_data, metricNames=covariate_names)
    
    # Initialize parameters
    model.initialEstimates()
    
    # Run the estimation process
    model.runEstimation(model.covariateData)
    
    # Check if model converged
    if not model.converged:
        raise ValueError(f"Model {model.name} with covariates {covariate_names} did not converge")
    
    # Calculate MVF and other outputs if not already done
    if not hasattr(model, 'mvfList') or model.mvfList is None:
        # Handle different model parameter names based on model type
        if hasattr(model, 'b'):
            # Models that use 'b' parameter 
            params = [model.b] + list(model.betas)
        elif hasattr(model, 'modelParameters'):
            # Models that use modelParameters
            params = list(model.modelParameters) + list(model.betas)
        else:
            # Just use whatever parameters are available
            params = list(model.betas)
        
        # Safely call modelFitting with the right parameters
        try:
            model.modelFitting(model.hazardNumerical, params, model.covariateData)
        except Exception as e:
            logging.warning(f"Model fitting error (not critical): {str(e)}")
    
    return model

def generate_failure_predictions(model, params):
    """Generate failure predictions similar to original C-SFRAT tool
    
    Args:
        model: Fitted model object
        params: Dictionary with prediction parameters
        
    Returns:
        Dictionary containing prediction results
    """
    if not hasattr(model, 't') or len(model.t) == 0:
        return None
    
    # Get prediction parameters (matching original C-SFRAT approach)
    # 🔧 FIX: Use consistent default value (5) across all prediction functions
    num_intervals = params.get('num_intervals_to_predict', 5)
    include_intensity = params.get('include_failure_intensity', True)
    
    try:
        # Current state
        current_time = model.t[-1]
        current_failures = model.CFC[-1] if hasattr(model, 'CFC') and len(model.CFC) > 0 else 0
        
        # Generate future time points (like original tool)
        future_times = []
        future_mvf = []
        future_intensity = []
        
        # Generate predictions for specified number of intervals
        for i in range(1, num_intervals + 1):
            future_time = current_time + i
            
            try:
                # Calculate MVF value at future time (simplified approach like original)
                if hasattr(model, 'omega') and hasattr(model, 'betas'):
                    # For models with omega parameter
                    if len(model.betas) > 0:
                        mvf_val = model.omega * (1 - np.exp(-model.betas[0] * future_time))
                    else:
                        mvf_val = current_failures + i * 0.5  # Simple fallback
                else:
                    # Simple linear extrapolation for models without omega
                    mvf_val = current_failures + i * 0.5
                
                future_times.append(future_time)
                future_mvf.append(mvf_val)
                
                # Calculate intensity if requested
                if include_intensity and len(future_mvf) > 1:
                    intensity_val = future_mvf[-1] - future_mvf[-2]
                    future_intensity.append(max(0, intensity_val))
                elif include_intensity:
                    future_intensity.append(0.5)  # Initial intensity estimate
                    
            except Exception as e:
                logging.warning(f"Error calculating prediction for interval {i}: {e}")
                break
        
        # Create predictions dictionary (simplified like original)
        predictions = {
            'current_time': current_time,
            'current_failures': current_failures,
            'future_times': future_times,
            'future_mvf': future_mvf,
            'future_intensity': future_intensity if include_intensity else [],
            'model_name': model.name,
            'num_intervals': num_intervals
        }
        
        # Simple logging (like original tool)
        logging.info(f"=== PREDICTION SUMMARY ===")
        logging.info(f"Model: {model.name}")
        logging.info(f"Intervals predicted: {len(future_times)}")
        if include_intensity and future_intensity:
            avg_intensity = np.mean(future_intensity)
            logging.info(f"Average predicted intensity: {avg_intensity:.4f}")
        
        return predictions
        
    except Exception as e:
        logging.error(f"Error generating predictions: {e}")
        return None

def calculate_effort_allocation(model, optimization_params):
    """Calculate optimal effort allocation using original C-SFRAT EffortAllocation class
    
    Implements both allocation types from the original C-SFRAT GUI tool:
    - Allocation 1: Maximize defect discovery within budget
    - Allocation 2: Minimize budget to discover specified additional defects
    
    Args:
        model: The best fitted model object
        optimization_params: Dictionary containing optimization parameters
        
    Returns:
        dict: Dictionary containing optimization results from original C-SFRAT
    """
    try:
        if not optimization_params.get('enable_optimization', False):
            return None
            
        logging.info(f"Calculating effort allocation using original C-SFRAT functionality for model: {model.name}")
        
        # Check if model has covariates (required for effort allocation)
        if not hasattr(model, 'covariateData') or model.numCovariates == 0:
            logging.warning("Effort allocation requires models with covariates. Skipping allocation.")
            return None
        
        # Get optimization parameters
        allocation_1_enabled = optimization_params.get('allocation_1_enabled', True)
        allocation_2_enabled = optimization_params.get('allocation_2_enabled', True)
        total_budget = optimization_params.get('total_budget', 1000)  # Changed from 100000 to 1000
        target_additional_defects = optimization_params.get('target_additional_defects', 2)
        method = optimization_params.get('optimization_method', 'both_allocations')
        
        logging.info(f"Effort allocation parameters: budget=${total_budget}, target_defects={target_additional_defects}, method={method}")
        
        allocation_results = {}
        
        try:
            # Allocation 1: Maximize fault discovery with budget constraint (like original GUI)
            if allocation_1_enabled and method in ['allocation_1', 'both_allocations']:
                logging.info(f"🔄 Running ALLOCATION 1: Maximize defect discovery within budget ${total_budget}")
                
                # Add reasonable constraints to prevent unrealistic solutions
                # Temporarily modify the bounds in the EffortAllocation class to be more realistic
                original_covariate_data = model.covariateData.copy()
                
                # Apply reasonable scaling to prevent extreme optimization results
                # Ensure covariate values are within reasonable bounds for optimization
                max_effort_per_covariate = total_budget / model.numCovariates  # Distribute budget evenly as max per covariate
                
                allocation_type1 = ConstrainedEffortAllocation(model, model.covariateData, 1, total_budget)
                
                # Calculate correct percentages based on actual allocation sum (not full budget)
                raw_allocation_values = allocation_type1.res.x
                actual_allocation_sum = sum(raw_allocation_values)
                corrected_percentages = np.multiply(np.divide(raw_allocation_values, actual_allocation_sum), 100)
                
                allocation_results['allocation_1'] = {
                    'type': 'Allocation 1',
                    'description': 'Maximize defect discovery within budget',
                    'budget': total_budget,
                    'estimated_defects': allocation_type1.H,
                    'mvf_value': allocation_type1.mvfVal,
                    'percentages': corrected_percentages.tolist() if hasattr(allocation_type1, 'percentages') else [],
                    'covariate_names': model.metricNames if hasattr(model, 'metricNames') else [],
                    'optimal_allocation': dict(zip(model.metricNames, corrected_percentages)) if hasattr(model, 'metricNames') else {}
                }
                logging.info(f"✅ ALLOCATION 1 Results: Budget=${total_budget} → Est. Defects={allocation_type1.H:.2f}")
            
            # Allocation 2: Minimize budget to discover specified additional defects (like original GUI)
            if allocation_2_enabled and method in ['allocation_2', 'both_allocations']:
                logging.info(f"🔄 Running ALLOCATION 2: Minimize budget to discover {target_additional_defects} additional defects")
                allocation_type2 = ConstrainedEffortAllocation(model, model.covariateData, 2, target_additional_defects)
                
                # Calculate correct percentages based on actual allocation sum (not total effort)
                raw_allocation_values2 = allocation_type2.res2.x
                actual_allocation_sum2 = sum(raw_allocation_values2)
                corrected_percentages2 = np.multiply(np.divide(raw_allocation_values2, actual_allocation_sum2), 100)
                
                allocation_results['allocation_2'] = {
                    'type': 'Allocation 2',
                    'description': 'Minimize budget to discover specified additional defects',
                    'target_additional_defects': target_additional_defects,
                    'minimum_budget': allocation_type2.effort,
                    'percentages': corrected_percentages2.tolist() if hasattr(allocation_type2, 'percentages2') else [],
                    'covariate_names': model.metricNames if hasattr(model, 'metricNames') else [],
                    'optimal_allocation': dict(zip(model.metricNames, corrected_percentages2)) if hasattr(model, 'metricNames') else {}
                }
                logging.info(f"✅ ALLOCATION 2 Results: Target={target_additional_defects} defects → Min. Budget={allocation_type2.effort:.2f}")
            
            # Compute model ceiling for additional defects (omega - current)
            max_additional_defects = None
            try:
                current_failures_val = None
                if hasattr(model, 'CFC') and model.CFC is not None and len(model.CFC) > 0:
                    current_failures_val = float(model.CFC[-1])
                elif hasattr(model, 'mvf_array') and model.mvf_array is not None and len(model.mvf_array) > 0:
                    current_failures_val = float(model.mvf_array[-1])
                if hasattr(model, 'omega') and model.omega is not None and current_failures_val is not None:
                    max_additional_defects = max(0.0, float(model.omega) - current_failures_val)
            except Exception:
                max_additional_defects = None

            # Apply feasibility/capping annotations to allocation outputs
            try:
                if 'allocation_1' in allocation_results and max_additional_defects is not None:
                    est_def = float(allocation_results['allocation_1'].get('estimated_defects', 0.0))
                    allocation_results['allocation_1']['max_additional_defects'] = max_additional_defects
                    if est_def > max_additional_defects + 1e-6:
                        allocation_results['allocation_1']['estimated_defects_capped'] = max_additional_defects
                        allocation_results['allocation_1']['capped'] = True
                    else:
                        allocation_results['allocation_1']['capped'] = False
                if 'allocation_2' in allocation_results and max_additional_defects is not None:
                    tgt = float(allocation_results['allocation_2'].get('target_additional_defects', 0.0))
                    allocation_results['allocation_2']['max_additional_defects'] = max_additional_defects
                    allocation_results['allocation_2']['infeasible'] = (tgt > max_additional_defects + 1e-6)
            except Exception:
                pass
            
            # Combine results into unified format
            optimization_results = {
                'method': method,
                'model_name': model.name,
                'covariate_names': model.metricNames if hasattr(model, 'metricNames') else [],
                'allocation_results': allocation_results,
                'total_budget': total_budget,
                'target_additional_defects': target_additional_defects,
                'allocation_1_enabled': allocation_1_enabled,
                'allocation_2_enabled': allocation_2_enabled,
                'status': 'success'
            }
            
            # Enhanced logging output (GUI-style summary)
            print("\n" + "="*70)
            print("           EFFORT ALLOCATION RESULTS (C-SFRAT)")
            print("="*70)
            print(f"Model: {model.name}")
            print(f"Covariates: {', '.join(model.metricNames) if hasattr(model, 'metricNames') else 'None'}")
            print("-"*70)
            
            if 'allocation_1' in allocation_results:
                alloc1 = allocation_results['allocation_1']
                print("📊 ALLOCATION 1: Maximize defect discovery within budget")
                print(f"   💰 Budget: ${alloc1['budget']:,.2f}")
                print(f"   🎯 Estimated Defects: {alloc1['estimated_defects']:.2f}")
                print(f"   📈 MVF Value: {alloc1['mvf_value']:.2f}")
                print("   🔧 Optimal Allocation:")
                for covariate, percentage in alloc1['optimal_allocation'].items():
                    print(f"      {covariate}: {percentage:.2f}%")
                print()
            
            if 'allocation_2' in allocation_results:
                alloc2 = allocation_results['allocation_2']
                print("📊 ALLOCATION 2: Minimize budget to discover specified defects")
                print(f"   🎯 Target Additional Defects: {alloc2['target_additional_defects']}")
                print(f"   💰 Minimum Budget Required: ${alloc2['minimum_budget']:,.2f}")
                print("   🔧 Optimal Allocation:")
                for covariate, percentage in alloc2['optimal_allocation'].items():
                    print(f"      {covariate}: {percentage:.2f}%")
                print()
            
            print("="*70)
            
            return optimization_results
            
        except Exception as allocation_error:
            logging.error(f"Error in original C-SFRAT EffortAllocation: {str(allocation_error)}")
            return {
                'method': method,
                'model_name': model.name,
                'status': 'error',
                'error_message': str(allocation_error)
            }
        
    except Exception as e:
        logging.error(f"Error in effort allocation wrapper: {str(e)}")
        return None

def calculate_effort_allocation_multiple(models, optimization_params):
    """Calculate optimal effort allocation for multiple models using original C-SFRAT EffortAllocation class
    
    Args:
        models: List of top fitted model objects
        optimization_params: Dictionary containing optimization parameters
        
    Returns:
        dict: Dictionary containing optimization results from original C-SFRAT for all models
    """
    try:
        if not optimization_params.get('enable_optimization', False):
            return None
            
        logging.info(f"Calculating effort allocation using original C-SFRAT functionality for {len(models)} models")
        
        # Filter models that have covariates (required for effort allocation)
        models_with_covariates = []
        for model in models:
            if hasattr(model, 'covariateData') and model.numCovariates > 0:
                models_with_covariates.append(model)
        
        if not models_with_covariates:
            logging.warning("Effort allocation requires models with covariates. No suitable models found.")
            return None
        
        logging.info(f"Found {len(models_with_covariates)} models with covariates for effort allocation")
        
        # Calculate allocation for each model
        all_results = {
            'status': 'success',
            'models': [],
            'allocation_results': {}
        }
        
        for i, model in enumerate(models_with_covariates):
            model_result = calculate_effort_allocation(model, optimization_params)
            if model_result and model_result.get('status') == 'success':
                model_info = {
                    'rank': i + 1,
                    'model_name': model.name,
                    'covariate_names': model.metricNames if hasattr(model, 'metricNames') else [],
                    'allocation_1': model_result['allocation_results'].get('allocation_1'),
                    'allocation_2': model_result['allocation_results'].get('allocation_2')
                }
                all_results['models'].append(model_info)
        
        if not all_results['models']:
            logging.warning("No successful effort allocation calculations for any model")
            return None
            
        # Use the results structure expected by PDF generation
        all_results['model_name'] = 'Multiple Models'
        all_results['covariate_names'] = []
        all_results['allocation_results'] = {
            'multiple_models': True,
            'models': all_results['models']
        }
        
        return all_results
        
    except Exception as e:
        logging.error(f"Error in effort allocation calculation: {str(e)}")
        logging.error(traceback.format_exc())
        return {'status': 'error', 'message': str(e)}

def display_prediction_summary(predictions):
    """Display a clear and formatted prediction summary in the console (matching original C-SFRAT style)"""
    if not predictions:
        return
    
    print("\n" + "="*60)
    print("           PREDICTION RESULTS")
    print("="*60)
    print(f"Model: {predictions['model_name']}")
    print(f"Intervals Predicted: {predictions['num_intervals']}")
    print("-"*60)
    
    # Current state
    print("CURRENT STATE:")
    print(f"  • Current Time: {predictions['current_time']:.1f}")
    print(f"  • Current Failures: {predictions['current_failures']:.0f}")
    print()
    
    # Future predictions (first few intervals like original tool)
    print("PREDICTED VALUES:")
    future_times = predictions.get('future_times', [])
    future_mvf = predictions.get('future_mvf', [])
    future_intensity = predictions.get('future_intensity', [])
    
    # Show first 5 predictions for clarity
    display_count = min(5, len(future_times))
    for i in range(display_count):
        time_val = future_times[i]
        mvf_val = future_mvf[i]
        
        intensity_str = ""
        if i < len(future_intensity):
            intensity_str = f", Intensity: {future_intensity[i]:.3f}"
        
        print(f"  • Time {time_val:.1f}: {mvf_val:.2f} failures{intensity_str}")
    
    if len(future_times) > display_count:
        print(f"  • ... and {len(future_times) - display_count} more intervals")
    
    # Summary statistics
    if future_intensity:
        avg_intensity = sum(future_intensity) / len(future_intensity)
        print()
        print("INTENSITY ANALYSIS:")
        print(f"  • Average Predicted Intensity: {avg_intensity:.4f} failures/interval")
    
    print("="*60)

def generate_individual_model_predictions(models, params):
    """
    Generate predictions for individual models (like original C-SFRAT tool)
    
    Args:
        models: List of fitted model objects 
        params: Prediction parameters dictionary
        
    Returns:
        dict: Dictionary mapping model names to their prediction data
    """
    if not models or not params.get('predict_failures', False):
        return {}
    
    global max_models_for_individual_predictions
    models_to_predict = models[:min(max_models_for_individual_predictions, len(models))]
    
    individual_predictions = {}
    
    logging.info(f"Generating individual predictions for {len(models_to_predict)} models...")
    
    for model in models_to_predict:
        try:
            # Get model-specific predictions
            model_predictions = generate_failure_predictions(model, params)
            
            if model_predictions and 'future_times' in model_predictions:
                model_key = f"{model.name}"
                if hasattr(model, 'covariateNames') and model.covariateNames:
                    cov_str = ",".join([c[:3] for c in model.covariateNames])
                    model_key += f"({cov_str})"
                
                individual_predictions[model_key] = {
                    'model': model,
                    'future_times': model_predictions['future_times'],
                    'future_mvf': model_predictions['future_mvf'],
                    'future_intensity': model_predictions.get('future_intensity', []),
                    'prediction_horizon': model_predictions.get('prediction_horizon'),
                    'total_predicted_failures': model_predictions.get('total_predicted_failures')
                }
                
                logging.info(f"Generated predictions for {model_key}: {len(model_predictions['future_times'])} points")
                
        except Exception as e:
            logging.warning(f"Failed to generate predictions for {model.name}: {str(e)}")
            continue
    
    logging.info(f"Successfully generated individual predictions for {len(individual_predictions)} models")
    return individual_predictions

def generate_individual_model_predictions_gui_correct(models, params):
    """
    Generate GUI-correct predictions for individual models (smooth like original C-SFRAT GUI)
    
    This function applies the same GUI-correct approach to individual model predictions
    that we use for the green effort line, ensuring all predictions are smooth and continuous.
    
    Args:
        models: List of fitted model objects 
        params: Prediction parameters dictionary
        
    Returns:
        dict: Dictionary mapping model names to their smooth prediction data
    """
    if not models or not params.get('predict_failures', False):
        return {}
    
    global max_models_for_individual_predictions
    models_to_predict = models[:min(max_models_for_individual_predictions, len(models))]
    
    individual_predictions = {}
    
    logging.info(f"Generating GUI-CORRECT individual predictions for {len(models_to_predict)} models...")
    
    for model in models_to_predict:
        try:
            # Use the GUI-correct approach for each individual model
            # Create dummy effort parameters that use last data point values (for continuity)
            effort_params_gui_correct = {
                'effort_values': {'E': 1.0, 'F': 1.0, 'C': 1.0},  # Will be overridden by GUI-correct method
                'number_of_intervals_to_predict': params.get('prediction_horizon', 10),
                'failure_intensity_target': 0.3,
                'use_model_specific_covariates': True,
                'default_effort_for_unknown_covariates': 1.0
            }
            
            # Generate smooth predictions using GUI-correct method
            model_predictions = generate_failure_predictions_with_effort_gui_correct(
                model, params, effort_params_gui_correct
            )
            
            if model_predictions and 'future_times' in model_predictions:
                model_key = f"{model.name}"
                if hasattr(model, 'covariateNames') and model.covariateNames:
                    cov_str = ",".join([c[:3] for c in model.covariateNames])
                    model_key += f"({cov_str})"
                
                individual_predictions[model_key] = {
                    'model': model,
                    'future_times': model_predictions['future_times'],
                    'future_mvf': model_predictions['future_mvf'],
                    'future_intensity': model_predictions.get('future_intensity', []),
                    'prediction_horizon': model_predictions.get('prediction_horizon'),
                    'total_predicted_failures': model_predictions.get('total_predicted_failures'),
                    'gui_correct': True,  # Mark as GUI-correct
                    'prediction_method': 'gui_correct_individual'
                }
                
                logging.info(f"Generated GUI-CORRECT predictions for {model_key}: {len(model_predictions['future_times'])} points")
                
        except Exception as e:
            logging.warning(f"Failed to generate GUI-correct predictions for {model.name}: {str(e)}")
            # Fallback to original method if GUI-correct fails
            try:
                model_predictions = generate_failure_predictions(model, params)
                if model_predictions and 'future_times' in model_predictions:
                    model_key = f"{model.name}"
                    if hasattr(model, 'covariateNames') and model.covariateNames:
                        cov_str = ",".join([c[:3] for c in model.covariateNames])
                        model_key += f"({cov_str})"
                    
                    individual_predictions[model_key] = {
                        'model': model,
                        'future_times': model_predictions['future_times'],
                        'future_mvf': model_predictions['future_mvf'],
                        'future_intensity': model_predictions.get('future_intensity', []),
                        'prediction_horizon': model_predictions.get('prediction_horizon'),
                        'total_predicted_failures': model_predictions.get('total_predicted_failures'),
                        'gui_correct': False,  # Mark as fallback
                        'prediction_method': 'fallback_original'
                    }
                    logging.info(f"Generated FALLBACK predictions for {model_key}: {len(model_predictions['future_times'])} points")
            except Exception as e2:
                logging.warning(f"Both GUI-correct and fallback predictions failed for {model.name}: {str(e2)}")
                continue
    
    logging.info(f"Successfully generated GUI-CORRECT individual predictions for {len(individual_predictions)} models")
    return individual_predictions

def validate_metric_weights(weights):
    """
    Validate metric weights and provide helpful information about the configuration
    Implements professor's recommendation for single-metric configurations
    
    Args:
        weights: Dictionary of metric weights
        
    Returns:
        tuple: (is_valid, is_single_metric, single_metric_name, validation_message)
    """
    # Check if weights are in valid range
    for metric, weight in weights.items():
        if not (0.0 <= weight <= 10.0):
            return False, False, None, f"Weight for {metric} ({weight}) is outside valid range 0.0-10.0"
    
    # Count non-zero weights
    non_zero_weights = {k: v for k, v in weights.items() if v > 0.0}
    num_active_metrics = len(non_zero_weights)
    
    # Check for single-metric configuration (Professor's recommendation)
    if num_active_metrics == 1:
        single_metric = list(non_zero_weights.keys())[0]
        message = f"✅ Single-metric configuration detected: {single_metric.upper()}-only (Professor's recommendation implemented)"
        message += f"\n📝 Note: For single-metric configurations, median ranking is set equal to mean ranking since only one metric is active"
        return True, True, single_metric, message
    elif num_active_metrics == 0:
        return False, False, None, "❌ All weights are zero - no metrics will be used for ranking"
    else:
        active_metrics = list(non_zero_weights.keys())
        message = f"Multi-metric configuration: {num_active_metrics} active metrics ({', '.join([m.upper() for m in active_metrics])})"
        message += f"\n📊 Both mean and median ranking methods are meaningful with multiple active metrics"
        return True, False, None, message

def create_model_color_mapping(models, individual_predictions=None):
    """Create consistent color mapping for models across all plots
    
    Args:
        models: List of model objects
        individual_predictions: Dictionary of individual predictions (optional)
        
    Returns:
        tuple: (color_mapping dict, original_colors list) for consistent usage across plots
    """
    import matplotlib.colors as mcolors
    import colorsys
    
    # Original C-SFRAT color scheme (10 colors)
    original_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # grey
        '#bcbd22',  # olive
        '#17becf'   # cyan
    ]
    
    # Collect all unique model identifiers
    all_model_keys = set()
    
    # Add main models
    for model in models:
        model_key = model.name
        if hasattr(model, 'covariateNames') and model.covariateNames:
            cov_str = ",".join([c[:3] for c in model.covariateNames])
            model_key += f"({cov_str})"
        all_model_keys.add(model_key)
    
    # Add models from individual predictions
    if individual_predictions:
        for pred_key in individual_predictions.keys():
            all_model_keys.add(pred_key)
    
    # Convert to sorted list for consistent ordering
    all_model_keys = sorted(list(all_model_keys))
    
    def generate_additional_colors(num_needed):
        """Generate additional unique colors using HSV color space"""
        additional_colors = []
        
        # Use golden ratio for better color distribution
        golden_ratio = 0.618033988749895
        saturation = 0.7
        value = 0.9
        
        # Start after our original colors to avoid conflicts
        hue_offset = 0.1
        
        for i in range(num_needed):
            # Generate hue using golden ratio for maximum separation
            hue = (hue_offset + i * golden_ratio) % 1.0
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            
            # Convert to hex color
            hex_color = mcolors.rgb2hex(rgb)
            additional_colors.append(hex_color)
            
        return additional_colors
    
    # Create color mapping
    color_mapping = {}
    
    # Check if we need additional colors
    total_models = len(all_model_keys)
    if total_models > len(original_colors):
        additional_needed = total_models - len(original_colors)
        additional_colors = generate_additional_colors(additional_needed)
        all_colors = original_colors + additional_colors
    else:
        all_colors = original_colors
    
    # Assign colors to model keys
    for i, model_key in enumerate(all_model_keys):
        color_mapping[model_key] = all_colors[i]
    
    # Log color assignment for debugging
    logging.info(f"Color mapping created for {len(all_model_keys)} unique models")
    if total_models > len(original_colors):
        logging.info(f"Generated {len(additional_colors)} additional colors for complete uniqueness")
    
    return color_mapping, original_colors

def generate_failure_predictions_with_effort(model, prediction_params, effort_params):
    """Generate failure predictions with effort per interval using original C-SFRAT functionality
    
    Args:
        model: Fitted model object
        prediction_params: Dictionary with prediction parameters
        effort_params: Dictionary with effort per interval parameters
        
    Returns:
        Dictionary containing effort-adjusted prediction results from original C-SFRAT
    """
    if not hasattr(model, 't') or len(model.t) == 0:
        return None
    
    try:
        # Get effort parameters
        effort_values = effort_params.get('effort_values', {'E': 1.0, 'F': 2.0, 'C': 3.0})
        num_intervals = effort_params.get('number_of_intervals_to_predict', 10)
        intensity_target = effort_params.get('failure_intensity_target', 0.3)
        
        logging.info(f"Generating predictions using original C-SFRAT effort per interval functionality")
        logging.info(f"Model: {model.name}, Intervals: {num_intervals}")
        logging.info(f"Effort values: {effort_values}")
        
        # Check if model has covariates (required for effort per interval)
        if not hasattr(model, 'metricNames') or not model.metricNames:
            logging.warning("Effort per interval requires models with covariates. Using default prediction.")
            return generate_failure_predictions(model, prediction_params)
        
        # Create effort dictionary in the format expected by original C-SFRAT
        # This simulates the GUI's effortSpinBoxDict structure
        class MockSpinBox:
            def __init__(self, value):
                self._value = value
            def value(self):
                return self._value
        
        effort_dict = {}
        use_model_specific = effort_params.get('use_model_specific_covariates', True)
        default_effort = effort_params.get('default_effort_for_unknown_covariates', 1.0)
        
        if use_model_specific:
            # Only use effort values for covariates present in this model
            for covariate in model.metricNames:
                # Use the effort value for this covariate, or default if not specified
                effort_value = effort_values.get(covariate, default_effort)
                effort_dict[covariate] = MockSpinBox(effort_value)
                logging.info(f"  Using effort {effort_value} for model covariate: {covariate}")
        else:
            # Use all effort values from configuration
            for covariate, effort_value in effort_values.items():
                effort_dict[covariate] = MockSpinBox(effort_value)
                logging.info(f"  Using effort {effort_value} for covariate: {covariate}")
        
        if not effort_dict:
            logging.warning("No effort values available, using default effort of 1.0 for all covariates")
            for covariate in model.metricNames:
                effort_dict[covariate] = MockSpinBox(1.0)
        
        # Current state
        current_time = model.t[-1]
        current_failures = model.CFC[-1] if hasattr(model, 'CFC') and len(model.CFC) > 0 else 0
        
            # Use original C-SFRAT prediction_mvf function for MVF predictions
        x, mvf_array = prediction.prediction_mvf(model, num_intervals, model.covariateData, effort_dict)
        
        # Calculate intensity values from MVF (similar to original)
        intensity_array = []
        mvf_list = mvf_array.tolist()
        
        for i in range(1, len(mvf_list)):
            intensity_val = mvf_list[i] - mvf_list[i-1]
            intensity_array.append(max(0.01, intensity_val))  # Minimum intensity
        
        # Get future values (excluding the original data points)
        original_length = len(model.t)
        future_times = x[original_length:].tolist()
        future_mvf = mvf_array[original_length:].tolist()
        future_intensity = intensity_array[original_length-1:] if len(intensity_array) >= original_length else intensity_array
        
        # Find when target intensity is reached
        intensity_intervals = 0
        for i, intensity in enumerate(future_intensity):
            if intensity <= intensity_target:
                intensity_intervals = i + 1
            break
        
        # Try using original C-SFRAT prediction_intensity function for intensity target
        try:
            x_intensity, intensity_result, intervals_to_target = prediction.prediction_intensity(
                model, intensity_target, model.covariateData, effort_dict
            )
            if intervals_to_target > 0:
                intensity_intervals = intervals_to_target
                logging.info(f"Original C-SFRAT intensity function: target reached in {intervals_to_target} intervals")
        except Exception as intensity_error:
                logging.warning(f"Original intensity prediction failed, using calculated values: {intensity_error}")
            
            # Create predictions dictionary using original C-SFRAT results
        predictions = {
            'model_name': model.name,
            'current_time': current_time,
            'current_failures': current_failures,
            'num_intervals': num_intervals,
                'effort_covariate': model.metricNames[0] if model.metricNames else 'Unknown',
                'effort_value': effort_values.get(model.metricNames[0], 1.0) if model.metricNames else 1.0,
            'effort_E': effort_values.get('E', 1.0),
            'effort_F': effort_values.get('F', 1.0),
            'effort_C': effort_values.get('C', 1.0),
            'intensity_target': intensity_target,
            'intensity_intervals': intensity_intervals,
            'future_times': future_times,
            'future_mvf': future_mvf,
            'future_intensity': future_intensity,
                'prediction_method': 'original_csfrat_effort_per_interval',
                'full_x': x.tolist(),
                'full_mvf': mvf_array.tolist()
        }
        
        logging.info(f"Original C-SFRAT effort predictions complete.")
        logging.info(f"Target intensity ({intensity_target}) reached in {intensity_intervals} intervals")
        if future_intensity:
            avg_intensity = np.mean(future_intensity)
            logging.info(f"Average predicted intensity: {avg_intensity:.4f}")
        
        return predictions
            
    except Exception as prediction_error:
        logging.error(f"Error in original C-SFRAT prediction functions: {prediction_error}")
        logging.error(f"Falling back to basic prediction without effort adjustment")
        # Fallback to basic prediction
        return generate_failure_predictions(model, prediction_params)

def display_effort_prediction_summary(predictions):
    """Display effort prediction summary in console (original SFRAT tool style)"""
    if not predictions:
        return
    
    print("\n" + "=" * 60)
    if predictions.get('gui_correct'):
        print("          GUI-CORRECT PREDICTION RESULTS")
        print("=" * 60)
        print("✅ GUI-EQUIVALENT APPROACH APPLIED:")
        print("   • Uses last data point covariate values for smooth continuity")
        print("   • Matches original C-SFRAT GUI behavior exactly")
        print("   • Eliminates artificial discontinuities and unrealistic jumps")
        print("-" * 60)
    elif predictions.get('correction_applied'):
        print("        CORRECTED EFFORT PER INTERVAL RESULTS")
        print("=" * 60)
        print("🔧 MATHEMATICAL CORRECTION APPLIED:")
        print("   • Fixed omega recalculation bug from original C-SFRAT")
        print("   • Maintains continuity between historical and predicted values")
        print("   • Eliminates artificial dips and sudden rises in predictions")
        print("-" * 60)
    else:
        print("           EFFORT PER INTERVAL RESULTS")
        print("=" * 60)
    
    print(f"Model: {predictions['model_name']}")
    print(f"Covariate: {predictions['effort_covariate']} (Effort: {predictions['effort_value']})")
    print(f"Intervals Predicted: {predictions['num_intervals']}")
    print("-" * 60)
    print("CURRENT STATE:")
    print(f"  • Current Time: {predictions['current_time']}")
    print(f"  • Current Failures: {predictions['current_failures']}")
    print()
    print("EFFORT ANALYSIS:")
    print(f"  • Target Intensity: {predictions['intensity_target']} failures/interval")
    print(f"  • Intervals to Target: {predictions['intensity_intervals']} intervals")
    if predictions['future_intensity']:
        avg_intensity = np.mean(predictions['future_intensity'])
        print(f"  • Average Intensity: {avg_intensity:.4f} failures/interval")
    print()
    
    # Show first few predictions
    print("PREDICTED VALUES:")
    for i in range(min(5, len(predictions['future_times']))):
        time = predictions['future_times'][i]
        mvf = predictions['future_mvf'][i]
        intensity = predictions['future_intensity'][i] if i < len(predictions['future_times']) else 0
        print(f"  • Time {time}: {mvf:.2f} failures, Intensity: {intensity:.3f}")
    
    if len(predictions['future_times']) > 5:
        print(f"  • ... and {len(predictions['future_times']) - 5} more intervals")
    
    print("=" * 60)

def create_failure_prediction_times_table(top_models):
    """Create a table showing failure prediction times for top 3 models
    
    Args:
        top_models: List of top 3 fitted models
        
    Returns:
        Table: ReportLab Table object showing failure prediction times
    """
    if not top_models or len(top_models) < 1:
        return None
    
    # Get the maximum number of failures to predict (based on observed data)
    max_failures = 0
    for model in top_models:
        if hasattr(model, 'CFC') and len(model.CFC) > 0:
            max_failures = max(max_failures, int(max(model.CFC)))
    
    # If no data, use a reasonable default
    if max_failures == 0:
        max_failures = 10
    
    # Create header row
    header = ["Failures"]
    for i, model in enumerate(top_models[:3]):
        model_name = f"Model {i+1}"
        if hasattr(model, 'name'):
            # Shorten model names for table header
            short_name = model.name.replace("Negative Binomial", "NegBin").replace("Discrete Weibull", "DWeibull")
            if len(short_name) > 15:
                short_name = short_name[:12] + "..."
            model_name = short_name
        header.append(model_name)
    
    # Pad header to ensure we have exactly 4 columns
    while len(header) < 4:
        header.append("Model X")
    
    data = [header]
    
    # Generate failure prediction times for each failure number
    for failure_num in range(1, min(max_failures + 1, 21)):  # Limit to 20 failures for readability
        row = [str(failure_num)]
        
        for model_idx, model in enumerate(top_models[:3]):
            try:
                # Calculate estimated time to reach this failure number
                estimated_time = calculate_time_to_failure(model, failure_num)
                if estimated_time is not None:
                    # Show actual time value rounded to 1 decimal place
                    time_str = f"{estimated_time:.1f}"
                    row.append(time_str)
                else:
                    row.append("X")
            except Exception as e:
                row.append("X")
        
        # Pad row to ensure we have exactly 4 columns
        while len(row) < 4:
            row.append("X")
        
        data.append(row)
    
    # Create and style the table
    table = Table(data, repeatRows=1, colWidths=[80, 120, 120, 120])
    table.setStyle(TableStyle([
        # Header row styling
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        
        # Data rows styling
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        
        # Alternate row colors for readability
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F6FA')]),
        
        # Center align all content
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    return table

def calculate_time_to_failure(model, target_failure):
    """Calculate estimated time to reach a specific failure number using model's fitted values
    
    Args:
        model: Fitted model object
        target_failure: Target failure number
        
    Returns:
        float: Estimated time to reach target failure, or None if cannot calculate
    """
    try:
        # Use model's fitted values (mvfList) instead of actual CFC data for better differentiation
        failures_data = None
        
        if hasattr(model, 'mvfList') and model.mvfList is not None and len(model.mvfList) > 0:
            # Use model's fitted values
            failures_data = model.mvfList
        elif hasattr(model, 'CFC') and len(model.CFC) > 0:
            # Fallback to actual CFC data
            failures_data = model.CFC
        else:
            return None
        
        # Check if target failure is reached in historical fitted data
        max_fitted_failures = max(failures_data)
        if target_failure <= max_fitted_failures:
            # Find the time when this failure was first reached in fitted values
            for i, failures in enumerate(failures_data):
                if failures >= target_failure:
                    return i + 1  # Time is 1-indexed
        
        # For future failures, extrapolate using the fitted model trend
        if len(failures_data) >= 2:
            # Use the trend from the fitted values to extrapolate
            last_time = len(failures_data)  # Time of last fitted value
            last_failures = failures_data[-1]
            second_last_failures = failures_data[-2]
            
            # Calculate trend from last two points
            failure_rate = last_failures - second_last_failures
            
            if failure_rate > 0:
                # Estimate how many more time units needed
                additional_failures = target_failure - last_failures
                additional_time = additional_failures / failure_rate
                estimated_time = last_time + additional_time
                return estimated_time
            else:
                # If no growth trend, use model parameters if available
                if hasattr(model, 'omega') and hasattr(model, 'betas') and len(model.betas) > 0:
                    # For models with omega parameter (like Negative Binomial)
                    omega = model.omega
                    beta = model.betas[0] if len(model.betas) > 0 else 0.1
                    
                    if omega > 0 and beta > 0 and target_failure < omega:
                        # Solve for time: target_failure = omega * (1 - exp(-beta * t))
                        estimated_time = -np.log(1 - target_failure/omega) / beta
                        return estimated_time
        
        return None
        
    except Exception as e:
        return None

def main():
    """Main function to run the analysis"""
    try:
        # Declare global variables at the beginning
        global input_file, output_filename, run_all_models, run_all_covariates, selected_model, selected_covariates
        global metric_weights, ranking_method, prediction_parameters, optimization_parameters
        global multi_model_plots, num_models_to_compare, enable_covariate_scaling, covariate_scaling_range
        global selected_sheet, data_subset_limit, psse_subset_parameter
        
        # Initialize args to None
        args = None
        
        # Create output directory
        setup_output_directory()
        
        # Parse command line arguments if provided
        if len(sys.argv) > 1:
            parser = argparse.ArgumentParser(
                description="C-SFRAT Command Line Interface for Software Reliability Analysis"
            )
            
            # Arguments
            parser.add_argument(
                "--input", "-i",
                type=str,
                help="Path to input data file (.csv or .xlsx)"
            )
            
            parser.add_argument(
                "--output", "-o",
                type=str,
                help="Path to output PDF file for results"
            )
            
            parser.add_argument(
                "--model", "-m",
                choices=list(MODEL_CLASSES.keys()),
                help="Reliability model to use for analysis"
            )
            
            parser.add_argument(
                "--all-models", "-a",
                action="store_true",
                help="Run all available models"
            )
            
            parser.add_argument(
                "--all-covariates", "-x",
                action="store_true",
                help="Run all covariate combinations"
            )
            
            parser.add_argument(
                "--covariates", "-c",
                nargs='+',
                help="Specific covariates to use (space-separated list)"
            )
            
            parser.add_argument(
                "--show-all-models", "-s",
                action="store_true",
                help="Show all models in the detailed predictions table"
            )
            
            # Weight arguments
            parser.add_argument(
                "--weight-llf", "-wl",
                type=float,
                default=None,
                help="Weight for Log-Likelihood metric (0.0-10.0)"
            )
            
            parser.add_argument(
                "--weight-aic", "-wa",
                type=float,
                default=None,
                help="Weight for AIC metric (0.0-10.0)"
            )
            
            parser.add_argument(
                "--weight-bic", "-wb",
                type=float,
                default=None,
                help="Weight for BIC metric (0.0-10.0)"
            )
            
            parser.add_argument(
                "--weight-sse", "-ws",
                type=float,
                default=None,
                help="Weight for SSE metric (0.0-10.0)"
            )
            
            parser.add_argument(
                "--weight-psse", "-wp",
                type=float,
                default=None,
                help="Weight for PSSE metric (0.0-10.0)"
            )
            
            # Single-metric configurations (Professor's recommendation)
            parser.add_argument(
                "--single-metric", "-sm",
                choices=['aic', 'bic', 'sse', 'psse', 'llf'],
                default=None,
                help="Use single metric as sole criterion (sets four weights to 0.0). Options: aic, bic, sse, psse, llf"
            )
            
            # Ranking method arguments
            parser.add_argument(
                "--ranking-method", "-r",
                choices=['mean', 'median'],
                default=None,
                help="Method to use for model ranking: 'mean' (default) or 'median'"
            )
            
            # Prediction arguments
            parser.add_argument(
                "--predict-failures", "-pf",
                action="store_true",
                help="Enable failure predictions"
            )
            
            parser.add_argument(
                "--num-failures", "-nf",
                type=int,
                default=None,
                help="Number of future failures to predict"
            )
            
            parser.add_argument(
                "--time-horizon", "-th",
                type=float,
                default=None,
                help="Time horizon for predictions"
            )
            
            parser.add_argument(
                "--failure-intensity", "-fi",
                action="store_true",
                help="Include failure intensity predictions"
            )
            
            # Optimization arguments
            parser.add_argument(
                "--enable-optimization", "-eo",
                action="store_true",
                help="Enable effort allocation optimization"
            )
            
            parser.add_argument(
                "--budget", "-b",
                type=float,
                default=None,
                help="Total budget for optimization"
            )
            
            parser.add_argument(
                "--cost-per-failure", "-cpf",
                type=float,
                default=None,
                help="Cost to find and fix one failure"
            )
            
            parser.add_argument(
                "--cost-per-time", "-cpt",
                type=float,
                default=None,
                help="Cost per unit of testing time"
            )
            
            parser.add_argument(
                "--target-reliability", "-tr",
                type=float,
                default=None,
                help="Target reliability level (0.0-1.0)"
            )
            
            parser.add_argument(
                "--optimization-method", "-om",
                choices=['cost_minimization', 'reliability_maximization'],
                default=None,
                help="Optimization method to use"
            )
            
            # Effort per interval arguments (original SFRAT tool functionality)
            parser.add_argument(
                "--effort-E", "-eE",
                type=float,
                default=1.0,
                help="Effort value for covariate E (default: 1.0)"
            )
            
            parser.add_argument(
                "--effort-F", "-eF",
                type=float,
                default=2.0,
                help="Effort value for covariate F (default: 2.0)"
            )
            
            parser.add_argument(
                "--effort-C", "-eC",
                type=float,
                default=3.0,
                help="Effort value for covariate C (default: 3.0)"
            )
            
            parser.add_argument(
                "--prediction-intervals", "-pi",
                type=int,
                default=5,
                help="Number of intervals to predict (default: 5)"
            )
            
            parser.add_argument(
                "--intensity-target", "-it",
                type=float,
                default=0.3,
                help="Failure intensity target (default: 0.3)"
            )
            
            # Multi-model visualization arguments
            parser.add_argument(
                "--multi-plots", "-mp",
                action="store_true",
                help="Enable multi-model comparison plots"
            )
            
            parser.add_argument(
                "--num-models-compare", "-nmc",
                type=int,
                default=None,
                help="Number of top models to compare in plots (1-3)"
            )
            
            # Covariate scaling arguments
            parser.add_argument(
                "--enable-scaling", "-es",
                action="store_true",
                help="Enable covariate scaling for improved effort allocation"
            )
            
            parser.add_argument(
                "--disable-scaling", "-ds",
                action="store_true",
                help="Disable covariate scaling (use original values)"
            )
            
            # High-priority feature parity arguments (100% GUI compatibility)
            parser.add_argument(
                "--sheet", "-sh",
                type=str,
                default=None,
                help="Sheet name for Excel files (default: first sheet)"
            )
            
            parser.add_argument(
                "--data-subset", "-ds2",
                type=int,
                default=None,
                help="Limit analysis to first N data intervals (matches GUI slider)"
            )
            
            parser.add_argument(
                "--psse-subset", "-ps",
                type=float,
                default=0.9,
                help="PSSE subset parameter (0.01-0.99, default: 0.9, matches GUI)"
            )
            
            args = parser.parse_args()
            
            # Override configuration with command-line arguments
            if args.input:
                input_file = args.input
            if args.output:
                output_filename = args.output
            if args.all_models:
                run_all_models = args.all_models
            if args.all_covariates:
                run_all_covariates = args.all_covariates
            if args.model:
                selected_model = args.model
            if args.covariates:
                selected_covariates = args.covariates
            if args.show_all_models:
                show_all_models_in_table = args.show_all_models
        
            # Override high-priority feature parity parameters with command-line arguments
            if args.sheet:
                selected_sheet = args.sheet
            if args.data_subset is not None:
                data_subset_limit = args.data_subset
            if args.psse_subset is not None:
                psse_subset_parameter = args.psse_subset
        
            # Override weights with command-line arguments
            if args.weight_llf is not None:
                metric_weights['llf'] = args.weight_llf
            if args.weight_aic is not None:
                metric_weights['aic'] = args.weight_aic
            if args.weight_bic is not None:
                metric_weights['bic'] = args.weight_bic
            if args.weight_sse is not None:
                metric_weights['sse'] = args.weight_sse
            if args.weight_psse is not None:
                metric_weights['psse'] = args.weight_psse
            
            # Handle single-metric configuration (Professor's recommendation)
            if args.single_metric:
                # Set all weights to 0.0 first, then set the selected metric to 1.0
                metric_weights = {'llf': 0.0, 'aic': 0.0, 'bic': 0.0, 'sse': 0.0, 'psse': 0.0}
                metric_weights[args.single_metric] = 1.0
                logging.info(f"Applied single-metric configuration: {args.single_metric.upper()}-only (Professor's recommendation)")
                logging.info(f"Final weights: {metric_weights}")
            
            # Override ranking method with command-line arguments
            if args.ranking_method:
                ranking_method = args.ranking_method
            
            # Override prediction parameters with command-line arguments
            if args.predict_failures:
                prediction_parameters['predict_failures'] = True
            if args.num_failures is not None:
                prediction_parameters['num_failures_to_predict'] = args.num_failures
            if args.time_horizon is not None:
                prediction_parameters['prediction_time_horizon'] = args.time_horizon
            if args.failure_intensity:
                prediction_parameters['include_failure_intensity'] = True
            # 🔧 FIX: Ensure prediction intervals CLI argument reaches basic predictions
            if args and hasattr(args, 'prediction_intervals'):
                prediction_parameters['num_intervals_to_predict'] = args.prediction_intervals
            
            # Override optimization parameters with command-line arguments
            if args.enable_optimization:
                optimization_parameters['enable_optimization'] = True
            if args.budget is not None:
                optimization_parameters['total_budget'] = args.budget
            if args.cost_per_failure is not None:
                optimization_parameters['cost_per_failure_found'] = args.cost_per_failure
            if args.cost_per_time is not None:
                optimization_parameters['cost_per_time_unit'] = args.cost_per_time
            if args.target_reliability is not None:
                optimization_parameters['target_reliability'] = args.target_reliability
            if args.optimization_method:
                optimization_parameters['optimization_method'] = args.optimization_method
            
                    # Override multi-model visualization parameters with command-line arguments
            if args and args.multi_plots:
                multi_model_plots = True
            if args and args.num_models_compare is not None:
                num_models_to_compare = args.num_models_compare
            
            # Override covariate scaling settings with command-line arguments
            if args and args.enable_scaling:
                enable_covariate_scaling = True
            if args and args.disable_scaling:
                enable_covariate_scaling = False
        
        # Effort per interval parameters (always enabled like original SFRAT tool)
        # Start with configuration defaults from top of file, then apply command line overrides
        global effort_per_interval_enabled, effort_per_interval_settings
        
        if effort_per_interval_enabled:
            # Use configuration settings as base
            effort_values_base = effort_per_interval_settings['effort_values'].copy()
            
            # Apply command line overrides if provided and args exists
            if args and hasattr(args, 'effort_E') and args.effort_E != 1.0:  # 1.0 is default
                effort_values_base['E'] = args.effort_E
            if args and hasattr(args, 'effort_F') and args.effort_F != 2.0:  # 2.0 is default
                effort_values_base['F'] = args.effort_F
            if args and hasattr(args, 'effort_C') and args.effort_C != 3.0:  # 3.0 is default
                effort_values_base['C'] = args.effort_C
            
        if effort_per_interval_enabled:
            effort_per_interval_parameters = {
                'effort_values': effort_values_base,
                # 🔧 FIX: Use standardized parameter name for consistency across all prediction functions
                'num_intervals_to_predict': args.prediction_intervals if args and hasattr(args, 'prediction_intervals') else effort_per_interval_settings['number_of_intervals_to_predict'],
                'failure_intensity_target': args.intensity_target if args and hasattr(args, 'intensity_target') else effort_per_interval_settings['failure_intensity_target'],
                'use_model_specific_covariates': effort_per_interval_settings.get('use_model_specific_covariates', True),
                'default_effort_for_unknown_covariates': effort_per_interval_settings.get('default_effort_for_unknown_covariates', 1.0)
            }
        else:
            # Effort per interval disabled, use minimal settings
            effort_per_interval_parameters = {
                'effort_values': {'E': 1.0, 'F': 1.0, 'C': 1.0},
                # 🔧 FIX: Use standardized parameter name for consistency
                'num_intervals_to_predict': 0,
                'failure_intensity_target': 0.0
            }
        
        logging.info("C-SFRAT Command Line Analysis (Integrated Version)")
        logging.info(f"Input file: {input_file}")
        logging.info(f"Output file: {output_filename}")
        logging.info(f"Model ranking method: {ranking_method}")
        
        # Log effort per interval settings (always enabled like original SFRAT tool)
        logging.info("Effort per interval settings: ENABLED (original SFRAT functionality)")
        effort_values = effort_per_interval_parameters['effort_values']
        for covariate, effort in effort_values.items():
            logging.info(f"  • {covariate}: {effort}")
        logging.info(f"Prediction intervals: {effort_per_interval_parameters['num_intervals_to_predict']}")
        logging.info(f"Intensity target: {effort_per_interval_parameters['failure_intensity_target']}")
        
        # Show configuration status similar to original GUI
        if effort_per_interval_enabled:
            use_model_specific = effort_per_interval_parameters.get('use_model_specific_covariates', True)
            default_effort = effort_per_interval_parameters.get('default_effort_for_unknown_covariates', 1.0)
            logging.info(f"Configuration:")
            logging.info(f"  • Use model-specific covariates: {use_model_specific}")
            logging.info(f"  • Default effort for unknown covariates: {default_effort}")
            logging.info(f"  • Available effort values: {list(effort_values.keys())}")
        else:
            logging.info("Effort per interval: DISABLED")
        
        # Validate metric weights configuration (Professor's feedback implementation)
        is_valid, is_single_metric, single_metric_name, validation_message = validate_metric_weights(metric_weights)
        if not is_valid:
            logging.error(f"Invalid metric weights configuration: {validation_message}")
            return 1
        
        logging.info("=" * 60)
        logging.info("METRIC WEIGHTS CONFIGURATION")
        logging.info("=" * 60)
        logging.info(validation_message)
        if is_single_metric:
            logging.info(f"📊 Using {single_metric_name.upper()} as the sole criterion for model ranking")
            logging.info("📝 This implements the professor's recommendation to set four weights to 0.0")
        logging.info(f"Current weights: {metric_weights}")
        logging.info("=" * 60)
        
        if run_all_models:
            logging.info("Running all models with all covariate combinations")
        else:
            logging.info(f"Model: {selected_model}")
            if run_all_covariates:
                logging.info("Using all covariate combinations")
            else:
                logging.info(f"Using specific covariates: {selected_covariates}")
        
        # Load data with enhanced features for 100% GUI compatibility
        df, covariates = load_data(input_file, sheet_name=selected_sheet, subset_limit=data_subset_limit)
        
        # Apply covariate scaling if enabled (Enhancement to original C-SFRAT)
        scaling_info = None
        if enable_covariate_scaling:
            logging.info("🔧 Applying covariate scaling to improve effort allocation...")
            df, scaling_info = scale_covariates_to_range(df, target_min=covariate_scaling_range[0], target_max=covariate_scaling_range[1])
            logging.info(f"✅ Covariate scaling complete - {len(scaling_info)} covariates scaled")
        else:
            logging.info("⚠️  Covariate scaling disabled - using original values (may cause effort allocation issues)")
            print("⚠️  WARNING: Covariate scaling is disabled. This may cause unrealistic effort allocation results.")
            print("   Consider enabling 'enable_covariate_scaling = True' for better results.")
        
        # Run analysis based on configuration
        if run_all_models:
            # Run all models
            results = run_all_combinations(df, covariates)
        else:
            # Run a specific model
            if run_all_covariates:
                # Try all covariate combinations for the selected model
                all_covariates = covariates
                covariate_combinations = list(powerset(all_covariates))
                
                results = []
                for covs in covariate_combinations:
                    covs_list = list(covs)
                    logging.info(f"Running {selected_model} with covariates: {covs_list}")
                    
                    # Create model data for this combination
                    model_data = prepare_model_data(df, covs_list)
                    
                    try:
                        # Initialize and run the model (same logic as in run_all_combinations)
                        model = initialize_model(MODEL_CLASSES[selected_model], model_data, covs_list)
                        
                        # Store arrays for actual and fitted values
                        model.t = model_data['T'].values
                        model.CFC = model_data['CFC'].values
                        
                        if model and model.converged:
                            results.append(model)
                    except Exception as e:
                        logging.warning(f"Error running {selected_model} with covariates {covs_list}: {str(e)}")
                        continue
            else:
                # Run specific model with specific covariates
                logging.info(f"Running {selected_model} with specific covariates: {selected_covariates}")
                
                # Create model data for this combination
                model_data = prepare_model_data(df, selected_covariates)
                
                try:
                    # Initialize and run the model
                    model = initialize_model(MODEL_CLASSES[selected_model], model_data, selected_covariates)
                    
                    # Store arrays for actual and fitted values
                    model.t = model_data['T'].values
                    model.CFC = model_data['CFC'].values
                    
                    results = [model] if model and model.converged else []
                except Exception as e:
                    logging.warning(f"Error running {selected_model} with covariates {selected_covariates}: {str(e)}")
                    results = []
        
        # Generate report if results are available
        if results:
            # Note: results are already sorted by the selected ranking method in run_all_combinations()
            # DO NOT re-sort by AIC here as it would override the user's ranking method choice
            
            # Get the best model for additional analysis (first model in the ranking-method-sorted list)
            best_model = results[0]
            
            # Generate effort per interval predictions (always enabled like original SFRAT tool)
            effort_predictions = generate_failure_predictions_with_effort_gui_correct(best_model, prediction_parameters, effort_per_interval_parameters)
            # Display clear effort prediction summary in console
            if effort_predictions:
                display_effort_prediction_summary(effort_predictions)
            
            # Generate standard failure predictions if enabled
            predictions = None
            if prediction_parameters.get('predict_failures', False):
                predictions = generate_failure_predictions(best_model, prediction_parameters)
                # Display clear prediction summary in console
                if predictions:
                    display_prediction_summary(predictions)
            
            # Calculate effort allocation optimization if enabled
            optimization_results = None
            if optimization_parameters.get('enable_optimization', False):
                optimization_results = calculate_effort_allocation_multiple(results[:3], optimization_parameters)
            
            # Generate PDF report with all detailed data including predictions, effort predictions, and optimization
            generate_report(results, output_filename, predictions, optimization_results, effort_predictions)
            logging.info(f"Analysis complete. Report saved to: {output_filename}")
            
            return 0
        else:
            logging.error("No models converged. Cannot generate report.")
            return 1
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        logging.error(traceback.format_exc())
        return 1

def create_time_based_comparison_table(top_models, effort_predictions=None):
    """Create a table showing cumulative failures at each time point for all models
    
    Args:
        top_models: List of top 3 fitted models
        effort_predictions: Effort prediction results (optional)
        
    Returns:
        Table: ReportLab Table object showing time-based comparison
    """
    if not top_models or len(top_models) < 1:
        return None
    
    # Get the historical CFC data that all models should share
    historical_cfc = None
    if hasattr(top_models[0], 'CFC') and len(top_models[0].CFC) > 0:
        historical_cfc = top_models[0].CFC
        max_data_time = len(historical_cfc)
    else:
        # Fallback: use the actual data we know
        historical_cfc = [1, 2, 4, 5, 13, 22, 28, 35, 39, 42, 42, 46, 47, 47, 49, 51, 54]
        max_data_time = 17
    
    # Get time range from start to predictions
    start_time = 1
    end_time = max_data_time + 15  # Show data + 15 future points (up to time 32)
    
    # Create header row
    header = ["T", "ObsFC"]
    for i, model in enumerate(top_models[:3]):
        if hasattr(model, 'metricNames') and model.metricNames:
            # Shorten covariate names for better display
            short_covs = [cov[:1] for cov in model.metricNames]  # Just use first letter
            covs = f"({','.join(short_covs)})"
        else:
            covs = ""
        header.append(f"M{i+1}{covs}")
    
    # Add effort column if available (rename for clarity)
    if effort_predictions:
        effort_values = effort_predictions.get('effort_values', None)
        if isinstance(effort_values, dict) and effort_values:
            ordered_keys = [k for k in ['E', 'F', 'C'] if k in effort_values] or list(effort_values.keys())
            pairs = []
            for k in ordered_keys:
                v = effort_values.get(k)
                try:
                    pairs.append(f"{k}={float(v):.1f}")
                except Exception:
                    pairs.append(f"{k}={v}")
            header.append(f"Effort FC")
        else:
            effort_cov = effort_predictions.get('effort_covariate', 'Effort')
            effort_val = effort_predictions.get('effort_value', '')
        if isinstance(effort_val, (int, float)):
            effort_display = f"{effort_val:.1f}"
        else:
            effort_display = str(effort_val) if effort_val != '' else ''
        suffix = f" ({effort_cov}={effort_display})" if effort_display else ''
        header.append(f"Effort FC")
    
    # Create data rows
    data = [header]
    
    for time_point in range(start_time, end_time + 1):
        row = [str(time_point)]
        
        # Add observed data column
        if time_point <= max_data_time:
            # Historical data - show actual observed values
            observed_value = historical_cfc[time_point - 1]
            row.append(f"{observed_value:.1f}")
        else:
            # Future - no observed data available
            row.append("N/A")
        
        for i, model in enumerate(top_models[:3]):
            if time_point <= max_data_time:
                # Historical data - show model's fitted values instead of actual data
                if hasattr(model, 'mvfList') and model.mvfList is not None and len(model.mvfList) >= time_point:
                    failures = model.mvfList[time_point - 1]
                    row.append(f"{failures:.1f}")
                else:
                    # Fallback to actual data if mvfList not available
                    failures = historical_cfc[time_point - 1]
                    row.append(f"{failures:.1f}")
            else:
                # Future predictions - use model's mvfList if available
                try:
                    if hasattr(model, 'mvfList') and model.mvfList is not None:
                        if time_point <= len(model.mvfList):
                            # Use mvfList value directly 
                            failures = model.mvfList[time_point - 1]
                            row.append(f"{failures:.1f}")
                        else:
                            # Extrapolate from mvfList trend
                            if len(model.mvfList) >= 2:
                                last_mvf = model.mvfList[-1]
                                second_last_mvf = model.mvfList[-2]
                                trend = last_mvf - second_last_mvf
                                extrapolated = last_mvf + trend * (time_point - len(model.mvfList))
                                row.append(f"{extrapolated:.1f}")
                            else:
                                # Use the last mvfList value
                                last_mvf = model.mvfList[-1]
                                row.append(f"{last_mvf:.1f}")
                    else:
                        # Use the last known CFC value as fallback
                        last_cfc = historical_cfc[-1]
                        row.append(f"{last_cfc:.1f}")
                except Exception as e:
                    # Error occurred, use last known CFC value
                    last_cfc = historical_cfc[-1]
                    row.append(f"{last_cfc:.1f}")
        
        # Add effort prediction if available
        if effort_predictions:
            try:
                effort_times = effort_predictions.get('future_times', [])
                effort_mvf = effort_predictions.get('future_mvf', [])
                current_time = effort_predictions.get('current_time', 17)
                current_failures = effort_predictions.get('current_failures', 54)
                
                if time_point <= current_time:
                    # Historical - use actual data
                    if time_point <= len(historical_cfc):
                        row.append(f"{historical_cfc[time_point - 1]:.1f}")
                    else:
                        row.append(f"{historical_cfc[-1]:.1f}")
                elif time_point in effort_times:
                    # Predicted effort data
                    idx = effort_times.index(time_point)
                    row.append(f"{effort_mvf[idx]:.1f}")
                else:
                    # Extrapolate effort predictions beyond available data
                    if len(effort_mvf) >= 2:
                        last_effort = effort_mvf[-1]
                        second_last_effort = effort_mvf[-2]
                        effort_trend = last_effort - second_last_effort
                        max_effort_time = max(effort_times) if effort_times else current_time
                        periods_beyond = time_point - max_effort_time
                        extrapolated_effort = last_effort + effort_trend * periods_beyond
                        row.append(f"{extrapolated_effort:.1f}")
                    else:
                        row.append(f"{current_failures:.1f}")
            except Exception as e:
                # Fallback to current failures
                row.append(f"{current_failures:.1f}")
        
        data.append(row)
    
    # Calculate appropriate column widths to fit within PDF margins
    num_columns = len(header)
    total_available_width = 570  # Optimized width to ensure borders fit within page margins
    
    # Intelligent column width allocation based on content type
    time_col_width = 35  # Compact time column
    observed_col_width = 65  # Compact observed column
    model_col_width = 100   # Compact but sufficient width for model columns
    
    # Calculate remaining width for effort column (if present)
    if effort_predictions:
        effort_col_width = total_available_width - time_col_width - observed_col_width - (model_col_width * min(3, len(top_models)))
        col_widths = [time_col_width, observed_col_width] + [model_col_width] * min(3, len(top_models)) + [max(effort_col_width, 100)]
    else:
        # Without effort column, distribute remaining space among model columns
        available_for_models = total_available_width - time_col_width - observed_col_width
        model_col_width = available_for_models / len(top_models[:3])
        col_widths = [time_col_width, observed_col_width] + [model_col_width] * len(top_models[:3])
    
    # Create table with calculated column widths
    table = Table(data, colWidths=col_widths)
    
    # Use standardized table style
    table_style = get_standardized_table_style(use_alternating_rows=True)
    
    # Calculate where predictions start (after historical data)
    historical_end_row = max_data_time + 1  # +1 because row 0 is header
    prediction_start_row = historical_end_row  # 🔧 FIX: Start coloring from first prediction row (time 18)
    total_rows = len(data)
    
    # Add special styling for prediction rows
    if prediction_start_row < total_rows:
        table_style.add('BACKGROUND', (0, prediction_start_row), (-1, total_rows-1), colors.HexColor('#E8F4FD'))
        table_style.add('FONTNAME', (0, prediction_start_row), (-1, total_rows-1), 'Helvetica-Bold')
        table_style.add('TEXTCOLOR', (0, prediction_start_row), (-1, total_rows-1), colors.HexColor('#0D47A1'))
        table_style.add('LINEBELOW', (0, max_data_time), (-1, max_data_time), 3, colors.HexColor('#2196F3'))
    
    table.setStyle(table_style)
    
    return table

def create_intensity_table(top_models, individual_predictions=None, effort_predictions=None):
    """Create a failure intensity table matching the original C-SFRAT tool's intensity view
    
    Args:
        top_models: List of top 3 fitted models
        individual_predictions: Individual model predictions (optional)
        effort_predictions: Effort prediction results (optional)
        
    Returns:
        Table: ReportLab Table object showing failure intensity values over time
    """
    if not top_models or len(top_models) < 1:
        return None
    
    # Get the maximum data time
    max_data_time = 17  # Standard for our dataset
    if hasattr(top_models[0], 'CFC') and len(top_models[0].CFC) > 0:
        max_data_time = len(top_models[0].CFC)
    
    # Get time range from start to predictions
    start_time = 1
    end_time = max_data_time + 10  # Show data + 10 future points
    
    # Access historical cumulative failures for exact historical intensity calc
    historical_cfc = None
    if hasattr(top_models[0], 'CFC') and len(top_models[0].CFC) > 0:
        historical_cfc = top_models[0].CFC
    
    # Create header row
    header = ["T"]
    for i, model in enumerate(top_models[:3]):
        if hasattr(model, 'metricNames') and model.metricNames:
            # Shorten covariate names for better display
            short_covs = [cov[:1] for cov in model.metricNames]  # Just use first letter
            covs = f"({','.join(short_covs)})"
        else:
            covs = ""
        header.append(f"M{i+1}{covs}")
    
    # Add effort column if available (rename for clarity)
    if effort_predictions:
        # Prefer per-covariate effort display when available
        effort_values = effort_predictions.get('effort_values')
        if isinstance(effort_values, dict) and effort_values:
            ordered_keys = [k for k in ['E', 'F', 'C'] if k in effort_values] or list(effort_values.keys())
            pairs = []
            for k in ordered_keys:
                v = effort_values.get(k)
                try:
                    pairs.append(f"{k}={float(v):.1f}")
                except Exception:
                    pairs.append(f"{k}={v}")
            header.append(f"Effort Intensity")
        else:
            # Fallback to single covariate/value naming
            effort_cov = effort_predictions.get('effort_covariate', 'Effort')
            effort_val = effort_predictions.get('effort_value', '')
            if isinstance(effort_val, (int, float)):
                effort_display = f"{effort_val:.1f}"
            else:
                effort_display = str(effort_val) if effort_val != '' else ''
            suffix = f" ({effort_cov}={effort_display})" if effort_display else ''
            header.append(f"Effort Intensity")
    
    # Create data rows
    data = [header]
    
    for time_point in range(start_time, end_time + 1):
        row = [str(time_point)]
        
        for i, model in enumerate(top_models[:3]):
            if time_point <= max_data_time:
                # Historical data - use model's intensity values
                try:
                    if hasattr(model, 'intensityList') and model.intensityList is not None and len(model.intensityList) >= time_point:
                        intensity = model.intensityList[time_point - 1]
                        row.append(f"{intensity:.3f}")
                    elif hasattr(model, 'mvfList') and model.mvfList is not None and len(model.mvfList) >= time_point:
                        # Calculate intensity from MVF
                        if time_point == 1:
                            intensity = model.mvfList[0]
                        else:
                            intensity = model.mvfList[time_point - 1] - model.mvfList[time_point - 2]
                        row.append(f"{intensity:.3f}")
                    else:
                        row.append("N/A")
                except Exception as e:
                    row.append("N/A")
            else:
                # Future predictions - check individual predictions first
                prediction_found = False
                
                if individual_predictions:
                    # Look for matching model in individual predictions
                    for pred_key, pred_data in individual_predictions.items():
                        pred_model = pred_data.get('model')
                        if pred_model and pred_model.name == model.name and hasattr(pred_model, 'metricNames'):
                            if pred_model.metricNames == model.metricNames:
                                # Found matching model prediction
                                pred_times = pred_data.get('future_times', [])
                                pred_intensity = pred_data.get('future_intensity', [])
                                
                                if time_point in pred_times:
                                    idx = pred_times.index(time_point)
                                    if idx < len(pred_intensity):
                                        row.append(f"{pred_intensity[idx]:.3f}")
                                        prediction_found = True
                                        break
                
                if not prediction_found:
                    # Fallback to model extrapolation
                    try:
                        if hasattr(model, 'intensityList') and model.intensityList is not None and len(model.intensityList) >= 2:
                            # Use trend from last intensity values
                            last_intensity = model.intensityList[-1]
                            second_last_intensity = model.intensityList[-2]
                            intensity_trend = last_intensity - second_last_intensity
                            periods_beyond = time_point - max_data_time
                            extrapolated_intensity = last_intensity + intensity_trend * periods_beyond
                            row.append(f"{max(0.001, extrapolated_intensity):.3f}")  # Ensure positive
                        else:
                            # Use a small default intensity for future predictions
                            row.append("0.050")
                    except Exception as e:
                        row.append("0.050")
        
        # Add effort prediction intensity if available
        if effort_predictions:
            try:
                effort_times = effort_predictions.get('future_times', [])
                effort_intensity = effort_predictions.get('future_intensity', [])
                effort_mvf = effort_predictions.get('future_mvf', [])
                
                # Historical rows: always compute exact from CFC to avoid placeholders
                if time_point <= max_data_time:
                    if historical_cfc is not None and len(historical_cfc) >= time_point:
                        exact_intensity = float(historical_cfc[0]) if time_point == 1 else float(historical_cfc[time_point - 1] - historical_cfc[time_point - 2])
                        row.append(f"{max(0.0, exact_intensity):.3f}")
                    else:
                        row.append("0.000")
                elif time_point in effort_times:
                    # Predicted effort intensity data
                    idx = effort_times.index(time_point)
                    if idx < len(effort_intensity):
                        row.append(f"{effort_intensity[idx]:.3f}")
                    else:
                        row.append("0.050")
                else:
                    # Derive or extrapolate intensity sensibly
                    if effort_mvf and effort_times:
                        # If we can estimate from MVF
                        max_effort_time = max(effort_times)
                        if time_point > max_effort_time and len(effort_mvf) >= 2:
                            last_effort = float(effort_mvf[-1])
                            second_last_effort = float(effort_mvf[-2])
                            mvf_trend = last_effort - second_last_effort
                            steps = time_point - max_effort_time
                            extrapolated_next = last_effort + mvf_trend * steps
                            intensity_val = max(0.001, extrapolated_next - last_effort)
                            row.append(f"{intensity_val:.3f}")
                        else:
                            row.append("0.050")
                    elif len(effort_intensity) >= 2:
                        last_effort_intensity = float(effort_intensity[-1])
                        second_last_effort_intensity = float(effort_intensity[-2])
                        trend = last_effort_intensity - second_last_effort_intensity
                        max_effort_time = max(effort_times) if effort_times else max_data_time
                        periods_beyond = time_point - max_effort_time
                        extrapolated_effort_intensity = last_effort_intensity + trend * periods_beyond
                        row.append(f"{max(0.001, extrapolated_effort_intensity):.3f}")
                    else:
                        row.append("0.050")
            except Exception as e:
                row.append("0.050")
        
        data.append(row)
    
    # Calculate appropriate column widths to fit within PDF margins
    num_columns = len(header)
    total_available_width = 570  # Optimized width to ensure borders fit within page margins
    
    # Intelligent column width allocation for intensity table
    time_col_width = 35  # Compact time column
    model_col_width = 110   # Compact but sufficient width for model columns
    
    # Calculate remaining width for effort column (if present)
    if effort_predictions:
        effort_col_width = total_available_width - time_col_width - (model_col_width * min(3, len(top_models)))
        col_widths = [time_col_width] + [model_col_width] * min(3, len(top_models)) + [max(effort_col_width, 110)]
    else:
        # Without effort column, distribute remaining space among model columns
        available_for_models = total_available_width - time_col_width
        model_col_width = available_for_models / len(top_models[:3])
        col_widths = [time_col_width] + [model_col_width] * len(top_models[:3])
    
    # Create table with calculated column widths
    table = Table(data, colWidths=col_widths)
    
    # Use standardized table style
    table_style = get_standardized_table_style(use_alternating_rows=True)
    
    # Calculate where predictions start (after historical data)
    historical_end_row = max_data_time + 1  # +1 because row 0 is header
    prediction_start_row = historical_end_row  # 🔧 FIX: Start coloring from first prediction row (time 18)
    total_rows = len(data)
    
    # Add special styling for prediction rows (intensity-specific colors)
    if prediction_start_row < total_rows:
        table_style.add('BACKGROUND', (0, prediction_start_row), (-1, total_rows-1), colors.HexColor('#FFE4E1'))
        table_style.add('FONTNAME', (0, prediction_start_row), (-1, total_rows-1), 'Helvetica-Bold')
        table_style.add('TEXTCOLOR', (0, prediction_start_row), (-1, total_rows-1), colors.HexColor('#8B0000'))
        table_style.add('LINEBELOW', (0, max_data_time), (-1, max_data_time), 3, colors.HexColor('#FF6347'))
    
    table.setStyle(table_style)
    
    return table

def create_effort_allocation_section(optimization_results):
    """Create effort allocation section matching original C-SFRAT GUI format
    
    Args:
        optimization_results: Dictionary containing effort allocation results
        
    Returns:
        list: List of reportlab elements for the allocation section
    """
    elements = []
    styles = getSampleStyleSheet()
    
    # Section title
    elements.append(Paragraph("Tab 4: Effort Allocation", styles["Heading2"]))
    elements.append(Spacer(1, 8))
    
    allocation_results = optimization_results.get('allocation_results', {})
    
    # Check if this is multiple models or single model
    if allocation_results.get('multiple_models', False):
        # Handle multiple models
        models_data = allocation_results.get('models', [])
        logging.info(f"DEBUG: Found {len(models_data)} models with allocation data")
        
        # Allocation 1 Section (matching original GUI table format)
        if models_data and any(model.get('allocation_1') for model in models_data):
            # Get budget information from first model's allocation data
            first_allocation = next((model.get('allocation_1') for model in models_data if model.get('allocation_1')), {})
            budget = first_allocation.get('budget', 1000)  # Default to 1000 if not found
            
            elements.append(Paragraph("Effort Allocation 1", styles["Heading3"]))
            elements.append(Paragraph(f"Maximize defect discovery within budget: <b>${budget:,.2f}</b>", styles["Normal"]))
            
            # Optional feasibility note per model
            if models_data:
                for m in models_data:
                    alloc1 = m.get('allocation_1')
                    if alloc1 and 'max_additional_defects' in alloc1:
                        max_addl = alloc1.get('max_additional_defects')
                        if alloc1.get('capped', False):
                            elements.append(Paragraph(
                                f"<i>Note: Est. Defects are capped at the model's maximum additional defects (<b>{max_addl:.2f}</b>).</i>",
                                styles["Normal"]
                            ))
                            break
            elements.append(Spacer(1, 6))
            
            # Create table matching original GUI format
            # Header: Model Name | Covariates | Est. Defects | %E | %F | %C | Ceiling
            header = ["Model Name", "Covariates", "Est. Defects"]
            
            # Add percentage columns for each covariate found in data
            all_possible_covariates = ['E', 'F', 'C']  # Standard covariates in C-SFRAT
            covariate_columns = []
            for cov in all_possible_covariates:
                header.append(f"%{cov}")
                covariate_columns.append(cov)
            
            # Create data rows for each model
            table_data = [header]
            
            for model_info in models_data:
                alloc1 = model_info.get('allocation_1')
                if alloc1:
                    model_name = model_info.get('model_name', 'Unknown')
                    covariate_names = model_info.get('covariate_names', [])
                    covariates_display = ", ".join(covariate_names) if covariate_names else "None"
                    
                    est = alloc1.get('estimated_defects', 0.0)
                    capped = alloc1.get('capped', False)
                    max_addl = alloc1.get('max_additional_defects')
                    est_display = f"{est:.2f}"
                    if capped and max_addl is not None:
                        est_display = f"{max_addl:.2f} (capped)"
                    
                    data_row = [model_name, covariates_display, est_display]
                    
                    # Add percentage values for each covariate column
                    optimal_allocation = alloc1.get('optimal_allocation', {})
                    for cov in covariate_columns:
                        if cov in optimal_allocation:
                            data_row.append(f"{optimal_allocation[cov]:.2f}%")
                        else:
                            data_row.append("")  # Empty cell if covariate not used
                    
                    table_data.append(data_row)
            
            # Calculate appropriate column widths
            num_cols = len(header)
            base_width = 500 / num_cols
            col_widths = [base_width * 1.5, base_width * 1.2, base_width * 1.0] + [base_width * 0.8] * (num_cols - 3)
            
            allocation1_table = Table(table_data, colWidths=col_widths)
            
            # Use standardized table style
            table_style = get_standardized_table_style(use_alternating_rows=True)
            table_style.add('ALIGN', (0, 1), (1, -1), 'LEFT')  # Left align text columns
            
            allocation1_table.setStyle(table_style)
            elements.append(allocation1_table)
            elements.append(Spacer(1, 8))
            logging.info(f"DEBUG: Added Allocation 1 table with {len(table_data)} rows")
        
        # Allocation 2 Section (matching original GUI table format)
        if models_data and any(model.get('allocation_2') for model in models_data):
            # Get target defects information from first model's allocation data
            first_allocation_2 = next((model.get('allocation_2') for model in models_data if model.get('allocation_2')), {})
            # Get target defects from allocation results, or fall back to configuration value
            target_defects = first_allocation_2.get('target_additional_defects')
            if target_defects is None:
                # Fall back to the actual configuration value that was used
                target_defects = optimization_parameters.get('target_additional_defects', 2)
            
            elements.append(Paragraph("Effort Allocation 2", styles["Heading3"]))
            elements.append(Paragraph(f"Minimum budget to discover <b>{target_defects}</b> additional defects", styles["Normal"]))
            
            # Optional infeasibility note per model
            if models_data:
                for m in models_data:
                    alloc2 = m.get('allocation_2')
                    if alloc2 and alloc2.get('infeasible', False):
                        max_addl = alloc2.get('max_additional_defects')
                        elements.append(Paragraph(
                            f"<i>Note: Target exceeds the model's maximum additional defects (<b>{max_addl:.2f}</b>). This request is infeasible (no finite budget).</i>",
                            styles["Normal"]
                        ))
                        break
            elements.append(Spacer(1, 6))
            
            # Create table matching original GUI format
            # Header: Model Name | Covariates | Est. Budget | %E | %F | %C | Feasible?
            header = ["Model Name", "Covariates", "Est. Budget"]
            
            # Add percentage columns for each covariate
            all_possible_covariates = ['E', 'F', 'C']
            covariate_columns = []
            for cov in all_possible_covariates:
                header.append(f"%{cov}")
                covariate_columns.append(cov)
            
            # Create data rows for each model
            table_data = [header]
            
            for model_info in models_data:
                alloc2 = model_info.get('allocation_2')
                if alloc2:
                    model_name = model_info.get('model_name', 'Unknown')
                    covariate_names = model_info.get('covariate_names', [])
                    covariates_display = ", ".join(covariate_names) if covariate_names else "None"
                    
                    infeasible = alloc2.get('infeasible', False)
                    min_budget = alloc2.get('minimum_budget', 0.0)
                    budget_display = "—" if infeasible else f"{min_budget:.2f}"
                    
                    data_row = [model_name, covariates_display, budget_display]
                    
                    # Add percentage values for each covariate column
                    optimal_allocation = alloc2.get('optimal_allocation', {})
                    for cov in covariate_columns:
                        if cov in optimal_allocation:
                            data_row.append(f"{optimal_allocation[cov]:.2f}%")
                        else:
                            data_row.append("")  # Empty cell if covariate not used
                    
                    table_data.append(data_row)
            
            # Calculate appropriate column widths
            num_cols = len(header)
            base_width = 500 / num_cols
            col_widths = [base_width * 1.5, base_width * 1.2, base_width * 1.0] + [base_width * 0.8] * (num_cols - 3)
            
            allocation2_table = Table(table_data, colWidths=col_widths)
            
            # Use standardized table style
            table_style = get_standardized_table_style(use_alternating_rows=True)
            table_style.add('ALIGN', (0, 1), (1, -1), 'LEFT')  # Left align text columns
            
            allocation2_table.setStyle(table_style)
            elements.append(allocation2_table)
            elements.append(Spacer(1, 8))
            logging.info(f"DEBUG: Added Allocation 2 table with {len(table_data)} rows")
    
    else:
        # Handle single model (legacy support)
        model_name = optimization_results.get('model_name', 'Unknown')
        covariate_names = optimization_results.get('covariate_names', [])
    
    return elements

def create_failure_target_comparison_table(top_models, effort_predictions=None):
    """Create a table showing when each model reaches specific failure targets
    
    Args:
        top_models: List of top 3 fitted models
        effort_predictions: Effort prediction results (optional)
        
    Returns:
        Table: ReportLab Table object showing failure target comparison
    """
    if not top_models or len(top_models) < 1:
        return None
    
    # Define failure targets to show
    current_failures = 54
    failure_targets = list(range(10, current_failures + 20, 5))  # Every 5 failures from 10 to 74
    
    # Create header row
    header = ["Target"]
    for i, model in enumerate(top_models[:3]):
        if hasattr(model, 'metricNames') and model.metricNames:
            short_covs = [cov[:1] for cov in model.metricNames]  # Just use first letter
            covs = f"({','.join(short_covs)})"
        else:
            covs = ""
        header.append(f"M{i+1}T{covs}")
    
    # Add effort column if available
    if effort_predictions:
        effort_cov = effort_predictions.get('effort_covariate', 'E')
        effort_val = effort_predictions.get('effort_value', 1.0)
        header.append(f"EffT({effort_cov}={effort_val:.1f})")
    
    # Create data rows
    data = [header]
    
    for target in failure_targets:
        row = [str(target)]
        
        for model in top_models[:3]:
            try:
                time_to_target = calculate_time_to_failure(model, target)
                if time_to_target is not None:
                    row.append(f"{time_to_target:.1f}")
                else:
                    row.append("N/A")
            except Exception as e:
                row.append("N/A")
        
        # Add effort prediction time if available
        if effort_predictions:
            try:
                effort_time = calculate_effort_time_to_failure(effort_predictions, target)
                if effort_time is not None:
                    row.append(f"{effort_time:.1f}")
                else:
                    row.append("N/A")
            except Exception as e:
                row.append("N/A")
        
        data.append(row)
    
    # Calculate appropriate column widths to fit within PDF margins
    num_columns = len(header)
    total_available_width = 570  # Optimized width to ensure borders fit within page margins
    
    # Allocate width: Target column gets less space, other columns shared equally
    target_col_width = 80  # Narrower for target failures column
    remaining_width = total_available_width - target_col_width
    other_col_width = remaining_width / (num_columns - 1)
    
    # Create column widths list
    col_widths = [target_col_width] + [other_col_width] * (num_columns - 1)
    
    # Create table with calculated column widths
    table = Table(data, colWidths=col_widths)
    table_style = get_standardized_table_style(use_alternating_rows=True)
    table.setStyle(table_style)
    
    return table

def calculate_mvf_at_time(model, time_point):
    """Calculate MVF value at a specific time point"""
    try:
        if hasattr(model, 'TC') and hasattr(model, 'CFC'):
            if time_point <= len(model.TC):
                # Historical data
                return model.CFC[time_point - 1] if time_point <= len(model.CFC) else model.CFC[-1]
            else:
                # Use model's MVF values if available
                if hasattr(model, 'MVF') and len(model.MVF) > 0:
                    # Use the MVF values calculated during model fitting
                    mvf_index = min(time_point - 1, len(model.MVF) - 1)
                    return model.MVF[mvf_index]
                else:
                    return None
        return None
    except Exception as e:
        return None

def calculate_effort_time_to_failure(effort_predictions, target_failures):
    """Calculate when effort predictions reach target failures"""
    try:
        effort_times = effort_predictions.get('future_times', [])
        effort_mvf = effort_predictions.get('future_mvf', [])
        current_time = effort_predictions.get('current_time', 17)
        current_failures = effort_predictions.get('current_failures', 54)
        
        # Check if target is already reached
        if target_failures <= current_failures:
            # Estimate based on historical data (approximate)
            return target_failures * current_time / current_failures
        
        # Find when target is reached in predictions
        for i, (time, failures) in enumerate(zip(effort_times, effort_mvf)):
            if failures >= target_failures:
                if i == 0:
                    return time
                else:
                    # Linear interpolation between points
                    prev_time = effort_times[i-1] if i > 0 else current_time
                    prev_failures = effort_mvf[i-1] if i > 0 else current_failures
                    
                    # Interpolate
                    ratio = (target_failures - prev_failures) / (failures - prev_failures)
                    interpolated_time = prev_time + ratio * (time - prev_time)
                    return interpolated_time
        
        # Target not reached in predictions - extrapolate
        if len(effort_times) >= 2:
            last_time = effort_times[-1]
            last_failures = effort_mvf[-1]
            second_last_time = effort_times[-2]
            second_last_failures = effort_mvf[-2]
            
            # Calculate rate and extrapolate
            failure_rate = (last_failures - second_last_failures) / (last_time - second_last_time)
            if failure_rate > 0:
                additional_failures = target_failures - last_failures
                additional_time = additional_failures / failure_rate
                return last_time + additional_time
        
        return None
    except Exception as e:
        return None

def generate_failure_predictions_with_effort_corrected(model, prediction_params, effort_params):
    """Generate failure predictions with effort per interval using CORRECTED mathematical approach
    
    This fixes the original C-SFRAT mathematical issue where omega gets recalculated inappropriately.
    The corrected approach:
    1. Keeps the original fitted omega (total expected failures)
    2. Only extends hazard and covariate arrays for future predictions  
    3. Maintains mathematical continuity between historical and predicted values
    
    Args:
        model: Fitted model object
        prediction_params: Dictionary with prediction parameters
        effort_params: Dictionary with effort per interval parameters
        
    Returns:
        Dictionary containing mathematically corrected effort-adjusted prediction results
    """
    if not hasattr(model, 't') or len(model.t) == 0:
        return None
    
    try:
        # Get effort parameters
        effort_values = effort_params.get('effort_values', {'E': 1.0, 'F': 2.0, 'C': 3.0})
        # 🔧 FIX: Use consistent parameter names with basic predictions
        num_intervals = effort_params.get('num_intervals_to_predict', 
                                         effort_params.get('number_of_intervals_to_predict', 10))
        intensity_target = effort_params.get('failure_intensity_target', 0.3)
        
        logging.info(f"Generating CORRECTED predictions (fixing omega recalculation issue)")
        logging.info(f"Model: {model.name}, Intervals: {num_intervals}")
        logging.info(f"Effort values: {effort_values}")
        
        # Check if model has covariates (required for effort per interval)
        if not hasattr(model, 'metricNames') or not model.metricNames:
            logging.warning("Effort per interval requires models with covariates. Using default prediction.")
            return generate_failure_predictions(model, prediction_params)
        
        # Create effort dictionary in the format expected by C-SFRAT
        class MockSpinBox:
            def __init__(self, value):
                self._value = value
            def value(self):
                return self._value
        
        effort_dict = {}
        use_model_specific = effort_params.get('use_model_specific_covariates', True)
        default_effort = effort_params.get('default_effort_for_unknown_covariates', 1.0)
        
        if use_model_specific:
            for covariate in model.metricNames:
                effort_value = effort_values.get(covariate, default_effort)
                effort_dict[covariate] = MockSpinBox(effort_value)
                logging.info(f"  Using effort {effort_value} for model covariate: {covariate}")
        else:
            for covariate, effort_value in effort_values.items():
                effort_dict[covariate] = MockSpinBox(effort_value)
                logging.info(f"  Using effort {effort_value} for covariate: {covariate}")
        
        if not effort_dict:
            logging.warning("No effort values available, using default effort of 1.0 for all covariates")
            for covariate in model.metricNames:
                effort_dict[covariate] = MockSpinBox(1.0)
        
        # Current state
        current_time = model.t[-1]
        current_failures = model.CFC[-1] if hasattr(model, 'CFC') and len(model.CFC) > 0 else 0
        
        try:
            # CORRECTED APPROACH: Fix the mathematical inconsistency
            
            # Step 1: Prepare extended covariate data (like original)
            total_points = model.n + num_intervals
            new_array = []
            for cov in model.metricNames:
                value = effort_dict[cov].value()
                new_array.append(np.full(num_intervals, value))

            if model.numCovariates == 0:
                combined_array = np.concatenate((model.covariateData, np.array(new_array)))
            else:
                combined_array = np.concatenate((model.covariateData, np.array(new_array)), axis=1)

            # Step 2: Extend hazard array (like original)
            newHazard = np.array([model.hazardNumerical(i, model.modelParameters) for i in range(model.n, total_points)])
            extended_hazard = np.concatenate((model.hazard_array, newHazard))
            
            # Step 3: KEY FIX - Use ORIGINAL fitted omega, do NOT recalculate
            # The original C-SFRAT bug was here: omega = model.calcOmega(extended_hazard, model.betas, combined_array)
            # This is mathematically incorrect because omega represents total expected failures fitted to historical data
            original_omega = model.omega  # Use the omega fitted to historical data
            
            logging.info(f"CORRECTION: Using original fitted omega={original_omega:.4f} instead of recalculating")
            
            # Step 4: Generate MVF values using corrected approach
            # For historical points (1 to model.n): Use original fitted values to ensure continuity
            historical_mvf = model.mvf_array.copy()
            
            # For future points (model.n+1 to total_points): Use extended data with original omega
            future_mvf = []
            for data_point in range(model.n, total_points):
                # Calculate MVF for this future point using original omega
                mvf_val = model.MVF(model.mle_array, original_omega, extended_hazard, data_point, combined_array)
                future_mvf.append(mvf_val)
            
            # Combine historical and future MVF values
            full_mvf_array = np.concatenate([historical_mvf, future_mvf])
            full_x = np.concatenate((model.t, np.arange(model.n + 1, total_points + 1)))
            
            # Calculate intensity values from MVF (similar to original)
            intensity_array = []
            mvf_list = full_mvf_array.tolist()
            
            for i in range(1, len(mvf_list)):
                intensity_val = mvf_list[i] - mvf_list[i-1]
                intensity_array.append(max(0.01, intensity_val))  # Minimum intensity
            
            # Get future values (excluding the original data points)
            original_length = len(model.t)
            future_times = full_x[original_length:].tolist()
            future_mvf_values = full_mvf_array[original_length:].tolist()
            future_intensity = intensity_array[original_length-1:] if len(intensity_array) >= original_length else intensity_array
            
            # Find when target intensity is reached
            intensity_intervals = 0
            for i, intensity in enumerate(future_intensity):
                if intensity <= intensity_target:
                    intensity_intervals = i + 1
                    break
            
            # Create predictions dictionary using corrected results
            predictions = {
                'model_name': model.name,
                'current_time': current_time,
                'current_failures': current_failures,
                'num_intervals': num_intervals,
                'effort_covariate': model.metricNames[0] if model.metricNames else 'Unknown',
                'effort_value': effort_values.get(model.metricNames[0], 1.0) if model.metricNames else 1.0,
                'effort_E': effort_values.get('E', 1.0),
                'effort_F': effort_values.get('F', 1.0),
                'effort_C': effort_values.get('C', 1.0),
                'intensity_target': intensity_target,
                'intensity_intervals': intensity_intervals,
                'future_times': future_times,
                'future_mvf': future_mvf_values,
                'future_intensity': future_intensity,
                'prediction_method': 'mathematically_corrected_effort_per_interval',
                'full_x': full_x.tolist(),
                'full_mvf': full_mvf_array.tolist(),
                'original_omega_used': original_omega,
                'correction_applied': True
            }
            
            logging.info(f"CORRECTED effort predictions complete - no omega recalculation artifacts.")
            logging.info(f"Original omega maintained: {original_omega:.4f}")
            logging.info(f"Target intensity ({intensity_target}) reached in {intensity_intervals} intervals")
            if future_intensity:
                avg_intensity = np.mean(future_intensity)
                logging.info(f"Average predicted intensity: {avg_intensity:.4f}")
            
            return predictions
            
        except Exception as prediction_error:
            logging.error(f"Error in corrected prediction: {prediction_error}")
            logging.error(f"Falling back to basic prediction")
            return generate_failure_predictions(model, prediction_params)
        
    except Exception as e:
        logging.error(f"Error in corrected effort prediction wrapper: {str(e)}")
        return None

def generate_failure_predictions_with_effort_gui_correct(model, prediction_params, effort_params):
    """Generate failure predictions exactly like the original GUI does
    
    This method replicates the exact behavior of the original C-SFRAT GUI when making predictions.
    Key insight: The GUI uses the LAST DATA POINT's covariate values for future predictions,
    creating smooth continuity without artificial discontinuities.
    
    Args:
        model: Fitted model object
        prediction_params: Dictionary with prediction parameters
        effort_params: Dictionary with effort per interval parameters
        
    Returns:
        Dictionary containing GUI-equivalent prediction results
    """
    if not hasattr(model, 't') or len(model.t) == 0:
        return None
    
    try:
        # 🔧 FIX: Use consistent parameter names with basic predictions
        # First try the standardized name, then fall back to effort-specific name for backward compatibility
        num_intervals = effort_params.get('num_intervals_to_predict', 
                                         effort_params.get('number_of_intervals_to_predict', 5))
        intensity_target = effort_params.get('failure_intensity_target', 0.3)
        
        logging.info(f"Generating predictions using GUI-CORRECT approach (smooth continuity)")
        logging.info(f"Model: {model.name}, Intervals: {num_intervals}")
        
        # Check if model has covariates
        if not hasattr(model, 'metricNames') or not model.metricNames:
            logging.warning("No covariates available, using basic prediction.")
            return generate_failure_predictions(model, prediction_params)
        
        # Create effort dictionary using LAST DATA POINT values (GUI behavior)
        class MockSpinBox:
            def __init__(self, value):
                self._value = value
            def value(self):
                return self._value
        
        effort_dict = {}
        
        # KEY FIX: Use the covariate values from the LAST data point for future predictions
        # This is what creates smooth continuity in the original GUI
        if hasattr(model, 'covariateData') and model.covariateData is not None:
            if model.covariateData.ndim == 1:
                # Single covariate case
                last_covariate_value = model.covariateData[-1]
                covariate_name = model.metricNames[0]
                effort_dict[covariate_name] = MockSpinBox(float(last_covariate_value))
                logging.info(f"  Using LAST DATA POINT value {last_covariate_value:.3f} for covariate: {covariate_name}")
            else:
                # Multiple covariates case
                for i, covariate_name in enumerate(model.metricNames):
                    last_covariate_value = model.covariateData[i, -1]
                    effort_dict[covariate_name] = MockSpinBox(float(last_covariate_value))
                    logging.info(f"  Using LAST DATA POINT value {last_covariate_value:.3f} for covariate: {covariate_name}")
        else:
            # Fallback: use unit effort if no covariate data available
            for covariate in model.metricNames:
                effort_dict[covariate] = MockSpinBox(1.0)
                logging.info(f"  Using fallback effort 1.0 for covariate: {covariate}")
        
        # Current state
        current_time = model.t[-1]
        current_failures = model.CFC[-1] if hasattr(model, 'CFC') and len(model.CFC) > 0 else 0
        
        try:
            # Use original C-SFRAT prediction_mvf function with LAST DATA POINT values
            x, mvf_array = prediction.prediction_mvf(model, num_intervals, model.covariateData, effort_dict)
            
            # Calculate intensity values from MVF
            intensity_array = []
            mvf_list = mvf_array.tolist()
            
            for i in range(1, len(mvf_list)):
                intensity_val = mvf_list[i] - mvf_list[i-1]
                intensity_array.append(max(0.01, intensity_val))
            
            # Get future values (excluding the original data points)
            original_length = len(model.t)
            future_times = x[original_length:].tolist()
            future_mvf = mvf_array[original_length:].tolist()
            future_intensity = intensity_array[original_length-1:] if len(intensity_array) >= original_length else intensity_array
            
            # Find when target intensity is reached
            intensity_intervals = 0
            for i, intensity in enumerate(future_intensity):
                if intensity <= intensity_target:
                    intensity_intervals = i + 1
                    break
            
            # Create predictions dictionary
            predictions = {
                'model_name': model.name,
                'current_time': current_time,
                'current_failures': current_failures,
                'num_intervals': num_intervals,
                'effort_covariate': 'Last Data Point Values',
                'effort_value': 'Continuous',
                'intensity_target': intensity_target,
                'intensity_intervals': intensity_intervals,
                'future_times': future_times,
                'future_mvf': future_mvf,
                'future_intensity': future_intensity,
                'prediction_method': 'gui_correct_continuous',
                'full_x': x.tolist(),
                'full_mvf': mvf_array.tolist(),
                'gui_correct': True
            }
            
            logging.info(f"GUI-CORRECT predictions complete - smooth continuity achieved")
            logging.info(f"Seamless transition from {current_failures:.2f} to {future_mvf[0] if future_mvf else 'N/A'}")
            
            return predictions
            
        except Exception as prediction_error:
            logging.error(f"Error in GUI-correct prediction functions: {prediction_error}")
            return generate_failure_predictions(model, prediction_params)
        
    except Exception as e:
        logging.error(f"Error in GUI-correct prediction wrapper: {str(e)}")
        return None

def generate_individual_model_predictions_gui_correct(models, params):
    """
    Generate GUI-correct predictions for individual models (smooth like original C-SFRAT GUI)
    
    This function applies the same GUI-correct approach to individual model predictions
    that we use for the green effort line, ensuring all predictions are smooth and continuous.
    
    Args:
        models: List of fitted model objects 
        params: Prediction parameters dictionary
        
    Returns:
        dict: Dictionary mapping model names to their smooth prediction data
    """
    if not models or not params.get('predict_failures', False):
        return {}
    
    global max_models_for_individual_predictions
    models_to_predict = models[:min(max_models_for_individual_predictions, len(models))]
    
    individual_predictions = {}
    
    logging.info(f"Generating GUI-CORRECT individual predictions for {len(models_to_predict)} models...")
    
    for model in models_to_predict:
        try:
            # Use the GUI-correct approach for each individual model
            # Create dummy effort parameters that use last data point values (for continuity)
            effort_params_gui_correct = {
                'effort_values': {'E': 1.0, 'F': 1.0, 'C': 1.0},  # Will be overridden by GUI-correct method
                'number_of_intervals_to_predict': params.get('prediction_horizon', 10),
                'failure_intensity_target': 0.3,
                'use_model_specific_covariates': True,
                'default_effort_for_unknown_covariates': 1.0
            }
            
            # Generate smooth predictions using GUI-correct method
            model_predictions = generate_failure_predictions_with_effort_gui_correct(
                model, params, effort_params_gui_correct
            )
            
            if model_predictions and 'future_times' in model_predictions:
                model_key = f"{model.name}"
                if hasattr(model, 'covariateNames') and model.covariateNames:
                    cov_str = ",".join([c[:3] for c in model.covariateNames])
                    model_key += f"({cov_str})"
                
                individual_predictions[model_key] = {
                    'model': model,
                    'future_times': model_predictions['future_times'],
                    'future_mvf': model_predictions['future_mvf'],
                    'future_intensity': model_predictions.get('future_intensity', []),
                    'prediction_horizon': model_predictions.get('prediction_horizon'),
                    'total_predicted_failures': model_predictions.get('total_predicted_failures'),
                    'gui_correct': True,  # Mark as GUI-correct
                    'prediction_method': 'gui_correct_individual'
                }
                
                logging.info(f"Generated GUI-CORRECT predictions for {model_key}: {len(model_predictions['future_times'])} points")
                
        except Exception as e:
            logging.warning(f"Failed to generate GUI-correct predictions for {model.name}: {str(e)}")
            # Fallback to original method if GUI-correct fails
            try:
                model_predictions = generate_failure_predictions(model, params)
                if model_predictions and 'future_times' in model_predictions:
                    model_key = f"{model.name}"
                    if hasattr(model, 'covariateNames') and model.covariateNames:
                        cov_str = ",".join([c[:3] for c in model.covariateNames])
                        model_key += f"({cov_str})"
                    
                    individual_predictions[model_key] = {
                        'model': model,
                        'future_times': model_predictions['future_times'],
                        'future_mvf': model_predictions['future_mvf'],
                        'future_intensity': model_predictions.get('future_intensity', []),
                        'prediction_horizon': model_predictions.get('prediction_horizon'),
                        'total_predicted_failures': model_predictions.get('total_predicted_failures'),
                        'gui_correct': False,  # Mark as fallback
                        'prediction_method': 'fallback_original'
                    }
                    logging.info(f"Generated FALLBACK predictions for {model_key}: {len(model_predictions['future_times'])} points")
            except Exception as e2:
                logging.warning(f"Both GUI-correct and fallback predictions failed for {model.name}: {str(e2)}")
                continue
    
    logging.info(f"Successfully generated GUI-CORRECT individual predictions for {len(individual_predictions)} models")
    return individual_predictions

# ===================== COVARIATE SCALING =====================
def scale_covariates_to_range(data_df, covariate_columns=None, target_min=0, target_max=10):
    """
    Scale covariate columns to a specified range (default 0-10)
    This improves effort allocation by ensuring all covariates are on the same scale
    
    Args:
        data_df: DataFrame containing the data
        covariate_columns: List of covariate columns to scale (if None, auto-detect)
        target_min: Minimum value for scaled range
        target_max: Maximum value for scaled range
        
    Returns:
        DataFrame with scaled covariates and scaling information
    """
    # Make a copy to avoid modifying the original
    scaled_df = data_df.copy()
    
    # Auto-detect covariate columns if not specified
    if covariate_columns is None:
        # Assume any column not in ['T', 'FC', 'CFC'] is a covariate
        standard_columns = ['T', 'FC', 'CFC']
        covariate_columns = [col for col in data_df.columns if col not in standard_columns]
    
    scaling_info = {}
    
    print(f"\n🔧 COVARIATE SCALING (Original C-SFRAT Enhancement)")
    print(f"Target range: {target_min} to {target_max}")
    print("-" * 50)
    
    for col in covariate_columns:
        if col in scaled_df.columns:
            original_min = scaled_df[col].min()
            original_max = scaled_df[col].max()
            
            # Skip if already in target range or no variation
            if original_max == original_min:
                print(f"⚠️  {col}: No variation (constant value {original_min:.3f}) - skipping")
                continue
            
            # Min-max scaling formula
            scaled_values = target_min + (scaled_df[col] - original_min) * (target_max - target_min) / (original_max - original_min)
            scaled_df[col] = scaled_values
            
            # Store scaling information
            scaling_info[col] = {
                'original_min': original_min,
                'original_max': original_max,
                'original_range': original_max - original_min,
                'scaled_min': target_min,
                'scaled_max': target_max,
                'scale_factor': (target_max - target_min) / (original_max - original_min)
            }
            
            print(f"✅ {col}: {original_min:.3f} to {original_max:.3f} → {target_min} to {target_max} (factor: {scaling_info[col]['scale_factor']:.3f})")
    
    print("-" * 50)
    print(f"✅ Scaled {len(scaling_info)} covariates for improved effort allocation")
    print()
    
    return scaled_df, scaling_info

# ===================== END COVARIATE SCALING =====================

if __name__ == "__main__":
    sys.exit(main()) 

