#!/usr/bin/env python3
"""
C-SFRAT Easy Configuration Script
=================================

This script provides a simple way to configure and run C-SFRAT analysis.
Just edit the settings below and run this file - no complex command-line needed!

Usage:
    python csfrat_config.py

Simply modify the configuration values below and run the script.
"""

import subprocess
import sys
import os

# ============================================================================
# 📊 TAB 1: DATA UPLOAD AND MODEL SELECTION (Original C-SFRAT Tab 1)
# ============================================================================
# This section mirrors the original GUI's first tab for data import and model setup

# 📁 Data Input Settings
INPUT_FILE = "C-SFRAT/datasets/ds1.csv"        # Your input data file (CSV or Excel)
OUTPUT_FILE = "outputs/config_demo_report.pdf" # Where to save the PDF report

# 📊 Excel File Options (only needed for Excel files)
EXCEL_SHEET = None                             # Excel sheet name to analyze 
                                              # Examples: "Sheet1", "Data", "Results" 
                                              # Leave as None to automatically use the first sheet

# 📏 Data Subsetting (matches GUI slider - limits how much historical data to analyze)  
DATA_LIMIT = None                              # Limit analysis to first N time intervals (None = use all data)
                                              # Examples: 10 (analyze first 10 intervals), 25 (first 25 intervals)
                                              # Useful for testing or focusing on early development phases

# 📊 Statistical Confidence Level (affects trend tests and confidence intervals)
CONFIDENCE_LEVEL = 0.95                       # Statistical confidence level (common values: 0.90, 0.95, 0.99)
                                              # 0.95 = 95% confidence (recommended for most analyses)

# 🔬 Reliability Model Selection (matches GUI model list)
RUN_ALL_MODELS = True                         # Analyze all reliability models: True (recommended) or False
SELECTED_MODEL = "Geometric"                  # If RUN_ALL_MODELS is False, choose ONE model:
                                             # "Geometric" (simple, good starting point)
                                             # "DiscreteWeibull2" (handles decreasing failure rates)  
                                             # "NegativeBinomial2" (handles increasing failure rates)
                                             # "S_Distribution" (S-shaped failure curves)
                                             # "IFR_SB" (increasing failure rate, Salvia-Bollinger)
                                             # "IFR_Generalized_SB" (generalized increasing failure rate)
                                             # "TruncatedLogistic" (logistic growth with bounds)

# 📈 Covariate Analysis (matches GUI metrics list - these are columns in your data file)  
RUN_ALL_COVARIATES = True                     # Test all covariate combinations: True (recommended) or False
SELECTED_COVARIATES = []                      # If RUN_ALL_COVARIATES is False, specify which covariates to use
                                             # Your data file MUST contain columns named exactly as listed here
                                             # Common examples: ["E"] (effort only), ["E", "F"] (effort + faults), ["E", "F", "C"] (all three)
                                             # ⚠️  Check your CSV/Excel column headers to see available covariate names

# ============================================================================
# 📈 TAB 2: MODEL RESULTS AND PREDICTIONS (Original C-SFRAT Tab 2)
# ============================================================================
# This section mirrors the original GUI's second tab for predictions and effort analysis

# 🔮 Future Failure Prediction Settings (matches GUI prediction controls)
ENABLE_PREDICTIONS = True                     # Generate future failure predictions: True (recommended) or False
PREDICTION_INTERVALS = 4                     # How many future time periods to predict (typical range: 3-20)
INTENSITY_TARGET = 0.3                       # Target failure intensity goal (failures per time interval)
                                             # Example: 0.1 = very reliable, 0.5 = moderate, 1.0 = high failure rate

# ⚡ Effort Per Interval Configuration (matches GUI effort controls)
# These values represent relative effort/resources allocated to each covariate
# ⚠️  IMPORTANT: Covariate names (E, F, C) must match your data file column headers exactly!
EFFORT_PER_INTERVAL_ENABLED = True           # Enable effort-based predictions: True (recommended) or False

# 💪 Effort Values (relative units - higher values = more effort/resources)
EFFORT_E = 1.00                              # Effort allocated to covariate E (commonly: development effort)
EFFORT_F = 2.00                              # Effort allocated to covariate F (commonly: fault correction effort)  
EFFORT_C = 3.00                              # Effort allocated to covariate C (commonly: code complexity effort)
                                            # 📝 If your data uses different covariate names (like "DevEffort", "TestHours"), 
                                            # you'll need to modify these parameter names accordingly

# 🎯 Advanced Effort Settings (usually keep these as default)
USE_MODEL_SPECIFIC_COVARIATES = True         # Use only covariates that each model actually supports: True (recommended) or False
DEFAULT_EFFORT_FOR_UNKNOWN_COVARIATES = 1.0 # Default effort for covariates not specified above (relative units)

# 🎨 Visualization Settings (matches GUI plot view options)
SHOW_ALL_MODELS = False                      # Show all models in detailed predictions table: True or False
MULTI_MODEL_PLOTS = True                     # Enable multi-model comparison plots: True or False
NUM_MODELS_COMPARE = 3                       # Number of top models to show in plots (1-3)
INDIVIDUAL_MODEL_PREDICTIONS = True          # Show individual model predictions: True or False
SHOW_MODEL_PREDICTIONS_SEPARATELY = True    # Show each model's predictions with different colors: True or False
MAX_MODELS_FOR_INDIVIDUAL_PREDICTIONS = 3   # Maximum number of models to show individual predictions for

# ============================================================================
# ⚖️ TAB 3: MODEL COMPARISON (Original C-SFRAT Tab 3)
# ============================================================================
# This section mirrors the original GUI's third tab for model comparison and ranking

# 📊 Ranking Method (matches GUI comparison options)
RANKING_METHOD = "mean"                      # Ranking criteria: "mean" or "median"
                                            # 'mean' - ranks by average critic values across all metrics
                                            # 'median' - ranks by median critic values across all metrics

# 📏 PSSE Calculation Settings (matches GUI PSSE group controls)
PSSE_SUBSET = 0.9                           # PSSE validation fraction (0.1 to 0.99, recommended: 0.9)
                                            # This determines what fraction of data is used for validation
                                            # 0.9 = use 90% of data for fitting, 10% for validation (recommended)
                                            # Higher values = more data for fitting, less for validation

# ⚖️ Model Ranking Weights (matches GUI weight controls)
# How much importance to give each quality metric when ranking models (0.0 to 10.0)
# 💡 TIP: Higher weight = more important for model selection
# 💡 TIP: Set all but one weight to 0.0 to use only that single metric
#
# 🎯 QUICK CONFIGURATIONS:
# For best statistical fit:        WEIGHT_AIC=2.0, others=0.0  
# For best prediction accuracy:    WEIGHT_PSSE=2.0, others=0.0
# For best goodness of fit:        WEIGHT_SSE=2.0, others=0.0
# For balanced approach (current): WEIGHT_AIC=2.0, WEIGHT_BIC=1.0, WEIGHT_SSE=1.0, WEIGHT_PSSE=1.0

WEIGHT_LLF = 0.0                            # Log-Likelihood: model fit quality (higher LLF = better fit)
WEIGHT_AIC = 2.0                            # AIC: model quality with complexity penalty (lower AIC = better model)  
WEIGHT_BIC = 1.0                            # BIC: conservative model selection (lower BIC = better model)
WEIGHT_SSE = 1.0                            # SSE: fitting accuracy (lower SSE = better fit to data)
WEIGHT_PSSE = 1.0                           # PSSE: prediction accuracy (lower PSSE = better predictions)

# 📋 Table Display Settings
MAX_MODELS_IN_COMPARISON_TABLE = 20         # Maximum number of models to display in comparison table (0 = show all)

# 🔧 Single Metric Mode (advanced users - overrides weight settings above)
SINGLE_METRIC = None                        # Force single metric ranking (leave as None to use weights above)
                                           # Options: None (use weights), "aic", "bic", "sse", "psse", "llf"
                                           # Example: "aic" ignores all weights and ranks models by AIC only

# ============================================================================
# 💰 TAB 4: EFFORT ALLOCATION (Original C-SFRAT Tab 4)
# ============================================================================
# This section mirrors the original GUI's fourth tab for budget optimization and effort allocation

# 🎯 Budget Optimization Settings (matches GUI allocation controls)
ENABLE_OPTIMIZATION = True                   # Run budget optimization analysis: True (recommended) or False

# 📊 Scenario 1: Maximize Defects within Fixed Budget (matches GUI "Allocation 1" tab)
ALLOCATION_1_ENABLED = True                  # Enable Scenario 1: True or False
TOTAL_BUDGET = 60.0                         # Your total available budget (in dollars, hours, or other units)
                                           # Example: 100 = $100, 1000 = $1000, 40 = 40 person-hours

# 📊 Scenario 2: Minimize Budget for Target Defects (matches GUI "Allocation 2" tab)  
ALLOCATION_2_ENABLED = True                  # Enable Scenario 2: True or False
TARGET_ADDITIONAL_DEFECTS = 2              # How many additional defects you want to find (positive integer)
                                           # Example: 5 = find 5 more defects, 10 = find 10 more defects

# ⚙️ Optimization Method (usually keep as default)
OPTIMIZATION_METHOD = 'both_allocations'     # Which scenarios to run:
                                           # 'allocation_1' = only Scenario 1 (maximize defects)
                                           # 'allocation_2' = only Scenario 2 (minimize budget) 
                                           # 'both_allocations' = run both scenarios (recommended)

# ============================================================================
# 🔧 ADVANCED SETTINGS - Enhancements beyond original C-SFRAT
# ============================================================================
# These settings provide additional functionality not available in the original GUI

# 📏 Covariate Scaling (Enhancement for better optimization)
ENABLE_COVARIATE_SCALING = True             # Normalize covariate values for better optimization: True (recommended) or False
COVARIATE_SCALING_RANGE = (0, 10)          # Scaling range: (minimum_value, maximum_value)
                                           # Example: (0, 10) scales all covariates to 0-10 range
                                           # This helps optimization algorithms work more effectively

# ============================================================================
# 🚀 EXECUTION CODE - DO NOT MODIFY BELOW THIS LINE
# ============================================================================

def validate_settings():
    """Check if all settings are valid."""
    errors = []
    
    # Check input file
    if not INPUT_FILE:
        errors.append("❌ INPUT_FILE cannot be empty")
    elif not os.path.exists(INPUT_FILE):
        errors.append(f"❌ Input file not found: {INPUT_FILE}")
    
    # Check output directory
    if OUTPUT_FILE:
        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"✅ Created output directory: {output_dir}")
            except Exception as e:
                errors.append(f"❌ Cannot create output directory: {e}")
    
    # Check model settings
    if not isinstance(RUN_ALL_MODELS, bool):
        errors.append("❌ RUN_ALL_MODELS must be True or False")
    
    if not isinstance(RUN_ALL_COVARIATES, bool):
        errors.append("❌ RUN_ALL_COVARIATES must be True or False")
    
    # Check weights
    weights = [WEIGHT_LLF, WEIGHT_AIC, WEIGHT_BIC, WEIGHT_SSE, WEIGHT_PSSE]
    for i, weight in enumerate(weights):
        if weight < 0 or weight > 10:
            metric_names = ["LLF", "AIC", "BIC", "SSE", "PSSE"]
            errors.append(f"❌ WEIGHT_{metric_names[i]} must be between 0.0 and 10.0")
    
    # Check PSSE subset
    if PSSE_SUBSET < 0.01 or PSSE_SUBSET > 0.99:
        errors.append("❌ PSSE_SUBSET must be between 0.01 and 0.99")
    
    # Check prediction intervals
    if PREDICTION_INTERVALS < 1 or PREDICTION_INTERVALS > 50:
        errors.append("❌ PREDICTION_INTERVALS must be between 1 and 50")
    
    # Check models compare
    if NUM_MODELS_COMPARE < 1 or NUM_MODELS_COMPARE > 3:
        errors.append("❌ NUM_MODELS_COMPARE must be 1, 2, or 3")
    
    return errors

def build_command():
    """Build the command to run the main script."""
    cmd = ["python", "csfrat_integrated.py"]
    
    # Basic settings
    cmd.extend(["--input", INPUT_FILE])
    cmd.extend(["--output", OUTPUT_FILE])
    
    if EXCEL_SHEET:
        cmd.extend(["--sheet", EXCEL_SHEET])
    
    if DATA_LIMIT:
        cmd.extend(["--data-subset", str(DATA_LIMIT)])
    
    # Model selection
    if RUN_ALL_MODELS:
        cmd.append("--all-models")
    else:
        cmd.extend(["--model", SELECTED_MODEL])
    
    # Covariate selection
    if RUN_ALL_COVARIATES:
        cmd.append("--all-covariates")
    else:
        cmd.extend(["--covariates"] + SELECTED_COVARIATES)
    
    # Ranking settings
    cmd.extend(["--ranking-method", RANKING_METHOD])
    cmd.extend(["--psse-subset", str(PSSE_SUBSET)])
    
    # Weights or single metric
    if SINGLE_METRIC:
        cmd.extend(["--single-metric", SINGLE_METRIC])
    else:
        cmd.extend(["--weight-llf", str(WEIGHT_LLF)])
        cmd.extend(["--weight-aic", str(WEIGHT_AIC)])
        cmd.extend(["--weight-bic", str(WEIGHT_BIC)])
        cmd.extend(["--weight-sse", str(WEIGHT_SSE)])
        cmd.extend(["--weight-psse", str(WEIGHT_PSSE)])
    
    # Prediction settings
    if ENABLE_PREDICTIONS:
        cmd.append("--predict-failures")
        cmd.extend(["--prediction-intervals", str(PREDICTION_INTERVALS)])
        cmd.extend(["--intensity-target", str(INTENSITY_TARGET)])
    
    # Effort values
    cmd.extend(["--effort-E", str(EFFORT_E)])
    cmd.extend(["--effort-F", str(EFFORT_F)])
    cmd.extend(["--effort-C", str(EFFORT_C)])
    
    # Optimization
    if ENABLE_OPTIMIZATION:
        cmd.append("--enable-optimization")
        cmd.extend(["--budget", str(TOTAL_BUDGET)])
    
    # Display settings
    if SHOW_ALL_MODELS:
        cmd.append("--show-all-models")
    
    if MULTI_MODEL_PLOTS:
        cmd.append("--multi-plots")
        cmd.extend(["--num-models-compare", str(NUM_MODELS_COMPARE)])
    
    if ENABLE_COVARIATE_SCALING:
        cmd.append("--enable-scaling")
    
    return cmd

def show_configuration():
    """Display current configuration."""
    print("📋 Current Configuration:")
    print("=" * 50)
    print(f"📁 Input File: {INPUT_FILE}")
    print(f"📄 Output File: {OUTPUT_FILE}")
    if EXCEL_SHEET:
        print(f"📊 Excel Sheet: {EXCEL_SHEET}")
    if DATA_LIMIT:
        print(f"🔢 Data Limit: {DATA_LIMIT} intervals")
    
    print(f"\n🔬 Run All Models: {RUN_ALL_MODELS}")
    if not RUN_ALL_MODELS:
        print(f"   Selected Model: {SELECTED_MODEL}")
    
    print(f"🧬 Run All Covariates: {RUN_ALL_COVARIATES}")
    if not RUN_ALL_COVARIATES:
        print(f"   Selected Covariates: {SELECTED_COVARIATES}")
    
    print(f"\n⚖️ Ranking Method: {RANKING_METHOD}")
    if SINGLE_METRIC:
        print(f"📊 Single Metric: {SINGLE_METRIC.upper()}")
    else:
        print(f"📊 Weights - LLF:{WEIGHT_LLF}, AIC:{WEIGHT_AIC}, BIC:{WEIGHT_BIC}, SSE:{WEIGHT_SSE}, PSSE:{WEIGHT_PSSE}")
    
    print(f"\n📈 Predictions: {'Enabled' if ENABLE_PREDICTIONS else 'Disabled'}")
    if ENABLE_PREDICTIONS:
        print(f"   Intervals: {PREDICTION_INTERVALS}, Target: {INTENSITY_TARGET}")
    
    print(f"💰 Optimization: {'Enabled' if ENABLE_OPTIMIZATION else 'Disabled'}")
    if ENABLE_OPTIMIZATION:
        print(f"   Budget: ${TOTAL_BUDGET}")
    
    print(f"\n🎨 Multi-plots: {'Enabled' if MULTI_MODEL_PLOTS else 'Disabled'}")
    print(f"🔧 Scaling: {'Enabled' if ENABLE_COVARIATE_SCALING else 'Disabled'}")
    print("=" * 50)

def main():
    """Main execution function."""
    print("🔬 C-SFRAT Easy Configuration Script")
    print("=" * 50)
    
    # Check for demo mode
    demo_mode = len(sys.argv) > 1 and sys.argv[1] == "--demo"
    
    # Show current configuration
    show_configuration()
    
    # Validate settings
    print("\n📋 Validating configuration...")
    errors = validate_settings()
    
    if errors:
        print("\n❌ Configuration errors found:")
        for error in errors:
            print(f"   {error}")
        print("\n🔧 Please fix the errors above and run again.")
        return 1
    
    print("✅ Configuration is valid!")
    
    # Build and show command
    print("\n🔧 Building command...")
    cmd = build_command()
    
    # Show command that would be executed
    print(f"\n🚀 Command that will be executed:")
    print("   " + " ".join(cmd))
    
    if demo_mode:
        print("\n🧪 DEMO MODE: Configuration validation complete!")
        print("✅ Your settings are valid and ready to use.")
        print("\nTo run the actual analysis, execute:")
        print("   python csfrat_config.py")
        return 0
    
    # Ask for confirmation
    print(f"\n🚀 Ready to run analysis!")
    print(f"📊 This will analyze: {INPUT_FILE}")
    print(f"📄 Report will be saved to: {OUTPUT_FILE}")
    
    response = input("\n▶️  Press Enter to start analysis (or 'q' to quit): ").strip().lower()
    if response == 'q':
        print("❌ Analysis cancelled by user.")
        return 0
    
    # Run the analysis
    print("\n⏳ Running C-SFRAT analysis...")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("🎉 Analysis completed successfully!")
            if os.path.exists(OUTPUT_FILE):
                abs_path = os.path.abspath(OUTPUT_FILE)
                print(f"📄 Report saved to: {abs_path}")
                print(f"📂 You can now open: {OUTPUT_FILE}")
            print("=" * 50)
            return 0
        else:
            print(f"\n❌ Analysis failed with return code: {result.returncode}")
            return result.returncode
            
    except FileNotFoundError:
        print("\n❌ Error: csfrat_integrated.py not found!")
        print("   Make sure this script is in the same directory as csfrat_integrated.py")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())