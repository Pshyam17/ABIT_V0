
#!/usr/bin/env python3
"""
ABIT Transform Comparison Framework
Main script to compare all transform combinations and orders
"""

import numpy as np
import pandas as pd
import itertools
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from config import TransformConfig
# CHANGE 1: Import BOTH data generators
from data_generator import load_or_generate_data  # Keep original
from data_generator_complex import (  # ADD THIS
    generate_complex_cookie_sales,
    generate_balanced_test_signal,
    get_comprehensive_test_patterns
)
from cascades import SingleTransformCascade, DualTransformCascade, TripleTransformCascade
from evaluation.metrics import PatternEvaluator
from evaluation.visualizer import plot_comparison_results

def run_comprehensive_comparison(data_complexity='balanced'):  # ADD PARAMETER
    """
    Run complete comparison of all transform combinations
    
    Args:
        data_complexity: Type of data to use
            - 'simple': Original periodic data (STFT favored)
            - 'balanced': Fair test with all pattern types
            - 'complex': Non-stationary patterns
            - 'realistic': Business-realistic mix
    """
    
    print("="*70)
    print("ABIT TRANSFORM COMPARISON FRAMEWORK")
    print(f"Data Complexity: {data_complexity.upper()}")  # Show which data we're using
    print("="*70)
    
    # CHANGE 2: Choose data based on complexity parameter
    if data_complexity == 'simple':
        # Use original data generator
        print("\n1. Loading SIMPLE data (STFT-favored)...")
        df = load_or_generate_data()
        
        # Configuration for simple data
        config = TransformConfig(
            minority_amplification=10.0,
            pattern_confidence_threshold=0.5
        )
        
        products = [
            ('chocochip', 'majority', 'Chocochip (Simple)'),
            ('oatmeal_raisin', 'minority', 'Oatmeal (Simple)')
        ]
        
    else:
        # Use new complex data generator
        print(f"\n1. Generating {data_complexity.upper()} data...")
        
        # Generate different complexity patterns
        if data_complexity == 'balanced':
            # Fair test - all transforms have opportunities
            signal_majority, comp_majority = generate_balanced_test_signal(
                n_points=500,
                scenario='balanced',
                sampling_rate=4.0
            )
            signal_minority, comp_minority = generate_balanced_test_signal(
                n_points=500,
                scenario='balanced',
                sampling_rate=4.0
            )
            # Add minority characteristics
            signal_minority = signal_minority * 0.1  # Scale down for minority
            
        elif data_complexity == 'complex':
            # Non-stationary, challenging patterns
            signal_majority, comp_majority = generate_complex_cookie_sales(
                n_months=120,
                product_type='majority',
                complexity='complex'
            )
            signal_minority, comp_minority = generate_complex_cookie_sales(
                n_months=120,
                product_type='minority',
                complexity='complex'
            )
            
        elif data_complexity == 'realistic':
            # Business-realistic patterns
            signal_majority, comp_majority = generate_balanced_test_signal(
                n_points=500,
                scenario='realistic',
                sampling_rate=4.0
            )
            signal_minority, comp_minority = generate_balanced_test_signal(
                n_points=500,
                scenario='realistic',
                sampling_rate=4.0
            )
            signal_minority = signal_minority * 0.1
            
        else:  # stft_friendly
            # Test that STFT works correctly
            signal_majority, comp_majority = generate_balanced_test_signal(
                n_points=500,
                scenario='stft_friendly',
                sampling_rate=4.0
            )
            signal_minority = signal_majority * 0.1
            comp_minority = comp_majority
        
        # Create DataFrame-like structure for compatibility
        df = pd.DataFrame({
            'complex_majority': signal_majority,
            'complex_minority': signal_minority
        })
        
        print(f"   Generated {len(signal_majority)} samples")
        print(f"   Pattern types: {comp_majority.get('pattern_types', 'mixed')}")
        
        # Configuration for complex data
        config = TransformConfig(
            minority_amplification=10.0 if data_complexity == 'complex' else 5.0,
            pattern_confidence_threshold=0.4,  # Lower threshold for complex patterns
            frequency_tolerance=0.03,  # More tolerance for varying frequencies
            stft_nperseg=48 if len(signal_majority) > 200 else 24  # Adjust window
        )
        
        products = [
            ('complex_majority', 'majority', f'Majority ({data_complexity.title()})'),
            ('complex_minority', 'minority', f'Minority ({data_complexity.title()})')
        ]
    
    evaluator = PatternEvaluator(config)
    
    # CHANGE 3: Rest of the code remains the same
    for column_name, signal_type, display_name in products:
        print(f"\n{'='*70}")
        print(f"Analyzing: {display_name}")
        print('='*70)
        
        signal = df[column_name].values
        all_results = {}
        
        # 1. Single transforms
        print("\n2. Testing SINGLE transforms...")
        for transform_type in ['hht', 'wavelet', 'stft']:
            cascade = SingleTransformCascade(transform_type, config)
            patterns = cascade.analyze(signal, signal_type)
            method_name = cascade.get_name()
            all_results[method_name] = patterns
            print(f"   {method_name:10} - Found {len(patterns)} patterns")
        
        # 2. Dual combinations
        print("\n3. Testing DUAL combinations...")
        transform_types = ['hht', 'wavelet', 'stft']
        for first, second in itertools.permutations(transform_types, 2):
            cascade = DualTransformCascade(first, second, config)
            patterns = cascade.analyze(signal, signal_type)
            method_name = cascade.get_name()
            all_results[method_name] = patterns
            print(f"   {method_name:20} - Found {len(patterns)} patterns")
        
        # 3. Triple combinations
        print("\n4. Testing TRIPLE combinations...")
        for perm in itertools.permutations(transform_types, 3):
            cascade = TripleTransformCascade(*perm, config)
            patterns = cascade.analyze(signal, signal_type)
            method_name = cascade.get_name()
            all_results[method_name] = patterns
            print(f"   {method_name:25} - Found {len(patterns)} patterns")
        
        # Evaluate all methods
        print("\n5. Evaluating performance...")
        all_metrics, best_method = evaluator.compare_methods(all_results)
        
        # Print summary
        print("\nTop 5 Methods by Combined Score:")
        print("-"*50)
        sorted_methods = sorted(all_metrics.items(), 
                              key=lambda x: x[1]['combined_score'], 
                              reverse=True)
        
        for i, (method, metrics) in enumerate(sorted_methods[:5], 1):
            print(f"{i}. {method:25} Score: {metrics['combined_score']:.3f}")
            print(f"   Detection: {metrics['detection_score']:.3f}, "
                  f"Noise Rej: {metrics['noise_rejection']:.3f}, "
                  f"Patterns: {metrics['num_patterns']}")
        
        print(f"\n🏆 WINNER: {best_method}")
        
        # Visualize results
        print("\n6. Creating visualizations...")
        plot_comparison_results(all_metrics, display_name)
    
    # Final analysis
    print("\n" + "="*70)
    print(f"ANALYSIS COMPLETE - {data_complexity.upper()} Data")
    print("="*70)

# CHANGE 4: Add new comparison function
def compare_data_complexities():
    """
    Run comparison across different data complexities to see how methods perform
    """
    print("\n" + "="*70)
    print("COMPARING ACROSS DATA COMPLEXITIES")
    print("="*70)
    
    complexities = ['simple', 'balanced', 'complex', 'realistic']
    
    for complexity in complexities:
        print(f"\n\n{'#'*70}")
        print(f"# Testing with {complexity.upper()} data")
        print('#'*70)
        run_comprehensive_comparison(data_complexity=complexity)
        
        input("\nPress Enter to continue to next complexity level...")

# CHANGE 5: Modify main execution
if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        complexity = sys.argv[1]
        if complexity in ['simple', 'balanced', 'complex', 'realistic', 'stft_friendly']:
            print(f"Running with {complexity} data...")
            run_comprehensive_comparison(data_complexity=complexity)
        elif complexity == 'compare_all':
            compare_data_complexities()
        else:
            print("Unknown complexity. Options: simple, balanced, complex, realistic, stft_friendly, compare_all")
    else:
        # Default to balanced (fair comparison)
        print("Running with BALANCED data (fair comparison)...")
        print("Tip: Use 'python run_comparison.py simple' for original data")
        print("     Use 'python run_comparison.py compare_all' to test all complexities")
        run_comprehensive_comparison(data_complexity='balanced')
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)