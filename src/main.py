#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional
import sys

from src.config.config_manager import config_manager
from src.utils.logging_config import app_logger
from src.generators.employment_generator import EmploymentDataGenerator
from src.generators.investment_generator import InvestmentDataGenerator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic financial data for fraud detection.'
    )
    parser.add_argument(
        '--size', 
        type=int, 
        default=1000,
        help='Number of records to generate'
    )
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output-format',
        choices=['csv', 'parquet', 'json'],
        default='csv',
        help='Output file format'
    )
    parser.add_argument(
        '--dataset-type',
        choices=['employment', 'investment'],
        default='investment',
        help='Type of dataset to generate'
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Initialize configuration
        config_manager.initialize(args.config)
        
        # Create generator based on dataset type
        if args.dataset_type == 'employment':
            generator = EmploymentDataGenerator()
        else:
            generator = InvestmentDataGenerator()
        
        # Generate data
        app_logger.info(f"Generating {args.size} records of {args.dataset_type} data")
        data = generator.generate(args.size)
        
        # Save data
        output_path = generator.save_dataset(data, args.output_format)
        app_logger.info(f"Data saved to {output_path}")
        
        return 0
    
    except Exception as e:
        app_logger.error(f"Error generating data: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 