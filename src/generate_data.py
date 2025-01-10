#!/usr/bin/env python3

import argparse
from pathlib import Path
from controllers.employment_generator import EmploymentGenerator
import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_dataset(size: int = 1000, preview: bool = False) -> str:
    """Generate synthetic employment dataset.
    
    Args:
        size: Number of records to generate
        preview: Whether to show data preview
        
    Returns:
        str: Path to the generated dataset
    """
    try:
        # Initialize the generator
        logger.info("Initializing Employment Data Generator...")
        generator = EmploymentGenerator()

        # Generate the dataset
        logger.info(f"Generating {size} employment records...")
        df = generator.generate(size)

        # Save the dataset
        logger.info("Saving dataset...")
        output_path = generator.save_industry_dataset()
        logger.info(f"Dataset saved to: {output_path}")

        # Display preview if requested
        if preview:
            show_preview(df)
            
        return output_path

    except Exception as e:
        logger.error(f"Error generating dataset: {str(e)}")
        raise

def show_preview(df: pd.DataFrame) -> None:
    """Display preview of the generated data.
    
    Args:
        df: Generated DataFrame to preview
    """
    logger.info("\nData Preview:")
    logger.info("-------------")
    
    logger.info("\nFirst 5 records:")
    print(df.head())
    
    logger.info("\nDataset Summary:")
    logger.info("---------------")
    print(f"Total Records: {len(df)}")
    
    logger.info("\nExperience Level Distribution:")
    exp_dist = df['experience_level'].value_counts(normalize=True).round(3)
    print(exp_dist)
    
    logger.info("\nSalary Statistics:")
    salary_stats = df.groupby('experience_level')['salary'].agg(['mean', 'min', 'max']).round(2)
    print(salary_stats)
    
    logger.info("\nTop 5 Industries by Count:")
    industry_counts = df['industry'].value_counts().head()
    print(industry_counts)

def main():
    """Main function to run the employment data generator."""
    parser = argparse.ArgumentParser(description='Generate synthetic employment data')
    parser.add_argument('--size', type=int, default=1000,
                       help='Number of records to generate (default: 1000)')
    parser.add_argument('--preview', action='store_true',
                       help='Preview the generated data')
    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory (optional)')
    args = parser.parse_args()

    try:
        output_path = generate_dataset(args.size, args.preview)
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main() 