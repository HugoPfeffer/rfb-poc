#!/usr/bin/env python3

import argparse
from pathlib import Path
from controllers.employment_generator import EmploymentGenerator
from controllers.investment_generator import InvestmentGenerator
import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_datasets(size: int = 1000, preview: bool = False) -> tuple[str, str]:
    """Generate synthetic employment and investment datasets.
    
    Args:
        size: Number of records to generate
        preview: Whether to show data preview
        
    Returns:
        tuple[str, str]: Paths to the generated employment and investment datasets
    """
    try:
        # Generate employment data
        logger.info("Initializing Employment Data Generator...")
        employment_generator = EmploymentGenerator()
        
        logger.info(f"Generating {size} employment records...")
        employment_df = employment_generator.generate(size)
        
        logger.info("Saving employment dataset...")
        employment_path = employment_generator.save_industry_dataset()
        logger.info(f"Employment dataset saved to: {employment_path}")

        # Generate investment data
        logger.info("\nInitializing Investment Data Generator...")
        investment_generator = InvestmentGenerator()
        
        logger.info(f"Generating {size} investment records...")
        investment_df = investment_generator.generate(size)
        
        # Save investment data
        output_dir = Path('./data/clean')
        output_dir.mkdir(parents=True, exist_ok=True)
        investment_path = output_dir / f'investment_dataset_{pd.Timestamp.now().strftime("%Y%m%d")}.csv'
        investment_generator.save(str(investment_path))
        logger.info(f"Investment dataset saved to: {investment_path}")

        # Display preview if requested
        if preview:
            show_preview(employment_df, investment_df)
            
        return employment_path, str(investment_path)

    except Exception as e:
        logger.error(f"Error generating datasets: {str(e)}")
        raise

def show_preview(employment_df: pd.DataFrame, investment_df: pd.DataFrame) -> None:
    """Display preview of the generated data.
    
    Args:
        employment_df: Generated employment DataFrame
        investment_df: Generated investment DataFrame
    """
    logger.info("\nEmployment Data Preview:")
    logger.info("------------------------")
    
    logger.info("\nFirst 5 employment records:")
    print(employment_df.head())
    
    logger.info("\nEmployment Dataset Summary:")
    logger.info("-------------------------")
    print(f"Total Records: {len(employment_df)}")
    
    logger.info("\nExperience Level Distribution:")
    exp_dist = employment_df['experience_level'].value_counts(normalize=True).round(3)
    print(exp_dist)
    
    logger.info("\nSalary Statistics:")
    salary_stats = employment_df.groupby('experience_level')['salary'].agg(['mean', 'min', 'max']).round(2)
    print(salary_stats)
    
    logger.info("\nTop 5 Industries by Count:")
    industry_counts = employment_df['industry'].value_counts().head()
    print(industry_counts)
    
    logger.info("\n\nInvestment Data Preview:")
    logger.info("----------------------")
    
    logger.info("\nFirst 5 investment records:")
    print(investment_df.head())
    
    logger.info("\nFraud Statistics:")
    logger.info("----------------")
    print("\nFraud Type Distribution:")
    fraud_dist = investment_df['fraud_type'].value_counts(normalize=True).round(3)
    print(fraud_dist)
    
    logger.info("\nSuspicious Activity Summary:")
    suspicious_count = investment_df['suspicious_lifestyle'].sum()
    total_count = len(investment_df)
    print(f"Suspicious Cases: {suspicious_count} ({(suspicious_count/total_count*100):.1f}%)")
    
    logger.info("\nAsset Allocation Summary:")
    allocation_stats = investment_df[['stocks', 'bonds', 'cash', 'real_estate']].agg(['mean', 'min', 'max']).round(3)
    print(allocation_stats)

def main():
    """Main function to run the data generators."""
    parser = argparse.ArgumentParser(description='Generate synthetic employment and investment data')
    parser.add_argument('--size', type=int, default=1000,
                       help='Number of records to generate (default: 1000)')
    parser.add_argument('--preview', action='store_true',
                       help='Preview the generated data')
    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory (optional)')
    args = parser.parse_args()

    try:
        employment_path, investment_path = generate_datasets(args.size, args.preview)
        logger.info("\nData generation completed successfully!")
        logger.info(f"Employment data: {employment_path}")
        logger.info(f"Investment data: {investment_path}")
    except Exception as e:
        logger.error(f"Failed to generate datasets: {str(e)}")
        raise

if __name__ == "__main__":
    main() 