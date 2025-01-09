#!/usr/bin/env python3

import argparse
from pathlib import Path
from controllers.employment_generator import EmploymentGenerator
import pandas as pd

def main():
    """Main function to run the employment data generator."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic employment data')
    parser.add_argument('--size', type=int, default=1000,
                       help='Number of records to generate (default: 1000)')
    parser.add_argument('--preview', action='store_true',
                       help='Preview the generated data')
    args = parser.parse_args()

    try:
        # Initialize the generator
        print("Initializing Employment Data Generator...")
        generator = EmploymentGenerator()

        # Generate the dataset
        print(f"\nGenerating {args.size} employment records...")
        df = generator.generate(args.size)

        # Save the dataset
        print("\nSaving dataset...")
        output_path = generator.save_industry_dataset()
        print(f"Dataset saved to: {output_path}")

        # Display preview if requested
        if args.preview:
            print("\nData Preview:")
            print("-------------")
            print("\nFirst 5 records:")
            print(df.head())
            
            print("\nDataset Summary:")
            print("---------------")
            print(f"Total Records: {len(df)}")
            
            print("\nExperience Level Distribution:")
            exp_dist = df['experience_level'].value_counts(normalize=True).round(3)
            print(exp_dist)
            
            print("\nSalary Statistics:")
            salary_stats = df.groupby('experience_level')['salary'].agg(['mean', 'min', 'max']).round(2)
            print(salary_stats)
            
            print("\nTop 5 Industries by Count:")
            industry_counts = df['industry'].value_counts().head()
            print(industry_counts)

    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main() 