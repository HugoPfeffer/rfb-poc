from classes.core.data_loader import DataLoader

def main():
    # Initialize the data loader
    loader = DataLoader()
    
    # Print the configured data path
    print(f"Configured data path: {loader.data_path}")
    
    try:
        # Load the data
        df = loader.load_data()
        print("\nData loaded successfully!")
        print(f"DataFrame shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure your CSV file exists at the path specified in settings.json")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main() 