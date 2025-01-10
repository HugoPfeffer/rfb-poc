# RFB-POC: Synthetic Financial Data Generator

A tool for generating synthetic financial data with fraud scenarios for testing and development purposes.

## Project Structure

```
rfb-poc/
├── src/                    # Source code
│   ├── generators/        # Data generation components
│   ├── config/           # Configuration management
│   ├── validation/       # Validation framework
│   ├── utils/           # Utilities and logging
│   └── main.py          # Entry point
├── tests/                # Test suite
│   ├── test_generators/ # Generator-specific tests
│   └── test_suite.py    # Main test runner
├── data/
│   ├── configs/         # Configuration files
│   ├── raw/            # Input data
│   └── generated/      # Generated datasets
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Features

- Generate synthetic employment data with realistic salary distributions
- Generate synthetic investment data with portfolio allocations
- Add fraud scenarios and anomalies for testing
- Configurable data generation parameters
- Comprehensive validation framework
- Detailed logging and error handling

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the generator:
```bash
# Generate investment data (default)
python src/main.py --size 1000

# Generate employment data
python src/main.py --size 1000 --dataset-type employment

# Specify output format
python src/main.py --size 1000 --output-format parquet

# Use custom configuration
python src/main.py --size 1000 --config path/to/config.json
```

3. Run tests:
```bash
python -m unittest discover tests
```

## TODO

- Create generator for Vehicle FIPE
- Split extra faker classes into separate files
- Add option to select year of data generation
- Add transactions generator with outliers
- Add randomizer for asset ownership
- Add number of dependents
- Add date of submission

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
