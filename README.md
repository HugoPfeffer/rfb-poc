# RFB-POC: Synthetic Financial Data Generator

A robust Python tool for generating realistic synthetic financial data with configurable fraud scenarios, designed for testing and development of financial fraud detection systems.

## Overview

This tool generates synthetic datasets that mimic real-world financial data patterns, including:
- Employment records with realistic salary distributions
- Investment portfolios with market-based asset allocations
- Fraud scenarios with configurable probabilities and patterns
- Lifestyle and spending pattern indicators

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rfb-poc.git
cd rfb-poc
```

2. Create and activate a virtual environment:
```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Or using conda
conda create -n rfb-poc python=3.11
conda activate rfb-poc
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
rfb-poc/
├── src/                    # Source code
│   ├── generators/        # Data generation components
│   │   ├── components.py       # Base generation classes
│   │   ├── employment_generator.py  # Employment data generator
│   │   └── investment_generator.py  # Investment data generator
│   ├── config/           # Configuration management
│   │   └── config_manager.py   # Configuration handling
│   ├── validation/       # Validation framework
│   │   └── validation_framework.py  # Data validation
│   ├── utils/           # Utilities
│   │   └── logging_config.py   # Logging setup
│   └── main.py          # Entry point
├── tests/               # Test suite
│   ├── test_generators/ # Generator tests
│   ├── test_utils.py   # Test utilities
│   └── test_suite.py   # Test runner
├── data/
│   ├── configs/        # Configuration files
│   │   ├── settings.json        # Main settings
│   │   └── industry_ranges.json # Industry data
│   └── generated/      # Output directory
└── docs/               # Documentation
```

## Usage

### Basic Usage

```python
from src.generators import InvestmentDataGenerator

# Initialize generator
generator = InvestmentDataGenerator()

# Generate dataset
data = generator.generate(size=1000)

# Save to file
generator.save_dataset(data, format='csv')
```

### Command Line Interface

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

### Configuration

The tool uses JSON configuration files in `data/configs/`:
- `settings.json`: Main configuration file
- `industry_ranges.json`: Industry-specific salary ranges

Example configuration:
```json
{
    "random_seed": 42,
    "fraud_scenarios": {
        "probability": 0.18,
        "salary_misreporting": {
            "probability": 0.5,
            "min_ratio": 0.7,
            "max_ratio": 0.9
        }
    }
}
```

## Testing

Run the test suite:
```bash
# Run all tests with detailed output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_generators/test_investment_generator.py

# Run with coverage report
python -m pytest --cov=src tests/
```

## Development Roadmap

### In Progress
- Vehicle FIPE data generator
- Transaction history generator
- Asset ownership randomizer

### Planned Features
- Year-specific data generation
- Dependent count generation
- Submission date handling
- Additional fraud patterns

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```
4. Make your changes and add tests
5. Run tests and linting:
```bash
python -m pytest tests/
python -m black src/ tests/
python -m flake8 src/ tests/
```
6. Commit your changes
7. Push to the branch
8. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Industry data ranges based on market research
- Fraud patterns developed with domain expertise
- Test suite inspired by pytest best practices
