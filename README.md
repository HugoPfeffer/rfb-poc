# RFB-POC: Data Science Learning Project

## Project Overview
A proof of concept project for learning SVD analysis and data processing.

## Project Structure
```
├── data/              # Data files
├── src/               # Source code
│   ├── data_processing/  # Scripts for data processing
│   ├── models/          # Model implementations
│   └── utils/           # Utility functions
└── tests/             # Test files
```

## Setup and Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd rfb-poc
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

## Usage

The project contains:
- Data generation utilities in `src/data_processing/`
- SVD analysis implementation in `src/models/`
- Tests in `tests/`

### Running Tests
```bash
pytest tests/
```
