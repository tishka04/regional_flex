# Regional Flexibility Optimization Model

This project implements a multi-regional energy flexibility optimization model for French power grid regions. The model optimizes dispatchable technologies, storage, interregional exchanges, and demand response based on residual demand for the full year of 2022 with half-hourly time resolution.

## Project Structure

- `src/`: Source code for the optimization model
  - `data/`: Data processing and management
  - `model/`: Core optimization model
  - `utils/`: Utility functions
- `tests/`: Unit tests
- `data/`: Input and output data
- `config/`: Configuration files
- `notebooks/`: Jupyter notebooks for analysis

## Features

- Multi-regional energy flexibility optimization for the full year of 2022
- Half-hourly (30-minute) time resolution analysis
- Demand response modeling and optimization
- Energy dispatch optimization with regional constraints
- Inter-regional energy exchange modeling
- Storage optimization (charge/discharge cycles)
- Renewable energy integration (solar, wind) with seasonal variations
- Grid constraints analysis with slack variables
- Advanced visualization of seasonal and yearly patterns
- Memory-optimized processing for large datasets (17,520 time periods per year)
- Comprehensive data quality control and outlier detection

## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Usage

1. Configure parameters in `config/config.yaml`
2. Place input data in `data/raw/`
3. Run the optimization model:
   ```bash
   python src/main.py
   ```

## License

MIT License
