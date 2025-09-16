# AI Project - Drone Trajectory Prediction

This project is about a machine learning system that predictes the drone trajectory (position + orintation) 
using the IMU sensor data which is: acceleration, gyroscope, and magnometer.

## Team Members

|   AC.NO   |         Name       |       Role        |                  Contributions                 |
|-----------|--------------------|-------------------|------------------------------------------------|
| 202270162 |   Laith Al-Jabri   |  Lead Developer   |   Project structure, core model development    |
| 202270158 |  Saleem Al-Soudi   | Feature Engineer  |    Data preprocessing, feature engineering     |
| 202270189 |   Osama Al-Jebzi   |    ML Engineer    |     Model optimization, evaluation metrics     |
| 202270305 |  Ayman Al-Dahmali  |   Data Analyst    | Exploratory Data Analysis (EDA), visualization |
| 202270130 | Mohammed Al-Wajeeh | Software Engineer |        Deployment, pipeline integration        |

## Installation and Setup

### Prerequisites
- Python 3.12 or higher
- UV package manager

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/EngLaith7/drone-trajectory-prediction.git
   cd drone-trajectory-prediction
   ```
2. Install dependencies using UV:
   ```bash
   uv sync
   ```

3. Run the project:
   ```bash
   uv run python main.py
   ```

## Project Structure

```
project-name/
├── README.md              # Project documentation
├── pyproject.toml         # UV project configuration
├── main.py               # Main application entry point
├── src/                  # Source code
│   ├── data/            # Data processing modules
│   ├── models/          # ML model implementations
│   └── utils/           # Utility functions
├── notebooks/           # Jupyter notebooks
├── data/               # Dataset files
└── docs/               # Additional documentation
```

## Usage

### Basic Usage
```python
from src.models.training import train_model
from src.data.clean_data import get_cleaned_data

# Load and preprocess data
df = get_cleaned_data(use_grid_search=True)

# Train model and Make predictions
train_model()
```
### Running Experiments
```bash
# Run the main application
uv run python main.py

# Run the training script
uv run python src/models/training.py

# Run Jupyter notebook
uv run jupyter notebook notebooks/EDA.ipynb
```
### Chat Applications
```bash
# First, train the model (if not already done)
uv run python main.py

# Web-based chat interface (Streamlit)
uv run streamlit run app.py
```
## Results

- **Model R2_score**: 0.8200
- **Training Time**: 69 minutes
- **Algorithm**: Random Forest Regressor
- **Key Findings**: 
  - Random Forest Regressor performed beter than linear regressor
  - Applying σ-rule (hyperparameterized) helps in filtering data for the model
  - Model suffer limitations becuase the using of data of single dorne flight
  
## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Commit changes: `git commit -m 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit a pull request 

## Acknowledgements

This project uses the **Drone Flight IMU Sensors Log Dataset** available on [Kaggle](https://www.kaggle.com/datasets/akkmit/drone-flight-imu-sensors-log), 
created and shared by [Akkmit](https://www.kaggle.com/akkmit).  

The dataset is licensed under the **Apache License 2.0**, and its terms apply to the use of the dataset.  
Our project code is licensed separately under the **MIT License** (see [LICENSE](./LICENSE) file).
