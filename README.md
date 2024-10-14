# Recommendation System with PySpark and Flask

This project is a recommendation system web application that uses PySpark's ALS (Alternating Least Squares) model for generating user-item recommendations. The application allows users to request item recommendations by providing their user ID. The model is pre-trained and loaded into the Flask web application.

## Features

- **User-specific Recommendations:** Enter a user ID to receive personalized item recommendations.
- **Pre-trained Model:** Uses a pre-trained ALS model for quick and efficient predictions.
- **Flask Web Application:** Simple and intuitive web interface for interacting with the recommendation engine.
- **PySpark Integration:** Spark is used for loading the model and running the recommendation engine.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- Flask (`pip install flask`)
- PySpark (`pip install pyspark`)
- Hadoop (for `winutils.exe` if running on Windows)

## Project Structure

```plaintext
recommendation-system/ 
│ 
├── recommandation.py # Main Flask application 
├── templates/ 
│ └── index.html # Frontend template for user input 
├── static/ 
│ └── style.css # Optional: CSS for styling the web page
├── als_model/ # Pre-trained ALS model (contains item factors, user factors, and metadata) 
├── README.md # Project documentation └── requirements.txt # Python dependencies
```
## Installation

**Clone the repository**:
```bash
git clone https://github.com/your-repo/recommendation-system
cd recommendation-system
```

**Configure Spark and Hadoop (for Windows users)**:
```bash
set HADOOP_HOME=C:\path\to\hadoop
set PATH=%PATH%;%HADOOP_HOME%\bin
```

**Run the Flask application**:
```bash
python recommandation.py
```
