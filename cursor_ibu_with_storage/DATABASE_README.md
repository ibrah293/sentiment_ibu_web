# Database Functionality for Sentiment Analysis Project

This document explains the database functionality that has been added to the sentiment analysis project.

## Overview

The project now includes a SQLite database to store and manage sentiment analysis and emotion detection results. This provides:

- **Data Persistence**: All analysis results are stored for future reference
- **History Tracking**: View past analyses and their results
- **Statistics**: Get insights into analysis patterns and model performance
- **Search Functionality**: Find specific analyses by text content
- **Export Capabilities**: Export data to CSV for external analysis
- **Data Management**: Clean old data and create backups

## Database Structure

The database (`sentiment_analysis.db`) contains four main tables:

### 1. sentiment_analysis
Stores sentiment analysis results:
- `id`: Unique identifier
- `text`: Original input text
- `processed_text`: Preprocessed text
- `sentiment`: Predicted sentiment (positive/negative/neutral)
- `confidence`: Confidence score (0-1)
- `model_type`: Model used (logistic_regression/random_forest/naive_bayes)
- `probabilities`: JSON string of class probabilities
- `timestamp`: Analysis timestamp

### 2. emotion_analysis
Stores emotion detection results:
- `id`: Unique identifier
- `text`: Original input text
- `processed_text`: Preprocessed text
- `emotion`: Predicted emotion (joy/sadness/anger/fear/surprise/disgust/neutral)
- `confidence`: Confidence score (0-1)
- `model_type`: Model used
- `probabilities`: JSON string of class probabilities
- `timestamp`: Analysis timestamp

### 3. model_performance
Stores model evaluation metrics:
- `id`: Unique identifier
- `model_type`: Model name
- `analysis_type`: Type of analysis (sentiment/emotion)
- `accuracy`: Model accuracy
- `precision`: Model precision
- `recall`: Model recall
- `f1_score`: F1 score
- `test_date`: Evaluation timestamp

### 4. batch_analysis
Stores batch processing results:
- `id`: Unique identifier
- `batch_id`: Batch identifier
- `text`: Input text
- `prediction`: Model prediction
- `confidence`: Confidence score
- `analysis_type`: Type of analysis
- `model_type`: Model used
- `timestamp`: Processing timestamp

## Usage

### 1. Basic Database Operations

The database is automatically integrated into the Flask application. When you perform sentiment or emotion analysis through the web interface, results are automatically stored.

### 2. Testing Database Functionality

Run the test script to verify database functionality:

```bash
python test_database.py
```

This script will:
- Test storing sentiment and emotion analyses
- Add sample data
- Test retrieval functions
- Test statistics and search
- Test export functionality

### 3. Database Management

Use the database manager for advanced operations:

```bash
python database_manager.py
```

This provides an interactive menu for:
- Viewing database information
- Viewing recent analyses
- Viewing model performance
- Viewing analysis trends
- Cleaning old data
- Creating backups
- Exporting data

### 4. Programmatic Usage

You can also use the database directly in your Python code:

```python
from database import db

# Store sentiment analysis
sentiment_id = db.store_sentiment_analysis(
    text="I love this product!",
    processed_text="love product",
    sentiment="positive",
    confidence=0.95,
    model_type="logistic_regression",
    probabilities={"positive": 0.95, "negative": 0.03, "neutral": 0.02}
)

# Get recent history
history = db.get_sentiment_history(10)

# Get statistics
stats = db.get_analysis_statistics()

# Search for specific text
results = db.search_analyses("love", "both", 5)

# Export to CSV
db.export_to_csv("my_export.csv")
```

## API Endpoints

The Flask application includes new API endpoints for database operations:

### GET /api/history
Get analysis history:
- `type`: Analysis type (sentiment/emotion/both)
- `limit`: Number of records to return

### GET /api/statistics
Get database statistics

### GET /api/search
Search for analyses:
- `q`: Search term
- `type`: Analysis type (sentiment/emotion/both)
- `limit`: Number of results to return

### GET /api/export
Export data to CSV

## Web Interface

The web interface includes a new "Database & Analytics" section with tabs for:

1. **History**: View recent analyses with filtering options
2. **Statistics**: View overall statistics and distributions
3. **Search**: Search for specific text in analyses

## Data Management

### Cleaning Old Data

To clean data older than a specified number of days:

```python
from database import db
db.clear_old_data(days=30)  # Delete data older than 30 days
```

### Backup

To create a backup of the database:

```python
from database_manager import DatabaseManager
manager = DatabaseManager()
backup_file = manager.backup_database()
```

### Export

To export data to CSV:

```python
from database import db
exported_file = db.export_to_csv("my_export.csv")
```

## Performance Considerations

- The database uses SQLite, which is suitable for small to medium-scale applications
- For large-scale deployments, consider migrating to PostgreSQL or MySQL
- Regular cleanup of old data helps maintain performance
- Indexes are automatically created on frequently queried columns

## Troubleshooting

### Common Issues

1. **Database not found**: The database file is created automatically when first used
2. **Permission errors**: Ensure write permissions in the project directory
3. **Import errors**: Make sure all required packages are installed

### Dependencies

The database functionality requires:
- `sqlite3` (built into Python)
- `pandas` (for data export and management)
- `json` (for storing probabilities)

## Future Enhancements

Potential improvements for the database system:

1. **User Management**: Add user accounts and authentication
2. **Advanced Analytics**: Add more sophisticated reporting
3. **Real-time Monitoring**: Add real-time analysis tracking
4. **Data Visualization**: Add charts and graphs to the web interface
5. **API Rate Limiting**: Add rate limiting for API endpoints
6. **Data Encryption**: Add encryption for sensitive data

## Conclusion

The database functionality provides a solid foundation for storing and managing sentiment analysis data. It enables data persistence, historical analysis, and insights into model performance, making the sentiment analysis project more comprehensive and useful for real-world applications. 