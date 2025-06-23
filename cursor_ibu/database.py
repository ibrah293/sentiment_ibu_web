import sqlite3
import json
import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

class SentimentDatabase:
    def __init__(self, db_path: str = "sentiment_analysis.db"):
        """
        Initialize the sentiment analysis database
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create sentiment_analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    processed_text TEXT,
                    sentiment TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    model_type TEXT NOT NULL,
                    probabilities TEXT,
                    analysis_type TEXT DEFAULT 'sentiment',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create emotion_analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS emotion_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    processed_text TEXT,
                    emotion TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    model_type TEXT NOT NULL,
                    probabilities TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create model_performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    test_date DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create batch_analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS batch_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    analysis_type TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def store_sentiment_analysis(self, text: str, processed_text: str, sentiment: str, 
                                confidence: float, model_type: str, probabilities: Dict[str, float]) -> int:
        """
        Store sentiment analysis result in database
        
        Returns:
            int: ID of the inserted record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sentiment_analysis 
                (text, processed_text, sentiment, confidence, model_type, probabilities)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (text, processed_text, sentiment, confidence, model_type, json.dumps(probabilities)))
            
            conn.commit()
            return cursor.lastrowid
    
    def store_emotion_analysis(self, text: str, processed_text: str, emotion: str, 
                              confidence: float, model_type: str, probabilities: Dict[str, float]) -> int:
        """
        Store emotion analysis result in database
        
        Returns:
            int: ID of the inserted record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO emotion_analysis 
                (text, processed_text, emotion, confidence, model_type, probabilities)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (text, processed_text, emotion, confidence, model_type, json.dumps(probabilities)))
            
            conn.commit()
            return cursor.lastrowid
    
    def store_model_performance(self, model_type: str, analysis_type: str, 
                               accuracy: float, precision: float, recall: float, f1_score: float):
        """Store model performance metrics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_performance 
                (model_type, analysis_type, accuracy, precision, recall, f1_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (model_type, analysis_type, accuracy, precision, recall, f1_score))
            
            conn.commit()
    
    def store_batch_analysis(self, batch_id: str, texts: List[str], predictions: List[str], 
                           confidences: List[float], analysis_type: str, model_type: str):
        """Store batch analysis results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for text, prediction, confidence in zip(texts, predictions, confidences):
                cursor.execute('''
                    INSERT INTO batch_analysis 
                    (batch_id, text, prediction, confidence, analysis_type, model_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (batch_id, text, prediction, confidence, analysis_type, model_type))
            
            conn.commit()
    
    def get_sentiment_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent sentiment analysis history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, text, sentiment, confidence, model_type, timestamp
                FROM sentiment_analysis 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_emotion_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent emotion analysis history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, text, emotion, confidence, model_type, timestamp
                FROM emotion_analysis 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_model_performance_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get model performance history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT model_type, analysis_type, accuracy, precision, recall, f1_score, test_date
                FROM model_performance 
                ORDER BY test_date DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get overall analysis statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Sentiment analysis stats
            cursor.execute('SELECT COUNT(*) FROM sentiment_analysis')
            sentiment_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(confidence) FROM sentiment_analysis')
            avg_sentiment_confidence = cursor.fetchone()[0] or 0
            
            # Emotion analysis stats
            cursor.execute('SELECT COUNT(*) FROM emotion_analysis')
            emotion_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(confidence) FROM emotion_analysis')
            avg_emotion_confidence = cursor.fetchone()[0] or 0
            
            # Most common sentiments
            cursor.execute('''
                SELECT sentiment, COUNT(*) as count 
                FROM sentiment_analysis 
                GROUP BY sentiment 
                ORDER BY count DESC
            ''')
            sentiment_distribution = dict(cursor.fetchall())
            
            # Most common emotions
            cursor.execute('''
                SELECT emotion, COUNT(*) as count 
                FROM emotion_analysis 
                GROUP BY emotion 
                ORDER BY count DESC
            ''')
            emotion_distribution = dict(cursor.fetchall())
            
            return {
                'total_sentiment_analyses': sentiment_count,
                'total_emotion_analyses': emotion_count,
                'avg_sentiment_confidence': round(avg_sentiment_confidence, 3),
                'avg_emotion_confidence': round(avg_emotion_confidence, 3),
                'sentiment_distribution': sentiment_distribution,
                'emotion_distribution': emotion_distribution
            }
    
    def search_analyses(self, search_term: str, analysis_type: str = 'both', limit: int = 50) -> List[Dict[str, Any]]:
        """Search for analyses containing specific text"""
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if analysis_type in ['sentiment', 'both']:
                cursor.execute('''
                    SELECT id, text, sentiment, confidence, model_type, timestamp
                    FROM sentiment_analysis 
                    WHERE text LIKE ? OR processed_text LIKE ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (f'%{search_term}%', f'%{search_term}%', limit))
                
                for row in cursor.fetchall():
                    results.append({
                        'id': row[0],
                        'text': row[1],
                        'prediction': row[2],
                        'confidence': row[3],
                        'model_type': row[4],
                        'timestamp': row[5],
                        'analysis_type': 'sentiment'
                    })
            
            if analysis_type in ['emotion', 'both']:
                cursor.execute('''
                    SELECT id, text, emotion, confidence, model_type, timestamp
                    FROM emotion_analysis 
                    WHERE text LIKE ? OR processed_text LIKE ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (f'%{search_term}%', f'%{search_term}%', limit))
                
                for row in cursor.fetchall():
                    results.append({
                        'id': row[0],
                        'text': row[1],
                        'prediction': row[2],
                        'confidence': row[3],
                        'model_type': row[4],
                        'timestamp': row[5],
                        'analysis_type': 'emotion'
                    })
        
        # Sort by timestamp and limit results
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        return results[:limit]
    
    def export_to_csv(self, filename: str = "sentiment_analysis_export.csv"):
        """Export all analysis data to CSV"""
        with sqlite3.connect(self.db_path) as conn:
            # Export sentiment analysis
            sentiment_df = pd.read_sql_query('''
                SELECT text, sentiment, confidence, model_type, timestamp
                FROM sentiment_analysis
                ORDER BY timestamp DESC
            ''', conn)
            
            # Export emotion analysis
            emotion_df = pd.read_sql_query('''
                SELECT text, emotion, confidence, model_type, timestamp
                FROM emotion_analysis
                ORDER BY timestamp DESC
            ''', conn)
            
            # Combine and export
            sentiment_df['analysis_type'] = 'sentiment'
            emotion_df['analysis_type'] = 'emotion'
            emotion_df = emotion_df.rename(columns={'emotion': 'sentiment'})
            
            combined_df = pd.concat([sentiment_df, emotion_df], ignore_index=True)
            combined_df.to_csv(filename, index=False)
            
            return filename
    
    def clear_old_data(self, days: int = 30):
        """Clear data older than specified days"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clear old sentiment analysis
            cursor.execute('''
                DELETE FROM sentiment_analysis 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days))
            
            # Clear old emotion analysis
            cursor.execute('''
                DELETE FROM emotion_analysis 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days))
            
            # Clear old batch analysis
            cursor.execute('''
                DELETE FROM batch_analysis 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days))
            
            conn.commit()

# Create a global database instance
db = SentimentDatabase()

if __name__ == "__main__":
    # Test the database
    print("Testing database functionality...")
    
    # Test storing sentiment analysis
    sentiment_id = db.store_sentiment_analysis(
        text="I love this product!",
        processed_text="love product",
        sentiment="positive",
        confidence=0.95,
        model_type="logistic_regression",
        probabilities={"positive": 0.95, "negative": 0.03, "neutral": 0.02}
    )
    print(f"Stored sentiment analysis with ID: {sentiment_id}")
    
    # Test storing emotion analysis
    emotion_id = db.store_emotion_analysis(
        text="I'm so happy today!",
        processed_text="happy today",
        emotion="joy",
        confidence=0.88,
        model_type="logistic_regression",
        probabilities={"joy": 0.88, "sadness": 0.05, "anger": 0.02, "fear": 0.01, "surprise": 0.02, "disgust": 0.01, "neutral": 0.01}
    )
    print(f"Stored emotion analysis with ID: {emotion_id}")
    
    # Test retrieving data
    sentiment_history = db.get_sentiment_history(5)
    print(f"Recent sentiment analyses: {len(sentiment_history)}")
    
    emotion_history = db.get_emotion_history(5)
    print(f"Recent emotion analyses: {len(emotion_history)}")
    
    # Test statistics
    stats = db.get_analysis_statistics()
    print(f"Database statistics: {stats}")
    
    print("Database test completed successfully!") 