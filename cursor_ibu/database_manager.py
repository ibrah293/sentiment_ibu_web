#!/usr/bin/env python3
"""
Database Manager for Sentiment Analysis Project
Provides utilities for managing the SQLite database.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from database import db
import os

class DatabaseManager:
    def __init__(self, db_path="sentiment_analysis.db"):
        self.db_path = db_path
    
    def get_database_size(self):
        """Get the size of the database file"""
        if os.path.exists(self.db_path):
            size_bytes = os.path.getsize(self.db_path)
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
        return 0
    
    def get_table_info(self):
        """Get information about all tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            table_info = {}
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                table_info[table_name] = {
                    'count': count,
                    'columns': [col[1] for col in columns]
                }
            
            return table_info
    
    def view_recent_analyses(self, limit=10, analysis_type='both'):
        """View recent analyses with detailed information"""
        if analysis_type == 'sentiment':
            query = f"""
                SELECT id, text, sentiment, confidence, model_type, timestamp
                FROM sentiment_analysis 
                ORDER BY timestamp DESC 
                LIMIT {limit}
            """
        elif analysis_type == 'emotion':
            query = f"""
                SELECT id, text, emotion, confidence, model_type, timestamp
                FROM emotion_analysis 
                ORDER BY timestamp DESC 
                LIMIT {limit}
            """
        else:
            # Combine both
            query = f"""
                SELECT id, text, sentiment as prediction, confidence, model_type, timestamp, 'sentiment' as type
                FROM sentiment_analysis 
                UNION ALL
                SELECT id, text, emotion as prediction, confidence, model_type, timestamp, 'emotion' as type
                FROM emotion_analysis 
                ORDER BY timestamp DESC 
                LIMIT {limit}
            """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
            return df
    
    def get_model_performance_summary(self):
        """Get summary of model performance"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT model_type, analysis_type, 
                       AVG(accuracy) as avg_accuracy,
                       AVG(precision) as avg_precision,
                       AVG(recall) as avg_recall,
                       AVG(f1_score) as avg_f1_score,
                       COUNT(*) as test_count,
                       MAX(test_date) as last_test
                FROM model_performance 
                GROUP BY model_type, analysis_type
                ORDER BY avg_accuracy DESC
            """
            df = pd.read_sql_query(query, conn)
            return df
    
    def clean_old_data(self, days=30):
        """Clean data older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count records to be deleted
            cursor.execute("""
                SELECT COUNT(*) FROM sentiment_analysis 
                WHERE timestamp < ?
            """, (cutoff_date,))
            sentiment_count = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM emotion_analysis 
                WHERE timestamp < ?
            """, (cutoff_date,))
            emotion_count = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM batch_analysis 
                WHERE timestamp < ?
            """, (cutoff_date,))
            batch_count = cursor.fetchone()[0]
            
            # Delete old records
            cursor.execute("""
                DELETE FROM sentiment_analysis 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            cursor.execute("""
                DELETE FROM emotion_analysis 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            cursor.execute("""
                DELETE FROM batch_analysis 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            conn.commit()
            
            return {
                'sentiment_deleted': sentiment_count,
                'emotion_deleted': emotion_count,
                'batch_deleted': batch_count,
                'total_deleted': sentiment_count + emotion_count + batch_count
            }
    
    def backup_database(self, backup_path=None):
        """Create a backup of the database"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"sentiment_analysis_backup_{timestamp}.db"
        
        with sqlite3.connect(self.db_path) as source_conn:
            with sqlite3.connect(backup_path) as backup_conn:
                source_conn.backup(backup_conn)
        
        return backup_path
    
    def get_analysis_trends(self, days=7):
        """Get analysis trends over the last N days"""
        start_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Sentiment trends
            sentiment_query = """
                SELECT DATE(timestamp) as date, 
                       sentiment, 
                       COUNT(*) as count,
                       AVG(confidence) as avg_confidence
                FROM sentiment_analysis 
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp), sentiment
                ORDER BY date DESC
            """
            sentiment_df = pd.read_sql_query(sentiment_query, conn, params=(start_date,))
            
            # Emotion trends
            emotion_query = """
                SELECT DATE(timestamp) as date, 
                       emotion, 
                       COUNT(*) as count,
                       AVG(confidence) as avg_confidence
                FROM emotion_analysis 
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp), emotion
                ORDER BY date DESC
            """
            emotion_df = pd.read_sql_query(emotion_query, conn, params=(start_date,))
            
            return {
                'sentiment_trends': sentiment_df,
                'emotion_trends': emotion_df
            }
    
    def get_model_usage_stats(self):
        """Get statistics about model usage"""
        with sqlite3.connect(self.db_path) as conn:
            # Sentiment model usage
            sentiment_query = """
                SELECT model_type, 
                       COUNT(*) as usage_count,
                       AVG(confidence) as avg_confidence
                FROM sentiment_analysis 
                GROUP BY model_type
                ORDER BY usage_count DESC
            """
            sentiment_df = pd.read_sql_query(sentiment_query, conn)
            
            # Emotion model usage
            emotion_query = """
                SELECT model_type, 
                       COUNT(*) as usage_count,
                       AVG(confidence) as avg_confidence
                FROM emotion_analysis 
                GROUP BY model_type
                ORDER BY usage_count DESC
            """
            emotion_df = pd.read_sql_query(emotion_query, conn)
            
            return {
                'sentiment_models': sentiment_df,
                'emotion_models': emotion_df
            }

def main():
    """Main function for database management"""
    manager = DatabaseManager()
    
    print("🗄️ Database Manager for Sentiment Analysis")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. View database information")
        print("2. View recent analyses")
        print("3. View model performance")
        print("4. View analysis trends")
        print("5. View model usage statistics")
        print("6. Clean old data")
        print("7. Backup database")
        print("8. Export data to CSV")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-8): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            print("\n📊 Database Information:")
            print(f"Database file: {manager.db_path}")
            print(f"Database size: {manager.get_database_size():.2f} MB")
            
            table_info = manager.get_table_info()
            print(f"\nTables:")
            for table_name, info in table_info.items():
                print(f"  {table_name}: {info['count']} records")
                print(f"    Columns: {', '.join(info['columns'])}")
        
        elif choice == "2":
            limit = int(input("How many records to show? (default 10): ") or "10")
            analysis_type = input("Analysis type (sentiment/emotion/both, default both): ") or "both"
            
            df = manager.view_recent_analyses(limit, analysis_type)
            print(f"\n📋 Recent Analyses ({len(df)} records):")
            print(df.to_string(index=False))
        
        elif choice == "3":
            df = manager.get_model_performance_summary()
            print("\n📈 Model Performance Summary:")
            print(df.to_string(index=False))
        
        elif choice == "4":
            days = int(input("Number of days to analyze (default 7): ") or "7")
            trends = manager.get_analysis_trends(days)
            
            print(f"\n📊 Sentiment Trends (last {days} days):")
            print(trends['sentiment_trends'].to_string(index=False))
            
            print(f"\n📊 Emotion Trends (last {days} days):")
            print(trends['emotion_trends'].to_string(index=False))
        
        elif choice == "5":
            stats = manager.get_model_usage_stats()
            
            print("\n🤖 Sentiment Model Usage:")
            print(stats['sentiment_models'].to_string(index=False))
            
            print("\n🤖 Emotion Model Usage:")
            print(stats['emotion_models'].to_string(index=False))
        
        elif choice == "6":
            days = int(input("Delete data older than how many days? (default 30): ") or "30")
            confirm = input(f"Are you sure you want to delete data older than {days} days? (y/n): ").strip().lower()
            
            if confirm == 'y':
                result = manager.clean_old_data(days)
                print(f"✅ Cleaned {result['total_deleted']} records:")
                print(f"  - Sentiment analyses: {result['sentiment_deleted']}")
                print(f"  - Emotion analyses: {result['emotion_deleted']}")
                print(f"  - Batch analyses: {result['batch_deleted']}")
            else:
                print("❌ Operation cancelled")
        
        elif choice == "7":
            backup_path = input("Backup path (default: auto-generated): ").strip() or None
            try:
                backup_file = manager.backup_database(backup_path)
                print(f"✅ Database backed up to: {backup_file}")
            except Exception as e:
                print(f"❌ Backup failed: {e}")
        
        elif choice == "8":
            filename = input("Export filename (default: export.csv): ").strip() or "export.csv"
            try:
                exported_file = db.export_to_csv(filename)
                print(f"✅ Data exported to: {exported_file}")
            except Exception as e:
                print(f"❌ Export failed: {e}")
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 