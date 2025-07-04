"""
Utilities for interacting with the MongoDB database.
"""
import os
from pymongo import MongoClient
from typing import List, Dict, Optional
from . import config

# --- Load Environment Variables ---
MONGO_URI = os.getenv("MONGODB_URI")

# --- Global Client ---
# Use a global client to avoid reconnecting on every function call
try:
    if not MONGO_URI:
        raise ConnectionError("MONGODB_URI not found in .env file.")
    
    client = MongoClient(MONGO_URI)
    # Test the connection
    client.admin.command('ping')
    print("MongoDB connection successful.")
    
    # Set the default database
    db = client.convfinqa

except Exception as e:
    print(f"Fatal: Could not connect to MongoDB upon application start. {e}")
    client = None
    db = None

def get_record_by_id(record_id: str, collection_name: str = "parent_docs") -> Optional[Dict]:
    """
    Retrieves a single record from the specified collection by its ID.
    """
    if db is None:
        print("Error: No database connection available.")
        return None
        
    try:
        collection = db[collection_name]
        record = collection.find_one({"id": record_id})
        return record
    except Exception as e:
        print(f"Error retrieving record from MongoDB: {e}")
        return None

def bulk_insert_data(documents: List[Dict], collection_name: str, clear_collection: bool = True) -> bool:
    """
    Inserts a list of documents into a specified collection, with an option to clear it first.
    """
    if db is None:
        print("Error: No database connection available.")
        return False
    
    try:
        collection = db[collection_name]
        if clear_collection:
            collection.delete_many({})
            print(f"Cleared existing documents in '{collection_name}'.")
        
        if not documents:
            print("No documents to insert.")
            return True

        collection.insert_many(documents)
        print(f"Successfully inserted {len(documents)} documents into '{collection_name}'.")
        return True
    except Exception as e:
        print(f"Error during MongoDB insertion: {e}")
        return False
