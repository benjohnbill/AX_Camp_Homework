import sqlite3
import numpy as np
import os

DB_PATH = "data/narrative.db"
DIM = 1536

def migrate_to_float32():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    print(f"Connecting to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Get all logs with embeddings
        cursor.execute("SELECT id, embedding FROM logs WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()
        
        migrated_count = 0
        skipped_count = 0
        
        print(f"Found {len(rows)} logs with embeddings. Checking types...")

        for log_id, blob in rows:
            if not blob:
                continue
                
            # Try float32 first (already migrated?)
            try:
                arr_f32 = np.frombuffer(blob, dtype=np.float32)
                if arr_f32.size == DIM:
                    skipped_count += 1
                    continue
            except:
                pass

            # Try float64 (needs migration)
            try:
                arr_f64 = np.frombuffer(blob, dtype=np.float64)
                if arr_f64.size == DIM:
                    # Convert to float32
                    arr_f32 = arr_f64.astype(np.float32)
                    new_blob = arr_f32.tobytes()
                    
                    cursor.execute("UPDATE logs SET embedding = ? WHERE id = ?", (new_blob, log_id))
                    migrated_count += 1
                else:
                    print(f"Warning: Log {log_id} has embedding of size {arr_f64.size} (expected {DIM})")
            except Exception as e:
                print(f"Error processing log {log_id}: {e}")

        conn.commit()
        print(f"Migration complete.\nMigrated: {migrated_count}\nSkipped (already float32): {skipped_count}")

    except Exception as e:
        print(f"Critical error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_to_float32()
