
import db_manager as db
import narrative_logic as logic
import pandas as pd
import os

def test_node_growth():
    print("\n--- Testing Node Growth (The Evolution) ---")
    counts = db.get_connection_counts()
    print(f"Connection Counts Sample: {list(counts.items())[:5] if counts else 'None'}")
    
    # Check if graph html generation uses these counts (indirectly)
    # We can't easily check internal variables, but we can check if it runs without error
    try:
        html = logic.generate_graph_html()
        print(f"Graph HTML generated successfully. Length: {len(html)}")
    except Exception as e:
        print(f"Graph Generation Error: {e}")

def test_daily_summary_data():
    print("\n--- Testing Weekly Summary Data (The Mirror) ---")
    logs = db.get_logs_last_7_days()
    print(f"Logs from last 7 days: {len(logs)}")
    if len(logs) > 0:
        print(f"Sample: {logs[0]['content'][:30]}...")

def test_pulse_data():
    print("\n--- Testing Activity Pulse Data ---")
    try:
        df = logic.get_activity_pulse()
        if not df.empty:
            print("Pulse Data Shape:", df.shape)
            print(df.head())
        else:
            print("Pulse Data is empty (No relevant logs)")
    except Exception as e:
        print(f"Pulse Data Error: {e}")

if __name__ == "__main__":
    test_node_growth()
    test_daily_summary_data()
    test_pulse_data()
