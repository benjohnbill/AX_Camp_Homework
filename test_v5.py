
import db_manager as db
import narrative_logic as logic
import pandas as pd
import os

def test_debt_mechanics():
    print("--- Testing Debt Mechanics ---")
    # Reset to known state
    db.reset_debt()
    print(f"Initial Debt: {db.get_debt_count()}")
    
    db.increment_debt(5)
    print(f"After Increment(5): {db.get_debt_count()}")
    
    db.decrement_debt(2)
    print(f"After Decrement(2): {db.get_debt_count()}")
    
    db.decrement_debt(10)
    print(f"After Decrement(10) [Should be 0]: {db.get_debt_count()}")
    
    db.reset_debt()

def test_analytics_logic():
    print("\n--- Testing Analytics Logic ---")
    # Mock data if DB is empty, but we can try calling the function
    try:
        df_heatmap = logic.get_temporal_patterns()
        print("Heatmap Data Shape:", df_heatmap.shape if not df_heatmap.empty else "Empty")
        if not df_heatmap.empty:
            print(df_heatmap.head())
            
        df_trend = logic.get_daily_apology_trend()
        print("Trend Data Shape:", df_trend.shape if not df_trend.empty else "Empty")
        if not df_trend.empty:
            print(df_trend.head())
            
    except Exception as e:
        print(f"Analytics Error: {e}")

if __name__ == "__main__":
    test_debt_mechanics()
    test_analytics_logic()
