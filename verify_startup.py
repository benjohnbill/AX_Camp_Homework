
import sys
import os

# Set up paths so we can import app modules
sys.path.append(os.getcwd())

print("--- Starting Verification ---")

try:
    print("1. Importing db_manager...")
    import db_manager as db
    print("   Success.")

    print("2. Importing narrative_logic...")
    import narrative_logic as logic
    print("   Success.")
    
    print("3. Checking DB functions...")
    if not hasattr(db, 'get_cores'):
        print("   ERROR: db.get_cores missing")
    else:
        print("   db.get_cores found.")
        
    print("4. Checking Logic functions...")
    if not hasattr(logic, 'check_streak_and_apply_penalty'):
        print("   ERROR: logic.check_streak_and_apply_penalty missing")
    else:
        print("   logic.check_streak_and_apply_penalty found.")
        
    print("5. Running Logic Diagnostics...")
    try:
        logic.run_startup_diagnostics()
        print("   Diagnostics ran successfully.")
    except Exception as e:
        print(f"   Diagnostics FAILED: {e}")

    print("6. Importing app (dry run)...")
    try:
        import app
        print("   app imported successfully.")
    except Exception as e:
        print(f"   app import FAILED: {e}")

    print("--- Verification Complete ---")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
