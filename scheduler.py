from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
import sys
import os

# Ensure we can import from current directory
sys.path.append(os.path.dirname(__file__))

try:
    from inference import run_pipeline
except ImportError:
    print("❌ Could not import run_pipeline from inference.py")
    def run_pipeline():
        print("Placeholder pipeline")

def start_scheduler():
    scheduler = BlockingScheduler()
    
    # Schedule to run at 11:00 AM and 5:00 PM (17:00) as requested
    scheduler.add_job(run_pipeline, 'cron', hour=11, minute=0)
    scheduler.add_job(run_pipeline, 'cron', hour=17, minute=0)
    
    print(f"[{datetime.now()}] Scheduler started. Jobs scheduled at 00:00, 06:00, 12:00, 18:00.")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")

if __name__ == "__main__":
    # Run once immediately for testing/initialization?
    # run_pipeline() 
    start_scheduler()
