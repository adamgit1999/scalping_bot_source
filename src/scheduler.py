"""
Cron-style or interval scheduler for auto-restarting/running the bot.
"""
import threading
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import pytz
from src.config import Config
from src.notifications import NotificationManager

def schedule_job(interval_seconds, job_fn):
    def loop():
        while True:
            time.sleep(interval_seconds)
            job_fn()
    t = threading.Thread(target=loop, daemon=True)
    t.start()

class Scheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.notification_manager = NotificationManager()
        self.jobs = {}
        self.config = Config.load_config()
    
    def start(self):
        """Start the scheduler"""
        if not self.scheduler.running:
            self.scheduler.start()
            self.notification_manager.send_notification(
                "Scheduler started",
                "Trading scheduler is now running"
            )
    
    def stop(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            self.notification_manager.send_notification(
                "Scheduler stopped",
                "Trading scheduler has been stopped"
            )
    
    def add_job(self, job_id, func, trigger_type, **trigger_args):
        """
        Add a new scheduled job
        
        Args:
            job_id (str): Unique identifier for the job
            func (callable): Function to execute
            trigger_type (str): 'cron' or 'interval'
            **trigger_args: Trigger-specific arguments
        """
        try:
            # Remove existing job if it exists
            if job_id in self.jobs:
                self.remove_job(job_id)
            
            # Create trigger
            if trigger_type == 'cron':
                trigger = CronTrigger(**trigger_args)
            elif trigger_type == 'interval':
                trigger = IntervalTrigger(**trigger_args)
            else:
                raise ValueError(f"Unsupported trigger type: {trigger_type}")
            
            # Add job to scheduler
            job = self.scheduler.add_job(
                func,
                trigger=trigger,
                id=job_id,
                replace_existing=True
            )
            
            self.jobs[job_id] = job
            
            self.notification_manager.send_notification(
                "Job scheduled",
                f"New job '{job_id}' has been scheduled"
            )
            
            return job
            
        except Exception as e:
            self.notification_manager.send_notification(
                "Scheduling error",
                f"Failed to schedule job '{job_id}': {str(e)}",
                level='error'
            )
            raise
    
    def remove_job(self, job_id):
        """Remove a scheduled job"""
        try:
            if job_id in self.jobs:
                self.scheduler.remove_job(job_id)
                del self.jobs[job_id]
                
                self.notification_manager.send_notification(
                    "Job removed",
                    f"Job '{job_id}' has been removed from schedule"
                )
                
        except Exception as e:
            self.notification_manager.send_notification(
                "Scheduling error",
                f"Failed to remove job '{job_id}': {str(e)}",
                level='error'
            )
            raise
    
    def get_jobs(self):
        """Get list of all scheduled jobs"""
        return [
            {
                'id': job.id,
                'next_run': job.next_run_time,
                'trigger': str(job.trigger)
            }
            for job in self.jobs.values()
        ]
    
    def pause_job(self, job_id):
        """Pause a scheduled job"""
        try:
            if job_id in self.jobs:
                self.jobs[job_id].pause()
                self.notification_manager.send_notification(
                    "Job paused",
                    f"Job '{job_id}' has been paused"
                )
                
        except Exception as e:
            self.notification_manager.send_notification(
                "Scheduling error",
                f"Failed to pause job '{job_id}': {str(e)}",
                level='error'
            )
            raise
    
    def resume_job(self, job_id):
        """Resume a paused job"""
        try:
            if job_id in self.jobs:
                self.jobs[job_id].resume()
                self.notification_manager.send_notification(
                    "Job resumed",
                    f"Job '{job_id}' has been resumed"
                )
                
        except Exception as e:
            self.notification_manager.send_notification(
                "Scheduling error",
                f"Failed to resume job '{job_id}': {str(e)}",
                level='error'
            )
            raise
    
    def modify_job(self, job_id, **job_args):
        """Modify an existing job's parameters"""
        try:
            if job_id in self.jobs:
                self.scheduler.modify_job(job_id, **job_args)
                self.notification_manager.send_notification(
                    "Job modified",
                    f"Job '{job_id}' has been modified"
                )
                
        except Exception as e:
            self.notification_manager.send_notification(
                "Scheduling error",
                f"Failed to modify job '{job_id}': {str(e)}",
                level='error'
            )
            raise

