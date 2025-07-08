import tkinter as tk
from tkinter import messagebox, filedialog
import tkinter.ttk as ttk
import csv
import datetime

class SettingsDialog:
    def __init__(self, parent, model_manager, gesture_processor):
        self.parent = parent
        self.model_manager = model_manager
        self.gesture_processor = gesture_processor
        
    def show(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.parent)
        settings_window.title("Recognition Settings")
        settings_window.geometry("400x400")
        
        # Confidence threshold setting
        tk.Label(settings_window, text="Confidence Threshold (0.0-1.0):", font=("Arial", 12)).pack(pady=(20, 5))
        confidence_slider = tk.Scale(settings_window, from_=0.0, to=1.0, resolution=0.01, 
                                    orient=tk.HORIZONTAL, length=300)
        confidence_slider.set(self.model_manager.confidence_threshold)
        confidence_slider.pack()
        
        # Minimum samples setting
        tk.Label(settings_window, text="Minimum Samples per User:", font=("Arial", 12)).pack(pady=(20, 5))
        samples_slider = tk.Scale(settings_window, from_=5, to=70, resolution=5, 
                                orient=tk.HORIZONTAL, length=300)
        samples_slider.set(self.model_manager.min_samples_per_user)
        samples_slider.pack()
        
        # Detection box size settings
        tk.Label(settings_window, text="Detection Box Width:", font=("Arial", 12)).pack(pady=(20, 5))
        width_slider = tk.Scale(settings_window, from_=100, to=400, resolution=10, 
                               orient=tk.HORIZONTAL, length=300)
        width_slider.set(self.gesture_processor.box_width)
        width_slider.pack()
        
        tk.Label(settings_window, text="Detection Box Height:", font=("Arial", 12)).pack(pady=(10, 5))
        height_slider = tk.Scale(settings_window, from_=100, to=400, resolution=10, 
                                orient=tk.HORIZONTAL, length=300)
        height_slider.set(self.gesture_processor.box_height)
        height_slider.pack()
        
        # Save button
        def save_settings():
            self.model_manager.update_settings(
                min_samples=samples_slider.get(),
                confidence_threshold=confidence_slider.get()
            )
            
            self.gesture_processor.update_box_size(
                width_slider.get(),
                height_slider.get()
            )
            
            messagebox.showinfo("Settings", "Settings saved successfully")
            settings_window.destroy()
            
        tk.Button(settings_window, text="Save", command=save_settings, 
                font=("Arial", 12), bg="#4CAF50", fg="white", padx=20).pack(pady=20)

class AccessLoggerDialog:
    def __init__(self, parent, conn):
        self.parent = parent
        self.conn = conn
        
    def show(self):
        """Show access logger window"""
        logger_window = tk.Toplevel(self.parent)
        logger_window.title("Access Logger")
        logger_window.geometry("800x500")
        
        # Create treeview
        columns = ("timestamp", "username", "confidence")
        tree = ttk.Treeview(logger_window, columns=columns, show="headings")
        
        # Define headings
        tree.heading("timestamp", text="Timestamp")
        tree.heading("username", text="Username")
        tree.heading("confidence", text="Confidence")
        
        # Set column widths
        tree.column("timestamp", width=200)
        tree.column("username", width=200)
        tree.column("confidence", width=100)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(logger_window, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Add refresh button
        def refresh_logs():
            # Clear existing items
            for item in tree.get_children():
                tree.delete(item)
            
            # Get logs from database
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT timestamp, username, confidence FROM access_logs ORDER BY timestamp DESC"
            )
            logs = cursor.fetchall()
            
            # Insert logs into treeview
            for log in logs:
                timestamp, username, confidence = log
                tree.insert("", tk.END, values=(timestamp, username, f"{confidence:.2f}"))
        
        # Add filter options
        filter_frame = tk.Frame(logger_window)
        filter_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(filter_frame, text="Filter by username:").pack(side=tk.LEFT, padx=5)
        username_var = tk.StringVar()
        username_entry = tk.Entry(filter_frame, textvariable=username_var)
        username_entry.pack(side=tk.LEFT, padx=5)
        
        # Date filtering
        tk.Label(filter_frame, text="Date range:").pack(side=tk.LEFT, padx=5)
        from_date_var = tk.StringVar()
        from_date_entry = tk.Entry(filter_frame, textvariable=from_date_var, width=10)
        from_date_entry.pack(side=tk.LEFT, padx=5)
        from_date_entry.insert(0, "YYYY-MM-DD")
        
        tk.Label(filter_frame, text="to").pack(side=tk.LEFT)
        to_date_var = tk.StringVar()
        to_date_entry = tk.Entry(filter_frame, textvariable=to_date_var, width=10)
        to_date_entry.pack(side=tk.LEFT, padx=5)
        to_date_entry.insert(0, "YYYY-MM-DD")
        
        def apply_filters():
            # Clear existing items
            for item in tree.get_children():
                tree.delete(item)
            
            # Construct query with filters
            query = "SELECT timestamp, username, confidence FROM access_logs WHERE 1=1"
            params = []
            
            if username_var.get():
                query += " AND username LIKE ?"
                params.append(f"%{username_var.get()}%")
            
            from_date = from_date_var.get()
            if from_date and from_date != "YYYY-MM-DD":
                query += " AND date(timestamp) >= ?"
                params.append(from_date)
                
            to_date = to_date_var.get()
            if to_date and to_date != "YYYY-MM-DD":
                query += " AND date(timestamp) <= ?"
                params.append(to_date)
                
            query += " ORDER BY timestamp DESC"
            
            # Execute query
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            logs = cursor.fetchall()
            
            # Insert logs into treeview
            for log in logs:
                timestamp, username, confidence = log
                tree.insert("", tk.END, values=(timestamp, username, f"{confidence:.2f}"))
            
            # Save the current filter for export
            self.current_filter_query = query
            self.current_filter_params = params
        
        def export_logs_to_csv():
            """Export access logs to a CSV file"""
            # Ask for file location
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"access_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            if not filename:
                return
            
            # Get logs from database using current filter if provided
            cursor = self.conn.cursor()
            
            if hasattr(self, 'current_filter_query') and self.current_filter_query:
                cursor.execute(self.current_filter_query, self.current_filter_params)
            else:
                cursor.execute(
                    "SELECT timestamp, username, confidence FROM access_logs ORDER BY timestamp DESC"
                )
            
            logs = cursor.fetchall()
            
            # Write to CSV
            try:
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Timestamp", "Username", "Confidence"])
                    for log in logs:
                        writer.writerow(log)
                        
                messagebox.showinfo("Export Successful", f"Logs exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting logs: {str(e)}")
        
        button_frame = tk.Frame(logger_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        filter_button = tk.Button(button_frame, text="Apply Filters", command=apply_filters,
                                bg="#2196F3", fg="white", padx=10)
        filter_button.pack(side=tk.LEFT, padx=5)
        
        refresh_button = tk.Button(button_frame, text="Refresh", command=refresh_logs,
                                bg="#4CAF50", fg="white", padx=10)
        refresh_button.pack(side=tk.LEFT, padx=5)
        
        export_button = tk.Button(button_frame, text="Export to CSV", command=export_logs_to_csv,
                                bg="#FF9800", fg="white", padx=10)
        export_button.pack(side=tk.LEFT, padx=5)
        
        # Initialize filter variables
        self.current_filter_query = None
        self.current_filter_params = None
        
        # Load logs initially
        refresh_logs()

class RegistrationControlPanel:
    def __init__(self, parent, training_manager):
        self.parent = parent
        self.training_manager = training_manager
        self.control_panel = None
        self.reg_status_label = None
        self.button_control_frame = None
        
    def create(self):
        """Create the bottom control panel for registration"""
        if self.control_panel:
            self.control_panel.destroy()
            
        self.control_panel = tk.Frame(self.parent, bg="#f0f0f0", relief=tk.RAISED, bd=1)
        
        # Registration status frame
        status_frame = tk.Frame(self.control_panel, bg="#f0f0f0")
        status_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y)
        
        self.reg_status_label = tk.Label(status_frame, text="Registration Status: Idle", 
                                        font=("Arial", 10), bg="#f0f0f0")
        self.reg_status_label.pack()
        
        # Control buttons frame (initially empty)
        self.button_control_frame = tk.Frame(self.control_panel, bg="#f0f0f0")
        self.button_control_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        
        return self.control_panel

    def update(self):
        """Update the registration control panel based on current state"""
        if not self.control_panel:
            return
            
        # Clear existing buttons
        for widget in self.button_control_frame.winfo_children():
            widget.destroy()
            
        if self.training_manager.recording:
            if self.training_manager.registration_paused:
                self.reg_status_label.config(text="Registration Status: PAUSED - No hand detected", fg="red")
                
                # Add continue and cancel buttons
                continue_btn = tk.Button(self.button_control_frame, text="Continue", 
                                       command=self.training_manager.continue_from_pause,
                                       font=("Arial", 10), bg="#4CAF50", fg="white", padx=15)
                continue_btn.pack(side=tk.LEFT, padx=5)
                
                cancel_btn = tk.Button(self.button_control_frame, text="Cancel Registration", 
                                     command=self.training_manager.cancel_registration,
                                     font=("Arial", 10), bg="#f44336", fg="white", padx=15)
                cancel_btn.pack(side=tk.LEFT, padx=5)
            else:
                rep_in_phase = (self.training_manager.samples_count % self.training_manager.repetitions_per_phase) + 1
                self.reg_status_label.config(text=f"Registration Status: Recording - Phase {self.training_manager.current_phase}/4, Rep {rep_in_phase}/{self.training_manager.repetitions_per_phase}", fg="green")
                
                # Add cancel button only
                cancel_btn = tk.Button(self.button_control_frame, text="Cancel Registration", 
                                     command=self.training_manager.cancel_registration,
                                     font=("Arial", 10), bg="#f44336", fg="white", padx=15)
                cancel_btn.pack(side=tk.LEFT, padx=5)
        else:
            self.reg_status_label.config(text="Registration Status: Idle", fg="black")
            # No buttons needed when not recording
