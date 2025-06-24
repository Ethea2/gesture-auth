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
        self.window = None
        
    def show(self):
        """Open access logger window to display recognition history with learning tracking"""
        self.window = tk.Toplevel(self.parent)
        self.window.title("🧠 Access Logger & Learning Tracker")
        self.window.geometry("1000x600")
        
        # Create main frame with tabs
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Access Logs Tab
        access_frame = ttk.Frame(notebook)
        notebook.add(access_frame, text="Access Logs")
        self.create_access_logs_tab(access_frame)
        
        # Learning Stats Tab
        learning_frame = ttk.Frame(notebook)
        notebook.add(learning_frame, text="🧠 Learning Statistics")
        self.create_learning_stats_tab(learning_frame)
        
    def create_access_logs_tab(self, parent):
        """Create the access logs tab"""
        # Create treeview with notes column
        columns = ("timestamp", "username", "confidence", "notes")
        self.tree = ttk.Treeview(parent, columns=columns, show="headings")
        
        # Define headings
        self.tree.heading("timestamp", text="Timestamp")
        self.tree.heading("username", text="Username")
        self.tree.heading("confidence", text="Confidence")
        self.tree.heading("notes", text="Notes")
        
        # Set column widths
        self.tree.column("timestamp", width=180)
        self.tree.column("username", width=150)
        self.tree.column("confidence", width=100)
        self.tree.column("notes", width=250)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(expand=True, fill=tk.BOTH, padx=(0, 10), pady=(0, 10))
        
        # Filter frame
        filter_frame = tk.Frame(parent)
        filter_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(filter_frame, text="Filter by username:").pack(side=tk.LEFT, padx=5)
        self.username_var = tk.StringVar()
        username_entry = tk.Entry(filter_frame, textvariable=self.username_var)
        username_entry.pack(side=tk.LEFT, padx=5)
        
        tk.Label(filter_frame, text="Type:").pack(side=tk.LEFT, padx=(20, 5))
        self.log_type_var = tk.StringVar()
        type_combo = ttk.Combobox(filter_frame, textvariable=self.log_type_var, width=18)
        type_combo['values'] = ('All', 'Recognition', 'Learning (Positive)', 'Learning (Corrective)', 'Learning (Negative)')
        type_combo.set('All')
        type_combo.pack(side=tk.LEFT, padx=5)
        
        # Date filtering
        tk.Label(filter_frame, text="Date range:").pack(side=tk.LEFT, padx=(20, 5))
        self.from_date_var = tk.StringVar()
        from_date_entry = tk.Entry(filter_frame, textvariable=self.from_date_var, width=10)
        from_date_entry.pack(side=tk.LEFT, padx=5)
        from_date_entry.insert(0, "YYYY-MM-DD")
        
        tk.Label(filter_frame, text="to").pack(side=tk.LEFT)
        self.to_date_var = tk.StringVar()
        to_date_entry = tk.Entry(filter_frame, textvariable=self.to_date_var, width=10)
        to_date_entry.pack(side=tk.LEFT, padx=5)
        to_date_entry.insert(0, "YYYY-MM-DD")
        
        # Button frame
        button_frame = tk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        filter_button = tk.Button(button_frame, text="Apply Filters", command=self.apply_filters,
                                bg="#2196F3", fg="white", padx=10)
        filter_button.pack(side=tk.LEFT, padx=5)
        
        refresh_button = tk.Button(button_frame, text="Refresh", command=self.refresh_logs,
                                bg="#4CAF50", fg="white", padx=10)
        refresh_button.pack(side=tk.LEFT, padx=5)
        
        export_button = tk.Button(button_frame, text="Export to CSV", command=self.export_logs_to_csv,
                                bg="#FF9800", fg="white", padx=10)
        export_button.pack(side=tk.LEFT, padx=5)
        
        clear_button = tk.Button(button_frame, text="Clear All Logs", command=self.clear_logs,
                               bg="#f44336", fg="white", padx=10)
        clear_button.pack(side=tk.RIGHT, padx=5)
        
        # Initialize filter variables
        self.current_filter_query = None
        self.current_filter_params = None
        
        # Load logs initially
        self.refresh_logs()

    def create_learning_stats_tab(self, parent):
        """Create the learning statistics tab"""
        # Main frame with scrollbar
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Title
        title_label = tk.Label(scrollable_frame, text="🧠 Learning System Statistics", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Stats container
        stats_container = tk.Frame(scrollable_frame)
        stats_container.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Get and display statistics
        self.display_learning_statistics(stats_container)
        
        # Refresh button for stats
        refresh_stats_button = tk.Button(scrollable_frame, text="🔄 Refresh Statistics", 
                                        command=lambda: self.display_learning_statistics(stats_container),
                                        bg="#4CAF50", fg="white", padx=20, pady=5,
                                        font=("Arial", 12))
        refresh_stats_button.pack(pady=20)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def display_learning_statistics(self, parent):
        """Display comprehensive learning statistics"""
        # Clear existing widgets
        for widget in parent.winfo_children():
            widget.destroy()
        
        cursor = self.conn.cursor()
        
        # Basic statistics
        basic_frame = tk.LabelFrame(parent, text="📊 Basic Statistics", font=("Arial", 12, "bold"))
        basic_frame.pack(fill=tk.X, pady=10)
        
        # Total recognitions
        cursor.execute("SELECT COUNT(*) FROM access_logs")
        total_recognitions = cursor.fetchone()[0]
        
        # Learning samples
        cursor.execute("SELECT COUNT(*) FROM access_logs WHERE notes LIKE '%Learning sample%'")
        learning_samples = cursor.fetchone()[0]
        
        # Positive vs corrective vs negative learning
        cursor.execute("SELECT COUNT(*) FROM access_logs WHERE notes LIKE '%positive%'")
        positive_samples = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM access_logs WHERE notes LIKE '%corrective%'")
        corrective_samples = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM access_logs WHERE notes LIKE '%negative%' OR notes LIKE '%unauthorized%'")
        negative_samples = cursor.fetchone()[0]
        
        # User statistics
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM users")
        total_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM gesture_samples")
        total_training_samples = cursor.fetchone()[0]
        
        basic_stats = [
            f"Total Recognition Attempts: {total_recognitions}",
            f"Learning Samples Added: {learning_samples}",
            f"  ├─ Positive Confirmations: {positive_samples}",
            f"  ├─ Corrective Samples: {corrective_samples}",
            f"  └─ Negative/Unauthorized: {negative_samples}",
            f"Registered Users: {total_users}",
            f"Total Training Samples: {total_training_samples}",
            f"Avg Samples per User: {total_training_samples/max(1, total_users):.1f}"
        ]
        
        for stat in basic_stats:
            tk.Label(basic_frame, text=stat, font=("Arial", 10), anchor="w").pack(anchor="w", padx=10, pady=2)
        
        # User-specific learning statistics
        user_frame = tk.LabelFrame(parent, text="👥 Per-User Learning Statistics", font=("Arial", 12, "bold"))
        user_frame.pack(fill=tk.X, pady=10)
        
        # Create treeview for user stats
        user_columns = ("username", "total_samples", "positive_learning", "corrective_learning", "last_activity")
        user_tree = ttk.Treeview(user_frame, columns=user_columns, show="headings", height=8)
        
        user_tree.heading("username", text="Username")
        user_tree.heading("total_samples", text="Total Samples")
        user_tree.heading("positive_learning", text="Positive Learning")
        user_tree.heading("corrective_learning", text="Corrective Learning")
        user_tree.heading("last_activity", text="Last Activity")
        
        user_tree.column("username", width=120)
        user_tree.column("total_samples", width=100)
        user_tree.column("positive_learning", width=120)
        user_tree.column("corrective_learning", width=120)
        user_tree.column("last_activity", width=150)
        
        # Get user statistics
        cursor.execute("""
            SELECT u.username, 
                   COUNT(gs.id) as total_samples,
                   COUNT(CASE WHEN al.notes LIKE '%positive%' THEN 1 END) as positive_learning,
                   COUNT(CASE WHEN al.notes LIKE '%corrective%' THEN 1 END) as corrective_learning,
                   MAX(al.timestamp) as last_activity
            FROM users u
            LEFT JOIN gesture_samples gs ON u.user_id = gs.user_id
            LEFT JOIN access_logs al ON u.user_id = al.user_id
            GROUP BY u.user_id, u.username
            ORDER BY u.username
        """)
        
        user_stats = cursor.fetchall()
        for stat in user_stats:
            username, total_samples, positive, corrective, last_activity = stat
            last_activity = last_activity if last_activity else "Never"
            user_tree.insert("", tk.END, values=(username, total_samples, positive, corrective, last_activity))
        
        user_tree.pack(fill=tk.X, padx=10, pady=10)
        
        # Learning trends
        trends_frame = tk.LabelFrame(parent, text="📈 Learning Trends (Last 30 Days)", font=("Arial", 12, "bold"))
        trends_frame.pack(fill=tk.X, pady=10)
        
        cursor.execute("""
            SELECT DATE(timestamp) as date, 
                   COUNT(CASE WHEN notes LIKE '%positive%' THEN 1 END) as positive,
                   COUNT(CASE WHEN notes LIKE '%corrective%' THEN 1 END) as corrective,
                   COUNT(CASE WHEN notes LIKE '%negative%' OR notes LIKE '%unauthorized%' THEN 1 END) as negative
            FROM access_logs 
            WHERE timestamp >= date('now', '-30 days') 
            AND notes LIKE '%Learning sample%'
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 10
        """)
        
        trends_data = cursor.fetchall()
        if trends_data:
            tk.Label(trends_frame, text="Recent Learning Activity:", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)
            for date, positive, corrective, negative in trends_data:
                trend_text = f"  {date}: {positive} positive, {corrective} corrective, {negative} negative"
                tk.Label(trends_frame, text=trend_text, font=("Arial", 9)).pack(anchor="w", padx=20, pady=1)
        else:
            tk.Label(trends_frame, text="No recent learning activity found.", 
                    font=("Arial", 10), fg="gray").pack(anchor="w", padx=10, pady=10)
        
        # Model performance insights
        performance_frame = tk.LabelFrame(parent, text="🎯 Model Performance Insights", font=("Arial", 12, "bold"))
        performance_frame.pack(fill=tk.X, pady=10)
        
        # Calculate learning ratio
        if total_recognitions > 0:
            learning_ratio = (learning_samples / total_recognitions) * 100
            if corrective_samples > 0:
                accuracy_estimate = (positive_samples / (positive_samples + corrective_samples)) * 100
            else:
                accuracy_estimate = 100 if positive_samples > 0 else 0
        else:
            learning_ratio = 0
            accuracy_estimate = 0
        
        performance_stats = [
            f"Learning Engagement: {learning_ratio:.1f}% of recognitions provide feedback",
            f"Estimated Accuracy: {accuracy_estimate:.1f}% (based on user feedback)",
            f"Learning Progress: {learning_samples} samples added through feedback",
        ]
        
        if corrective_samples > positive_samples:
            performance_stats.append("⚠️ High corrective feedback - consider collecting more training data")
        elif learning_samples < 10:
            performance_stats.append("💡 Tip: More learning feedback will improve accuracy")
        else:
            performance_stats.append("✅ Good learning feedback ratio!")
        
        for stat in performance_stats:
            color = "red" if stat.startswith("⚠️") else "blue" if stat.startswith("💡") else "green" if stat.startswith("✅") else "black"
            tk.Label(performance_frame, text=stat, font=("Arial", 10), anchor="w", fg=color).pack(anchor="w", padx=10, pady=2)

    def refresh_logs(self):
        """Refresh the access logs display"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Get logs from database
        cursor = self.conn.cursor()
        
        # Check if notes column exists, if not add it
        try:
            cursor.execute("SELECT timestamp, username, confidence, notes FROM access_logs ORDER BY timestamp DESC")
        except:
            # Notes column doesn't exist, add it
            try:
                cursor.execute("ALTER TABLE access_logs ADD COLUMN notes TEXT")
                self.conn.commit()
            except:
                pass  # Column might already exist
            cursor.execute("SELECT timestamp, username, confidence, COALESCE(notes, '') as notes FROM access_logs ORDER BY timestamp DESC")
        
        logs = cursor.fetchall()
        
        # Insert logs into treeview with color coding
        for log in logs:
            timestamp, username, confidence, notes = log
            confidence_str = f"{confidence:.2f}" if confidence > 0 else "N/A"
            notes_str = notes if notes else ""
            
            item = self.tree.insert("", tk.END, values=(timestamp, username, confidence_str, notes_str))
            
            # Color code based on notes
            if "positive" in notes_str.lower():
                self.tree.set(item, "notes", "🟢 " + notes_str)
            elif "corrective" in notes_str.lower():
                self.tree.set(item, "notes", "🔴 " + notes_str)
            elif "negative" in notes_str.lower() or "unauthorized" in notes_str.lower():
                self.tree.set(item, "notes", "🚫 " + notes_str)
            elif "learning" in notes_str.lower():
                self.tree.set(item, "notes", "🧠 " + notes_str)

    def apply_filters(self):
        """Apply filters to the access logs"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Construct query with filters
        query = "SELECT timestamp, username, confidence, COALESCE(notes, '') as notes FROM access_logs WHERE 1=1"
        params = []
        
        if self.username_var.get():
            query += " AND username LIKE ?"
            params.append(f"%{self.username_var.get()}%")
        
        # Filter by log type
        log_type = self.log_type_var.get()
        if log_type == "Recognition":
            query += " AND (notes IS NULL OR notes = '' OR NOT notes LIKE '%Learning sample%')"
        elif log_type == "Learning (Positive)":
            query += " AND notes LIKE '%positive%'"
        elif log_type == "Learning (Corrective)":
            query += " AND notes LIKE '%corrective%'"
        elif log_type == "Learning (Negative)":
            query += " AND (notes LIKE '%negative%' OR notes LIKE '%unauthorized%')"
        
        from_date = self.from_date_var.get()
        if from_date and from_date != "YYYY-MM-DD":
            query += " AND date(timestamp) >= ?"
            params.append(from_date)
            
        to_date = self.to_date_var.get()
        if to_date and to_date != "YYYY-MM-DD":
            query += " AND date(timestamp) <= ?"
            params.append(to_date)
            
        query += " ORDER BY timestamp DESC"
        
        # Execute query
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        logs = cursor.fetchall()
        
        # Insert filtered logs into treeview
        for log in logs:
            timestamp, username, confidence, notes = log
            confidence_str = f"{confidence:.2f}" if confidence > 0 else "N/A"
            notes_str = notes if notes else ""
            
            item = self.tree.insert("", tk.END, values=(timestamp, username, confidence_str, notes_str))
            
            # Color code based on notes
            if "positive" in notes_str.lower():
                self.tree.set(item, "notes", "🟢 " + notes_str)
            elif "corrective" in notes_str.lower():
                self.tree.set(item, "notes", "🔴 " + notes_str)
            elif "negative" in notes_str.lower() or "unauthorized" in notes_str.lower():
                self.tree.set(item, "notes", "🚫 " + notes_str)
            elif "learning" in notes_str.lower():
                self.tree.set(item, "notes", "🧠 " + notes_str)
        
        # Save the current filter for export
        self.current_filter_query = query
        self.current_filter_params = params

    def clear_logs(self):
        """Clear all access logs after confirmation"""
        result = messagebox.askyesno("Clear Logs", 
                                   "Are you sure you want to clear ALL access logs?\n\nThis action cannot be undone!")
        if result:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM access_logs")
            self.conn.commit()
            self.refresh_logs()
            messagebox.showinfo("Logs Cleared", "All access logs have been cleared.")

    def export_logs_to_csv(self):
        """Export access logs to a CSV file"""
        # Ask for file location
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"access_logs_with_learning_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if not filename:
            return
        
        # Get logs from database using current filter if provided
        cursor = self.conn.cursor()
        
        if self.current_filter_query and self.current_filter_params:
            cursor.execute(self.current_filter_query, self.current_filter_params)
        else:
            cursor.execute(
                "SELECT timestamp, username, confidence, COALESCE(notes, '') as notes FROM access_logs ORDER BY timestamp DESC"
            )
        
        logs = cursor.fetchall()
        
        # Write to CSV
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Timestamp", "Username", "Confidence", "Notes"])
                for log in logs:
                    writer.writerow(log)
                    
            messagebox.showinfo("Export Successful", f"Logs exported to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting logs: {str(e)}")

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