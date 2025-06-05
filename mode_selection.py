import tkinter as tk

class ModeSelectionDialog:
    def __init__(self):
        self.mode = None
        self.root = tk.Tk()
        self.root.title("Hand Gesture Recognition - Mode Selection")
        self.root.geometry("400x300")
        self.root.transient()
        self.root.grab_set()
        
        # Center the window
        self.root.geometry("+%d+%d" % (
            (self.root.winfo_screenwidth() // 2) - 200,
            (self.root.winfo_screenheight() // 2) - 150
        ))
        
        self.create_ui()
        
    def create_ui(self):
        # Title
        title_label = tk.Label(self.root, text="Hand Gesture Recognition System", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Description
        desc_label = tk.Label(self.root, 
                             text="Choose your mode:",
                             font=("Arial", 12))
        desc_label.pack(pady=10)
        
        # Training mode button
        train_frame = tk.Frame(self.root)
        train_frame.pack(pady=15, padx=20, fill=tk.X)
        
        train_btn = tk.Button(train_frame, text="Training Mode", 
                             command=self.select_training,
                             font=("Arial", 14), bg="#4CAF50", fg="white", 
                             padx=30, pady=10)
        train_btn.pack(fill=tk.X)
        
        train_desc = tk.Label(train_frame, 
                             text="Register new users and manage the gesture database",
                             font=("Arial", 10), fg="gray")
        train_desc.pack(pady=(5, 0))
        
        # Recognition mode button
        recog_frame = tk.Frame(self.root)
        recog_frame.pack(pady=15, padx=20, fill=tk.X)
        
        recog_btn = tk.Button(recog_frame, text="Recognition Mode", 
                             command=self.select_recognition,
                             font=("Arial", 14), bg="#2196F3", fg="white", 
                             padx=30, pady=10)
        recog_btn.pack(fill=tk.X)
        
        recog_desc = tk.Label(recog_frame, 
                             text="Continuous gesture recognition for access control",
                             font=("Arial", 10), fg="gray")
        recog_desc.pack(pady=(5, 0))
        
        # Exit button
        exit_btn = tk.Button(self.root, text="Exit", command=self.root.quit,
                            font=("Arial", 12), bg="#f44336", fg="white", padx=20)
        exit_btn.pack(pady=20)
        
    def select_training(self):
        self.mode = "training"
        self.root.destroy()
        
    def select_recognition(self):
        self.mode = "recognition"
        self.root.destroy()
        
    def show(self):
        self.root.mainloop()
        return self.mode