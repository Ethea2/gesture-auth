from mode_selection import ModeSelectionDialog
from gesture_app import GestureRecognitionApp

def main():
    print("Starting Hand Gesture Recognition System...")
    
    # Show mode selection dialog
    mode_dialog = ModeSelectionDialog()
    selected_mode = mode_dialog.show()
    
    if selected_mode:
        print(f"Selected mode: {selected_mode}")
        
        # Create and run the main application
        app = GestureRecognitionApp(mode=selected_mode)
        app.run()
    else:
        print("No mode selected. Exiting.")

if __name__ == "__main__":
    main()
