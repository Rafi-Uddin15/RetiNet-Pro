import os

def setup_kaggle_and_download():
    # 1. Setup Kaggle Config
    # Check if kaggle.json exists in the current directory
    if os.path.exists('kaggle.json'):
        print("Found kaggle.json.")
        
        # Create the hidden directory .kaggle if it doesn't exist
        # In Colab, this is usually /root/.kaggle
        # On Windows local, it's C:\Users\Username\.kaggle
        
        kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
        os.makedirs(kaggle_dir, exist_ok=True)
        
        # Move/Copy the file
        import shutil
        target_path = os.path.join(kaggle_dir, 'kaggle.json')
        shutil.copy('kaggle.json', target_path)
        
        # Set permissions (important for Linux/Colab)
        if os.name == 'posix':
            os.chmod(target_path, 0o600)
            
        print(f"Kaggle API key moved to {target_path}")
    else:
        print("WARNING: 'kaggle.json' not found in current directory.")
        print("Please download it from your Kaggle account settings and place it here.")
        return

    # 2. Download Dataset
    print("Downloading dataset...")
    # Using the 'os.system' to run the kaggle command line tool
    # Make sure 'kaggle' is installed: pip install kaggle
    exit_code = os.system('kaggle datasets download -d sovitrath/diabetic-retinopathy-224x224-gaussian-filtered-limit')
    
    if exit_code == 0:
        print("Download complete. Unzipping...")
        os.system('unzip -q diabetic-retinopathy-224x224-gaussian-filtered-limit.zip -d dataset')
        print("Dataset ready in 'dataset/' folder.")
    else:
        print("Download failed. Please check your Kaggle API key and internet connection.")

if __name__ == "__main__":
    setup_kaggle_and_download()
