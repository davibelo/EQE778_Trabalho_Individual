import os

def delete_files():
    # Define the files and extensions to delete
    log_files = [file for file in os.listdir('.') if file.endswith('.log')]

    # Delete all .log files
    for file in log_files:
        try:
            os.remove(file)
            print(f"Deleted {file}")
        except Exception as e:
            print(f"Failed to delete {file}: {e}")

if __name__ == '__main__':
    delete_files()
