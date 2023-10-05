import os
# Define the directory path you want to work with
directory_path = r'D:\Users\jacob\Project\Data\val\Labels'

# Check if the directory exists
if not os.path.exists(directory_path):
    print(f"The directory '{directory_path}' does not exist.")
else:
    # Iterate through the files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file name contains spaces
        if ' ' in filename and (filename[-3:]== 'txt' or filename[-3:]== 'png'):
            # Create the new file name by replacing spaces with underscores
            new_filename = filename.replace(' ', '_')
            
            # Construct the full paths for the old and new file names
            old_filepath = os.path.join(directory_path, filename)
            new_filepath = os.path.join(directory_path, new_filename)
            
            try:
                # Rename the file
                os.rename(old_filepath, new_filepath)
                print(f"Renamed '{filename}' to '{new_filename}'.")
            except Exception as e:
                print(f"Failed to rename '{filename}' to '{new_filename}': {str(e)}")