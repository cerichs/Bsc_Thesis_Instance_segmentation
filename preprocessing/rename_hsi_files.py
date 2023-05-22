import os
import shutil
        
def rename_files(PATH):
    base_path = PATH  # Test, Training or Validation path
    directory_name = os.path.basename(base_path)

    to_move = []
    
    for entry in os.listdir(base_path):
        full_entry_path = os.path.join(base_path, entry)
        # Check if the entry is a directory before proceeding
        if os.path.isdir(full_entry_path):
            for files in os.listdir(full_entry_path):
                if files.endswith("npy"):
                    # Only rename files that do not contain "sub" and "Multiplied"
                    if "sub" not in files and "Multiplied" not in files:
                        # Avoid duplication of grain type in filename
                        new_filename = (directory_name + "_" + entry + "_" + files) if directory_name not in entry else (entry + "_" + files)
                        old_path = os.path.join(base_path, entry, files)
                        new_path = os.path.join(base_path, entry, new_filename)
                        os.rename(old_path, new_path)
                        # Add the new path (with new filename) to the move list
                        to_move.append((new_path, os.path.join(base_path, new_filename)))
                    else:
                        # If the file is not renamed, add the original path to the move list
                        to_move.append((os.path.join(base_path, entry, files), os.path.join(base_path, files)))
                
    # Now move the files
    for old_path, new_path in to_move:
        shutil.move(old_path, new_path)
                
    # Remove the now empty directories
    for entry in os.listdir(base_path):
        if entry!="windows":
            full_entry_path = os.path.join(base_path, entry)
            if os.path.isdir(full_entry_path):
                os.rmdir(full_entry_path)

    # Create new directories
    os.makedirs(os.path.join(base_path, "windows", "masks"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "windows", "PLS_eval_img"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "windows", "PLS_eval_img_rgb"), exist_ok=True)
