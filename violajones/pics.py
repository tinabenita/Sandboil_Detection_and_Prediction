import os

def extract_filenames(folder_path):
    # List to store file names without extension
    filenames = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image file (you can add more extensions if needed)
        if filename.endswith(('.jpg')):
            # Extract file name without extension
            name_without_extension = os.path.splitext(filename)[0]
            # Append to the list
            filenames.append(name_without_extension)

    return filenames

def save_to_text_file(filenames, output_file):
    with open(output_file, 'w') as f:
        for name in filenames:
            f.write(name + '\n')

if __name__ == "__main__":
    # Folder containing the images
    folder_path = 'D:/PROJECTS/Sandboil_Detection_and_Prediction/violajones/test-run/test-data'

    # Output text file path
    output_file = 'test-list.txt'

    # Extract filenames
    filenames = extract_filenames(folder_path)

    # Save filenames to a text file
    save_to_text_file(filenames, output_file)
    