import glob
import csv
import os

# Specify the folder containing your CSV files
folder_path = r"C:\Users\Li_mi\Desktop\materials"
new_path = r"C:\Users\Li_mi\Desktop\materials new"

# Use glob to find all CSV files in the folder
for filepath in glob.glob(os.path.join(folder_path, "*.csv")):
    data = []

    # Read the CSV file
    with open(filepath, 'r', newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            # Convert values to float for proper numeric sorting
            row['wl'] = float(row['wl'])
            row['n'] = float(row['n'])
            row['k'] = float(row['k'])
            data.append(row)
    
    # Sort by wl in descending order
    data.sort(key=lambda x: x['wl'], reverse=True)
    
    # Prepare output filename
    filename = os.path.basename(filepath)  # e.g. "data.csv"
    base, ext = os.path.splitext(filename) # e.g. "data", ".csv"
    outname = os.path.join(new_path, f"{base}.csv")
    
    # Write the transformed data
    with open(outname, 'w', newline='') as outfile:
        # First block: wl,n
        outfile.write("wl,n\n")
        for row in data:
            outfile.write(f"{row['wl']},{row['n']}\n")
        
        # Blank line separator
        outfile.write("\n")
        
        # Second block: wl,k
        outfile.write("wl,k\n")
        for row in data:
            outfile.write(f"{row['wl']},{row['k']}\n")
    
    print(f"Processed: {filepath} -> {outname}")
