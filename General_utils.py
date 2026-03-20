import csv
import nibabel as nib
import numpy as np
import pandas as pd
import os
import sys
from subprocess import run
import shlex
from colors import bcolors
from itertools import product
from pathlib import Path
import shutil
import glob
import random
import tarfile
import io
from PIL import Image,ImageDraw, ImageFont

def read_tsv(file_path):
    """
    Reads a TSV file and prints each row.
    
    :param file_path: Path to the TSV file
    """
    values = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                #print(row)  # row is a list of columns
                values.append(row[0])
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
    return values

import re

def get_actual_count(txt_path: str) -> int:
    """
    Read a text file and return the integer value from
    the line 'actual count in file: <number>'.

    Parameters
    ----------
    txt_path : str
        Path to the text file

    Returns
    -------
    int
        Actual count found in the file

    Raises
    ------
    ValueError
        If the actual count entry is not found
    """
    with open(txt_path, "r") as f:
        for line in f:
            match = re.search(r"actual\s+count\s+in\s+file\s*:\s*(\d+)", line, re.IGNORECASE)
            if match:
                return int(match.group(1))

    raise ValueError("actual count in file entry not found")

def read_one_column_tsv( filepath ) :
    outlist = []
    with open( filepath, "r" ) as f :
        for line in f :
            line = line.strip()
            if line :
                outlist.append( line )

    return outlist


def return_voxel_stats(filepath,stats):
    """
    Reads a stats .txt file and returns count * mean.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    headers = lines[0].split()
    values = lines[1].split()
    values.remove('[')
    values.remove(']')
    if stats=='count':
        count = float(values[headers.index("count")])
        return count 
    else:
        return
    
def mrstats_custom(infile,outfile,stats,verbose=False):
    if os.path.exists(infile):
        data = nib.load(infile).get_fdata()
        mask = data > 0
        data_dict = {}
        if data[mask].size > 0 :
            data_dict['count']  = data[mask].size
            data_dict['sum']    = np.sum(data[mask])
            data_dict['mean']   = np.mean(data[mask])
            data_dict['median'] = np.median(data[mask])
            data_dict['min']    = np.amin(data[mask])
            data_dict['max']    = np.amax(data[mask])
            
        else:
            data_dict['count']  = 0
            data_dict['sum']    = 0
            data_dict['mean']   = 0
            data_dict['median'] = 0
            data_dict['min']    = 0
            data_dict['max']    = 0

        df = pd.DataFrame([data_dict])
        df.to_csv(outfile, index=False)

        if verbose == True:
            print(df)

        if stats   == 'count':
            return data_dict['count']
        elif stats =='sum':
            return data_dict['sum']
        elif stats =='mean':
            return data_dict['mean']
        elif stats =='median':
            return data_dict['median']
        elif stats =='min':
            return data_dict['min']
        elif stats =='max':
            return data_dict['max']
    else:
        return np.nan
    
def close_logs():
    sys.stdout.close()
    sys.stderr.close()


""" def mesh_smooth_folder(input_folder,output_folder,avoid1,avoid2):
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        path_dico     = f'/neurospin/dico/'
        path_MRtrix   = f'/home_local/cm283129/mrtrix3/MRtrix3.sif'
        nifti_files = [f for f in os.listdir(input_folder) if f.endswith('.nii.gz') and avoid1 not in f and avoid2 not in f] 
        for nifti in nifti_files:
            run(shlex.split(f'singularity run -B {path_dico} {path_MRtrix} voxel2mesh {input_folder}/{nifti} {output_folder}/{nifti.split('.nii.gz')[0]}.obj -blocky -force -nthreads 20'),stderr=sys.stderr,stdout=sys.stdout)
            run(shlex.split(f'singularity run -B {path_dico} {path_MRtrix} meshfilter {output_folder}/{nifti.split('.nii.gz')[0]}.obj smooth {output_folder}/{nifti.split('.nii.gz')[0]}_smoothed.obj -force -nthreads 20'),stderr=sys.stderr,stdout=sys.stdout)
        print(f'{bcolors.GREEN}OK: Meshed and smoothed folder of nifti files{bcolors.RESET}')
    except:
        print(f'{bcolors.RED}Fail: Meshed and smoothed folder of nifti files{bcolors.RESET}') """

def read_val_from_txt(file_path):
    with open(file_path, "r") as file:
        content = file.read().strip()
    return float(content)


def read_one_column_tsv( filepath ) :
    outlist = []
    with open( filepath, "r" ) as f :
        for line in f :
            line = line.strip()
            if line :
                outlist.append( line )

    return outlist


def compute_dice_voxel(density_1, density_2):
    """
    Compute the overlap (dice coefficient) between two
    density maps (or binary).

    Parameters
    ----------
    density_1: ndarray
        Density (or binary) map computed from the first bundle
    density_2: ndarray
        Density (or binary) map computed from the second bundle

    Returns
    -------
    A tuple containing:

    - float: Value between 0 and 1 that represent the spatial aggrement
        between both bundles.
    - float: Value between 0 and 1 that represent the spatial aggrement
        between both bundles, weighted by streamlines density.
    """
    overlap_idx = np.nonzero(density_1 * density_2)
    numerator = 2 * len(overlap_idx[0])
    denominator = np.count_nonzero(density_1) + np.count_nonzero(density_2)

    if denominator > 0:
        dice = numerator / float(denominator)
    else:
        dice = np.nan

    overlap_1 = density_1[overlap_idx]
    overlap_2 = density_2[overlap_idx]
    w_dice = np.sum(overlap_1) + np.sum(overlap_2)
    denominator = np.sum(density_1) + np.sum(density_2)
    if denominator > 0:
        w_dice /= denominator
    else:
        w_dice = np.nan

    return dice


def compute_correlation(density_1, density_2):
    """
    Compute the overlap (dice coefficient) between two density
    maps (or binary). Correlation being less robust to extreme
    case (no overlap, identical array), a lot of check a needed to prevent NaN.
    Parameters
    ----------
    density_1: ndarray
        Density (or binary) map computed from the first bundle
    density_2: ndarray
        Density (or binary) map computed from the second bundle
    Returns
    -------
    float: Value between 0 and 1 that represent the spatial aggrement
        between both bundles taking into account density.
    """
    indices = np.where(density_1 + density_2 > 0)
    if np.array_equal(density_1, density_2):
        density_correlation = 1
    elif (np.sum(density_1) > 0 and np.sum(density_2) > 0) \
            and np.count_nonzero(density_1 * density_2):
        density_correlation = np.corrcoef(density_1[indices],
                                          density_2[indices])[0, 1]
    else:
        density_correlation = 0

    return max(0, density_correlation)

def overlaps_to_matrix(data, fill_diagonal=1.0, symmetric=True, csv_path=None):
    """
    Convert a list of (region1, region2, overlap) tuples into a square DataFrame
    and optionally save it as a CSV file.

    Parameters:
    - data: list of tuples, each (region1, region2, overlap)
    - fill_diagonal: value to fill on the diagonal (self-overlap)
    - symmetric: if True, mirror values across the diagonal
    - csv_path: str or None. If provided, saves the DataFrame to this path.

    Returns:
    - pandas.DataFrame: square matrix regions x regions
    """
    # Step 1: Create DataFrame
    df = pd.DataFrame(data, columns=['regions1', 'regions2', 'overlap'])
    
    # Step 2: Get all unique regions
    regions = sorted(set(df['regions1']).union(df['regions2']))
    
    # Step 3: Create empty square matrix
    matrix = pd.DataFrame(0, index=regions, columns=regions)
    
    # Step 4: Fill values
    for r1, r2, val in data:
        matrix.loc[r1, r2] = val
        if symmetric:
            matrix.loc[r2, r1] = val
    
    # Step 5: Fill diagonal
    for r in regions:
        matrix.loc[r, r] = fill_diagonal
    
    # Step 6: Save to CSV if path is provided
    if csv_path is not None:
        matrix.to_csv(csv_path)
    
    return matrix



def is_intra_inter_bundle(
    end1_info,
    end2_info,
    mode,
    region,
    threshold1=90.0,
    threshold2=90.0,
    verbose=False):
    """
    Check whether a bundle is intra-bundle based on overlap information.

    Parameters
    ----------
    end1_info : dict
        Dict with keys like (something, region) and numeric values.
    end2_info : dict
        Same structure as end1_info.
    region : str
        Region name to check.
    threshold : float, optional
        Overlap threshold (default: 90.0).
    verbose : bool, optional
        Print debug information.

    Returns
    -------
    bool
        True if intra-bundle condition is met, False otherwise.
    """

    if mode == 'Intra-ROI':
        # Filter by threshold
        end1_filtered = {k: v for k, v in end1_info.items() if v > threshold1}
        end2_filtered = {k: v for k, v in end2_info.items() if v > threshold2}
        
        # Extract regions
        regions_1 = [k[1] for k in end1_filtered.keys()]
        regions_2 = [k[1] for k in end2_filtered.keys()]

        #print('end1_filtered_keys',end1_filtered).keys()

        if region in regions_1 and region in regions_2:
            #print('Found Intra-ROI bundle, continue...')
            return True
        else:
            return False


    elif mode == 'Inter-ROI':
        # Filter by threshold
        end1_filtered_tmp = {k: v for k, v in end1_info.items() if v > threshold1}
        end2_filtered_tmp = {k: v for k, v in end2_info.items() if v > threshold2}
    
        # Extract regions
        regions_1 = [k[1] for k in end1_filtered_tmp.keys()]
        regions_2 = [k[1] for k in end2_filtered_tmp.keys()]

        if region in regions_1 and region in regions_2:
            print(region,regions_1,regions_2)
            print(f'{bcolors.RED}ERROR bundle should have been labeled as Intra-ROI already{bcolors.RESET}')
            return False
        elif region in regions_1 or region in regions_2:
            #print('Found Inter-ROI bundle, continue...')
            return True
        

def edit_assignments(input_csv, output_csv):
    with open(input_csv, newline='', encoding='utf-8') as f_in:
        reader = csv.reader(f_in)
        rows = list(reader)

    header = rows[0]
    data = rows[1:]
    print(data)
    for row in data:
        # Asumimos que los assignments están en la primera columna
        valores = row[0].strip().split()
        #print(valores)
        if len(valores) == 2:
            v1, v2 = valores
            if (v1 == '1' and v2 == '1') or (v1 == '2' and v2 == '2'):
                row[0] = '0 0'  
        
    print(data)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out,lineterminator='\n')
        writer.writerow(header)
        writer.writerows(data) 
    
def edit_assignments_pd(input_csv, output_csv):
    # Read CSV
    df = pd.read_csv(input_csv, dtype=str)

    # Name of the first column
    col = df.columns[0]

    # Split the values
    vals = df[col].str.strip().str.split(expand=True)

    # Replace (1 1) and (2 2) with (0 0)
    mask = ((vals[0] == '1') & (vals[1] == '1')) | \
           ((vals[0] == '2') & (vals[1] == '2'))

    df.loc[mask, col] = '0 0'

    # Write CSV (this WILL add a final \n — expected)
    df.to_csv(output_csv, index=False)

def edit_assignments_mrtrix(input_csv, output_csv):
    # Read the file as strings (important to preserve spaces)
    df = pd.read_csv(input_csv, dtype=str)
    #print('df\n',df)
    # Name of the first column (assignments)
    col = df.columns[0]
    #print('col\n',col)
    # Split each row into two columns for comparison
    vals = df[col].str.strip().str.split(expand=True)
    #print('df[col]\n',df[col])
    #print('df[col].str.strip()\n',df[col].str.strip().str.split(expand=True).shape)
    #print('vals\n',vals)
    # Create a mask for rows that are "1 1" or "2 2"
    mask = ((vals[0] == '1') & (vals[1] == '1')) | ((vals[0] == '2') & (vals[1] == '2'))
    #print('mask\n',mask)
    # Replace those rows with "0 0"
    df.loc[mask, col] = '0 0'
    #print(df.shape)
    # Convert DataFrame to CSV string without index and without extra blank lines
    csv_str = df.to_csv(index=False)

    # Strip **only trailing newline characters** to avoid empty last line
    csv_str = csv_str.rstrip('\n')

    # Write to file (MRtrix-safe)
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write(csv_str)

def img2mesh(input_img,extension):
    
    if not os.path.exists(input_img):
        print('Img does not exists')
        return 
    
    try:
        p = Path(input_img)
        parent_dir = p.parent
        base = p.name.replace(".nii.gz", "")

        tmp_dir = f'{parent_dir}/{base}_tmp'
        if not os.path.exists(f'{tmp_dir}'):
            os.makedirs(f'{tmp_dir}')

        img_thresholded = f'{tmp_dir}/{base}_thresholded.nii.gz'
        mesh            = f'{tmp_dir}/{base}.{extension}'
        mesh_final      = f'{parent_dir}/{base}.{extension}'

        run(shlex.split(f'AimsThreshold -i {input_img} -o {img_thresholded} -b -m gt -t 0 --fg 1'))
        run(shlex.split(f'AimsMesh -i {img_thresholded} -o {mesh} --smooth True --smoothIt 50 --smoothType laplacian'))
        meshes = glob.glob(f"{tmp_dir}/*.{extension}")
        cmd = ["AimsZCat", "-i", *meshes, "-o", mesh_final]
        run(cmd, check=True)
        shutil.rmtree(Path(f'{tmp_dir}'))
    except:
        print(f'im2mesh exception... Return')
        shutil.rmtree(Path(f'{tmp_dir}'))
        return
    
def generate_random_color():
    while True:
        # Generate a random color with RGB values between 0 and 1
        color = [random.random() for _ in range(3)]
        
        # Check if the color is black, gray, or white
        # Black: [0, 0, 0], Gray: [x, x, x], White: [1, 1, 1]
        if not all(0.1 <= c <= 0.9 for c in color):  # Check if any value is too close to black or white
            continue  # Regenerate the color if it's black, gray, or white
        
        return color
    
def assign_colors_to_files(files):
    color_dict = {}
    
    # Create a list of files sorted into left and right groups
    left_files = {file: None for file in files if 'Left' in file or '_lh_' in file}
    right_files = {file: None for file in files if 'Right' in file or '_rh_' in file}
    Comm_files = {file: None for file in files if 'Comm' in file}

    for left in left_files:
        if 'Left' in left:
            right = left.replace('Left', 'Right')
        elif '_lh_' in left:
            right = left.replace('_lh_', '_rh_')

        if right in right_files:
            # If the symmetric right file exists, assign the same color to both
            color = generate_random_color()
            color_dict[left] = color
            color_dict[right] = color
            right_files.pop(right)  # Remove the right file after pairing
        else:
            # If no symmetric right file, assign a unique color to the left file
            color_dict[left] = generate_random_color()
    
    # For any remaining right files without a pair, assign them a unique color
    for right in right_files:
        color_dict[right] = generate_random_color()

    for comm in Comm_files:
        color_dict[comm] = generate_random_color()

    return color_dict


def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def chunks(lst, size=100):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def percentage(nums):
    if not nums:
        return 0.0
    return (sum(n > 10 for n in nums) / len(nums)) * 100


def remove_tck_files_safe(folder_path):
    # Ensure the provided folder path exists
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Error: The specified folder '{folder_path}' does not exist.")
        return

    # Ensure the folder is a directory
    if not folder_path.is_dir():
        print(f"Error: The specified path '{folder_path}' is not a directory.")
        return
    
    # Use glob to find all .tck files recursively in the folder and subfolders
    tck_files = glob.glob(str(folder_path / '**' / '*.tck'), recursive=True)

    if not tck_files:
        print("No .tck files found.")
        return


    # Iterate over the found files
    for file_path_str in tck_files:
        file_path = Path(file_path_str)
        try:
            # Check if the path is a file (to avoid directories or symlinks)
            if file_path.is_file():
                # Delete the file using unlink
                file_path.unlink()

        except Exception as e:
            print(f"Error removing {file_path}: {e}")

def log_exception(exception_message, filename="error_log.txt"):
    with open(filename, "a") as file:
        file.write(exception_message + "\n")

def check_file_in_tar(sub,tar_gz_path, filename='x'):
    """
    Opens a .tar.gz file, checks if a specific file ('x' by default) exists, 
    and prints its content if it does.
    """
    try:
        with tarfile.open(tar_gz_path, "r:gz") as tar:
            # Check if the specific file exists in the tar archive
            if filename in tar.getnames():
                return (sub,1)
            else:
                return (sub,0)
                
    except Exception as e:
        print(f"An error occurred: {e}")

def get_actual_count_from_folder(tar_gz_path,  encoding="utf-8"):
    """
    Searches all .txt files under a folder inside a tar.gz
    and extracts the integer after 'actual count:'.
    """
    pattern = re.compile(r"actual count in file: \s*(\d+)", re.IGNORECASE)

    # Ensure folder path ends with '/'
    #folder_path = folder_path.rstrip("/") + "/"
    data_dict = {}
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        for member in tar:
            
            if (
                member.isfile()
                #and member.name.startswith(folder_path)
                and member.name.endswith("_slcount_QC.txt")):
                #print(member)
                f = tar.extractfile(member)
                if not f:
                    continue

                with io.TextIOWrapper(f, encoding=encoding) as text_file:
                    for line in text_file:
                        match = pattern.search(line)
                        if match:
                            #print(f.name,int(match.group(1)))
                            data_dict[f.name] = int(match.group(1))
                            #return int(match.group(1))

    #raise ValueError("'actual count:' not found in folder")
    return data_dict


def write_one_column_tsv(filename,data):
    """
    Writes a list to a one-column TSV file.
    
    Parameters:
        data (list): List of values to write
        filename (str): Output TSV file path
    """
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(f"{item}\n")



def adjust_in_shape(config):
    dims=[]
    for idx in range(1, 4):
        dim = config.in_shape[idx]
        r = dim%(2**config.depth)
        if r!=0:
            dim+=(2**config.depth-r)
        dims.append(dim)
    return((1, dims[0]+4, dims[1], dims[2]))




