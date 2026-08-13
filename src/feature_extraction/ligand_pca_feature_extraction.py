from ase.io import read
import numpy as np  
import os
import csv  
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ase import Atoms

# === Define Paths (customize for your environment) ===
folder_path = 'your_ligand_trajectory_folder'
file_names = ['file1.pdb', 'file2.pdb', 'file3.pdb']  # Replace with your filenames
output_plot_folder = 'output_plot_folder_path'
csv_output_file = 'PCA_summary.csv'

os.makedirs(output_plot_folder, exist_ok=True)

# === Open CSV Writer ===
with open(csv_output_file, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["File", "Explained Variance Ratio", "PCA Components Shape", "Average Structure Shape"])

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        file_identifier = file_name.split('_')[-1].replace('.pdb', '')

        # Load MD trajectory
        trajectory = read(file_path, index=':')
        valid_frames = [atoms for i, atoms in enumerate(trajectory) if len(atoms) > 0 and i != 1001]

        if len(valid_frames) == 0:
            print(f"No valid frames found for {file_name}. Skipping.")
            continue

        num_valid_frames = len(valid_frames)
        num_atoms = len(valid_frames[0])
        data = np.zeros((num_valid_frames, num_atoms * 3))

        for i, atoms in enumerate(valid_frames):
            positions = atoms.get_positions().flatten()
            if positions.size == num_atoms * 3:
                data[i, :] = positions
            else:
                print(f"Frame {i} in {file_name} has incorrect number of positions. Skipping.")
                continue

        # PCA Analysis
        pca = PCA(n_components=3)
        pca.fit(data)

        explained_variance_ratio = pca.explained_variance_ratio_
        pca_components = pca.components_.reshape((pca.n_components_, num_atoms, 3))
        first_pc_motion = pca_components[0]
        average_structure = np.mean(data, axis=0).reshape((num_atoms, 3))
        displaced_structure = average_structure + first_pc_motion

        # Plot PCA results
        plt.figure(figsize=(8, 6))
        plt.scatter(pca.transform(data)[:, 0], pca.transform(data)[:, 1], alpha=0.7)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'PCA of MD Frames_{file_identifier}')
        plot_filename = f'PCA_{file_identifier}.png'
        plt.savefig(os.path.join(output_plot_folder, plot_filename), format='png')
        plt.close()

        # Save results to CSV
        csv_writer.writerow([
            file_name,
            explained_variance_ratio.tolist(),
            pca.components_.shape,
            average_structure.shape
        ])

        print(f"Processed and saved PCA results for {file_name}.")
