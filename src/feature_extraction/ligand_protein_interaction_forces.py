import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
import numpy as np
from collections import defaultdict
import os
import csv

def calculate_angle(atom1, atom2, atom3):
    vec1 = atom1.position - atom2.position
    vec2 = atom3.position - atom2.position
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

folder_path = 'your_trajectory_folder_path'  # <-- Replace this with your path
file_names = ['your_pdb_file_1.pdb', 'your_pdb_file_2.pdb']  # <-- Replace with actual filenames

cutoffs = {"H-bond": 2.5, "π-Cation": 4.5, "Hydrophobic": 3.6, "Ionic": 3.7, "Metal": 3.4, "Water Bridge": 2.8}
force_values = {"H-bond": 6.0, "π-Cation": 3.5, "Hydrophobic": 1.5, "Ionic": 4.0, "Water Bridge": 2.0}
angle_cutoffs = {"H-bond Donor": 120, "H-bond Acceptor": 90, "Water Donor": 110, "Water Acceptor": 90}
hydrophobic_residues = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR"}
charged_residues = {"ARG", "LYS", "ASP", "GLU"}

output_csv = 'interaction_results.csv'  # <-- Will save to the current directory

with open(output_csv, mode='w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['File Name', 'Residue', 'Average Force Per Frame'])

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        u = mda.Universe(file_path)
        ligand_sel = "resname K8J"
        protein_sel = "protein"
        ligand = u.select_atoms(ligand_sel)
        protein = u.select_atoms(protein_sel)

        print(f"Processing file: {file_name}")
        print(f"Number of ligand atoms: {ligand.n_atoms}")
        print(f"Number of protein atoms: {protein.n_atoms}")

        residue_forces = defaultdict(float)
        all_residues = {res.resname for res in protein.residues}

        for ts in u.trajectory:
            frame_forces = defaultdict(float)
            hbond_count = defaultdict(int)
            hydrophobic_count = defaultdict(int)
            ionic_count = defaultdict(int)
            water_bridge_count = defaultdict(int)

            distances = distance_array(ligand.positions, protein.positions)

            interacting_pairs = np.where(distances < cutoffs["H-bond"])
            for lig_idx, prot_idx in zip(interacting_pairs[0], interacting_pairs[1]):
                prot_atom = protein[prot_idx]
                resname = prot_atom.residue.resname
                donor_angle = calculate_angle(ligand[lig_idx], prot_atom, protein[prot_idx])
                acceptor_angle = calculate_angle(protein[prot_idx], ligand[lig_idx], ligand[lig_idx])
                if donor_angle >= angle_cutoffs["H-bond Donor"] and acceptor_angle >= angle_cutoffs["H-bond Acceptor"]:
                    hbond_count[resname] += 1

            hydrophobic_pairs = np.where(distances < cutoffs["Hydrophobic"])
            for lig_idx, prot_idx in zip(hydrophobic_pairs[0], hydrophobic_pairs[1]):
                prot_atom = protein[prot_idx]
                resname = prot_atom.residue.resname
                if resname in hydrophobic_residues:
                    hydrophobic_count[resname] += 1

            ionic_pairs = np.where(distances < cutoffs["Ionic"])
            for lig_idx, prot_idx in zip(ionic_pairs[0], ionic_pairs[1]):
                prot_atom = protein[prot_idx]
                resname = prot_atom.residue.resname
                if resname in charged_residues:
                    ionic_count[resname] += 1

            water_bridge_pairs = np.where(distances < cutoffs["Water Bridge"])
            for lig_idx, prot_idx in zip(water_bridge_pairs[0], water_bridge_pairs[1]):
                prot_atom = protein[prot_idx]
                resname = prot_atom.residue.resname
                donor_angle = calculate_angle(ligand[lig_idx], prot_atom, protein[prot_idx])
                acceptor_angle = calculate_angle(protein[prot_idx], ligand[lig_idx], ligand[lig_idx])
                if donor_angle >= angle_cutoffs["Water Donor"] and acceptor_angle >= angle_cutoffs["Water Acceptor"]:
                    water_bridge_count[resname] += 1

            for resname, count in hbond_count.items():
                frame_forces[resname] += count * force_values["H-bond"]
            for resname, count in hydrophobic_count.items():
                frame_forces[resname] += count * force_values["Hydrophobic"]
            for resname, count in ionic_count.items():
                frame_forces[resname] += count * force_values["Ionic"]
            for resname, count in water_bridge_count.items():
                frame_forces[resname] += count * force_values["Water Bridge"]

            for resname in all_residues:
                residue_forces[resname] += frame_forces[resname]

        num_frames = len(u.trajectory)
        print(f"\nResults for file: {file_name}")
        print("Average Force Interactions Per Residue Per Frame (Alphabetical Order):")
        for resname in sorted(all_residues):
            avg_force_per_frame = residue_forces[resname] / num_frames
            print(f"Residue {resname}: {avg_force_per_frame:.2f} average force interactions per frame")
            csvwriter.writerow([file_name, resname, f"{avg_force_per_frame:.2f}"])
        print(f"Finished processing file: {file_name}\n")
