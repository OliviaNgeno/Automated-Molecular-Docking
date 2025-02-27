import yaml
import argparse
import logging
import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import Descriptors, AllChem, Draw
from openbabel import openbabel
import pybel
import re
import os
import matplotlib.pyplot as plt
from rdkit.Chem.MolStandardize import rdMolStandardize
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import subprocess
from pathlib import Path
import pymol
from pymol import cmd
from Bio import PDB
from typing import Tuple, Dict, List, Union
from math import ceil
from PIL import Image
from vina import Vina
import random
from rdkit.Chem import AllChem
import py3Dmol
import sys
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from tabulate import tabulate



# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_input_lists(yaml_file_path):
    """
    Read lists of ligands and proteins from a YAML file.
    """
    try:
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        if 'ligands' not in data or 'proteins' not in data:
            raise KeyError("The YAML file must contain both 'ligands' and 'proteins' keys with respective lists.")
        
        ligands_df = pd.DataFrame({'name': data['ligands']})
        proteins_df = pd.DataFrame({'pdb_id': data['proteins']})
        
        return ligands_df, proteins_df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {yaml_file_path} was not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing the YAML file: {e}")
    
def get_smiles(ligand_name, max_retries=3):
    """
    Retrieve the SMILES string for a given ligand name.
    
    Parameters:
    ligand_name (str): Name of the ligand
    max_retries (int): Maximum number of retry attempts
    
    Returns:
    str or None: Canonical SMILES string if found, None otherwise
    """
    for attempt in range(max_retries):
        try:
            # Search compound name
            compounds = pcp.get_compounds(ligand_name.strip(), 'name')
            if compounds:
                # Validate SMILES using RDKit
                smiles = compounds[0].canonical_smiles
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    return Chem.MolToSmiles(mol, isomericSmiles=True)
            return None
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Error retrieving SMILES for {ligand_name} after {max_retries} attempts: {e}")
                return None
            continue

def process_ligands(ligand_df):
    """
    Process the ligand DataFrame to add SMILES strings.
    
    Parameters:
    ligand_df (pd.DataFrame): DataFrame containing ligand information
    
    Returns:
    pd.DataFrame: Updated DataFrame with SMILES information
    """
    tqdm.pandas(desc="Retrieving SMILES")
    ligand_df['SMILES'] = ligand_df['name'].progress_apply(get_smiles)
    return ligand_df.dropna(subset=['SMILES'])

def compute_descriptors(smiles):
    """
    Compute molecular descriptors for a given SMILES string.
    
    Parameters:
    smiles (str): SMILES string of the molecule
    
    Returns:
    dict or None: Dictionary of computed descriptors or None if invalid SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MolWt': Descriptors.MolWt(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'MolLogP': Descriptors.MolLogP(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'AromaticProportion': Descriptors.FractionCSP3(mol)
    }

def transform_name(name):
    """
    Transform ligand name to a format suitable for file naming.
    
    Parameters:
    name (str): Original ligand name
    
    Returns:
    str: Transformed name
    """
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def process_ligands_with_descriptors(ligand_df):
    """
    Process ligands DataFrame: compute descriptors and transform names.
    
    Parameters:
    lig_df (pd.DataFrame): DataFrame containing ligand information with SMILES
    
    Returns:
    pd.DataFrame: Processed DataFrame with descriptors and transformed names
    """
    # Compute descriptors
    descriptors_df = ligand_df['SMILES'].apply(lambda x: compute_descriptors(x) if pd.notna(x) else None)
    descriptors_expanded = pd.DataFrame(descriptors_df.tolist(), index=ligand_df.index)
    
    # Merge descriptors with ligand DataFrame
    ligand_df = pd.concat([ligand_df, descriptors_expanded], axis=1)
    
    # Transform names
    ligand_df['name_transformed'] = ligand_df['name'].apply(transform_name)
    
    return ligand_df.dropna(subset=['SMILES'])  # Remove entries with invalid SMILES


def generate_3d_structure(smiles, name, forcefield='mmff94s', steps=500):
    """
    Generate a 3D structure for a ligand from its SMILES string.
    
    Parameters:
    smiles (str): SMILES string of the ligand
    name (str): Name of the ligand (used for file naming)
    output_dir (str): Directory to save the output files
    forcefield (str): Force field to use for 3D structure generation and optimization
    steps (int): Number of steps for local optimization
    
    Returns:
    str: Path to the generated mol2 file, or None if generation failed
    """
    try:
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=steps)

        # Convert RDKit molecule to MOL block (for Open Babel conversion)
        mol_block = Chem.MolToMolBlock(mol)
        
        # Use Open Babel to convert MOL block to MOL2 format
        obconversion = openbabel.OBConversion()
        obmol = openbabel.OBMol()
        obconversion.SetInFormat("mol")
        obconversion.ReadString(obmol, mol_block)  # Read the MOL block into OBMol object
        
        # Write the molecule to a MOL2 file
        output_filename = f"{name}.mol2"
        obconversion.SetOutFormat("mol2")  # Set the output format to MOL2
        obconversion.WriteFile(obmol, output_filename)
        
        logging.info(f"Ligand {name} processed successfully. Output file: {output_filename}")
        
        return output_filename
    
    except Exception as e:
        logging.error(f"Error processing ligand {name}: {e}")
        return None
    
def process_ligands_3d(ligand_df):
    """
    Process all ligands in the DataFrame to generate 3D structures.
    
    Parameters:
    ligand_df (pd.DataFrame): DataFrame containing ligand information
    output_dir (str): Directory to save the output files
    
    Returns:
    pd.DataFrame: Updated DataFrame with paths to generated mol2 files
    """
    structure_paths = []
    
    for _, row in tqdm(ligand_df.iterrows(), total=len(ligand_df), desc="Generating 3D structures"):
        path = generate_3d_structure(row['SMILES'], row['name_transformed'])
        structure_paths.append(path)
    
    ligand_df['mol2_path'] = structure_paths

    return ligand_df.dropna(subset=['mol2_path'])  # Remove entries where 3D generation failed


def generate_structure_grid(ligand_df, output_dir, max_mols=50, mols_per_row=5):
    """
    Generate a grid image of 3D structures for ligands and save it to a file.
    
    Parameters:
    ligand_df (pd.DataFrame): DataFrame containing ligand information including mol2_path
    output_dir (str): Directory to save the output image
    max_mols (int): Maximum number of molecules to include in the grid
    mols_per_row (int): Number of molecules per row in the grid
    
    Returns:
    str: Path to the saved image file
    """
    mols = []
    names = []
    
    for _, row in tqdm(ligand_df.iterrows(), total=min(len(ligand_df), max_mols), desc="Preparing structures for visualization"):
        if len(mols) >= max_mols:
            break
        
        mol = Chem.MolFromMol2File(row['mol2_path'])
        if mol:
            AllChem.Compute2DCoords(mol)
            mols.append(mol)
            names.append(row['name_transformed'])
    
    # Create a grid image of molecules
    img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(300, 300), legends=names)
    
    # Save the image
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ligand_structures_grid.png')
    img.save(output_path)
    
    return output_path

def save_descriptors_plot(ligand_df, output_dir):
    """
    Generate and save a plot of key molecular descriptors.
    
    Parameters:
    ligand_df (pd.DataFrame): DataFrame containing ligand information and descriptors
    output_dir (str): Directory to save the output image
    
    Returns:
    str: Path to the saved image file
    """
    descriptors = ['MolWt', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'MolLogP']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, descriptor in enumerate(descriptors):
        ligand_df[descriptor].hist(ax=axes[i], bins=20)
        axes[i].set_title(descriptor)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'descriptor_distributions.png')
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def visualize_ligands(ligand_df, output_dir='ligand_visualizations'):
    """
    Generate and save visualizations for ligands.
    
    Parameters:
    ligand_df (pd.DataFrame): DataFrame containing ligand information
    output_dir (str): Directory to save the output images
    
    Returns:
    dict: Paths to the saved image files
    """
    structure_grid_path = generate_structure_grid(ligand_df, output_dir)
    descriptors_plot_path = save_descriptors_plot(ligand_df, output_dir)
    
    return {
        'structure_grid': structure_grid_path,
        'descriptors_plot': descriptors_plot_path
    }

def read_mol2_file(mol2_path):
    try:
        # Initialize Open Babel conversion object
        obconversion = openbabel.OBConversion()
        
        # Set the input format to MOL2
        obconversion.SetInFormat("mol2")
        
        # Create an empty OBMol object to store the molecule data
        mol = openbabel.OBMol()
        
        # Read the MOL2 file into the OBMol object
        if not obconversion.ReadFile(mol, mol2_path):
            raise ValueError(f"Failed to read MOL2 file: {mol2_path}")
        
        return mol
    
    except Exception as e:
        logging.error(f"Error reading MOL2 file {mol2_path}: {e}")
        return None


def analyze_protonation_state(mol2_path):
    """
    Analyze the protonation state of a molecule from a mol2 file.
    
    Parameters:
    mol2_path (str): Path to the mol2 file
    
    Returns:
    dict: Dictionary containing protonation state analysis
    """
    # Load the molecule from the MOL2 file
    mol = Chem.MolFromMol2File(mol2_path, removeHs=False)
    
    # Initialize the analysis dictionary
    analysis = {
        'file': mol2_path,
        'total_atoms': len(mol.GetAtoms()),  # Use GetAtoms() to count atoms
        'total_hydrogens': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1),  # Count hydrogens
        'polar_hydrogens': [],
        'charged_atoms': [],
        'pH_sensitive_groups': []
    }

    # Iterate over atoms to analyze protonation state
    for atom in mol.GetAtoms():
        # Identify polar hydrogens
        if atom.GetAtomicNum() == 1:  # Hydrogen atom
            # Check if hydrogen is attached to N, O, or F (polar atoms)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() in [7, 8, 9]:  # Nitrogen, Oxygen, Fluorine
                    analysis['polar_hydrogens'].append({
                        'index': atom.GetIdx(),
                        'attached_to': neighbor.GetAtomicNum(),
                        'attached_type': neighbor.GetSymbol()
                    })
        
        # Identify charged atoms
        formal_charge = atom.GetFormalCharge()
        if formal_charge != 0:
            analysis['charged_atoms'].append({
                'index': atom.GetIdx(),
                'element': atom.GetSymbol(),
                'charge': formal_charge
            })
        
        # Identify pH-sensitive groups (simplified example for alcohols and amines)
        if atom.GetSymbol() == 'O' and atom.GetDegree() == 1:  # Alcohol group (OH)
            analysis['pH_sensitive_groups'].append({
                'index': atom.GetIdx(),
                'type': 'Alcohol'
            })
        elif atom.GetSymbol() == 'N' and atom.GetDegree() == 3:  # Amine group (NH2)
            analysis['pH_sensitive_groups'].append({
                'index': atom.GetIdx(),
                'type': 'Amine'
            })
    
    return analysis

def analyze_ligand_set(ligand_df, output_dir, output_filename="ligand_analysis.csv"):
    """
    Analyze the protonation states for a set of ligands.
    
    Parameters:
    ligand_df (pd.DataFrame): DataFrame containing ligand information including mol2_path
    
    Returns:
    pd.DataFrame: DataFrame with protonation state analysis for each ligand
    """
    analyses = []
    
    for _, row in tqdm(ligand_df.iterrows(), total=len(ligand_df), desc="Analyzing protonation states"):
        analysis = analyze_protonation_state(row['mol2_path'])
        analyses.append({
            'name': row['name_transformed'],
            'mol2_path': row['mol2_path'],
            'total_atoms': analysis['total_atoms'],
            'total_hydrogens': analysis['total_hydrogens'],
            'num_polar_hydrogens': len(analysis['polar_hydrogens']),
            'num_charged_atoms': len(analysis['charged_atoms']),
            'num_pH_sensitive_groups': len(analysis['pH_sensitive_groups']),
            'formal_charge': analysis.get('formal_charge', 'N/A'),
            'num_rotatable_bonds': analysis.get('num_rotatable_bonds', 'N/A'),
        })
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_path = os.path.join(output_dir, output_filename)
    
    # Save analysis to CSV
    analysis_df = pd.DataFrame(analyses)
    analysis_df.to_csv(output_path, index=False)

    print(f"Analysis saved to {output_path}")
    return analysis_df


def perform_additional_checks(mol, mol2_path):
    """
    Perform additional checks on the molecule before PDBQT conversion.
    
    Parameters:
    mol (rdkit.Chem.rdchem.Mol): RDKit molecule object
    mol2_path (str): Path to the original mol2 file
    
    Returns:
    dict: Results of the additional checks
    """
    results = {
        'mol2_path': mol2_path,
        'passed_all_checks': True,
        'warnings': []
    }

    # 1. Structural Integrity - Check for multiple fragments
    if len(Chem.GetMolFrags(mol)) > 1:
        results['warnings'].append("Multiple fragments detected")

    # 2. Atom checks: Unusual valence and zero coordinates
    for atom in mol.GetAtoms():
        # Check for atoms with unusual valence
        if atom.GetNumImplicitHs() < 0:
            results['warnings'].append(f"Unusual valence for atom index {atom.GetIdx()}")
        
        # Check for atoms with zero coordinates
        conf = mol.GetConformer()
        pos = conf.GetAtomPosition(atom.GetIdx())
        if pos.x == 0 and pos.y == 0 and pos.z == 0:
            results['warnings'].append(f"Atom at index {atom.GetIdx()} has zero coordinates")
    
    # 3. Energy Check: Perform UFF optimization
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception as e:
        results['warnings'].append(f"Failed to perform energy minimization: {str(e)}")
    
    # 4. Tautomers: Check for tautomeric forms
    enumerator = rdMolStandardize.TautomerEnumerator()
    tautomers = list(enumerator.Enumerate(mol))
    if len(tautomers) > 1:
        results['warnings'].append(f"Multiple tautomers possible ({len(tautomers)} found)")

    # 6. Ring Conformations: Check for ring issues
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() > 0 and not AllChem.EmbedMolecule(mol, randomSeed=42):
        results['warnings'].append("Potential issues with ring conformations")

    # 7. Ligand Flexibility: Check the number of rotatable bonds
    n_rotatable = AllChem.CalcNumRotatableBonds(mol)
    if n_rotatable > 10:
        results['warnings'].append(f"High number of rotatable bonds ({n_rotatable})")

    # 8. Atom Types: Check for unusual atoms (atomic number > 18)
    unusual_atoms = [atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 18]
    if unusual_atoms:
        results['warnings'].append(f"Unusual atoms present: {', '.join(unusual_atoms)}")
    
    # If there are any warnings, mark the checks as failed
    if results['warnings']:
        results['passed_all_checks'] = False
    
    return results


def check_ligands_before_conversion(ligand_df):
    """
    Perform additional checks on all ligands before PDBQT conversion.
    
    Parameters:
    ligand_df (pd.DataFrame): DataFrame containing ligand information including mol2_path
    
    Returns:
    pd.DataFrame: DataFrame with results of additional checks for each ligand
    """
    check_results = []
    
    for _, row in tqdm(ligand_df.iterrows(), total=len(ligand_df), desc="Performing additional checks"):
        mol = Chem.MolFromMol2File(row['mol2_path'], removeHs=False)
        if mol:
            results = perform_additional_checks(mol, row['mol2_path'])
            results['name'] = row['name_transformed']
            check_results.append(results)
        else:
            check_results.append({
                'mol2_path': row['mol2_path'],
                'name': row['name_transformed'],
                'passed_all_checks': False,
                'warnings': ['Failed to read molecule']
            })
    
    return pd.DataFrame(check_results)


def convert_to_pdbqt(mol2_file, prepare_ligand_script):
    """Convert MOL2 file to PDBQT."""
    pdbqt_file = mol2_file.with_suffix('.pdbqt') 
    command = [prepare_ligand_script, '-v', '-l', mol2_file, '-o', pdbqt_file]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return pdbqt_file
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting {mol2_file} to PDBQT: {e}")
        logging.error(f"Error output: {e.stderr}")
        return None


def convert_ligands_to_pdbqt(ligand_df):
    """Convert all ligands from MOL2 to PDBQT format using ADFRsuite."""
    
    # Dynamically find ADFRsuite's bin directory
    adfrsuite_path = Path(__file__).resolve().parent / "ADFRsuite" / "bin"
    prepare_ligand_script = str(adfrsuite_path / "prepare_ligand")

    if not os.path.exists(prepare_ligand_script):
        raise FileNotFoundError(f"Error: {prepare_ligand_script} not found! Make sure ADFRsuite is installed.")

    os.chmod(prepare_ligand_script, 0o755)  # Ensure it's executable

    pdbqt_files = []
    for _, row in ligand_df.iterrows():
        mol2_file = Path(row['mol2_path']).resolve()  # Ensure absolute path
        pdbqt_file = convert_to_pdbqt(mol2_file, prepare_ligand_script)
        pdbqt_files.append(pdbqt_file)
    
    ligand_df['pdbqt_path'] = pdbqt_files
    return ligand_df.dropna(subset=['pdbqt_path'])

def fetch_and_process_proteins(proteins_df):
    """
    Fetch and process proteins using PyMOL and save cleaned PDB files.
    """

    # Start with a clean PyMOL session.
    cmd.reinitialize()

    pdb_paths = []

    for idx, row in proteins_df.iterrows():
        protein_id = row["pdb_id"]
        cmd.fetch(protein_id, type='pdb1')
        
        # Select only the protein portion.
        cmd.select(name='Prot', selection='polymer.protein')
        
        # Define the output filename (saved in the current working directory).
        pdb_filename = f"{protein_id}.pdb"
        pdb_file_path = Path.cwd() / pdb_filename
        
        # Save the processed protein.
        cmd.save(filename=str(pdb_file_path), format='pdb', selection='Prot')
        
        # Clear the PyMOL session for the next protein.
        cmd.delete('all')
        
        # Append the absolute path (as string) to our list.
        pdb_paths.append(str(pdb_file_path))
    
    # Add the new column to the original DataFrame.
    proteins_df["pdb_path"] = pdb_paths
    
    return proteins_df

def clean_proteins_with_lepro(proteins_df):
    """
    Clean protein PDB files using the local 'lepro' executable and update the
    proteins DataFrame with a new column 'cleaned_path' that points to the cleaned PDB file.
    
    """
    # Locate the 'lepro' executable in the local 'bin' folder.
    bin_dir = Path(__file__).resolve().parent / "bin"
    lepro = bin_dir / "lepro_linux_x86"
    
    # Ensure lepro is executable.
    os.chmod(lepro, 0o755)
    
    cleaned_paths = []
    
    for idx, row in proteins_df.iterrows():
        protein_id = row["pdb_id"]
        
        # Determine the input file.
        # Use the 'pdb_path' column if available; otherwise default to f"{protein_id}.pdb"
        if "pdb_path" in proteins_df.columns and pd.notna(row["pdb_path"]):
            pdb_input = Path(row["pdb_path"]).resolve()
        else:
            pdb_input = Path.cwd() / f"{protein_id}.pdb"
        
        if not pdb_input.exists():
            logging.error(f"PDB file not found for protein {protein_id}: {pdb_input}. Skipping.")
            cleaned_paths.append(None)
            continue
        
        # Run lepro on the input PDB file.
        lepro_command = f'"{lepro}" "{pdb_input}"'
        
        try:
            subprocess.run(lepro_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Lepro failed for protein {protein_id} with error: {e}")
            cleaned_paths.append(None)
            continue
        
        # Lepro output is 'pro.pdb' in the current directory.
        temp_output = Path.cwd() / "pro.pdb"
        if temp_output.exists():
            cleaned_file = Path.cwd() / f"{protein_id}_cleaned.pdb"
            try:
                temp_output.rename(cleaned_file)
                logging.info(f"Protein {protein_id} cleaned successfully")
                cleaned_paths.append(str(cleaned_file.resolve()))
            except Exception as e:
                logging.error(f"Failed to rename {temp_output} to {cleaned_file}: {e}")
                cleaned_paths.append(None)
        else:
            logging.error(f"Expected output 'pro.pdb' not found after processing {pdb_input} for protein {protein_id}")
            cleaned_paths.append(None)
    
    # Add the new 'cleaned_path' column to the DataFrame.
    proteins_df["cleaned_path"] = cleaned_paths
    
    # Optionally, drop rows where cleaning failed.
    return proteins_df.dropna(subset=["cleaned_path"])


def convert_receptors_to_pdbqt(proteins_df):
    """
    Convert cleaned protein PDB files to PDBQT format using the ADFRsuite prepare_receptor script.
    
    """
    
    # Dynamically locate the ADFRsuite bin directory and prepare_receptor script.
    adfrsuite_bin = Path(__file__).resolve().parent / "ADFRsuite" / "bin"
    prepare_receptor_script = adfrsuite_bin / "prepare_receptor"
    
    if not prepare_receptor_script.exists():
        raise FileNotFoundError(f"Error: {prepare_receptor_script} not found! "
                                "Make sure ADFRsuite is installed in the expected location.")
    
    # Ensure the prepare_receptor script is executable.
    os.chmod(prepare_receptor_script, 0o755)
    
    pdbqt_paths = []
    
    for idx, row in proteins_df.iterrows():
        protein_id = row["pdb_id"]
        
        # Construct the input (cleaned PDB) and output (PDBQT) filenames.
        cleaned_pdb_file = Path.cwd() / f"{protein_id}_cleaned.pdb"
        output_pdbqt_file = Path.cwd() / f"{protein_id}.pdbqt"
        
        if not cleaned_pdb_file.exists():
            logging.error(f"Cleaned PDB file not found for protein {protein_id}")
            pdbqt_paths.append(None)
            continue
        
        # Build the prepare_receptor command.
        command = [
            str(prepare_receptor_script),
            "-v",
            "-r", str(cleaned_pdb_file),
            "-o", str(output_pdbqt_file)
        ]
        
        try:
            subprocess.run(command, shell=False, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Conversion to pdbqt failed for {protein_id}: {e}")
            logging.error(f"Error output: {e.stderr}")
            pdbqt_paths.append(None)
            continue
        
        # Check if the output file was created.
        if output_pdbqt_file.exists():
            logging.info(f"'{cleaned_pdb_file}' converted to '{output_pdbqt_file}'")
            pdbqt_paths.append(str(output_pdbqt_file.resolve()))
        else:
            logging.error(f"Error: {output_pdbqt_file} was not created for protein {protein_id}.")
            pdbqt_paths.append(None)
    
    # Update the DataFrame with the new 'pdbqt_path' column.
    proteins_df["pdbqt_path"] = pdbqt_paths
    
    # Optionally, drop any rows where conversion failed.
    return proteins_df.dropna(subset=["pdbqt_path"])

def getbox(selection: str = 'sele', extending: float = 1.0, software: str = 'vina') -> Union[
    Tuple[Dict[str, float], Dict[str, float]],
    Tuple[Dict[str, float], Dict[str, float], Dict[str, float]],
    Tuple[Tuple[Dict[str, float], Dict[str, float]], Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]]
]:
    """
    Calculate the grid box coordinates for a given PyMOL selection.

    Args:
        selection (str): The PyMOL selection to calculate the extent for.
        extending (float): The extension (in Å) to add to each side of the box.
        software (str): 'vina', 'ledock', or 'both'. Determines the format of the returned values.

    """
    # Get the extent of the selection (returns two lists: min and max coordinates)
    ([minX, minY, minZ], [maxX, maxY, maxZ]) = cmd.get_extent(selection)

    # Extend the boundaries
    minX -= extending
    minY -= extending
    minZ -= extending
    maxX += extending
    maxY += extending
    maxZ += extending
    
    # Calculate box dimensions and center for Vina
    SizeX = maxX - minX
    SizeY = maxY - minY
    SizeZ = maxZ - minZ
    CenterX = (maxX + minX) / 2
    CenterY = (maxY + minY) / 2
    CenterZ = (maxZ + minZ) / 2
    
    if software == 'vina':
        return (
            {'center_x': CenterX, 'center_y': CenterY, 'center_z': CenterZ},
            {'size_x': SizeX, 'size_y': SizeY, 'size_z': SizeZ}
        )
    elif software == 'ledock':
        return (
            {'minX': minX, 'maxX': maxX},
            {'minY': minY, 'maxY': maxY},
            {'minZ': minZ, 'maxZ': maxZ}
        )
    elif software == 'both':
        vina_info = (
            {'center_x': CenterX, 'center_y': CenterY, 'center_z': CenterZ},
            {'size_x': SizeX, 'size_y': SizeY, 'size_z': SizeZ}
        )
        ledock_info = (
            {'minX': minX, 'maxX': maxX},
            {'minY': minY, 'maxY': maxY},
            {'minZ': minZ, 'maxZ': maxZ}
        )
        return vina_info, ledock_info
    else:
        raise ValueError('software options must be "vina", "ledock" or "both"')

def calculate_box_coordinates(proteins_df, output_dir: str, extending: float = 1.0, software: str = 'vina') -> pd.DataFrame:
    """
    Calculate the grid box coordinates for each protein in the list and save to a CSV file.

    Args:
        protein_list (List[str]): List of protein IDs.
        output_dir (str): Directory where the CSV file will be saved.
        extending (float): Extension to add to the bounding box.
        software (str): 'vina', 'ledock', or 'both'.

    Returns:
        pd.DataFrame: DataFrame containing the box coordinates for each protein.
    """
    # Launch PyMOL in headless mode
    pymol.finish_launching()

    protein_boxes = []

    for idx, row in proteins_df.iterrows():
        protein_id = row["pdb_id"]
        pdb_file = f"{protein_id}.pdb"
        
        # Load the protein file into PyMOL
        cmd.load(filename=pdb_file, format='pdb', object='prot')
        
        # Calculate the box using the defined selection "prot"
        box_info = getbox(selection='prot', extending=extending, software=software)
        
        if software == 'vina':
            center, size = box_info
            protein_boxes.append([
                protein_id, 
                center['center_x'], center['center_y'], center['center_z'],
                size['size_x'], size['size_y'], size['size_z']
            ])
        elif software == 'ledock':
            x, y, z = box_info
            protein_boxes.append([
                protein_id,
                x['minX'], x['maxX'],
                y['minY'], y['maxY'],
                z['minZ'], z['maxZ']
            ])
        elif software == 'both':
            vina_info, ledock_info = box_info
            center, size = vina_info
            x, y, z = ledock_info
            protein_boxes.append([
                protein_id,
                center['center_x'], center['center_y'], center['center_z'],
                size['size_x'], size['size_y'], size['size_z'],
                x['minX'], x['maxX'],
                y['minY'], y['maxY'],
                z['minZ'], z['maxZ']
            ])
        
        # Clear the PyMOL session before processing the next protein
        cmd.delete('all')

    # Define the DataFrame columns based on the software option
    if software == 'vina':
        columns = ['protein_id', 'center_x', 'center_y', 'center_z', 'size_x', 'size_y', 'size_z']
    elif software == 'ledock':
        columns = ['protein_id', 'minX', 'maxX', 'minY', 'maxY', 'minZ', 'maxZ']
    elif software == 'both':
        columns = ['protein_id', 'center_x', 'center_y', 'center_z', 'size_x', 'size_y', 'size_z',
                   'minX', 'maxX', 'minY', 'maxY', 'minZ', 'maxZ']
    
    protein_boxes_df = pd.DataFrame(protein_boxes, columns=columns)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    csv_path = Path(output_dir) / f"protein_boxes_{software}.csv"
    protein_boxes_df.to_csv(csv_path, index=False)
    logging.info(f"Box coordinates saved to {csv_path}")
    
    return protein_boxes_df


def generate_protein_grid(proteins_df: pd.DataFrame,
                          output_dir: str,
                          max_proteins: int = 20,
                          proteins_per_row: int = 5,
                          thumbnail_size: tuple = (300, 300)) -> str:
    """
    Generate a grid image of protein structure snapshots.
    
    """
    # Create a temporary directory to store individual protein images.
    temp_dir = Path(output_dir) / "temp_protein_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    protein_image_paths = []
    count = 0
    
    # Iterate over the proteins.
    for idx, row in proteins_df.iterrows():
        if count >= max_proteins:
            break
        
        protein_id = row["pdb_id"]
        pdb_filename = f"{protein_id}_cleaned.pdb"
        pdb_path = Path.cwd() / pdb_filename
        
        if not pdb_path.exists():
            logging.error(f"Protein PDB file not found: {pdb_path}. Skipping {protein_id}.")
            continue
        
        # Reinitialize PyMOL for a clean session.
        cmd.reinitialize()
        # Load the protein.
        cmd.load(str(pdb_path), protein_id)
        # Set a simple visual style.
        cmd.hide("everything", "all")
        cmd.show("cartoon", protein_id)
        cmd.bg_color("white")
        cmd.orient(protein_id)
        
        # Define the output image filename.
        image_file = temp_dir / f"{protein_id}.png"
        # Render a PNG snapshot using PyMOL.
        # Adjust width and height based on thumbnail_size.
        cmd.png(str(image_file), width=thumbnail_size[0], height=thumbnail_size[1], ray=1)
        
        if image_file.exists():
            protein_image_paths.append(str(image_file))
            logging.info(f"Generated image for protein {protein_id} at {image_file}")
            count += 1
        else:
            logging.error(f"Failed to generate image for protein {protein_id}")
    
    if not protein_image_paths:
        logging.error("No protein images were generated. Exiting grid generation.")
        return ""
    
    # Determine the grid dimensions.
    n_images = len(protein_image_paths)
    n_rows = ceil(n_images / proteins_per_row)
    grid_width = proteins_per_row * thumbnail_size[0]
    grid_height = n_rows * thumbnail_size[1]
    
    # Create a new blank image to serve as the grid.
    grid_img = Image.new("RGB", (grid_width, grid_height), "white")
    
    for i, img_path in enumerate(protein_image_paths):
        try:
            img = Image.open(img_path)
            # Resize image to thumbnail_size (if necessary).
            img = img.resize(thumbnail_size)
            row_idx = i // proteins_per_row
            col_idx = i % proteins_per_row
            x = col_idx * thumbnail_size[0]
            y = row_idx * thumbnail_size[1]
            grid_img.paste(img, (x, y))
        except Exception as e:
            logging.error(f"Error processing image {img_path}: {e}")
    
    # Save the grid image.
    grid_output_path = Path(output_dir) / "protein_structures_grid.png"
    grid_img.save(str(grid_output_path))
    logging.info(f"Protein grid image saved at {grid_output_path}")
    
    return str(grid_output_path)

def visualize_proteins(proteins_df: pd.DataFrame, output_dir: str = "protein_visualizations") -> dict:
    """
    Generate a grid visualization for proteins.
    
    Parameters:
        proteins_df (pd.DataFrame): DataFrame containing protein information.
        output_dir (str): Directory to save the output images.
    
    Returns:
        dict: A dictionary containing the path to the grid image.
    """
    grid_image_path = generate_protein_grid(proteins_df, output_dir)
    
    return {
        "protein_grid": grid_image_path
    }

def dock_protein_with_all_ligands(protein_id: str,
                                  receptor_pdbqt: str,
                                  grid_center: dict,
                                  grid_size: dict,
                                  ligands_df: pd.DataFrame,
                                  output_dir: str,
                                  exhaustiveness: int = 10,
                                  n_poses: int = 10) -> list:
    """
    Dock a single receptor (protein) with all ligands using AutoDock Vina.

    """
    os.makedirs(output_dir, exist_ok=True)
    docking_results = []

    for _, ligand in ligands_df.iterrows():
        # Use the transformed ligand name to build the file name.
        ligand_name = ligand["name_transformed"]
        ligand_pdbqt = f"{ligand_name}.pdbqt"

        # Define the output filename and path.
        out_filename = f"{protein_id}_{ligand_name}_vina_out.pdbqt"
        output_file = Path(output_dir) / out_filename

        try:
            # Create and configure the Vina object.
            v = Vina(sf_name='vina')
            v.set_receptor(receptor_pdbqt)
            v.set_ligand_from_file(ligand_pdbqt)
            v.compute_vina_maps(center=[
                                    grid_center['center_x'],
                                    grid_center['center_y'],
                                    grid_center['center_z']],
                                box_size=[
                                    grid_size['size_x'],
                                    grid_size['size_y'],
                                    grid_size['size_z']])
            # Perform the docking.
            v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            v.write_poses(str(output_file), n_poses=n_poses, overwrite=True)

            docking_results.append({
                "protein_id": protein_id,
                "ligand_name": ligand_name,
                "docked_file": str(output_file.resolve())
            })
        except Exception as e:
            logging.error(f"Docking failed for protein {protein_id} with ligand {ligand_name}: {e}")

    return docking_results


def perform_vina_docking(proteins_df: pd.DataFrame,
                         ligands_df: pd.DataFrame,
                         protein_boxes_df: pd.DataFrame,
                         output_dir: str,
                         exhaustiveness: int = 10,
                         n_poses: int = 5) -> list:
    """
    Iterate over all proteins and dock each with all ligands using Vina.
    
    """
    all_results = []

    for _, protein in proteins_df.iterrows():
        protein_id = protein["pdb_id"]
        receptor_pdbqt = protein["pdbqt_path"]

        # Look up grid parameters for this protein in protein_boxes_df.
        grid_info = protein_boxes_df[protein_boxes_df["protein_id"] == protein_id]
        if grid_info.empty:
            logging.error(f"No grid parameters found for protein {protein_id}. Skipping docking for this receptor.")
            continue

        grid_center = {
            "center_x": grid_info.iloc[0]["center_x"],
            "center_y": grid_info.iloc[0]["center_y"],
            "center_z": grid_info.iloc[0]["center_z"]
        }
        grid_size = {
            "size_x": grid_info.iloc[0]["size_x"],
            "size_y": grid_info.iloc[0]["size_y"],
            "size_z": grid_info.iloc[0]["size_z"]
        }

        # Dock the current protein with all ligands.
        results = dock_protein_with_all_ligands(protein_id, receptor_pdbqt, grid_center, grid_size,
                                                  ligands_df, output_dir, exhaustiveness, n_poses)
        all_results.extend(results)

    return all_results

def pdbqt_to_sdf(pdbqt_file=None, output=None):
    """
    Convert a PDBQT file to an SDF file.
    
    For each pose, the function will:
      - Create a new data field "Pose" with the value from "MODEL".
      - Create a new data field "Score" by splitting the value from "REMARK" and taking the third token.
      - Remove the fields "MODEL", "REMARK", and "TORSDO".
      
    The resulting molecules are written to the output SDF file.
    
    Parameters:
      pdbqt_file (str): Path to the input PDBQT file.
      output (str): Path to the output SDF file.
    """
    from openbabel import openbabel as ob

    def update_mol_data(mol):
        # Update "MODEL" -> "Pose"
        data_model = mol.GetData("MODEL")
        if data_model:
            model_value = data_model.GetValue()
            new_data = ob.OBPairData()
            new_data.SetAttribute("Pose")
            new_data.SetValue(model_value)
            mol.CloneData(new_data)
            mol.DeleteData("MODEL")
        
        # Update "REMARK": extract third token as score and add as "Score"
        data_remark = mol.GetData("REMARK")
        if data_remark:
            remark_value = data_remark.GetValue()
            parts = remark_value.split()
            if len(parts) >= 3:
                score = parts[2]
                new_data = ob.OBPairData()
                new_data.SetAttribute("Score")
                new_data.SetValue(score)
                mol.CloneData(new_data)
            mol.DeleteData("REMARK")
        
        # Remove "TORSDO" if it exists.
        mol.DeleteData("TORSDO")
    
    conv = ob.OBConversion()
    conv.SetInFormat("pdbqt")
    conv.SetOutFormat("sdf")
    
    mol = ob.OBMol()
    output_str = ""
    
    # Read the first molecule
    if not conv.ReadFile(mol, pdbqt_file):
        raise Exception(f"Unable to read any molecule from {pdbqt_file}")
    
    # Process each molecule (pose) in the input file.
    while True:
        update_mol_data(mol)
        # Use WriteString to get the SDF output as a string.
        sdf_str = conv.WriteString(mol)
        output_str += sdf_str
        if not conv.Read(mol):
            break
    
    # Write the collected SDF output to file.
    with open(output, "w") as out_f:
        out_f.write(output_str)



def convert_all_docking_outputs_to_sdf(proteins_df, ligands_df, docking_dir="docking_results"):
    """
    Convert all docking output PDBQT files (in docking_dir) to SDF format.
    
    """
    conversion_results = []
    
    for _, prot_row in proteins_df.iterrows():
        protein_id = prot_row["pdb_id"]
        for _, lig_row in ligands_df.iterrows():
            ligand_name = lig_row["name_transformed"]
            
            # Construct the expected docking output filename.
            pdbqt_filename = f"{protein_id}_{ligand_name}_vina_out.pdbqt"
            sdf_filename = pdbqt_filename.replace(".pdbqt", ".sdf")
            
            # Build the relative file paths.
            pdbqt_path = os.path.join(docking_dir, pdbqt_filename)
            sdf_path = os.path.join(docking_dir, sdf_filename)
            
            result = {
                "protein_id": protein_id,
                "ligand_name": ligand_name,
                "pdbqt_file": pdbqt_filename,
                "sdf_file": None
            }
            
            if not os.path.exists(pdbqt_path):
                logging.error(f"Docking output file not found: {pdbqt_path}. Skipping conversion for {protein_id} and {ligand_name}.")
            else:
                try:
                    # Call your conversion function.
                    pdbqt_to_sdf(pdbqt_file=pdbqt_path, output=sdf_path)
                    result["sdf_file"] = sdf_filename
                except Exception as e:
                    logging.error(f"Error converting {pdbqt_filename} to SDF: {e}")
            
            conversion_results.append(result)
    
    return conversion_results

def combine_ligand_mol2_files(ligands_df, output_file, input_dir="."):
    """
    Combine individual MOL2 files for ligands into a single multi-molecule MOL2 file.
    
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    
    with output_file.open("w") as outfile:
        # Iterate over each ligand
        for _, row in ligands_df.iterrows():
            ligand_name = row["name_transformed"]
            ligand_file = input_dir / f"{ligand_name}.mol2"
            
            if not ligand_file.exists():
                print(f"Warning: {ligand_file} does not exist and will be skipped.")
                continue
            
            content = ligand_file.read_text().strip()
            # Write the entire content and add a couple of newlines to separate molecules.
            outfile.write(content)
            outfile.write("\n\n")
    
    return str(output_file)


def perform_smina_docking(protein_boxes_df, docking_output_dir="smina_results", smina_executable=None):
    """
    For each protein in protein_boxes_df, run Smina docking using the combined ligand file.
    
    """
    logging.info("Starting Smina docking...")
    
    docking_output_dir = Path(docking_output_dir)
    docking_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine the Smina executable path.
    if smina_executable is None:
        smina_executable = Path(__file__).resolve().parent / "bin" / "smina"
    else:
        smina_executable = Path(smina_executable)
    
    # Ensure the Smina executable has proper permissions.
    os.chmod(smina_executable, 0o755)
    
    # Combined ligand file is assumed to be in the current directory.
    ligand_file = Path.cwd() / "InputMols.mol2"
    if not ligand_file.exists():
        logging.error(f"Combined ligand file not found: {ligand_file}")
        return []
    
    results = []
    
    # Iterate over each protein.
    for index, row in protein_boxes_df.iterrows():
        protein_id = row["protein_id"]
        center_x = row["center_x"]
        center_y = row["center_y"]
        center_z = row["center_z"]
        size_x = row["size_x"]
        size_y = row["size_y"]
        size_z = row["size_z"]
        
        # Receptor file is assumed to be "<protein_id>.pdb" in the current directory.
        receptor_file = Path.cwd() / f"{protein_id}_cleaned.pdb"
        if not receptor_file.exists():
            logging.error(f"Receptor file not found: {receptor_file}. Skipping protein {protein_id}.")
            continue
        
        # Construct the output file name: replace ".pdb" with "_smina_out.sdf"
        output_file = docking_output_dir / receptor_file.name.replace("_cleaned.pdb", "_smina_out.sdf")
        
        # Build the Smina command.
        command = [
            str(smina_executable),
            "-r", str(receptor_file),
            "-l", str(ligand_file),
            "-o", str(output_file),
            "--center_x", str(center_x),
            "--center_y", str(center_y),
            "--center_z", str(center_z),
            "--size_x", str(size_x),
            "--size_y", str(size_y),
            "--size_z", str(size_z),
            "--exhaustiveness", "8",
            "--num_modes", "5",
            "--seed", "1676539924"
        ]
        
        
        try:
            proc = subprocess.run(command, capture_output=True, text=True)
            logging.info(f"Smina docking completed for protein {protein_id} with return code {proc.returncode}")
            if proc.stdout:
                logging.info(f"Stdout: {proc.stdout}")
            if proc.stderr:
                logging.error(f"Stderr: {proc.stderr}")
            results.append({
                "protein_id": protein_id,
                "receptor_file": str(receptor_file),
                "output_file": str(output_file),
                "command": " ".join(command),
                "returncode": proc.returncode
            })
        except Exception as e:
            logging.error(f"Error running Smina for protein {protein_id}: {e}")
            results.append({
                "protein_id": protein_id,
                "receptor_file": str(receptor_file),
                "output_file": str(output_file),
                "command": " ".join(command),
                "returncode": None,
                "error": str(e)
            })
    
    return results

def visualize_and_save(protein_df, output_dir="docking_results", save_json=True):
    """
    Visualizes docking results, saves images, and prints docking scores.

    Parameters:
    - protein_df (pd.DataFrame): Must contain 'pdb_id'.
    - output_dir (str): Folder to save outputs.
    - save_json (bool): Whether to save results as JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for _, row in protein_df.iterrows():
        protein_id = row['pdb_id']
        pdb_file = f"{protein_id}_cleaned.pdb"
        sdf_filename = Path.cwd() / "smina_results" / f"{protein_id}_smina_out.sdf"

        print(f"Processing: {protein_id}...")

        if not os.path.exists(pdb_file):
            print(f"[WARNING] {pdb_file} not found, skipping.")
            continue

        # Initialize Py3Dmol viewer
        view = py3Dmol.view(width=800, height=600)
        view.removeAllModels()
        view.setViewStyle({'style': 'outline', 'color': 'black', 'width': 0.1})

        # Load Protein Model
        with open(pdb_file, 'r') as file:
            view.addModel(file.read(), format='pdb')
        Prot = view.getModel()
        Prot.setStyle({'cartoon': {'arrows': True, 'tubes': False, 'style': 'oval', 'color': 'white'}})

        docking_data = []
        if sdf_filename.exists():
            poses = Chem.SDMolSupplier(str(sdf_filename), sanitize=True)
            for p in list(poses)[::5]:  
                pose_1 = Chem.MolToMolBlock(p)
                ligand_name = p.GetProp('_Name')
                score = p.GetProp('minimizedAffinity')

                docking_data.append({"Ligand": ligand_name, "Score": float(score)})

                # Random color for ligand
                color = "#" + ''.join(random.choices('0123456789ABCDEF', k=6))
                view.addModel(pose_1, 'mol')
                z = view.getModel()
                z.setStyle({}, {'stick': {'color': color, 'radius': 0.05, 'opacity': 0.6}})
        else:
            print(f"[WARNING] {sdf_filename} not found, skipping docking visualization.")

        # Finalize View
        view.zoomTo()

        # Save HTML visualization
        html_output = f"{output_dir}/{protein_id}_view.html"
        with open(html_output, "w") as file:
            file.write(view._make_html())

        # Capture Screenshot Using Selenium
        screenshot_path = f"{output_dir}/{protein_id}_view.png"
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=800x600")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(f"file://{os.path.abspath(html_output)}")
        driver.save_screenshot(screenshot_path)
        driver.quit()

        # Store results
        results.append({
            "Protein": protein_id,
            "PDB File": pdb_file,
            "SDF File": str(sdf_filename),
            "Docking Results": docking_data,
            "HTML View": html_output,
            "Image": screenshot_path
        })

        print(f"✅ Saved: {screenshot_path}")

    # Save results as JSON
    if save_json:
        json_output = os.path.join(output_dir, "docking_summary.json")
        with open(json_output, "w") as json_file:
            json.dump(results, json_file, indent=4)
        print(f"✅ Docking results saved to {json_output}")

    # Print Summary Table
    table_data = []
    for result in results:
        for ligand in result["Docking Results"]:
            table_data.append([result["Protein"], ligand["Ligand"], ligand["Score"]])

    print("\n📊 Docking Summary:\n")
    print(tabulate(table_data, headers=["Protein", "Ligand", "Score"], tablefmt="grid"))





def main():
    """Main function."""

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process ligands, Process proteins, prepare them for docking (PDBQT conversion).")
    parser.add_argument("yaml_file", help="Path to the YAML file containing ligand and protein lists.")
    parser.add_argument("--output_dir", default="output", help="Base directory for output files (default: output)")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ligand_structures_dir = output_dir / 'ligand_structures'
    ligand_structures_dir.mkdir(parents=True, exist_ok=True)

    # Read input lists
    ligands_df, proteins_df = read_input_lists(args.yaml_file)
    logging.info("Ligands loaded successfully.")
    # --- Ligand Processing Steps ---
    logging.info("Processing ligands...")
    ligands_df = process_ligands(ligands_df)
    ligands_df = process_ligands_with_descriptors(ligands_df)
    # Generate 3D structures
    logging.info("Generating 3D structures...")
    ligands_df = process_ligands_3d(ligands_df)
    # Visualize ligands
    visualize_ligands(ligands_df)
    logging.info("Ligand visualization complete.")

    # Additional checks before conversion
    logging.info("Performing additional checks before conversion...")
    analyses = analyze_ligand_set(ligands_df, output_dir)
    check_results_df = check_ligands_before_conversion(ligands_df)

    # Convert to PDBQT
    logging.info("Converting to PDBQT format...")
    ligands_df = convert_ligands_to_pdbqt(ligands_df)

    # --- Save Results ---
    analyses.to_csv(output_dir / 'analyses_ligands.csv', index=False)
    check_results_df.to_csv(output_dir / 'check_results.csv', index=False)
    ligands_df.to_csv(output_dir / 'processed_ligands.csv', index=False)
    logging.info(f"Processed ligand data saved to {output_dir / 'processed_ligands.csv'}")

    logging.info("Ligand processing complete.")

    # retrieving proteins from pdb using PyMOL.
    logging.info("Fetching and processing proteins from pdb...")
    proteins_df = fetch_and_process_proteins(proteins_df)

    # clean protein pdbs with lepro
    logging.info("Cleaning protein structures with lepro...")
    proteins_df = clean_proteins_with_lepro(proteins_df)

    # convert protein structures to pdbqt
    logging.info("Converting protein structures to pdbqt...")
    proteins_df = convert_receptors_to_pdbqt(proteins_df)

    # Calculate box coordinates for Vina
    logging.info("Calculating box coordinates for Vina...")
    protein_boxes_df = calculate_box_coordinates(proteins_df, output_dir, extending=5.0, software='vina')

    # visualize proteins
    logging.info("Generating pymol images for proteins...")
    visualize_proteins(proteins_df, output_dir="protein_visualizations")
    logging.info("Protein grid visualization saved")

    # Docking with vina
    #logging.info("Docking with Vina...")
    #docking_output_directory = "docking_results"
    
    #perform_vina_docking(proteins_df, ligands_df, protein_boxes_df,
                                           #docking_output_directory, exhaustiveness=10, n_poses=5)

    # Converting vina resutls to sdf
    #logging.info("Converting vina resutls to sdf...")
    #convert_all_docking_outputs_to_sdf(proteins_df, ligands_df, docking_dir="docking_results")

    # combined mol2 file
    combined_file = combine_ligand_mol2_files(ligands_df, output_file="InputMols.mol2", input_dir=".")
    logging.info(f"Combined MOL2 file created: {combined_file}")

    # Docking with smina
    logging.info("Docking with Smina...")
    smina_results = perform_smina_docking(protein_boxes_df, docking_output_dir="smina_results")
    for res in smina_results:
        print(f"Protein {res['protein_id']} docking output: {res['output_file']}")

    # results visualization
    #index_html = generate_all_docking_views(protein_boxes_df)
    #print(f"All docking views available in: {index_html}")

    #grid_image_path = generate_docking_grid(proteins_df, output_dir="docking_visualizations",
                                              #max_proteins=20, proteins_per_row=5,
                                              #thumbnail_size=(300, 300))
    #print(f"Docking grid image created: {grid_image_path}")

    visualize_and_save(proteins_df, output_dir="docking_results", save_json=True)





if __name__ == "__main__":
    main()
