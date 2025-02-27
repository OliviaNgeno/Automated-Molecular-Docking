### **📌 Automated Molecular Docking Pipeline**
#### High-throughput ligand docking & protein preparation using ADFRSuite, AutoDock Vina, Open Babel, PyMOL, RDKit, and Smina.

---

## **📚 Overview**
This **Automated Molecular Docking Pipeline** is designed to streamline the process of:
- **Ligand preparation** (SMILES retrieval, descriptor calculation, 3D structure generation)
- **Protein processing** (retrieval from PDB, cleaning, PDBQT conversion)
- **Molecular docking** using **AutoDock Vina** and **Smina**
- **Visualization of results** using **PyMOL, Py3Dmol**, and **matplotlib**
- **Analysis and conversion** of docking results into SDF format.

The pipeline automates the preparation of input files, performs docking, and outputs **binding scores, docked structures, and visualizations**.

---

## **💂️ Repository Structure**
```
Molecular_Docking_Pipeline/
│── bin/                  # Executables (ADFRSuite, LePro, Smina, etc.)
│── data/                 # Example input ligand & protein lists
│── output/               # Generated results (docking, structures, etc.)
│── scripts/              # Core processing scripts
│── setup.sh              # Installation script for ADFRSuite
│── requirements.txt      # Python dependencies
│── environment.yml       # Conda environment file (alternative to requirements.txt)
│── Molecular_Docking_Pipeline.py  # Main pipeline script
│── README.md             # Project documentation
│── LICENSE               # License file
```

---

## **🚀 Installation**
### **🔹 Step 1: Clone the Repository**
```sh
git clone https://github.com/yourusername/Molecular_Docking_Pipeline.git
cd Molecular_Docking_Pipeline
```

### **🔹 Step 2: Install Dependencies**
You can install dependencies via **pip** or **conda**:

#### **Using pip**
```sh
pip install -r requirements.txt
```

#### **Using Conda (Recommended)**
```sh
conda env create -f environment.yml
conda activate docking_pipeline
```

### **🔹 Step 3: Install ADFRSuite**
```sh
bash setup.sh
```
This will download and set up **AutoDockTools, AutoGrid, AutoDock, and AutoDock Vina**.

### **🔹 Step 4: Ensure Executables Are in `bin/`**
Make sure the following executables are available in the `bin/` folder:
- `lepro` (for protein cleaning)
- `smina` (for docking with Smina)
- `prepare_ligand` and `prepare_receptor` from **ADFRSuite**

If needed, update permissions:
```sh
chmod +x bin/*
```

---

## **🗂️ Input Files**
The pipeline expects a **YAML configuration file** specifying the **ligands** and **proteins** to be processed.

Example **input.yaml**:
```yaml
ligands:
  - aspirin
  - ibuprofen
  - caffeine

proteins:
  - 6LU7  # PDB ID
  - 1HSG
  - 2AZ8
```

---

## **⚡ Usage**
```sh
python Molecular_Docking_Pipeline.py input.yaml --output_dir results/
```

### **Pipeline Workflow**
1. **Ligand Processing** (SMILES retrieval, descriptor calculation, 3D structure generation, MOL2/PDBQT conversion)
2. **Protein Preparation** (Retrieve, clean, convert proteins to PDBQT, generate docking grid box)
3. **Molecular Docking** using **AutoDock Vina** & **Smina**
4. **Post-Docking Analysis** (Convert results to SDF, generate visualizations, export data)

---

## **📁 Output Files**

### 🔹 **Ligand Processing**
```
output/
├── processed_ligands.csv    # Ligand info with descriptors
├── ligand_structures/       # Generated ligand 3D structures
├── ligand_visualizations/   # Ligand 2D/3D visualization images
```

### 🔹 **Protein Preparation**
```
output/
├── processed_proteins.csv  # Protein information
├── protein_pdbqt/         # PDBQT files for docking
├── protein_visualizations/  # Protein structure images
```

### 🔹 **Docking Results**
```
output/
├── docking_results/
├── docking_visualizations/
```

---

## **🔧 Troubleshooting**
- **ADFRSuite is not found** → Ensure scripts are installed in `bin/`.
- **LePro or Smina not found** → Check `bin/` for executables.
- **Missing dependencies** → Try reinstalling via `pip` or `conda`.
- **Errors during docking** → Increase `exhaustiveness` in **Vina/Smina settings**.

---

## **👨‍👩‍👦 Contributors**
- **Your Name** (@yourusername)
- Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## **📝 License**
This project is licensed under the **MIT License**.

---

## **🌟 Acknowledgments**
Utilizes:
- **AutoDock Vina & ADFRSuite** for docking.
- **RDKit & Open Babel** for molecular processing.
- **PyMOL & Py3Dmol** for visualization.

