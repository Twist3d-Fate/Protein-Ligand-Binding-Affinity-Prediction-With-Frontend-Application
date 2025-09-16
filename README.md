Protein-Ligand Binding Affinity Predictor
A comprehensive web application for predicting protein-ligand binding affinity using Graph Neural Networks (GNNs) and molecular descriptors.

🚀 Features

- GNN-based Predictions: Uses PyTorch Geometric for molecular graph processing
- SMILES Processing: RDKit integration for molecular structure analysis
- 3D Visualization: NGL Viewer for interactive protein structure display
- Modern UI: React frontend with Tailwind CSS and shadcn/ui components
- Real-time Results: Instant binding affinity predictions with confidence scores
- Molecular Descriptors: Detailed molecular property analysis
  
🏗️ Architecture

Backend (Flask)
- Framework: Flask with CORS support
- ML Stack: PyTorch Geometric, RDKit
- Model: Graph Convolutional Network (GCN) for binding affinity prediction
- API Endpoints:
  - GET /health - Health check
  - POST /predict - Binding affinity prediction
    
Frontend (React)
- Framework: React 18 with TypeScript
- UI Library: shadcn/ui components with Tailwind CSS
- 3D Viewer: NGL Viewer for protein structure visualization
- State Management: React hooks for local state

Directory Tree

├── backend/

│   ├── app.py                # Main Flask application with GNN model and API endpoints

│   ├── requirements.txt      # Python dependencies for backend

├── public/

│   ├── favicon.svg           # Application favicon

├── src/

│   ├── components/           # React components for UI

│   ├── pages/                # React pages

│   └── main.tsx              # Entry point for React application

├── index.html                # Main HTML file

├── package.json              # Frontend dependencies

└── README.md                 # Project documentation


📁 File Descriptions

- backend/app.py: Contains the Flask application, defines the GNN model, and sets up API endpoints for predictions and health checks.
- backend/requirements.txt: Lists the required Python packages for the backend.
- src/components/: Contains React components for the UI, including forms and result displays.
- src/pages/: Contains the main application pages.
- index.html: The main HTML file that serves the React application.
- package.json: Lists the frontend dependencies.

💻 Tech Stack

Backend:

- Flask
- PyTorch Geometric
- RDKit
  
Frontend:
- React
- TypeScript
- Tailwind CSS
- shadcn/ui components
- Axios
  
📦 Installation & Setup

Prerequisites
- Python 3.8+ (for backend)
- Node.js 16+ (for frontend)
- pnpm (recommended) or npm

Backend Setup
- Navigate to backend directory:

```bash 
cd backend
```
Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

- Start Flask server:

```bash
python app.py
```
The backend will be available at http://localhost:5000

Frontend Setup

- Install dependencies:

```bash
pnpm install
```

- Start development server:

```bash
pnpm run dev
```
The frontend will be available at http://localhost:5173

🔬 Usage

Basic Workflow
- Enter SMILES String: Input a molecular SMILES representation
  - Example: CCO (Ethanol)
  - Use provided example buttons for quick testing
- Specify PDB ID: Enter a 4-character PDB identifier
  - Example: 1HTM
  - Must be a valid PDB ID from RCSB database
- Optional PDB Upload: Upload local PDB file for custom structures
- Run Prediction: Click “Predict Binding Affinity” button
- View Results:
  - Binding affinity score (kcal/mol)
  - Molecular descriptors (MW, LogP, HBD/HBA, etc.)
  - Confidence score and interpretation
  - 3D protein structure visualization

Example Inputs
  - SMILES Examples:
    - CCO - Ethanol (simple alcohol)
    - CC(C)CC1=CC=C(C=C1)C(C)C(=O)O - Ibuprofen (NSAID)
    - CN1C=NC2=C1C(=O)N(C(=O)N2C)C - Caffeine (stimulant)
  - PDB Examples:
    - 1HTM - HIV-1 protease
    - 3ERT - Estrogen receptor
    - 1A28 - Thrombin

🧬 Model Details

Graph Neural Network Architecture

- Input: Molecular graph with atomic features (78 dimensions)
- Layers: 3 Graph Convolutional layers (128, 128, 64 hidden units)
- Output: Single binding affinity value
- Activation: ReLU with dropout (0.2)

Molecular Features
- Atomic number, degree, formal charge
- Hybridization state, aromaticity
- Ring membership, valence
- Additional RDKit descriptors

Prediction Range
- Binding Affinity: -15 to 0 kcal/mol
- Interpretation: Lower (more negative) = stronger binding
- Confidence: 0.0 to 1.0 (based on molecular complexity)
  
🔧 API Reference

Health Check
```bash
GET /health
```
Response:

```bash
{
  "status": "healthy",
  "model_loaded": true,
  "backend": "Flask",
  "ml_framework": "PyTorch Geometric",
  "version": "1.0.0"
}
```
Predict Binding Affinity
```bash
POST /predict
Content-Type: application/json

{
  "smiles": "CCO",
  "pdb_id": "1HTM"
}
```
Response:
```bash
{
  "binding_affinity": -7.25,
  "units": "kcal/mol",
  "smiles": "CCO",
  "pdb_id": "1HTM",
  "pdb_url": "https://files.rcsb.org/download/1HTM.pdb",
  "molecular_descriptors": {
    "molecular_weight": 46.07,
    "logp": -0.31,
    "num_hbd": 1,
    "num_hba": 1,
    "tpsa": 20.23,
    "num_rotatable_bonds": 0,
    "num_aromatic_rings": 0
  },
  "prediction_confidence": 0.85,
  "interpretation": {
    "binding_strength": "Moderate",
    "note": "Lower (more negative) values indicate stronger binding"
  }
}
```

🎨 UI Components

PredictionForm
- SMILES input with validation
- PDB ID input with format checking
- File upload for local PDB files
- Example buttons for quick testing
  
ResultsCard
- Binding affinity display with color coding
- Molecular descriptor breakdown
- Confidence scoring
- Interpretation guidelines
  
ProteinViewer
- NGL-based 3D visualization
- Interactive controls (rotate, zoom, pan)
- Multiple representation modes
- Screenshot export functionality
  
🚨 Demo Mode
If the Flask backend is not running, the frontend will automatically switch to demo mode with mock predictions. This allows you to explore the UI without setting up the backend.

Demo Features:

- Realistic mock binding affinity values
- Generated molecular descriptors
- Full UI functionality
- 3D structure loading from RCSB PDB
  
🔍 Troubleshooting

Common Issues

- Backend not starting:
  - Check Python version (3.8+ required)
  - Verify all dependencies installed
  - Ensure port 5000 is available
    
- Frontend build errors:
  - Clear node_modules and reinstall
  - Check Node.js version (16+ required)
  - Verify TypeScript compilation
    
CORS errors:
  - Ensure Flask-CORS is installed
  - Check backend URL in frontend code
  - Verify both servers are running
    
3D viewer not loading:
  - Check internet connection (NGL loads from CDN)
  - Verify PDB ID is valid
  - Try uploading local PDB file

📚 Dependencies

Backend
- Flask 2.3.3
- PyTorch 2.0.1
- PyTorch Geometric 2.3.1
- RDKit 2023.3.2
- NumPy 1.24.3

Frontend
- React 18.2.0
- TypeScript 5.0+
- Tailwind CSS 3.3.0
- shadcn/ui components
- Axios 1.12.2
- Lucide React (icons)
  
🤝 Contributing
- Fork the repository
- Create feature branch
- Make changes with tests
- Submit pull request
  
📄 License
- This project is licensed under the MIT License.
- Uses the GEMS repository for the underlying prediction model.
```bash
@article {Graber2024.12.09.627482,
	author = {Graber, David and Stockinger, Peter and Meyer, Fabian and Mishra, Siddhartha and Horn, Claus and Buller, Rebecca M. U.},
	title = {GEMS: A Generalizable GNN Framework For Protein-Ligand Binding Affinity Prediction Through Robust Data Filtering and Language Model Integration},
	elocation-id = {2024.12.09.627482},
	year = {2024},
	doi = {10.1101/2024.12.09.627482},
}
```

🔬 Scientific Background
- This application demonstrates the use of Graph Neural Networks for drug discovery applications. The GNN processes molecular structures as graphs, where atoms are nodes and bonds are edges, allowing for sophisticated analysis of molecular properties and their relationship to biological activity.

Key Concepts:

- Binding Affinity: Measure of interaction strength between protein and ligand
- SMILES: Simplified molecular-input line-entry system for chemical notation
- PDB: Protein Data Bank format for 3D molecular structures
- Graph Convolution: Neural network operation on graph-structured data
  
For more information on the underlying science, refer to recent literature on graph neural networks in drug discovery and molecular property prediction.
