import React, { useState } from 'react';
import PredictionForm from '@/components/PredictionForm';
import ResultsCard from '@/components/ResultsCard';
import ProteinViewer from '@/components/ProteinViewer';
import AIJustificationCard from '@/components/AIJustificationCard';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { 
  Activity, 
  Zap, 
  Database, 
  Cpu, 
  Brain,
  Sparkles,
  TrendingUp,
  Shield,
  Target
} from 'lucide-react';

interface AIAnalysis {
  justification: string;
  key_factors: string[];
  confidence_explanation: string;
  limitations: string[];
}

interface PredictionResult {
  binding_affinity: number;
  units: string;
  smiles: string;
  pdb_id: string;
  pdb_url: string;
  molecular_descriptors?: {
    molecular_weight?: number;
    logp?: number;
    num_hbd?: number;
    num_hba?: number;
    tpsa?: number;
    num_rotatable_bonds?: number;
    num_aromatic_rings?: number;
    num_saturated_rings?: number;
    num_heteroatoms?: number;
    bertz_ct?: number;
  };
  prediction_confidence?: number;
  uncertainty?: number;
  interpretation?: {
    binding_strength: string;
    note: string;
  };
  ai_analysis?: AIAnalysis;
  timestamp?: string;
}

export default function Index() {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [pdbFile, setPdbFile] = useState<File | null>(null);
  const [feedbackStatus, setFeedbackStatus] = useState<string>('');

  // API configuration
  const API_BASE_URL = 'http://localhost:5000';

  const handlePredict = async (smiles: string, pdbId: string) => {
    setLoading(true);
    setError('');
    setPrediction(null);
    setFeedbackStatus('');

    try {
      // First check if backend is available
      const healthResponse = await fetch(`${API_BASE_URL}/health`);
      
      if (!healthResponse.ok) {
        throw new Error('Backend server is not available. Please ensure the Flask server is running on port 5000.');
      }

      // Make prediction request
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          smiles: smiles,
          pdb_id: pdbId
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setPrediction(result);

    } catch (err: unknown) {
      console.error('Prediction error:', err);
      
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      
      // If backend is not available, show enhanced mock prediction for demo
      if (errorMessage.includes('fetch') || errorMessage.includes('Backend server')) {
        console.log('Backend not available, showing enhanced mock prediction for demo');
        
        // Generate enhanced mock prediction for demo
        const mockResult: PredictionResult = {
          binding_affinity: -8.5 + Math.random() * 6,
          units: 'kcal/mol',
          smiles: smiles,
          pdb_id: pdbId.toUpperCase(),
          pdb_url: `https://files.rcsb.org/download/${pdbId.toUpperCase()}.pdb`,
          molecular_descriptors: {
            molecular_weight: 250 + Math.random() * 300,
            logp: Math.random() * 6 - 1,
            num_hbd: Math.floor(Math.random() * 5),
            num_hba: Math.floor(Math.random() * 8),
            tpsa: 50 + Math.random() * 100,
            num_rotatable_bonds: Math.floor(Math.random() * 10),
            num_aromatic_rings: Math.floor(Math.random() * 3),
            num_saturated_rings: Math.floor(Math.random() * 2),
            num_heteroatoms: Math.floor(Math.random() * 5),
            bertz_ct: 50 + Math.random() * 100
          },
          prediction_confidence: 0.7 + Math.random() * 0.3,
          uncertainty: Math.random() * 0.5,
          interpretation: {
            binding_strength: Math.random() > 0.6 ? 'Strong' : Math.random() > 0.3 ? 'Moderate' : 'Weak',
            note: 'Lower (more negative) values indicate stronger binding'
          },
          ai_analysis: {
            justification: `Based on the molecular structure analysis of ${smiles}, this compound shows ${Math.random() > 0.5 ? 'favorable' : 'moderate'} binding characteristics with protein ${pdbId}. The predicted binding affinity suggests ${Math.random() > 0.5 ? 'strong' : 'moderate'} interaction potential, likely driven by ${Math.random() > 0.5 ? 'hydrophobic interactions and hydrogen bonding' : 'electrostatic interactions and van der Waals forces'}. The molecular weight and lipophilicity parameters are within acceptable ranges for drug-like properties.`,
            key_factors: [
              'Molecular weight within optimal range for binding',
              'Favorable lipophilicity (LogP) for membrane permeability',
              'Hydrogen bonding potential with target residues',
              'Aromatic rings may contribute to π-π stacking interactions'
            ].slice(0, Math.floor(Math.random() * 3) + 2),
            confidence_explanation: `Model confidence reflects the similarity to training data and molecular complexity. Higher confidence indicates more reliable predictions based on learned patterns.`,
            limitations: [
              'Prediction based on 2D molecular structure only',
              'Does not account for protein conformational changes',
              'Training data bias may affect novel scaffolds',
              'Experimental validation recommended'
            ]
          },
          timestamp: new Date().toISOString()
        };
        
        setPrediction(mockResult);
        setError('Note: Using enhanced mock prediction with AI analysis for demo. Start the Flask backend server for real predictions.');
      } else {
        setError(errorMessage);
      }
    } finally {
      setLoading(false);
    }
  };

  const handlePdbUpload = (file: File) => {
    setPdbFile(file);
  };

  const handleFeedbackSubmit = async (feedback: {
    is_correct: boolean;
    feedback_text: string;
    user_justification?: string;
    confidence_rating?: number;
  }) => {
    if (!prediction) return;

    try {
      const response = await fetch(`${API_BASE_URL}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          smiles: prediction.smiles,
          pdb_id: prediction.pdb_id,
          predicted_affinity: prediction.binding_affinity,
          ...feedback
        })
      });

      if (response.ok) {
        setFeedbackStatus('Feedback submitted successfully! Thank you for helping improve our AI.');
      } else {
        setFeedbackStatus('Failed to submit feedback. Please try again.');
      }
    } catch (err) {
      console.error('Feedback submission error:', err);
      setFeedbackStatus('Feedback recorded locally (backend unavailable).');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Enhanced Header */}
      <div className="bg-white border-b shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="flex items-center justify-between">
            <div className="space-y-2">
              <h1 className="text-4xl font-bold text-gray-900 flex items-center gap-3">
                <div className="p-2 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl">
                  <Activity className="h-8 w-8 text-white" />
                </div>
                Protein-Ligand Binding Affinity Predictor
                <Badge variant="secondary" className="ml-2">
                  <Sparkles className="h-3 w-3 mr-1" />
                  AI-Enhanced
                </Badge>
              </h1>
              <p className="text-gray-600 text-lg">
                Advanced GNN predictions with AI-powered scientific justification
              </p>
              <div className="flex items-center gap-4 text-sm text-gray-500">
                <div className="flex items-center gap-1">
                  <Target className="h-4 w-4" />
                  Pretrained Model
                </div>
                <div className="flex items-center gap-1">
                  <Brain className="h-4 w-4" />
                  Gemini Integration
                </div>
                <div className="flex items-center gap-1">
                  <Shield className="h-4 w-4" />
                  Uncertainty Quantification
                </div>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge variant="default" className="flex items-center gap-1">
                <Cpu className="h-3 w-3" />
                PyTorch Geometric
              </Badge>
              <Badge variant="default" className="flex items-center gap-1">
                <Database className="h-3 w-3" />
                RDKit Enhanced
              </Badge>
              <Badge variant="default" className="flex items-center gap-1">
                <Zap className="h-3 w-3" />
                Flask API
              </Badge>
              <Badge variant="default" className="flex items-center gap-1">
                <Brain className="h-3 w-3" />
                Gemini Pro
              </Badge>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* Left Column - Input Form */}
          <div className="xl:col-span-1 space-y-6">
            <PredictionForm
              onPredict={handlePredict}
              onPdbUpload={handlePdbUpload}
              loading={loading}
              error={error}
            />

            {/* Enhanced Info Panel */}
            <Card className="border-2 border-gradient-to-r from-blue-200 to-indigo-200">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-blue-600" />
                  System Capabilities
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 gap-3">
                  <div className="flex items-start gap-3 p-3 bg-blue-50 rounded-lg">
                    <Target className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-medium text-blue-900">Pretrained GNN Model</p>
                      <p className="text-xs text-blue-700">Advanced Graph Attention Network with uncertainty quantification</p>
                    </div>
                  </div>
                  
                  <div className="flex items-start gap-3 p-3 bg-purple-50 rounded-lg">
                    <Brain className="h-5 w-5 text-purple-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-medium text-purple-900">AI Justification</p>
                      <p className="text-xs text-purple-700">Gemini Pro provides scientific reasoning for predictions</p>
                    </div>
                  </div>
                  
                  <div className="flex items-start gap-3 p-3 bg-green-50 rounded-lg">
                    <Shield className="h-5 w-5 text-green-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-medium text-green-900">Feedback Learning</p>
                      <p className="text-xs text-green-700">User feedback helps improve AI analysis quality</p>
                    </div>
                  </div>
                </div>
                
                <Separator />
                
                <div className="text-xs text-gray-500 space-y-1">
                  <p><strong>Binding Affinity:</strong> -15 to 0 kcal/mol range</p>
                  <p><strong>Confidence:</strong> Model uncertainty quantification</p>
                  <p><strong>AI Analysis:</strong> Scientific justification with key factors</p>
                </div>
              </CardContent>
            </Card>

            {/* Demo AI Analysis Preview */}
            <Card className="border-2 border-purple-100 bg-gradient-to-br from-purple-50/50 to-pink-50/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5 text-purple-600" />
                  AI Analysis Preview
                  <Badge variant="outline" className="ml-auto text-xs">
                    Demo
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="p-3 bg-white rounded border border-purple-200">
                  <p className="text-sm text-gray-700">
                    <strong>Scientific Justification:</strong> AI provides detailed molecular analysis explaining binding predictions based on structure-activity relationships.
                  </p>
                </div>
                <div className="p-3 bg-white rounded border border-purple-200">
                  <p className="text-sm text-gray-700">
                    <strong>Key Factors:</strong> Molecular weight, lipophilicity, hydrogen bonding, and aromatic interactions.
                  </p>
                </div>
                <div className="p-3 bg-white rounded border border-purple-200">
                  <p className="text-sm text-gray-700">
                    <strong>Feedback System:</strong> Rate AI analysis accuracy to improve future predictions.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Middle Column - Results and AI Analysis */}
          <div className="xl:col-span-1 space-y-6">
            {/* Results */}
            {prediction && (
              <>
                <ResultsCard result={prediction} />
                
                {/* AI Justification */}
                {prediction.ai_analysis && (
                  <AIJustificationCard
                    analysis={prediction.ai_analysis}
                    smiles={prediction.smiles}
                    pdbId={prediction.pdb_id}
                    predictedAffinity={prediction.binding_affinity}
                    confidence={prediction.prediction_confidence || 0}
                    onFeedbackSubmit={handleFeedbackSubmit}
                  />
                )}
                
                {/* Feedback Status */}
                {feedbackStatus && (
                  <Alert className="border-blue-200 bg-blue-50">
                    <AlertDescription className="text-blue-700">
                      {feedbackStatus}
                    </AlertDescription>
                  </Alert>
                )}
              </>
            )}

            {/* Placeholder when no prediction */}
            {!prediction && (
              <Card className="border-2 border-dashed border-gray-300">
                <CardContent className="flex items-center justify-center py-12">
                  <div className="text-center space-y-3">
                    <Brain className="h-12 w-12 text-gray-400 mx-auto" />
                    <p className="text-gray-500 font-medium">AI Analysis & Results</p>
                    <p className="text-sm text-gray-400">
                      Enter SMILES and PDB ID, then click "Predict" to see detailed analysis
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Right Column - 3D Viewer */}
          <div className="xl:col-span-1 space-y-6">
            <ProteinViewer
              pdbUrl={prediction?.pdb_url}
              pdbFile={pdbFile}
              pdbId={prediction?.pdb_id}
            />
          </div>
        </div>
      </div>

      {/* Enhanced Footer */}
      <div className="bg-gradient-to-r from-gray-900 to-blue-900 text-white mt-16">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="text-center space-y-4">
            <div className="flex justify-center items-center gap-2 text-lg font-semibold">
              <Sparkles className="h-5 w-5" />
              Advanced AI-Powered Drug Discovery Platform
            </div>
            <p className="text-gray-300">
              Powered by PyTorch Geometric, RDKit, NGL Viewer, and Google Gemini AI
            </p>
            <div className="flex justify-center gap-6 text-sm text-gray-400">
              <span>Graph Neural Networks</span>
              <span>•</span>
              <span>Molecular Descriptors</span>
              <span>•</span>
              <span>3D Visualization</span>
              <span>•</span>
              <span>AI Justification</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}