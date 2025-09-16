import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Upload, Beaker, Database } from 'lucide-react';

interface PredictionFormProps {
  onPredict: (smiles: string, pdbId: string) => void;
  onPdbUpload: (file: File) => void;
  loading: boolean;
  error: string;
}

const PredictionForm: React.FC<PredictionFormProps> = ({
  onPredict,
  onPdbUpload,
  loading,
  error
}) => {
  const [smiles, setSmiles] = useState('');
  const [pdbId, setPdbId] = useState('');

  const exampleSmiles = [
    { smiles: 'CCO', name: 'Ethanol' },
    { smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', name: 'Ibuprofen' },
    { smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', name: 'Caffeine' },
    { smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O', name: 'Aspirin' },
    { smiles: 'CN(C)CCOC(C1=CC=CC=C1)C2=CC=CC=C2', name: 'Diphenhydramine' }
  ];

  const examplePdbIds = ['1HTM', '3ERT', '1A28', '2XYZ', '4ABC'];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (smiles.trim() && pdbId.trim()) {
      onPredict(smiles.trim(), pdbId.trim().toUpperCase());
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.name.endsWith('.pdb')) {
      onPdbUpload(file);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Beaker className="h-5 w-5" />
          Input Parameters
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* SMILES Input */}
          <div className="space-y-2">
            <Label htmlFor="smiles" className="text-sm font-medium">
              SMILES String
            </Label>
            <Input
              id="smiles"
              type="text"
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="Enter SMILES string (e.g., CCO)"
              className="w-full"
            />
            <div className="flex flex-wrap gap-2 mt-2">
              <span className="text-xs text-muted-foreground">Examples:</span>
              {exampleSmiles.map((example, idx) => (
                <Button
                  key={idx}
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => setSmiles(example.smiles)}
                  className="text-xs h-6 px-2"
                >
                  {example.name}
                </Button>
              ))}
            </div>
          </div>

          {/* PDB ID Input */}
          <div className="space-y-2">
            <Label htmlFor="pdbId" className="text-sm font-medium flex items-center gap-2">
              <Database className="h-4 w-4" />
              PDB ID
            </Label>
            <Input
              id="pdbId"
              type="text"
              value={pdbId}
              onChange={(e) => setPdbId(e.target.value.toUpperCase())}
              placeholder="Enter PDB ID (e.g., 1HTM)"
              maxLength={4}
              className="w-full"
            />
            <div className="flex flex-wrap gap-2 mt-2">
              <span className="text-xs text-muted-foreground">Examples:</span>
              {examplePdbIds.map((example, idx) => (
                <Button
                  key={idx}
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => setPdbId(example)}
                  className="text-xs h-6 px-2"
                >
                  {example}
                </Button>
              ))}
            </div>
            <p className="text-xs text-muted-foreground">
              4-character PDB identifier from RCSB Protein Data Bank
            </p>
          </div>

          {/* PDB File Upload */}
          <div className="space-y-2">
            <Label htmlFor="pdbFile" className="text-sm font-medium flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Upload PDB File (Optional)
            </Label>
            <Input
              id="pdbFile"
              type="file"
              accept=".pdb"
              onChange={handleFileUpload}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Upload a local PDB file for 3D visualization
            </p>
          </div>

          {/* Error Display */}
          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Predict Button */}
          <Button
            type="submit"
            disabled={loading || !smiles.trim() || !pdbId.trim()}
            className="w-full"
            size="lg"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Predicting...
              </>
            ) : (
              'Predict Binding Affinity'
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};

export default PredictionForm;