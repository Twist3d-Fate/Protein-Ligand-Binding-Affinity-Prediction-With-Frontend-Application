import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { TrendingDown, TrendingUp, Minus, Activity, Atom, Info } from 'lucide-react';

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
  };
  prediction_confidence?: number;
  interpretation?: {
    binding_strength: string;
    note: string;
  };
}

interface ResultsCardProps {
  result: PredictionResult;
}

const ResultsCard: React.FC<ResultsCardProps> = ({ result }) => {
  const getAffinityIcon = (affinity: number) => {
    if (affinity < -10) return <TrendingDown className="h-4 w-4 text-green-600" />;
    if (affinity < -5) return <Minus className="h-4 w-4 text-yellow-600" />;
    return <TrendingUp className="h-4 w-4 text-red-600" />;
  };

  const getAffinityColor = (affinity: number) => {
    if (affinity < -10) return 'text-green-600 bg-green-50 border-green-200';
    if (affinity < -5) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-red-600 bg-red-50 border-red-200';
  };

  const getStrengthBadgeVariant = (strength: string) => {
    switch (strength.toLowerCase()) {
      case 'strong': return 'default';
      case 'moderate': return 'secondary';
      case 'weak': return 'outline';
      default: return 'secondary';
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Prediction Results
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Main Result */}
        <div className={`p-4 rounded-lg border-2 ${getAffinityColor(result.binding_affinity)}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {getAffinityIcon(result.binding_affinity)}
              <span className="text-sm font-medium">Binding Affinity</span>
            </div>
            <Badge variant={getStrengthBadgeVariant(result.interpretation?.binding_strength || '')}>
              {result.interpretation?.binding_strength || 'Unknown'}
            </Badge>
          </div>
          <div className="mt-2">
            <span className="text-2xl font-bold">
              {result.binding_affinity} {result.units}
            </span>
            {result.prediction_confidence && (
              <div className="text-sm text-muted-foreground mt-1">
                Confidence: {(result.prediction_confidence * 100).toFixed(1)}%
              </div>
            )}
          </div>
        </div>

        {/* Input Summary */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Atom className="h-4 w-4" />
              <span className="text-sm font-medium">Ligand (SMILES)</span>
            </div>
            <div className="p-2 bg-muted rounded text-sm font-mono break-all">
              {result.smiles}
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4" />
              <span className="text-sm font-medium">Protein (PDB)</span>
            </div>
            <div className="p-2 bg-muted rounded text-sm font-mono">
              {result.pdb_id}
            </div>
          </div>
        </div>

        {/* Molecular Descriptors */}
        {result.molecular_descriptors && (
          <>
            <Separator />
            <div className="space-y-3">
              <h4 className="text-sm font-medium">Molecular Descriptors</h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                {result.molecular_descriptors.molecular_weight && (
                  <div>
                    <span className="text-muted-foreground">MW:</span>
                    <span className="ml-1 font-medium">
                      {result.molecular_descriptors.molecular_weight.toFixed(1)} Da
                    </span>
                  </div>
                )}
                {result.molecular_descriptors.logp && (
                  <div>
                    <span className="text-muted-foreground">LogP:</span>
                    <span className="ml-1 font-medium">
                      {result.molecular_descriptors.logp.toFixed(2)}
                    </span>
                  </div>
                )}
                {result.molecular_descriptors.num_hbd !== undefined && (
                  <div>
                    <span className="text-muted-foreground">HBD:</span>
                    <span className="ml-1 font-medium">
                      {result.molecular_descriptors.num_hbd}
                    </span>
                  </div>
                )}
                {result.molecular_descriptors.num_hba !== undefined && (
                  <div>
                    <span className="text-muted-foreground">HBA:</span>
                    <span className="ml-1 font-medium">
                      {result.molecular_descriptors.num_hba}
                    </span>
                  </div>
                )}
                {result.molecular_descriptors.tpsa && (
                  <div>
                    <span className="text-muted-foreground">TPSA:</span>
                    <span className="ml-1 font-medium">
                      {result.molecular_descriptors.tpsa.toFixed(1)} Ų
                    </span>
                  </div>
                )}
                {result.molecular_descriptors.num_rotatable_bonds !== undefined && (
                  <div>
                    <span className="text-muted-foreground">RotBonds:</span>
                    <span className="ml-1 font-medium">
                      {result.molecular_descriptors.num_rotatable_bonds}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* Interpretation */}
        {result.interpretation && (
          <>
            <Separator />
            <div className="flex items-start gap-2 p-3 bg-blue-50 rounded-lg border border-blue-200">
              <Info className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
              <div className="text-sm">
                <p className="font-medium text-blue-900 mb-1">Interpretation</p>
                <p className="text-blue-700">{result.interpretation.note}</p>
                <p className="text-blue-600 mt-1 text-xs">
                  Typical range: -15 to 0 kcal/mol (lower values = stronger binding)
                </p>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default ResultsCard;