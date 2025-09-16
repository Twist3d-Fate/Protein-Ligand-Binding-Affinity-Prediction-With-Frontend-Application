import React, { useRef, useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Eye, RotateCcw, Maximize, Download } from 'lucide-react';

interface ProteinViewerProps {
  pdbUrl?: string;
  pdbFile?: File;
  pdbId?: string;
}

// Type declarations for NGL
declare global {
  interface Window {
    NGL: {
      Stage: new (element: HTMLElement, params?: Record<string, unknown>) => {
        removeAllComponents: () => void;
        loadFile: (file: string | Blob, options?: Record<string, unknown>) => Promise<{
          addRepresentation: (type: string, params?: Record<string, unknown>) => void;
        }>;
        autoView: () => void;
        toggleFullscreen: () => void;
        makeImage: (options?: Record<string, unknown>) => Promise<Blob>;
        mouseControls: {
          add: (action: string, handler: unknown) => void;
        };
      };
      MouseActions: {
        rotateDrag: unknown;
        panDrag: unknown;
        zoomScroll: unknown;
      };
    };
  }
}

const ProteinViewer: React.FC<ProteinViewerProps> = ({ pdbUrl, pdbFile, pdbId }) => {
  const viewerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<InstanceType<typeof window.NGL.Stage> | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [nglLoaded, setNglLoaded] = useState(false);

  // Load NGL viewer dynamically
  useEffect(() => {
    const loadNGL = async () => {
      try {
        // Load NGL from CDN
        if (!window.NGL) {
          const script = document.createElement('script');
          script.src = 'https://unpkg.com/ngl@2.0.0-dev.39/dist/ngl.js';
          script.onload = () => {
            setNglLoaded(true);
          };
          script.onerror = () => {
            setError('Failed to load NGL viewer library');
          };
          document.head.appendChild(script);
        } else {
          setNglLoaded(true);
        }
      } catch (err) {
        setError('Failed to initialize 3D viewer');
      }
    };

    loadNGL();
  }, []);

  // Initialize NGL stage
  useEffect(() => {
    if (nglLoaded && viewerRef.current && !stageRef.current && window.NGL) {
      try {
        stageRef.current = new window.NGL.Stage(viewerRef.current, {
          backgroundColor: 'white',
          quality: 'medium'
        });
        
        // Add mouse controls info
        stageRef.current.mouseControls.add('drag-left', window.NGL.MouseActions.rotateDrag);
        stageRef.current.mouseControls.add('drag-right', window.NGL.MouseActions.panDrag);
        stageRef.current.mouseControls.add('scroll', window.NGL.MouseActions.zoomScroll);
        
      } catch (err) {
        setError('Failed to initialize 3D viewer stage');
      }
    }
  }, [nglLoaded]);

  // Load structure when URL or file changes
  useEffect(() => {
    if (!stageRef.current || !nglLoaded) return;

    const loadStructure = async () => {
      setIsLoading(true);
      setError('');

      try {
        // Clear existing components
        stageRef.current!.removeAllComponents();

        let loadPromise;

        if (pdbFile) {
          // Load from uploaded file
          const fileContent = await readFileAsText(pdbFile);
          const blob = new Blob([fileContent], { type: 'text/plain' });
          loadPromise = stageRef.current!.loadFile(blob, { ext: 'pdb' });
        } else if (pdbUrl) {
          // Load from URL
          loadPromise = stageRef.current!.loadFile(pdbUrl);
        } else {
          setIsLoading(false);
          return;
        }

        const component = await loadPromise;
        
        // Add representations
        component.addRepresentation('cartoon', { 
          color: 'chainname',
          opacity: 0.8
        });
        
        component.addRepresentation('ball+stick', { 
          sele: 'hetero and not water',
          color: 'element',
          scale: 0.8
        });

        // Auto-view the structure
        stageRef.current!.autoView();
        
      } catch (err) {
        console.error('Failed to load structure:', err);
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        setError(`Failed to load protein structure: ${errorMessage}`);
      } finally {
        setIsLoading(false);
      }
    };

    if (pdbUrl || pdbFile) {
      loadStructure();
    }
  }, [pdbUrl, pdbFile, nglLoaded]);

  const readFileAsText = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target?.result as string);
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  };

  const handleReset = () => {
    if (stageRef.current) {
      stageRef.current.autoView();
    }
  };

  const handleFullscreen = () => {
    if (stageRef.current) {
      stageRef.current.toggleFullscreen();
    }
  };

  const handleDownload = () => {
    if (stageRef.current) {
      stageRef.current.makeImage({
        factor: 2,
        antialias: true,
        trim: true,
        transparent: false
      }).then((blob: Blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `protein_${pdbId || 'structure'}.png`;
        a.click();
        URL.revokeObjectURL(url);
      });
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            3D Protein Structure
            {pdbId && <span className="text-sm font-normal text-muted-foreground">({pdbId})</span>}
          </CardTitle>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleReset}
              disabled={!stageRef.current || isLoading}
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleFullscreen}
              disabled={!stageRef.current || isLoading}
            >
              <Maximize className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleDownload}
              disabled={!stageRef.current || isLoading}
            >
              <Download className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        
        <div className="relative">
          <div 
            ref={viewerRef}
            className="w-full h-96 border border-border rounded-lg bg-muted/30"
            style={{ minHeight: '400px' }}
          />
          
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/80 rounded-lg">
              <div className="text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
                <p className="text-sm text-muted-foreground">Loading 3D structure...</p>
              </div>
            </div>
          )}
          
          {!pdbUrl && !pdbFile && !isLoading && (
            <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
              <div className="text-center">
                <Eye className="h-12 w-12 mx-auto mb-2 opacity-50" />
                <p className="text-sm">Upload a PDB file or run prediction to view 3D structure</p>
              </div>
            </div>
          )}
        </div>
        
        <div className="mt-4 text-xs text-muted-foreground space-y-1">
          <p><strong>Controls:</strong> Left click + drag to rotate, Right click + drag to pan, Scroll to zoom</p>
          <p><strong>Visualization:</strong> Protein shown in cartoon representation, ligands in ball-and-stick</p>
        </div>
      </CardContent>
    </Card>
  );
};

export default ProteinViewer;