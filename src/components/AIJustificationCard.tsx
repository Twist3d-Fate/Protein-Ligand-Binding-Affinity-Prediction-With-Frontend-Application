import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Brain, 
  ThumbsUp, 
  ThumbsDown, 
  MessageSquare, 
  AlertTriangle,
  Lightbulb,
  Send,
  CheckCircle
} from 'lucide-react';

interface AIAnalysis {
  justification: string;
  key_factors: string[];
  confidence_explanation: string;
  limitations: string[];
}

interface AIJustificationCardProps {
  analysis: AIAnalysis;
  smiles: string;
  pdbId: string;
  predictedAffinity: number;
  confidence: number;
  onFeedbackSubmit: (feedback: {
    is_correct: boolean;
    feedback_text: string;
    user_justification?: string;
    confidence_rating?: number;
  }) => void;
}

const AIJustificationCard: React.FC<AIJustificationCardProps> = ({
  analysis,
  smiles,
  pdbId,
  predictedAffinity,
  confidence,
  onFeedbackSubmit
}) => {
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackText, setFeedbackText] = useState('');
  const [userJustification, setUserJustification] = useState('');
  const [confidenceRating, setConfidenceRating] = useState<number>(3);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  const handleFeedbackSubmit = (isCorrect: boolean) => {
    if (!feedbackText.trim()) {
      return;
    }

    onFeedbackSubmit({
      is_correct: isCorrect,
      feedback_text: feedbackText,
      user_justification: userJustification,
      confidence_rating: confidenceRating
    });

    setFeedbackSubmitted(true);
    setShowFeedback(false);
    
    // Reset form after a delay
    setTimeout(() => {
      setFeedbackText('');
      setUserJustification('');
      setConfidenceRating(3);
      setFeedbackSubmitted(false);
    }, 3000);
  };

  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.8) return 'text-green-600 bg-green-50 border-green-200';
    if (conf >= 0.6) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-red-600 bg-red-50 border-red-200';
  };

  return (
    <Card className="w-full border-2 border-blue-100 bg-gradient-to-br from-blue-50/50 to-indigo-50/50">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-blue-600" />
          AI Analysis & Justification
          <Badge variant="secondary" className="ml-auto">
            Powered by Gemini Pro
          </Badge>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Main Justification */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Lightbulb className="h-4 w-4 text-amber-500" />
            <span className="font-medium">Scientific Justification</span>
          </div>
          <div className="p-4 bg-white rounded-lg border border-blue-200 shadow-sm">
            <p className="text-sm leading-relaxed whitespace-pre-wrap">
              {analysis.justification}
            </p>
          </div>
        </div>

        {/* Key Factors */}
        {analysis.key_factors && analysis.key_factors.length > 0 && (
          <div className="space-y-3">
            <h4 className="font-medium flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-orange-500" />
              Key Contributing Factors
            </h4>
            <div className="grid gap-2">
              {analysis.key_factors.map((factor, index) => (
                <div key={index} className="flex items-start gap-2 p-2 bg-white rounded border border-orange-100">
                  <div className="w-2 h-2 bg-orange-400 rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-sm">{factor}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Confidence Explanation */}
        <div className="space-y-2">
          <h4 className="font-medium">Confidence Analysis</h4>
          <div className={`p-3 rounded-lg border ${getConfidenceColor(confidence)}`}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Model Confidence</span>
              <span className="text-lg font-bold">{(confidence * 100).toFixed(1)}%</span>
            </div>
            <p className="text-xs">{analysis.confidence_explanation}</p>
          </div>
        </div>

        {/* Limitations */}
        {analysis.limitations && analysis.limitations.length > 0 && (
          <div className="space-y-2">
            <h4 className="font-medium text-gray-600">Limitations & Considerations</h4>
            <div className="space-y-1">
              {analysis.limitations.map((limitation, index) => (
                <div key={index} className="flex items-start gap-2 text-xs text-gray-500">
                  <div className="w-1 h-1 bg-gray-400 rounded-full mt-2 flex-shrink-0"></div>
                  <span>{limitation}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <Separator />

        {/* Feedback Section */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="font-medium flex items-center gap-2">
              <MessageSquare className="h-4 w-4" />
              Help Improve AI Analysis
            </h4>
            
            {!showFeedback && !feedbackSubmitted && (
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setShowFeedback(true)}
                  className="text-green-600 border-green-200 hover:bg-green-50"
                >
                  <ThumbsUp className="h-3 w-3 mr-1" />
                  Correct
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setShowFeedback(true)}
                  className="text-red-600 border-red-200 hover:bg-red-50"
                >
                  <ThumbsDown className="h-3 w-3 mr-1" />
                  Incorrect
                </Button>
              </div>
            )}
          </div>

          {feedbackSubmitted && (
            <Alert className="border-green-200 bg-green-50">
              <CheckCircle className="h-4 w-4 text-green-600" />
              <AlertDescription className="text-green-700">
                Thank you for your feedback! This helps improve our AI analysis.
              </AlertDescription>
            </Alert>
          )}

          {showFeedback && (
            <div className="space-y-4 p-4 bg-gray-50 rounded-lg border">
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Please explain your feedback:
                </label>
                <Textarea
                  value={feedbackText}
                  onChange={(e) => setFeedbackText(e.target.value)}
                  placeholder="Describe why the analysis is correct/incorrect..."
                  className="min-h-[80px]"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Your scientific justification (optional):
                </label>
                <Textarea
                  value={userJustification}
                  onChange={(e) => setUserJustification(e.target.value)}
                  placeholder="Provide your own scientific reasoning..."
                  className="min-h-[60px]"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Rate your confidence in this feedback (1-5):
                </label>
                <div className="flex gap-2">
                  {[1, 2, 3, 4, 5].map((rating) => (
                    <Button
                      key={rating}
                      size="sm"
                      variant={confidenceRating === rating ? "default" : "outline"}
                      onClick={() => setConfidenceRating(rating)}
                      className="w-8 h-8 p-0"
                    >
                      {rating}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="flex gap-2 pt-2">
                <Button
                  size="sm"
                  onClick={() => handleFeedbackSubmit(true)}
                  disabled={!feedbackText.trim()}
                  className="bg-green-600 hover:bg-green-700"
                >
                  <Send className="h-3 w-3 mr-1" />
                  Submit as Correct
                </Button>
                <Button
                  size="sm"
                  onClick={() => handleFeedbackSubmit(false)}
                  disabled={!feedbackText.trim()}
                  className="bg-red-600 hover:bg-red-700"
                >
                  <Send className="h-3 w-3 mr-1" />
                  Submit as Incorrect
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setShowFeedback(false)}
                >
                  Cancel
                </Button>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default AIJustificationCard;