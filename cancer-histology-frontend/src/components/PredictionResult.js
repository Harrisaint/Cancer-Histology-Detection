import {
  Box,
  Card,
  CardContent,
  Typography,
  Alert,
  LinearProgress,
  Chip,
  Fade,
  Grow,
  CircularProgress
} from '@mui/material';
import {
  Psychology,
  CheckCircle,
  Error,
  TrendingUp,
  Science
} from '@mui/icons-material';
import { useEffect, useState } from 'react';

// Mock prediction function - simulates the model prediction
const mockPredict = (image) => {
  const isCorrect = Math.random() > 0.3;
  const actualLabel = image.category || 'uploaded';
  const predictedLabel = isCorrect ? actualLabel : (actualLabel === 'benign' ? 'malignant' : 'benign');
  const confidence = Math.random() * 0.4 + 0.6;
  return {
    predictedLabel,
    confidence,
    actualLabel,
    isCorrect
  };
};

export default function PredictionResult({ selectedImage }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!selectedImage) return;
    setResult(null);
    setError('');

    if (selectedImage.isUploaded && selectedImage.file) {
      setLoading(true);
      const formData = new FormData();
      formData.append('image', selectedImage.file);

      console.log("Sending image to backend:", selectedImage.file);

      fetch('https://histology-backend.onrender.com', {
        method: 'POST',
        body: formData,
      })
        .then(res => {
          if (!res.ok) throw new Error('API error');
          return res.json();
        })
        .then(data => {
          console.log("Received prediction:", data);
          setResult({
            predictedLabel: data.predictedLabel || null,
            confidence: data.confidence,
            actualLabel: data.actualLabel || selectedImage.category || 'uploaded',
            isCorrect: undefined
          });
          setLoading(false);
        })
        .catch((err) => {
          console.error("Prediction failed:", err);
          setError('Prediction failed. Please try again.');
          setLoading(false);
        });
    } else {
      setResult(mockPredict(selectedImage));
    }
  }, [selectedImage]);

  const getLabelColor = (label) => {
    return label === 'benign' ? '#4CAF50'
         : label === 'malignant' ? '#F44336'
         : '#FF6B00';
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#4CAF50';
    if (confidence >= 0.6) return '#FF9800';
    return '#F44336';
  };

  if (!selectedImage) return null;

  return (
    <Fade in={true} timeout={1000}>
      <Box sx={{ maxWidth: 800, margin: '2rem auto', padding: '0 1rem' }}>
        <Card sx={{
          background: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
        }}>
          <CardContent sx={{ p: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Psychology sx={{ fontSize: 32, color: '#FF6B00', mr: 2 }} />
              <Typography variant="h4" sx={{ fontWeight: 600, color: '#1A1A1A' }}>
                🧠 AI Prediction Results
              </Typography>
            </Box>

            {loading && (
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', my: 6 }}>
                <CircularProgress color="secondary" size={60} thickness={5} />
                <Typography variant="h6" sx={{ mt: 2, color: '#FF6B00', fontWeight: 600 }}>
                  Analyzing image...
                </Typography>
              </Box>
            )}

            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
            )}

            {result && !loading && !error && (
              <>
                <Box sx={{ mb: 4 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: '#1A1A1A' }}>
                      Predicted Classification:
                    </Typography>
                    <Chip
                      label={
                        typeof result.predictedLabel === 'string'
                          ? result.predictedLabel.charAt(0).toUpperCase() + result.predictedLabel.slice(1)
                          : 'N/A'
                      }
                      size="large"
                      icon={<Science />}
                      sx={{
                        backgroundColor: getLabelColor(result.predictedLabel),
                        color: 'white',
                        fontWeight: 700,
                        fontSize: '1rem',
                        padding: '8px 16px',
                      }}
                    />
                  </Box>

                  <Box sx={{ mb: 3 }}>
                    <Typography variant="body1" sx={{ mb: 1, fontWeight: 500, color: '#1A1A1A' }}>
                      Model Confidence: {result.confidence ? (result.confidence * 100).toFixed(1) : '--'}%
                    </Typography>
                    <Box sx={{ position: 'relative', mb: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={result.confidence ? result.confidence * 100 : 0}
                        sx={{
                          height: 12,
                          borderRadius: 6,
                          backgroundColor: 'rgba(0,0,0,0.1)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: getConfidenceColor(result.confidence),
                            borderRadius: 6,
                          }
                        }}
                      />
                      <Box sx={{
                        position: 'absolute',
                        right: 0,
                        top: -20,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 0.5
                      }}>
                        <TrendingUp sx={{ fontSize: 16, color: getConfidenceColor(result.confidence) }} />
                        <Typography variant="caption" sx={{ color: getConfidenceColor(result.confidence), fontWeight: 600 }}>
                          {result.confidence >= 0.8 ? 'High' : result.confidence >= 0.6 ? 'Medium' : 'Low'}
                        </Typography>
                      </Box>
                    </Box>
                  </Box>

                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                    <Typography variant="body1" sx={{ fontWeight: 500, color: '#1A1A1A' }}>
                      Actual Label:
                    </Typography>
                    <Chip
                      label={
                        typeof result.actualLabel === 'string'
                          ? result.actualLabel.charAt(0).toUpperCase() + result.actualLabel.slice(1)
                          : 'Unknown'
                      }
                      size="medium"
                      sx={{
                        backgroundColor: getLabelColor(result.actualLabel),
                        color: 'white',
                        fontWeight: 600,
                      }}
                    />
                  </Box>
                </Box>

                {result.isCorrect !== undefined && (
                  <Grow in={true} timeout={1500}>
                    <Box>
                      {result.isCorrect ? (
                        <Alert
                          severity="success"
                          icon={<CheckCircle />}
                          sx={{
                            borderRadius: 3,
                            backgroundColor: 'rgba(76, 175, 80, 0.1)',
                            border: '1px solid rgba(76, 175, 80, 0.3)',
                            '& .MuiAlert-icon': {
                              color: '#4CAF50',
                            }
                          }}
                        >
                          <Typography variant="body1" sx={{ fontWeight: 600, color: '#2E7D32' }}>
                            ✅ Excellent! The AI model predicted correctly with high confidence.
                          </Typography>
                        </Alert>
                      ) : (
                        <Alert
                          severity="error"
                          icon={<Error />}
                          sx={{
                            borderRadius: 3,
                            backgroundColor: 'rgba(244, 67, 54, 0.1)',
                            border: '1px solid rgba(244, 67, 54, 0.3)',
                            '& .MuiAlert-icon': {
                              color: '#F44336',
                            }
                          }}
                        >
                          <Typography variant="body1" sx={{ fontWeight: 600, color: '#C62828' }}>
                            ❌ The AI model made an incorrect prediction. This highlights the importance of human expert review.
                          </Typography>
                        </Alert>
                      )}
                    </Box>
                  </Grow>
                )}

                <Box sx={{ mt: 3, p: 2, backgroundColor: 'rgba(255, 107, 0, 0.05)', borderRadius: 2 }}>
                  <Typography variant="body2" sx={{ color: '#666666', fontStyle: 'italic' }}>
                    💡 <strong>Note:</strong> This is a demonstration. Uploaded images are sent to the backend for real prediction. In a real medical setting, AI predictions should always be reviewed by qualified healthcare professionals.
                  </Typography>
                </Box>
              </>
            )}
          </CardContent>
        </Card>
      </Box>
    </Fade>
  );
}
