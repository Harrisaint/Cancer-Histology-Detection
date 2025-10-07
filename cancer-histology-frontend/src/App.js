import { useState, useMemo, useRef } from 'react';
import {
  Container,
  CssBaseline,
  ThemeProvider,
  createTheme,
  Box,
  useMediaQuery,
  Typography, 
  Button, 
  Grid,
  Card,
  CardContent,
  useTheme,
  Chip,
  LinearProgress
} from '@mui/material';
import { Global } from '@emotion/react';
import { motion } from 'framer-motion';
import { 
  Science, 
  Speed, 
  Psychology,
  ArrowDownward,
  CloudUpload,
  AutoAwesome,
  CheckCircle,
  Error
} from '@mui/icons-material';

// Import components
import Header from './components/Header';
import ImageSelector from './components/ImageSelector';
import ImageDisplay from './components/ImageDisplay';
import PredictionResult from './components/PredictionResult';

// Import Google Fonts
import '@fontsource/poppins/300.css';
import '@fontsource/poppins/400.css';
import '@fontsource/poppins/500.css';
import '@fontsource/poppins/600.css';
import '@fontsource/poppins/700.css';

const getDesignTokens = () => ({
  palette: {
    mode: 'light',
    primary: {
      main: '#0D47A1',
      contrastText: '#fff',
    },
    background: {
      default: '#FFFFFF',
      paper: '#FFFFFF',
    },
    text: {
      primary: '#212121',
      secondary: '#666666',
    },
    secondary: {
      main: '#1976D2',
      contrastText: '#fff',
    },
    accent: {
      main: '#0D47A1',
    },
  },
  typography: {
    fontFamily: '"Poppins", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '3rem',
      '@media (max-width:600px)': {
        fontSize: '2rem',
      },
    },
    h2: {
      fontWeight: 600,
      fontSize: '2.5rem',
      '@media (max-width:600px)': {
        fontSize: '1.8rem',
      },
    },
    h3: {
      fontWeight: 600,
      fontSize: '2rem',
      '@media (max-width:600px)': {
        fontSize: '1.5rem',
      },
    },
    h4: {
      fontWeight: 500,
      fontSize: '1.5rem',
    },
    h5: {
      fontWeight: 500,
      fontSize: '1.25rem',
    },
    h6: {
      fontWeight: 500,
      fontSize: '1rem',
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.5,
    },
    button: {
      fontWeight: 600,
      textTransform: 'none',
    },
  },
  shape: {
    borderRadius: 20,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 25,
          padding: '12px 24px',
          fontSize: '1rem',
          fontWeight: 600,
          textTransform: 'none',
          boxShadow: '0 4px 12px rgba(13, 71, 161, 0.15)',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 6px 20px rgba(13, 71, 161, 0.2)',
          },
        },
        containedPrimary: {
          background: 'linear-gradient(45deg, #0D47A1 30%, #1976D2 90%)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 24,
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.08)',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: '0 12px 40px rgba(0, 0, 0, 0.12)',
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          borderRadius: 0,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
        },
      },
    },
    MuiFab: {
      styleOverrides: {
        root: {
          borderRadius: '50%',
          boxShadow: '0 6px 20px rgba(13, 71, 161, 0.2)',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'scale(1.1)',
            boxShadow: '0 8px 25px rgba(13, 71, 161, 0.3)',
          },
        },
      },
    },
  },
});


function App() {
  const theme = useMemo(() => createTheme(getDesignTokens()), []);
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageSelect = (image) => {
    setSelectedImage(image);
    setPrediction(null);
    
    // Simulate prediction for demo
    if (image) {
      setLoading(true);
      setTimeout(() => {
        const isCorrect = Math.random() > 0.3;
        const actualLabel = image.actualLabel || 'uploaded';
        const predictedLabel = isCorrect ? actualLabel : (actualLabel === 'benign' ? 'malignant' : 'benign');
        const confidence = Math.random() * 0.4 + 0.6;
        
        setPrediction({
          predictedLabel,
          confidence,
          actualLabel,
          isCorrect
        });
        setLoading(false);
      }, 2000);
    }
  };

  const features = [
    {
      icon: <Science sx={{ fontSize: 32, color: '#0D47A1' }} />,
      title: 'AI-Powered Analysis',
      description: 'Advanced machine learning algorithms'
    },
    {
      icon: <Speed sx={{ fontSize: 32, color: '#0D47A1' }} />,
      title: 'Instant Results',
      description: 'Get predictions in seconds'
    },
    {
      icon: <Psychology sx={{ fontSize: 32, color: '#0D47A1' }} />,
      title: 'Smart Detection',
      description: 'Precise benign vs malignant classification'
    }
  ];

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Global styles={{
        body: {
          margin: 0,
          padding: 0,
          background: 'linear-gradient(135deg, #FFFFFF 0%, #E3F2FD 100%)',
          minHeight: '100vh',
          fontFamily: "'Poppins', sans-serif",
        }
      }} />
        <div className="App">
        <Header />
        
        {/* Split Screen Layout */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1 }}
        >
          <Box
            sx={{
              minHeight: 'calc(100vh - 64px)',
              display: 'flex',
              flexDirection: isMobile ? 'column' : 'row',
              position: 'relative',
              '&::before': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: 'radial-gradient(circle at 20% 80%, rgba(13, 71, 161, 0.03) 0%, transparent 50%), radial-gradient(circle at 80% 20%, rgba(25, 118, 210, 0.08) 0%, transparent 50%)',
                zIndex: 0,
              }
            }}
          >
            {/* Left Side - Text Section */}
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              style={{ 
                flex: isMobile ? 'none' : '1',
                display: 'flex',
                alignItems: 'center',
                padding: isMobile ? '2rem 1rem' : '4rem 2rem'
              }}
            >
              <Container maxWidth="sm" sx={{ position: 'relative', zIndex: 1 }}>
                <Box sx={{ textAlign: isMobile ? 'center' : 'left' }}>
                  {/* Logo */}
                  <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.4 }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3, justifyContent: isMobile ? 'center' : 'flex-start' }}>
                      <Science sx={{ fontSize: 40, color: '#0D47A1', mr: 2 }} />
                      <Typography variant="h5" sx={{ fontWeight: 600, color: '#0D47A1' }}>
                        AI Cancer Detection
                      </Typography>
                    </Box>
                  </motion.div>

                  {/* Headline */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.6 }}
                  >
                    <Typography
                      variant="h2"
                      sx={{
                        fontWeight: 700,
                        mb: 2,
                        fontSize: isMobile ? '2.5rem' : '3.5rem',
                        lineHeight: 1.2,
                        color: '#212121',
                      }}
                    >
                      Detect Cancer with
                      <Box component="span" sx={{ 
                        background: 'linear-gradient(45deg, #0D47A1, #1976D2)',
                        backgroundClip: 'text',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        display: 'block'
                      }}>
                        AI Precision
                      </Box>
                    </Typography>
                  </motion.div>
                  
                  {/* Subheadline */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.8 }}
                  >
                    <Typography
                      variant="h6"
                      sx={{
                        mb: 4,
                        color: '#666666',
                        fontWeight: 400,
                        lineHeight: 1.6,
                      }}
                    >
                      Upload histology images to receive instant benign vs malignant analysis using our deep learning model.
                    </Typography>
                  </motion.div>

                  {/* Feature Highlights */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 1 }}
                  >
                    <Box sx={{ mb: 4 }}>
                      {features.map((feature, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: 0.4, delay: 1.2 + index * 0.1 }}
                        >
                          <Box sx={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            mb: 2,
                            justifyContent: isMobile ? 'center' : 'flex-start'
                          }}>
                            <Box sx={{ 
                              mr: 2, 
                              p: 1, 
                              borderRadius: '50%', 
                              backgroundColor: 'rgba(13, 71, 161, 0.1)',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center'
                            }}>
                              {feature.icon}
                            </Box>
                            <Box>
                              <Typography variant="h6" sx={{ fontWeight: 600, color: '#212121', mb: 0.5 }}>
                                {feature.title}
                              </Typography>
                              <Typography variant="body2" sx={{ color: '#666666' }}>
                                {feature.description}
                              </Typography>
                            </Box>
                          </Box>
                        </motion.div>
                      ))}
                    </Box>
                  </motion.div>

                </Box>
              </Container>
            </motion.div>

            {/* Right Side - Functional Section */}
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              style={{ 
                flex: isMobile ? 'none' : '1',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: isMobile ? '2rem 1rem' : '4rem 2rem',
                background: isMobile ? 'transparent' : 'linear-gradient(135deg, rgba(227, 242, 253, 0.6) 0%, rgba(187, 222, 251, 0.3) 100%)'
              }}
            >
              <Container maxWidth="sm" sx={{ position: 'relative', zIndex: 1 }}>
                {/* Upload Card */}
                <motion.div
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.8 }}
                >
                  <Card
                    sx={{
                      mb: 3,
                      borderRadius: 4,
                      background: 'rgba(255, 255, 255, 0.95)',
                      backdropFilter: 'blur(10px)',
                      border: '1px solid rgba(255, 255, 255, 0.2)',
                      boxShadow: '0 12px 40px rgba(25, 118, 210, 0.1)',
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        boxShadow: '0 16px 50px rgba(25, 118, 210, 0.15)',
                        transform: 'translateY(-2px)',
                      }
                    }}
                  >
                    <CardContent sx={{ p: 4 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                        <CloudUpload sx={{ fontSize: 28, color: '#0D47A1', mr: 2 }} />
                        <Typography variant="h5" sx={{ fontWeight: 600, color: '#212121' }}>
                          Upload Image
                        </Typography>
                      </Box>

                      {/* Image Preview */}
                      {selectedImage && (
                        <motion.div
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ duration: 0.4 }}
                        >
                          <Box sx={{ 
                            mb: 3, 
                            borderRadius: 3, 
                            overflow: 'hidden',
                            boxShadow: '0 8px 25px rgba(0,0,0,0.1)'
                          }}>
                            <img
                              src={selectedImage.path}
                              alt="Selected"
                              style={{
                                width: '100%',
                                height: '200px',
                                objectFit: 'cover',
                                display: 'block'
                              }}
                            />
                          </Box>
                        </motion.div>
                      )}

                      {/* Upload Button */}
                      <ImageSelector selectedImage={selectedImage} onImageSelect={handleImageSelect} />
                    </CardContent>
                  </Card>
                </motion.div>

                {/* Analysis Output Card */}
                {(loading || prediction) && (
                  <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.2 }}
                  >
                    <Card
                      sx={{
                        borderRadius: 4,
                        background: 'rgba(255, 255, 255, 0.95)',
                        backdropFilter: 'blur(10px)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        boxShadow: '0 12px 40px rgba(25, 118, 210, 0.1)',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          boxShadow: '0 16px 50px rgba(25, 118, 210, 0.15)',
                          transform: 'translateY(-2px)',
                        }
                      }}
                    >
                      <CardContent sx={{ p: 4 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                          <AutoAwesome sx={{ fontSize: 28, color: '#0D47A1', mr: 2 }} />
                          <Typography variant="h5" sx={{ fontWeight: 600, color: '#212121' }}>
                            Analysis Results
                          </Typography>
                        </Box>

                        {loading && (
                          <Box sx={{ textAlign: 'center', py: 4 }}>
                            <Box sx={{ 
                              width: 60, 
                              height: 60, 
                              borderRadius: '50%', 
                              background: 'linear-gradient(45deg, #0D47A1, #1976D2)',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              margin: '0 auto 2rem',
                              animation: 'pulse 2s infinite',
                              '@keyframes pulse': {
                                '0%': { transform: 'scale(1)', opacity: 1 },
                                '50%': { transform: 'scale(1.1)', opacity: 0.7 },
                                '100%': { transform: 'scale(1)', opacity: 1 },
                              }
                            }}>
                              <Science sx={{ fontSize: 30, color: 'white' }} />
                            </Box>
                            <Typography variant="h6" sx={{ color: '#0D47A1', fontWeight: 600 }}>
                              Analyzing image...
                            </Typography>
                          </Box>
                        )}

                        {prediction && !loading && (
                          <motion.div
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.6 }}
                          >
                            <Box sx={{ mb: 3 }}>
                              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600, color: '#212121' }}>
                                Predicted Classification:
                              </Typography>
                              <Chip
                                label={prediction.predictedLabel?.charAt(0).toUpperCase() + prediction.predictedLabel?.slice(1)}
                                size="large"
                                sx={{
                                  backgroundColor: prediction.predictedLabel === 'benign' ? '#4CAF50' : '#F44336',
                                  color: 'white',
                                  fontWeight: 700,
                                  fontSize: '1.1rem',
                                  padding: '8px 16px',
                                  mb: 2
                                }}
                              />
                            </Box>

                            <Box sx={{ mb: 3 }}>
                              <Typography variant="body1" sx={{ mb: 1, fontWeight: 500, color: '#212121' }}>
                                Confidence: {(prediction.confidence * 100).toFixed(1)}%
                              </Typography>
                              <LinearProgress
                                variant="determinate"
                                value={prediction.confidence * 100}
                                sx={{
                                  height: 8,
                                  borderRadius: 4,
                                  backgroundColor: 'rgba(13, 71, 161, 0.1)',
                                  '& .MuiLinearProgress-bar': {
                                    backgroundColor: prediction.confidence >= 0.8 ? '#4CAF50' : prediction.confidence >= 0.6 ? '#FF9800' : '#F44336',
                                    borderRadius: 4,
                                  }
                                }}
                              />
                            </Box>

                            {prediction.isCorrect !== undefined && (
                              <Box sx={{
                                p: 2,
                                borderRadius: 2,
                                backgroundColor: prediction.isCorrect ? 'rgba(76, 175, 80, 0.1)' : 'rgba(244, 67, 54, 0.1)',
                                border: `1px solid ${prediction.isCorrect ? 'rgba(76, 175, 80, 0.3)' : 'rgba(244, 67, 54, 0.3)'}`
                              }}>
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                  {prediction.isCorrect ? (
                                    <CheckCircle sx={{ color: '#4CAF50', mr: 1 }} />
                                  ) : (
                                    <Error sx={{ color: '#F44336', mr: 1 }} />
                                  )}
                                  <Typography variant="body2" sx={{ 
                                    color: prediction.isCorrect ? '#2E7D32' : '#C62828',
                                    fontWeight: 600 
                                  }}>
                                    {prediction.isCorrect ? 'Correct prediction!' : 'Prediction differs from ground truth'}
                                  </Typography>
                                </Box>
                              </Box>
                            )}
                          </motion.div>
                        )}
                      </CardContent>
                    </Card>
                  </motion.div>
                )}
              </Container>
            </motion.div>
          </Box>
        </motion.div>

        {/* Footer */}
        <motion.footer
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <Box sx={{ 
            py: 4, 
            backgroundColor: '#0D47A1',
            color: 'white',
            textAlign: 'center'
          }}>
            <Container maxWidth="lg">
              <Typography variant="body1" sx={{ fontWeight: 500 }}>
                © 2024 Cancer Histology Detection. Powered by AI for medical research and education.
              </Typography>
            </Container>
          </Box>
        </motion.footer>
        </div>
    </ThemeProvider>
  );
}

export default App;
