import { 
  Box, 
  Typography, 
  Button, 
  Container, 
  Grid,
  Card,
  CardContent,
  useTheme,
  useMediaQuery
} from '@mui/material';
import { 
  Science, 
  AutoAwesome, 
  Speed, 
  Psychology,
  ArrowForward
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

export default function HeroSection() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const navigate = useNavigate();

  const features = [
    {
      icon: <Science sx={{ fontSize: 40, color: '#FF6B00' }} />,
      title: 'AI-Powered Analysis',
      description: 'Advanced machine learning algorithms for accurate diagnosis'
    },
    {
      icon: <Speed sx={{ fontSize: 40, color: '#FF6B00' }} />,
      title: 'Instant Results',
      description: 'Get predictions in seconds with high confidence scores'
    },
    {
      icon: <Psychology sx={{ fontSize: 40, color: '#FF6B00' }} />,
      title: 'Smart Detection',
      description: 'Distinguish between benign and malignant cells with precision'
    }
  ];

  return (
    <Box
      sx={{
        minHeight: 'calc(100vh - 64px)',
        display: 'flex',
        alignItems: 'center',
        position: 'relative',
        overflow: 'hidden',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'radial-gradient(circle at 20% 80%, rgba(255, 107, 0, 0.1) 0%, transparent 50%), radial-gradient(circle at 80% 20%, rgba(255, 133, 51, 0.1) 0%, transparent 50%)',
          zIndex: 0,
        }
      }}
    >
      <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
        <Grid container spacing={4} alignItems="center">
          <Grid item xs={12} md={6}>
            <Box sx={{ textAlign: isMobile ? 'center' : 'left' }}>
              <Typography
                variant="h1"
                sx={{
                  fontWeight: 700,
                  mb: 2,
                  background: 'linear-gradient(45deg, #1A1A1A, #333333)',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  fontSize: isMobile ? '2.5rem' : '3.5rem',
                  lineHeight: 1.2,
                }}
              >
                Detect Cancer with
                <Box component="span" sx={{ 
                  background: 'linear-gradient(45deg, #FF6B00, #FF8533)',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}>
                  {' '}AI Precision
                </Box>
              </Typography>
              
              <Typography
                variant="h5"
                sx={{
                  mb: 4,
                  color: '#666666',
                  fontWeight: 400,
                  lineHeight: 1.5,
                }}
              >
                Upload histology images and get instant, accurate predictions for benign vs malignant classification using our advanced deep learning model.
              </Typography>

              <Button
                variant="contained"
                size="large"
                onClick={() => navigate('/analyze')}
                endIcon={<ArrowForward />}
                sx={{
                  fontSize: '1.1rem',
                  padding: '16px 32px',
                  borderRadius: 30,
                  background: 'linear-gradient(45deg, #FF6B00 30%, #FF8533 90%)',
                  boxShadow: '0 8px 25px rgba(255, 107, 0, 0.4)',
                  '&:hover': {
                    background: 'linear-gradient(45deg, #E55A00 30%, #FF6B00 90%)',
                    transform: 'translateY(-3px)',
                    boxShadow: '0 12px 35px rgba(255, 107, 0, 0.5)',
                  },
                }}
              >
                Image Analysis
              </Button>
            </Box>
          </Grid>

          <Grid item xs={12} md={6}>
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'center',
              position: 'relative'
            }}>
              <Box
                sx={{
                  width: 300,
                  height: 300,
                  borderRadius: '50%',
                  background: 'linear-gradient(135deg, rgba(255, 107, 0, 0.1), rgba(255, 133, 51, 0.1))',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  position: 'relative',
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    width: 250,
                    height: 250,
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, rgba(255, 107, 0, 0.2), rgba(255, 133, 51, 0.2))',
                    animation: 'pulse 3s infinite',
                  },
                  '@keyframes pulse': {
                    '0%': { transform: 'scale(1)', opacity: 1 },
                    '50%': { transform: 'scale(1.1)', opacity: 0.7 },
                    '100%': { transform: 'scale(1)', opacity: 1 },
                  }
                }}
              >
                <AutoAwesome 
                  sx={{ 
                    fontSize: 80, 
                    color: '#FF6B00',
                    zIndex: 1,
                    position: 'relative'
                  }} 
                />
              </Box>
            </Box>
          </Grid>
        </Grid>

        {/* Features Section */}
        <Box sx={{ mt: 8 }}>
          <Typography
            variant="h3"
            sx={{
              textAlign: 'center',
              mb: 4,
              fontWeight: 600,
              color: '#1A1A1A',
            }}
          >
            Why Choose Our Platform?
          </Typography>
          
          <Grid container spacing={3}>
            {features.map((feature, index) => (
              <Grid item xs={12} md={4} key={index}>
                <Card
                  sx={{
                    height: '100%',
                    textAlign: 'center',
                    background: 'rgba(255, 255, 255, 0.9)',
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                  }}
                >
                  <CardContent sx={{ p: 4 }}>
                    <Box sx={{ mb: 2 }}>
                      {feature.icon}
                    </Box>
                    <Typography
                      variant="h5"
                      sx={{
                        fontWeight: 600,
                        mb: 2,
                        color: '#1A1A1A',
                      }}
                    >
                      {feature.title}
                    </Typography>
                    <Typography
                      variant="body1"
                      sx={{
                        color: '#666666',
                        lineHeight: 1.6,
                      }}
                    >
                      {feature.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      </Container>
    </Box>
  );
} 