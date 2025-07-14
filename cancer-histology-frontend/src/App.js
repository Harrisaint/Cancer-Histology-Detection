import { useState, useMemo } from 'react';
import {
  Container,
  CssBaseline,
  ThemeProvider,
  createTheme,
  Box,
  useMediaQuery,
  IconButton
} from '@mui/material';
import { Global } from '@emotion/react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import Header from './components/Header';
import HeroSection from './components/HeroSection';
import ImageSelector from './components/ImageSelector';
import ImageDisplay from './components/ImageDisplay';
import PredictionResult from './components/PredictionResult';
import FloatingActionButton from './components/FloatingActionButton';
import About from './components/About';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';

// Import Google Fonts
import '@fontsource/poppins/300.css';
import '@fontsource/poppins/400.css';
import '@fontsource/poppins/500.css';
import '@fontsource/poppins/600.css';
import '@fontsource/poppins/700.css';

const getDesignTokens = (mode) => ({
  palette: {
    mode,
    primary: {
      main: mode === 'dark' ? '#1A1A1A' : '#fff',
      contrastText: mode === 'dark' ? '#fff' : '#1A1A1A',
    },
    background: {
      default: mode === 'dark' ? '#1A1A1A' : '#fff',
      paper: mode === 'dark' ? '#232323' : '#fff',
    },
    text: {
      primary: mode === 'dark' ? '#fff' : '#1A1A1A',
      secondary: mode === 'dark' ? '#ccc' : '#666',
    },
    secondary: {
      main: '#FF6B00',
      contrastText: '#fff',
    },
    accent: {
      main: '#FF6B00',
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
          boxShadow: '0 4px 12px rgba(255, 107, 0, 0.15)',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 6px 20px rgba(255, 107, 0, 0.2)',
          },
        },
        containedSecondary: {
          background: 'linear-gradient(45deg, #FF6B00 30%, #FF8533 90%)',
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
          boxShadow: '0 6px 20px rgba(255, 107, 0, 0.2)',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'scale(1.1)',
            boxShadow: '0 8px 25px rgba(255, 107, 0, 0.3)',
          },
        },
      },
    },
  },
});

function ImageAnalysisPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const handleImageSelect = (image) => setSelectedImage(image);
  return (
    <Container maxWidth="lg" sx={{ mt: 2, pb: 8 }}>
      <ImageSelector selectedImage={selectedImage} onImageSelect={handleImageSelect} />
      <ImageDisplay selectedImage={selectedImage} />
      <PredictionResult selectedImage={selectedImage} />
    </Container>
  );
}

function App() {
  const [mode, setMode] = useState('light');
  const theme = useMemo(() => createTheme(getDesignTokens(mode)), [mode]);
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const handleToggleMode = () => {
    setMode((prev) => (prev === 'light' ? 'dark' : 'light'));
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Global styles={{
        body: {
          margin: 0,
          padding: 0,
          background: mode === 'dark'
            ? 'linear-gradient(135deg, #1A1A1A 0%, #232323 100%)'
            : 'linear-gradient(135deg, #fff 0%, #f7f7f7 100%)',
          minHeight: '100vh',
          fontFamily: "'Poppins', sans-serif",
        }
      }} />
      <Router>
        <div className="App">
          <Header mode={mode} onToggleMode={handleToggleMode} />
          <Box sx={{ position: 'fixed', top: 16, right: 24, zIndex: 2000 }}>
            <IconButton onClick={handleToggleMode} color="secondary" size="large">
              {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
            </IconButton>
          </Box>
          <Routes>
            <Route path="/" element={<HeroSection />} />
            <Route path="/analyze" element={<ImageAnalysisPage />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;
