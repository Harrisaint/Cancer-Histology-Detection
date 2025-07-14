import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Box, 
  Button,
  useTheme,
  useMediaQuery
} from '@mui/material';
import { Science } from '@mui/icons-material';
import { Link, useLocation } from 'react-router-dom';

export default function Header({ mode, onToggleMode }) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const location = useLocation();

  const navLinks = [
    { label: 'Home', to: '/' },
    { label: 'About', to: '/about' },
  ];

  return (
    <AppBar 
      position="static" 
      sx={{
        background: theme.palette.mode === 'dark'
          ? 'linear-gradient(45deg, #1A1A1A 30%, #232323 90%)'
          : 'linear-gradient(45deg, #fff 30%, #f7f7f7 90%)',
        color: theme.palette.text.primary,
        boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
      }}
    >
      <Toolbar sx={{ justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Science 
            sx={{ 
              fontSize: 32, 
              color: '#FF6B00',
              animation: 'pulse 2s infinite',
              '@keyframes pulse': {
                '0%': { transform: 'scale(1)' },
                '50%': { transform: 'scale(1.1)' },
                '100%': { transform: 'scale(1)' },
              }
            }} 
          />
          <Box>
            <Typography 
              variant="h6" 
              component="div" 
              sx={{ 
                fontWeight: 700,
                background: 'linear-gradient(45deg, #FF6B00, #FF8533)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                fontSize: isMobile ? '1.2rem' : '1.5rem'
              }}
            >
              🔬 Cancer Histology Detection
            </Typography>
            {!isMobile && (
              <Typography 
                variant="body2" 
                component="div"
                sx={{ 
                  color: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.7)',
                  fontSize: '0.8rem'
                }}
              >
                AI-Powered Medical Image Analysis
              </Typography>
            )}
          </Box>
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          {navLinks.map((nav) => (
            <Button
              key={nav.to}
              component={Link}
              to={nav.to}
              color="inherit"
              sx={{
                borderRadius: 20,
                fontWeight: 600,
                px: 2,
                background: location.pathname === nav.to ? 'rgba(255,107,0,0.12)' : 'transparent',
                color: location.pathname === nav.to ? '#FF6B00' : theme.palette.text.primary,
                '&:hover': {
                  backgroundColor: 'rgba(255, 107, 0, 0.15)',
                  color: '#FF6B00',
                },
                transition: 'all 0.2s',
              }}
            >
              {nav.label}
            </Button>
          ))}
        </Box>
      </Toolbar>
    </AppBar>
  );
} 