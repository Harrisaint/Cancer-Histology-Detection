import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Box, 
  useTheme,
  useMediaQuery
} from '@mui/material';
import { Science } from '@mui/icons-material';

export default function Header() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  return (
    <AppBar 
      position="static" 
      sx={{
        background: 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid rgba(13, 71, 161, 0.1)',
        boxShadow: '0 4px 20px rgba(13, 71, 161, 0.1)',
      }}
    >
      <Toolbar sx={{ justifyContent: 'center' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Science 
            sx={{ 
              fontSize: 32, 
              color: '#0D47A1',
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
                background: 'linear-gradient(45deg, #0D47A1, #1976D2)',
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
                  color: 'rgba(0,0,0,0.7)',
                  fontSize: '0.8rem'
                }}
              >
                AI-Powered Medical Image Analysis
              </Typography>
            )}
          </Box>
        </Box>
      </Toolbar>
    </AppBar>
  );
} 