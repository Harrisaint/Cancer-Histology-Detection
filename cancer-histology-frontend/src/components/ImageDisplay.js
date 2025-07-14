import { 
  Box, 
  Card, 
  CardMedia, 
  CardContent, 
  Typography,
  Chip,
  Fade
} from '@mui/material';
import { PhotoCamera } from '@mui/icons-material';

export default function ImageDisplay({ selectedImage }) {
  if (!selectedImage) {
    return null;
  }

  const getLabelColor = (label) => {
    return label === 'benign' ? '#4CAF50' : '#F44336';
  };

  return (
    <Fade in={true} timeout={800}>
      <Box sx={{ maxWidth: 800, margin: '2rem auto', padding: '0 1rem' }}>
        <Card sx={{ 
          background: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          overflow: 'hidden',
        }}>
          <Box sx={{ position: 'relative' }}>
            <CardMedia
              component="img"
              height="400"
              image={selectedImage.path}
              alt={selectedImage.filename}
              sx={{ 
                objectFit: 'contain', 
                backgroundColor: '#f8f9fa',
                borderBottom: '1px solid rgba(0,0,0,0.1)',
              }}
              onError={(e) => {
                // Fallback to a placeholder if image doesn't exist
                e.target.src = 'https://via.placeholder.com/400x400?text=Histology+Image&bg=FF6B00&color=FFFFFF';
              }}
            />
            <Box sx={{ 
              position: 'absolute', 
              top: 16, 
              right: 16,
              display: 'flex',
              gap: 1,
            }}>
              <Chip 
                icon={<PhotoCamera />}
                label="Histology"
                size="small"
                sx={{ 
                  backgroundColor: 'rgba(255, 107, 0, 0.9)',
                  color: 'white',
                  fontWeight: 600,
                  backdropFilter: 'blur(10px)',
                }}
              />
              <Chip 
                label={selectedImage.actualLabel}
                size="small"
                sx={{ 
                  backgroundColor: getLabelColor(selectedImage.actualLabel),
                  color: 'white',
                  fontWeight: 600,
                  backdropFilter: 'blur(10px)',
                }}
              />
            </Box>
          </Box>
          
          <CardContent sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Typography variant="h6" sx={{ fontWeight: 600, color: '#1A1A1A' }}>
                Selected Image
              </Typography>
              <Typography variant="body2" sx={{ color: '#666666' }}>
                {selectedImage.filename}
              </Typography>
            </Box>
            
            <Typography variant="body2" sx={{ mt: 1, color: '#666666' }}>
              This image will be analyzed by our AI model to determine if the cells are benign or malignant.
            </Typography>
          </CardContent>
        </Card>
      </Box>
    </Fade>
  );
} 