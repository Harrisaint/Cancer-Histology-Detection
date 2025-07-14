import { Box, Typography, Container, Card, CardContent } from '@mui/material';
import { EmojiObjects } from '@mui/icons-material';

export default function About() {
  return (
    <Container maxWidth="md" sx={{ mt: 6, mb: 6 }}>
      <Card sx={{
        background: 'rgba(255,255,255,0.97)',
        borderRadius: 4,
        boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
        p: 4,
      }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <EmojiObjects sx={{ fontSize: 40, color: '#FF6B00', mr: 2 }} />
            <Typography variant="h3" sx={{ fontWeight: 700, color: '#1A1A1A' }}>
              About This Project
            </Typography>
          </Box>
          <Typography variant="body1" sx={{ color: '#333', fontSize: '1.2rem', mb: 2 }}>
            <b>Cancer Histology Detection</b> is a playful, modern demo app for classifying breast cancer histology images as benign or malignant using AI. It features a vibrant, energetic UI with a bold orange accent, and supports both light and dark modes for a fun, accessible experience.
          </Typography>
          <Typography variant="body1" sx={{ color: '#333', fontSize: '1.1rem', mb: 2 }}>
            This project is for demonstration and educational purposes only. The predictions are simulated. In real-world medical settings, always consult a qualified healthcare professional.
          </Typography>
          <Typography variant="body2" sx={{ color: '#FF6B00', fontWeight: 600 }}>
            Made with ❤️ and React + Material UI
          </Typography>
        </CardContent>
      </Card>
    </Container>
  );
} 