import { Fab, Zoom } from '@mui/material';
import { Home } from '@mui/icons-material';

export default function FloatingActionButton({ onClick, show }) {
  return (
    <Zoom in={show}>
      <Fab
        color="primary"
        aria-label="home"
        onClick={onClick}
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          background: 'linear-gradient(45deg, #FF6B00 30%, #FF8533 90%)',
          boxShadow: '0 6px 20px rgba(255, 107, 0, 0.4)',
          '&:hover': {
            background: 'linear-gradient(45deg, #E55A00 30%, #FF6B00 90%)',
            transform: 'scale(1.1)',
            boxShadow: '0 8px 25px rgba(255, 107, 0, 0.5)',
          },
          transition: 'all 0.3s ease',
        }}
      >
        <Home />
      </Fab>
    </Zoom>
  );
} 