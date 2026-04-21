import {
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  Box,
  Card,
  CardContent,
  Chip,
  Button,
  CircularProgress
} from '@mui/material';
import { Image, Science, UploadFile } from '@mui/icons-material';
import { useRef, useState, useEffect } from 'react';

export default function ImageSelector({ selectedImage, onImageSelect }) {
  const fileInput = useRef();
  const [uploadedImage, setUploadedImage] = useState(null);
  const [holdoutImages, setHoldoutImages] = useState([]);
  const [loading, setLoading] = useState(true);

  const getLabelColor = (label) => {
    return label === 'benign' ? '#4CAF50'
         : label === 'malignant' ? '#F44336'
         : '#0D47A1'; // uploaded
  };

  useEffect(() => {
    fetch('/holdout_image_index.json')
      .then((res) => res.json())
      .then((data) => {
        setHoldoutImages(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("❌ Failed to load holdout image index:", err);
        setLoading(false);
      });
  }, []);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const previewUrl = URL.createObjectURL(file);
      const uploaded = {
        filename: file.name,
        actualLabel: 'uploaded',
        path: previewUrl,
        file,
        isUploaded: true,
      };
      setUploadedImage(uploaded);
      onImageSelect(uploaded);
    }
  };

  const handleDropdownChange = (e) => {
    const selected = holdoutImages.find(img => img.filename === e.target.value);
    if (selected) {
      const withPath = {
        ...selected,
        path: selected.url, // assign cloudinary url as .path
        isUploaded: false,
      };
      setUploadedImage(null);
      onImageSelect(withPath);
    }
  };

  return (
    <Box>
      <Typography variant="body1" sx={{ mb: 2, color: '#666666', textAlign: 'center' }}>
        Select a histology image from our test set or upload your own image to analyze with our AI model.
      </Typography>

          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel sx={{ color: '#666666' }}>Select image</InputLabel>
            <Select
              value={selectedImage && !selectedImage.isUploaded ? selectedImage.filename : ''}
              label="Select image"
              onChange={handleDropdownChange}
              sx={{
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: '#0D47A1',
                },
                '&:hover .MuiOutlinedInput-notchedOutline': {
                  borderColor: '#1E88E5',
                },
                '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                  borderColor: '#0D47A1',
                },
              }}
            >
              {loading ? (
                <MenuItem disabled>
                  <CircularProgress size={20} sx={{ mr: 2 }} /> Loading images...
                </MenuItem>
              ) : (
                holdoutImages.map((image) => (
                  <MenuItem key={image.filename} value={image.filename}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <img
                          src={image.url}
                          alt={image.filename}
                          style={{ width: 24, height: 24, objectFit: 'cover', marginRight: 10, borderRadius: 4 }}
                        />
                        <Typography>{image.filename}</Typography>
                      </Box>
                      <Chip
                        label={image.actualLabel}
                        size="small"
                        sx={{
                          backgroundColor: getLabelColor(image.actualLabel),
                          color: 'white',
                          fontWeight: 600,
                        }}
                      />
                    </Box>
                  </MenuItem>
                ))
              )}
            </Select>
          </FormControl>

      <Button
        variant="outlined"
        startIcon={<UploadFile />}
        component="label"
        fullWidth
        sx={{
          borderRadius: 20,
          borderColor: '#0D47A1',
          color: '#0D47A1',
          fontWeight: 600,
          mb: 2,
          py: 1.5,
          '&:hover': {
            backgroundColor: 'rgba(13, 71, 161, 0.08)',
            borderColor: '#1976D2',
          },
        }}
      >
        Upload Image
        <input
          type="file"
          accept="image/*"
          hidden
          ref={fileInput}
          onChange={handleFileChange}
        />
      </Button>

          {uploadedImage && (
            <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
              <Chip
                icon={<UploadFile />}
                label={uploadedImage.filename}
                color="secondary"
                sx={{ fontWeight: 600 }}
              />
              <Typography variant="body2" color="text.secondary">
                (Uploaded image selected)
              </Typography>
            </Box>
          )}

      <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap', justifyContent: 'center' }}>
        <Chip label="Benign" size="small" sx={{ backgroundColor: '#4CAF50', color: 'white', fontWeight: 600 }} />
        <Chip label="Malignant" size="small" sx={{ backgroundColor: '#F44336', color: 'white', fontWeight: 600 }} />
        <Chip label="Uploaded" size="small" sx={{ backgroundColor: '#0D47A1', color: 'white', fontWeight: 600 }} />
      </Box>
    </Box>
  );
}
