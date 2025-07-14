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
  Button
} from '@mui/material';
import { Image, Science, UploadFile } from '@mui/icons-material';
import { useRef, useState } from 'react';

// Mock data - in real app, this would come from the holdout_test_set directory
const mockImageOptions = [
  { filename: "benign_sample_1.jpg", actualLabel: "benign", path: "/sample-images/benign_1.jpg" },
  { filename: "benign_sample_2.jpg", actualLabel: "benign", path: "/sample-images/benign_2.jpg" },
  { filename: "malignant_sample_1.jpg", actualLabel: "malignant", path: "/sample-images/malignant_1.jpg" },
  { filename: "malignant_sample_2.jpg", actualLabel: "malignant", path: "/sample-images/malignant_2.jpg" },
  { filename: "benign_sample_3.jpg", actualLabel: "benign", path: "/sample-images/benign_3.jpg" },
  { filename: "malignant_sample_3.jpg", actualLabel: "malignant", path: "/sample-images/malignant_3.jpg" },
];

export default function ImageSelector({ selectedImage, onImageSelect }) {
  const getLabelColor = (label) => {
    return label === 'benign' ? '#4CAF50' : '#F44336';
  };
  const fileInput = useRef();
  const [uploadedImage, setUploadedImage] = useState(null);

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

  return (
    <Box sx={{ maxWidth: 800, margin: '2rem auto', padding: '0 1rem' }}>
      <Card sx={{ 
        background: 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.2)',
      }}>
        <CardContent sx={{ p: 4 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <Image sx={{ fontSize: 32, color: '#FF6B00', mr: 2 }} />
            <Typography variant="h4" sx={{ fontWeight: 600, color: '#1A1A1A' }}>
              Choose or Upload an Image
            </Typography>
          </Box>
          
          <Typography variant="body1" sx={{ mb: 3, color: '#666666' }}>
            Select a histology image from our test set or upload your own image to analyze with our AI model.
          </Typography>

          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel sx={{ color: '#666666' }}>Select image</InputLabel>
            <Select
              value={selectedImage && !selectedImage.isUploaded ? selectedImage.filename : ''}
              label="Select image"
              onChange={(e) => {
                const selected = mockImageOptions.find(img => img.filename === e.target.value);
                onImageSelect(selected);
                setUploadedImage(null);
              }}
              sx={{
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: '#FF6B00',
                },
                '&:hover .MuiOutlinedInput-notchedOutline': {
                  borderColor: '#FF8533',
                },
                '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                  borderColor: '#FF6B00',
                },
              }}
            >
              {mockImageOptions.map((image) => (
                <MenuItem key={image.filename} value={image.filename}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Science sx={{ mr: 2, color: '#FF6B00' }} />
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
              ))}
            </Select>
          </FormControl>

          <Button
            variant="outlined"
            startIcon={<UploadFile />}
            component="label"
            sx={{
              borderRadius: 20,
              borderColor: '#FF6B00',
              color: '#FF6B00',
              fontWeight: 600,
              mb: 2,
              '&:hover': {
                backgroundColor: 'rgba(255, 107, 0, 0.08)',
                borderColor: '#FF8533',
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

          <Box sx={{ mt: 3, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Chip 
              label="Benign" 
              size="small" 
              sx={{ 
                backgroundColor: '#4CAF50',
                color: 'white',
                fontWeight: 600,
              }}
            />
            <Chip 
              label="Malignant" 
              size="small" 
              sx={{ 
                backgroundColor: '#F44336',
                color: 'white',
                fontWeight: 600,
              }}
            />
            <Chip 
              label="Uploaded" 
              size="small" 
              sx={{ 
                backgroundColor: '#FF6B00',
                color: 'white',
                fontWeight: 600,
              }}
            />
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
} 