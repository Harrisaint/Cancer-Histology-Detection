# Cancer Histology Detection - React Frontend

A React-based frontend that mirrors the functionality of the original Streamlit app for breast cancer histology image classification.

## Features

- **Modern, Playful Theme**: Bold orange (#FF6B00) and black (#1A1A1A) color palette, bubbly UI, and smooth animations
- **Image Selection & Upload**: Dropdown to select from a list of sample histology images **or upload your own image**
- **Image Display**: Shows the selected or uploaded image with filename caption
- **Prediction Results**: Displays predicted label (benign/malignant), confidence level, and actual label
- **Success/Error Feedback**: Visual indicators for correct/incorrect predictions
- **Real API Integration**: Uploaded images are sent to the backend for real prediction
- **Modern UI**: Built with Material-UI for a clean, professional look

## Current Status

- **Frontend supports both mock data and real API integration.**
- If you upload an image, it will be sent to the backend `/api/predict` endpoint for prediction.
- If you select a sample image, a mock prediction is shown.

## Getting Started

1. **Install dependencies** (if not already done):
   ```bash
   npm install
   npm install @fontsource/poppins @mui/icons-material
   ```

2. **Start the development server**:
   ```bash
   npm start
   ```

3. **Open your browser** and navigate to [http://localhost:3000](http://localhost:3000)

## API Integration

### **Image Upload & Prediction**
- When a user uploads an image, the frontend sends a POST request to `/api/predict` with the image as form data:
  - **Endpoint:** `POST /api/predict`
  - **Form field:** `image` (the uploaded file)
- The backend should respond with JSON:
  ```json
  {
    "predictedLabel": "benign" | "malignant",
    "confidence": 0.92,
    "actualLabel": "benign" // optional
  }
  ```
- If the backend is not running, the frontend will show an error for uploaded images.
- Selecting a sample image will always use mock prediction logic.

### **Connecting the Backend**
- Make sure your backend exposes a `/api/predict` endpoint that accepts image uploads and returns the expected JSON.
- The frontend and backend should be served from the same domain (or use a proxy in development).

## Component Structure

- `Header.js` - App title and description
- `HeroSection.js` - Hero landing section with call to action and illustration
- `ImageSelector.js` - Dropdown for image selection and upload
- `ImageDisplay.js` - Image display with caption
- `PredictionResult.js` - Prediction results and feedback (calls real API for uploads)
- `FloatingActionButton.js` - Orange-themed floating action button
- `App.js` - Main app component with state management and theme
- `About.js` - About page

## Theme & Styling

- **Primary color:** Black/white (backgrounds, text)
- **Accent color:** Bright orange (#FF6B00)
- **Font:** [Poppins](https://fonts.google.com/specimen/Poppins) via `@fontsource/poppins`
- **Material UI v5** with custom theme and large border radii
- **Icons:** Material UI Icons (`@mui/icons-material`)
- **Animations:** MUI Fade, Grow, and Zoom

## Troubleshooting

**Missing font or icon errors?**
- Make sure you have installed both:
  ```bash
  npm install @fontsource/poppins @mui/icons-material
  ```

**Global style errors?**
- The app uses the `Global` component from `@emotion/react` for global styles. If you see errors, ensure you are not using `createGlobalStyle` (which is from styled-components).

**Other issues?**
- Try deleting `node_modules` and running `npm install` again.
- Make sure you are running Node.js 16+ and npm 7+.

## Backend Integration

When ready to connect to the backend:
1. Ensure the backend exposes a `/api/predict` endpoint as described above.
2. The frontend will automatically POST uploaded images to this endpoint.
3. No frontend code changes are needed for basic integration.

## Technologies Used

- React 19
- Material-UI (MUI) v5
- Emotion (for styling)
- Poppins Google Font
- Material UI Icons
