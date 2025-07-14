# Conversation Monitoring Application

## 1. Project Overview

The Conversation Monitoring application is a full-stack platform for tracking, analyzing, and annotating client conversations across industries such as healthcare and veterinary services. It provides a dashboard for high-level monitoring, detailed conversation views, collaborative comment threads, annotation tools, and performance metrics to help teams improve client interactions and outcomes.

### **Primary Features**
- **Home Screen**: Dashboard showing all clients, categories (e.g., healthcare, veterinary), and summary statistics (e.g., total conversations, flagged items, performance trends).
- **Conversations Screen**: View individual conversations, add threaded comments, annotate transcripts, and review performance metrics (e.g., sentiment, response time).

---

## 2. Tech Stack

- **Frontend**: React, TailwindCSS
- **Backend**: Node.js, Express, Sequelize
- **Database**: PostgreSQL
- **Dev Tools**: Docker, Vite, ESLint, Prettier

---

## 3. Development Environment Setup

### 🔧 **Prerequisites**
- **Node.js**: v18+ recommended
- **PostgreSQL**: v13+ (local or Docker)
- **(Optional) Docker**: For containerized development

### 📦 **Backend Setup**
```bash
cd backend
npm install
# Set environment variables in .env (see .env.example)
npx sequelize-cli db:migrate
npm run dev
```

### 💻 **Frontend Setup**
```bash
cd frontend
npm install
npm run dev
```

---

## 4. Running the Application

- **Frontend** runs on [http://localhost:3000](http://localhost:3000)
- **Backend** runs on [http://localhost:5000](http://localhost:5000)
- The frontend is configured to proxy API requests to the backend (see `frontend/vite.config.js` or `package.json` proxy field).

To run both simultaneously:
- Start the backend (`npm run dev` in `backend/`)
- Start the frontend (`npm run dev` in `frontend/`)

---

## 5. API Endpoints

### **/api/clients**
- **GET /api/clients**
  - Description: Fetch all clients
  - Output: List of client objects

### **/api/comments**
- **GET /api/comments/:interactionId**
  - Description: Fetch all comments for a given interaction
  - Input: `interactionId` (URL param)
  - Output: List of comment objects
- **POST /api/comments**
  - Description: Add a new comment to an interaction
  - Input: `{ interactionId, author, content, comment_type }`
  - Output: Created comment object

### **/api/interactions**
- **GET /api/interactions**
  - Description: Fetch all interactions
  - Output: List of interaction objects
- **GET /api/interactions/:id**
  - Description: Fetch a single interaction by ID
  - Input: `id` (URL param)
  - Output: Interaction object

### **/api/performance**
- **GET /api/performance/:clientId**
  - Description: Fetch performance metrics for a client
  - Input: `clientId` (URL param)
  - Output: Performance metrics object

---

## 6. Data Models / Schemas

### **Comment**
- `id`: integer, primary key
- `interactionId`: integer, foreign key
- `author`: string
- `content`: text
- `comment_type`: enum ('general', 'flag', 'note')
- `exported`: boolean
- `created_at`: timestamp

### **Client**
- `id`: integer, primary key
- `name`: string
- `category`: enum ('healthcare', 'veterinary', ...)
- `created_at`: timestamp

### **Interaction**
- `id`: integer, primary key
- `clientId`: integer, foreign key
- `transcript`: text
- `priority`: enum ('low', 'medium', 'high')
- `created_at`: timestamp

### **Enums**
- `priority`: 'low', 'medium', 'high'
- `comment_type`: 'general', 'flag', 'note'
- `category`: 'healthcare', 'veterinary', ...

---

## 7. Troubleshooting

- **Database not running**: Ensure PostgreSQL is running locally or via Docker.
- **CORS errors**: Check proxy settings in frontend config.
- **Migrations failing**: Check your `.env` DB credentials and run:
  ```bash
  npx sequelize-cli db:drop && npx sequelize-cli db:create && npx sequelize-cli db:migrate
  ```
- **Resetting your local DB**: Use the command above to drop, create, and migrate the database.

---

## 8. Contributing

- **Branching strategy**: Use `main` for production, `dev` for integration, and feature branches (`feature/xyz`) for new work.
- **Linting/formatting**: Run `npm run lint` and `npm run format` before submitting a PR.
- **Pull Requests**: Submit PRs to the `dev` branch. Include a clear description and reference any related issues.
- **Code reviews**: At least one approval required before merging.
- **Contact**: For questions or approvals, contact the project maintainer or lead developer listed in `CONTRIBUTORS.md`.

---

**Happy coding!**
