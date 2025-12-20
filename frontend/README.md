# CQA Chat Frontend - React Application

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Project Structure

```
frontend/
├── src/
│   ├── components/     # Reusable React components
│   │   ├── Navbar.jsx
│   │   ├── Sidebar.jsx
│   │   ├── Message.jsx
│   │   ├── MessageList.jsx
│   │   └── InputArea.jsx
│   ├── pages/          # Page components
│   │   └── ChatPage.jsx
│   ├── App.jsx         # Main app component
│   ├── main.jsx        # Entry point
│   └── index.css       # Global styles
├── index.html
├── package.json
└── vite.config.js
```

## Features

- ✅ Modular React components
- ✅ Scrollable message area
- ✅ Responsive design
- ✅ Smooth animations
- ✅ Auto-scroll to latest message

