/**
 * Frontend Configuration Constants
 * Centralized configuration for React frontend
 * 
 * Can be overridden via environment variables (VITE_*)
 */

// Backend API Configuration (can be overridden via VITE_BACKEND_HOST and VITE_BACKEND_PORT)
const BACKEND_HOST = import.meta.env.VITE_BACKEND_HOST || 'localhost';
const BACKEND_PORT = import.meta.env.VITE_BACKEND_PORT || '5000';
const BACKEND_BASE_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`;

export const API_CONFIG = {
    BASE_URL: BACKEND_BASE_URL,
    CHAT_ENDPOINT: '/api/chat',
    STATUS_ENDPOINT: '/api/status',
    HEALTH_ENDPOINT: '/health',
    DOCS_URL: '/docs',
};

// Full API endpoint
export const API_ENDPOINT = `${API_CONFIG.BASE_URL}${API_CONFIG.CHAT_ENDPOINT}`;

// Frontend Configuration (can be overridden via VITE_FRONTEND_PORT)
export const FRONTEND_CONFIG = {
    PORT: parseInt(import.meta.env.VITE_FRONTEND_PORT || '3000'),
    HOST: 'localhost',
    BASE_URL: `http://localhost:${import.meta.env.VITE_FRONTEND_PORT || '3000'}`,
};

// UI Configuration
export const UI_CONFIG = {
    SIDEBAR_WIDTH: 300,
    NAVBAR_HEIGHT: 52,
    MAX_INPUT_WIDTH: 800,
    ICON_BUTTON_SIZE: 45,
    TOGGLE_SIZE: 30,
};

// Message Configuration
export const MESSAGE_CONFIG = {
    MAX_WIDTH_PERCENT: 70,
    ANIMATION_DURATION: 400, // milliseconds
    TYPING_DELAY: 300, // milliseconds before showing bot response
};

// Colors (matching CSS variables)
export const COLORS = {
    PRIMARY_GRADIENT: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    SECONDARY_GRADIENT: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    ACCENT_GRADIENT: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    BG_DARK: '#1e1e1e',
    BG_MEDIUM: '#252526',
    TEXT_LIGHT: '#f0f0f0',
    TEXT_FAINT: '#999',
};
