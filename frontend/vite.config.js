import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Configuration (can be overridden via environment variables)
// VITE_BACKEND_PORT links to BACKEND_PORT for consistency
// Falls back to BACKEND_PORT if VITE_BACKEND_PORT is not set
const BACKEND_PORT = process.env.VITE_BACKEND_PORT || process.env.BACKEND_PORT || 5000
const FRONTEND_PORT = process.env.VITE_FRONTEND_PORT || process.env.FRONTEND_PORT || 3000
const BACKEND_HOST = process.env.VITE_BACKEND_HOST || process.env.BACKEND_HOST || 'localhost'

export default defineConfig({
    plugins: [react()],
    server: {
        port: parseInt(FRONTEND_PORT),
        proxy: {
            '/api': {
                target: `http://${BACKEND_HOST}:${BACKEND_PORT}`,
                changeOrigin: true
            }
        }
    }
})

