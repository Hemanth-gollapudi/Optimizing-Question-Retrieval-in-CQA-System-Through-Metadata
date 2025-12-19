import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Import config (note: this runs in Node.js context, so we need to read from a JS file)
// For now, we'll use environment variables or hardcode, but you can also create a vite.config.constants.js
const BACKEND_PORT = process.env.VITE_BACKEND_PORT || 5000
const FRONTEND_PORT = process.env.VITE_FRONTEND_PORT || 3000
const BACKEND_HOST = process.env.VITE_BACKEND_HOST || 'localhost'

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

