/**
 * Utility functions for chat management
 */

/**
 * Generate a title from the first user message
 * @param {string} message - The first user message
 * @returns {string} - A shortened title (max 50 chars)
 */
export function generateChatTitle(message) {
    if (!message || !message.trim()) {
        return "New Chat"
    }

    // Remove extra whitespace and limit length
    const cleaned = message.trim().replace(/\s+/g, ' ')
    const maxLength = 50

    if (cleaned.length <= maxLength) {
        return cleaned
    }

    // Truncate and add ellipsis
    return cleaned.substring(0, maxLength - 3) + "..."
}

/**
 * Load chats from localStorage
 * @returns {Array} - Array of chat objects
 */
export function loadChatsFromStorage() {
    try {
        const stored = localStorage.getItem('chatHistory')
        if (stored) {
            const chats = JSON.parse(stored)
            // Ensure backward compatibility: add favorite property if missing
            return chats.map(chat => ({
                ...chat,
                favorite: chat.favorite || false
            }))
        }
    } catch (error) {
        console.error('Error loading chats from storage:', error)
    }
    return []
}

/**
 * Save chats to localStorage
 * @param {Array} chats - Array of chat objects
 */
export function saveChatsToStorage(chats) {
    try {
        localStorage.setItem('chatHistory', JSON.stringify(chats))
    } catch (error) {
        console.error('Error saving chats to storage:', error)
    }
}

/**
 * Create a new chat object
 * @param {string} id - Unique chat ID
 * @returns {Object} - New chat object
 */
export function createNewChat(id) {
    return {
        id,
        title: "New Chat",
        messages: [
            { text: "Hello! I am the bot. My messages are on the left.", sender: 'bot' }
        ],
        favorite: false,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
    }
}

/**
 * Sort chats: favorites first, then by updatedAt (most recent first)
 * @param {Array} chats - Array of chat objects
 * @returns {Array} - Sorted array of chats
 */
export function sortChats(chats) {
    return [...chats].sort((a, b) => {
        // Favorites first
        if (a.favorite && !b.favorite) return -1
        if (!a.favorite && b.favorite) return 1
        // Then by most recently updated
        return new Date(b.updatedAt) - new Date(a.updatedAt)
    })
}

