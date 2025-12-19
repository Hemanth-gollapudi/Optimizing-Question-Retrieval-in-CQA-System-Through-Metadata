import React, { useState, useEffect } from 'react'
import Navbar from './components/Navbar'
import Sidebar from './components/Sidebar'
import ChatPage from './pages/ChatPage'
import { loadChatsFromStorage, saveChatsToStorage, createNewChat, generateChatTitle, sortChats } from './utils/chatUtils'
import './App.css'

function App() {
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
    const [currentPage, setCurrentPage] = useState('chat')
    const [chats, setChats] = useState([])
    const [activeChatId, setActiveChatId] = useState(null)

    // Load chats from localStorage on mount
    useEffect(() => {
        const loadedChats = loadChatsFromStorage()
        if (loadedChats.length > 0) {
            setChats(loadedChats)
            setActiveChatId(loadedChats[0].id)
        } else {
            // Create first chat if none exist
            const firstChat = createNewChat(Date.now().toString())
            setChats([firstChat])
            setActiveChatId(firstChat.id)
        }
    }, [])

    // Save chats to localStorage whenever they change
    useEffect(() => {
        if (chats.length > 0) {
            saveChatsToStorage(chats)
        }
    }, [chats])

    // Create a new chat
    const handleNewChat = () => {
        const newChatId = Date.now().toString()
        const newChat = createNewChat(newChatId)
        setChats(prev => [newChat, ...prev])
        setActiveChatId(newChatId)
    }

    // Switch to a different chat
    const handleSwitchChat = (chatId) => {
        setActiveChatId(chatId)
    }

    // Toggle favorite status of a chat
    const handleToggleFavorite = (chatId) => {
        setChats(prev => prev.map(chat => {
            if (chat.id === chatId) {
                return {
                    ...chat,
                    favorite: !chat.favorite,
                    updatedAt: new Date().toISOString()
                }
            }
            return chat
        }))
    }

    // Delete a chat
    const handleDeleteChat = (chatId) => {
        if (window.confirm('Are you sure you want to delete this chat?')) {
            setChats(prev => {
                const filtered = prev.filter(chat => chat.id !== chatId)
                // If we deleted the active chat, switch to another one
                if (chatId === activeChatId) {
                    if (filtered.length > 0) {
                        setActiveChatId(filtered[0].id)
                    } else {
                        // Create a new chat if all were deleted
                        const newChat = createNewChat(Date.now().toString())
                        setActiveChatId(newChat.id)
                        return [newChat]
                    }
                }
                return filtered
            })
        }
    }

    // Update messages for the active chat
    const handleUpdateMessages = (chatId, messages) => {
        setChats(prev => prev.map(chat => {
            if (chat.id === chatId) {
                const updatedChat = {
                    ...chat,
                    messages,
                    updatedAt: new Date().toISOString()
                }

                // Update title from first user message if it's still "New Chat"
                if (updatedChat.title === "New Chat" || updatedChat.title === "New Chat...") {
                    const firstUserMessage = messages.find(msg => msg.sender === 'user')
                    if (firstUserMessage) {
                        updatedChat.title = generateChatTitle(firstUserMessage.text)
                    }
                }

                return updatedChat
            }
            return chat
        }))
    }

    const activeChat = chats.find(chat => chat.id === activeChatId)

    return (
        <div className="app">
            <Navbar
                sidebarCollapsed={sidebarCollapsed}
                onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
                currentPage={currentPage}
                onNavigate={setCurrentPage}
            />
            <div className="content-area">
                <Sidebar
                    collapsed={sidebarCollapsed}
                    chats={sortChats(chats)}
                    activeChatId={activeChatId}
                    onNewChat={handleNewChat}
                    onSwitchChat={handleSwitchChat}
                    onToggleFavorite={handleToggleFavorite}
                    onDeleteChat={handleDeleteChat}
                />
                <div className="main-content">
                    {currentPage === 'chat' && activeChat && (
                        <ChatPage
                            key={activeChat.id} // Force remount when chatId changes to prevent state leaks
                            chatId={activeChat.id}
                            messages={activeChat.messages}
                            onUpdateMessages={handleUpdateMessages}
                        />
                    )}
                </div>
            </div>
        </div>
    )
}

export default App

