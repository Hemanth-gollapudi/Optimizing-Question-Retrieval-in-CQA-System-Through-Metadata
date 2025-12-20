import React, { useState, useEffect, useCallback, useMemo } from 'react'
import Navbar from './components/Navbar'
import Sidebar from './components/Sidebar'
import ChatPage from './pages/ChatPage'
import ModelPerformance from './pages/ModelPerformance'
import ConfirmModal from './components/ConfirmModal'
import { loadChatsFromStorage, saveChatsToStorage, createNewChat, generateChatTitle, sortChats } from './utils/chatUtils'
import './App.css'

function App() {
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
    const [currentPage, setCurrentPage] = useState(() => {
        // Load saved page from localStorage on initial render
        const savedPage = localStorage.getItem('currentPage')
        return savedPage === 'performance' ? 'performance' : 'chat'
    })
    const [chats, setChats] = useState([])
    const [activeChatId, setActiveChatId] = useState(null)
    const [deleteConfirm, setDeleteConfirm] = useState({ isOpen: false, chatId: null })

    // Save current page to localStorage whenever it changes
    useEffect(() => {
        localStorage.setItem('currentPage', currentPage)
    }, [currentPage])

    // Load chats from localStorage on mount
    useEffect(() => {
        const loadedChats = loadChatsFromStorage()
        if (loadedChats.length > 0) {
            setChats(loadedChats)
            // Only set activeChatId if we're on chat page
            if (currentPage === 'chat') {
                setActiveChatId(loadedChats[0].id)
            }
        } else {
            // Create first chat if none exist and we're on chat page
            if (currentPage === 'chat') {
                const firstChat = createNewChat(Date.now().toString())
                setChats([firstChat])
                setActiveChatId(firstChat.id)
            }
        }
    }, []) // eslint-disable-line react-hooks/exhaustive-deps

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
        setDeleteConfirm({ isOpen: true, chatId })
    }

    const confirmDelete = () => {
        const { chatId } = deleteConfirm
        if (chatId) {
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
        setDeleteConfirm({ isOpen: false, chatId: null })
    }

    const cancelDelete = () => {
        setDeleteConfirm({ isOpen: false, chatId: null })
    }

    // Update messages for the active chat
    // Use useCallback to prevent creating new function reference on every render
    const handleUpdateMessages = useCallback((chatId, messages) => {
        setChats(prev => prev.map(chat => {
            if (chat.id === chatId) {
                // Only update if messages actually changed (deep comparison)
                const currentMessagesStr = JSON.stringify(chat.messages)
                const newMessagesStr = JSON.stringify(messages)

                // If messages are the same, don't update (prevents unnecessary re-renders)
                if (currentMessagesStr === newMessagesStr) {
                    return chat
                }

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
    }, [])

    const activeChat = useMemo(() => {
        return chats.find(chat => chat.id === activeChatId)
    }, [chats, activeChatId])

    // Memoize messages to prevent unnecessary re-renders
    const activeChatMessages = useMemo(() => {
        return activeChat?.messages || []
    }, [activeChat?.messages])

    return (
        <div className="app">
            <Navbar
                sidebarCollapsed={sidebarCollapsed}
                onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
                currentPage={currentPage}
                onNavigate={setCurrentPage}
            />
            <div className="content-area">
                {currentPage !== 'performance' && (
                    <Sidebar
                        collapsed={sidebarCollapsed}
                        chats={sortChats(chats)}
                        activeChatId={activeChatId}
                        onNewChat={handleNewChat}
                        onSwitchChat={handleSwitchChat}
                        onToggleFavorite={handleToggleFavorite}
                        onDeleteChat={handleDeleteChat}
                    />
                )}
                <div className={`main-content ${currentPage === 'performance' ? 'full-width' : ''}`}>
                    {currentPage === 'chat' && activeChat && (
                        <ChatPage
                            key={activeChat.id} // Force remount when chatId changes to prevent state leaks
                            chatId={activeChat.id}
                            messages={activeChatMessages}
                            onUpdateMessages={handleUpdateMessages}
                        />
                    )}
                    {currentPage === 'performance' && (
                        <ModelPerformance />
                    )}
                </div>
            </div>
            <ConfirmModal
                isOpen={deleteConfirm.isOpen}
                title="Delete Chat"
                message="Are you sure you want to delete this chat? This action cannot be undone and all messages in this chat will be permanently deleted."
                confirmText="Delete"
                cancelText="Cancel"
                confirmColor="danger"
                onConfirm={confirmDelete}
                onCancel={cancelDelete}
            />
        </div>
    )
}

export default App
