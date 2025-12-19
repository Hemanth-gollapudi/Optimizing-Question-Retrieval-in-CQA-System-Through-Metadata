import React, { useState } from 'react'
import './Sidebar.css'

function Sidebar({ collapsed, chats = [], activeChatId, onNewChat, onSwitchChat, onToggleFavorite, onDeleteChat }) {
    const [hoveredChatId, setHoveredChatId] = useState(null)

    const handleChatClick = (e, chatId) => {
        // Don't switch if clicking on action buttons
        if (e.target.closest('.chat-actions')) {
            return
        }
        onSwitchChat(chatId)
    }

    const handleFavoriteClick = (e, chatId) => {
        e.stopPropagation()
        onToggleFavorite(chatId)
    }

    const handleDeleteClick = (e, chatId) => {
        e.stopPropagation()
        onDeleteChat(chatId)
    }

    return (
        <div id="side-chat-history" className={collapsed ? 'collapsed' : ''}>
            <div className="sidebar-header">
                <div id="history-title">Chat History</div>
                <button
                    className="new-chat-button"
                    onClick={onNewChat}
                    title="New Chat"
                    aria-label="New Chat"
                >
                    +
                </button>
            </div>
            <div className="chat-list">
                {chats.length === 0 ? (
                    <div className="no-chats">No chats yet. Click + to start!</div>
                ) : (
                    chats.map((chat) => (
                        <div
                            key={chat.id}
                            className={`history-item ${chat.id === activeChatId ? 'active' : ''} ${chat.favorite ? 'favorite' : ''}`}
                            onClick={(e) => handleChatClick(e, chat.id)}
                            onMouseEnter={() => setHoveredChatId(chat.id)}
                            onMouseLeave={() => setHoveredChatId(null)}
                            title={chat.title}
                        >
                            <span className="chat-title">{chat.favorite && '⭐ '}{chat.title}</span>
                            {hoveredChatId === chat.id && (
                                <div className="chat-actions" onClick={(e) => e.stopPropagation()}>
                                    <button
                                        className="chat-action-btn favorite-btn"
                                        onClick={(e) => handleFavoriteClick(e, chat.id)}
                                        title={chat.favorite ? "Remove from favorites" : "Add to favorites"}
                                        aria-label={chat.favorite ? "Remove from favorites" : "Add to favorites"}
                                    >
                                        {chat.favorite ? '★' : '☆'}
                                    </button>
                                    <button
                                        className="chat-action-btn delete-btn"
                                        onClick={(e) => handleDeleteClick(e, chat.id)}
                                        title="Delete chat"
                                        aria-label="Delete chat"
                                    >
                                        🗑️
                                    </button>
                                </div>
                            )}
                        </div>
                    ))
                )}
            </div>
        </div>
    )
}

export default Sidebar

