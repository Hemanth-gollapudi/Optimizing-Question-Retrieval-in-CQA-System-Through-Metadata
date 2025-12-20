import React from 'react'
import Message from './Message'
import './MessageList.css'

function MessageList({ messages, messageEndRef, isLoading = false }) {
    return (
        <div id="message-display">
            {messages.map((msg, index) => (
                <Message key={index} text={msg.text} sender={msg.sender} />
            ))}
            {isLoading && (
                <div className="message-row bot-row">
                    <div className="profile-icon bot-icon">🤖</div>
                    <div className="message bot-message typing-indicator">
                        <span className="typing-dot"></span>
                        <span className="typing-dot"></span>
                        <span className="typing-dot"></span>
                    </div>
                </div>
            )}
            <div ref={messageEndRef} />
        </div>
    )
}

export default MessageList

