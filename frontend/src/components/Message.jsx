import React from 'react'
import './Message.css'

function Message({ text, sender }) {
    return (
        <div className={`message-row ${sender}-row`}>
            {sender === 'bot' && (
                <div className="profile-icon bot-icon">🤖</div>
            )}
            <div className={`message ${sender}-message`}>
                {text}
            </div>
            {sender === 'user' && (
                <div className="profile-icon user-icon">👤</div>
            )}
        </div>
    )
}

export default Message

