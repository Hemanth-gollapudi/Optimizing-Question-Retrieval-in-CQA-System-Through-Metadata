import React from 'react'
import Message from './Message'
import './MessageList.css'

function MessageList({ messages, messageEndRef }) {
    return (
        <div id="message-display">
            {messages.map((msg, index) => (
                <Message key={index} text={msg.text} sender={msg.sender} />
            ))}
            <div ref={messageEndRef} />
        </div>
    )
}

export default MessageList

