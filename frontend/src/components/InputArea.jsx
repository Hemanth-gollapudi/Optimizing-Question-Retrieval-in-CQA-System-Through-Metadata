import React, { useState } from 'react'
import './InputArea.css'

function InputArea({ onSendMessage }) {
    const [input, setInput] = useState('')

    const handleSubmit = (e) => {
        e.preventDefault()
        if (input.trim()) {
            onSendMessage(input)
            setInput('')
        }
    }

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSubmit(e)
        }
    }

    return (
        <div id="input-area">
            <form id="input-wrapper" onSubmit={handleSubmit}>
                <button type="button" id="search-icon" aria-label="Search">🔍</button>
                <input
                    type="text"
                    id="user-input"
                    placeholder="Type your message..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    autoComplete="off"
                />
                <button type="submit" id="send-button">Send</button>
            </form>
        </div>
    )
}

export default InputArea

