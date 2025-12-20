import React, { useState } from 'react'
import './InputArea.css'

function InputArea({ onSendMessage, disabled = false, isLoading = false, onAbandon }) {
    const [input, setInput] = useState('')

    const handleSubmit = (e) => {
        e.preventDefault()
        if (input.trim() && !disabled) {
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

    const handleAbandon = () => {
        if (onAbandon) {
            onAbandon()
        }
    }

    return (
        <div id="input-area">
            <form id="input-wrapper" onSubmit={handleSubmit}>
                <button type="button" id="search-icon" aria-label="Search">🔍</button>
                <input
                    type="text"
                    id="user-input"
                    placeholder={disabled ? "Waiting for response..." : "Type your message..."}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    autoComplete="off"
                    disabled={disabled}
                />
                {isLoading && (
                    <button
                        type="button"
                        id="abandon-button"
                        onClick={handleAbandon}
                        aria-label="Abandon request"
                        title="Abandon this request"
                    >
                        <span className="abandon-icon"></span>
                    </button>
                )}
                <button type="submit" id="send-button" disabled={disabled}>
                    {disabled ? "..." : "Send"}
                </button>
            </form>
        </div>
    )
}

export default InputArea

