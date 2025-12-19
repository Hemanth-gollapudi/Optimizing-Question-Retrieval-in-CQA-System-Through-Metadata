import React, { useState, useRef, useEffect } from 'react'
import MessageList from '../components/MessageList'
import InputArea from '../components/InputArea'
import { API_ENDPOINT } from '../config'
import './ChatPage.css'

function ChatPage({ chatId, messages: initialMessages, onUpdateMessages }) {
    const [messages, setMessages] = useState(initialMessages || [])
    const messageEndRef = useRef(null)
    const currentChatIdRef = useRef(chatId)
    const abortControllerRef = useRef(null)
    const isMountedRef = useRef(true)

    // Initialize messages when chatId changes (switching to different chat)
    useEffect(() => {
        // Only reset messages when chatId actually changes (not just when initialMessages updates)
        if (currentChatIdRef.current !== chatId) {
            setMessages(initialMessages || [])
            currentChatIdRef.current = chatId

            // Cancel any pending requests when switching chats
            if (abortControllerRef.current) {
                abortControllerRef.current.abort()
                abortControllerRef.current = null
            }
        } else {
            // If same chatId but initialMessages updated (from parent), sync if local state is behind
            // This handles cases where parent was updated but local state wasn't
            if (initialMessages && initialMessages.length > messages.length) {
                setMessages(initialMessages)
            }
        }
    }, [chatId, initialMessages, messages.length])

    // Track mount status separately (only on actual mount/unmount)
    useEffect(() => {
        isMountedRef.current = true
        return () => {
            isMountedRef.current = false
            if (abortControllerRef.current) {
                abortControllerRef.current.abort()
                abortControllerRef.current = null
            }
        }
    }, [])


    // Notify parent when messages change (but only if we're still on the same chat)
    useEffect(() => {
        if (onUpdateMessages && chatId && currentChatIdRef.current === chatId) {
            onUpdateMessages(chatId, messages)
        }
    }, [messages, chatId, onUpdateMessages])

    const scrollToBottom = () => {
        messageEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    const sendMessage = async (text) => {
        if (!text.trim()) return

        // Capture the chatId at the time of sending (important for race condition prevention)
        const sendingChatId = chatId
        currentChatIdRef.current = sendingChatId

        // Create abort controller for this request
        const abortController = new AbortController()
        abortControllerRef.current = abortController

        // Add user message
        const userMessage = { text, sender: 'user' }
        const updatedMessages = [...messages, userMessage]

        // Only update if we're still on the same chat
        if (currentChatIdRef.current === sendingChatId) {
            setMessages(updatedMessages)
        }

        // Always update parent with the correct chatId (even if user switched)
        // This ensures the message is saved to the correct chat
        if (onUpdateMessages && sendingChatId) {
            onUpdateMessages(sendingChatId, updatedMessages)
        }

        try {
            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: text }),
                signal: abortController.signal
            })

            // Check if request was aborted (user switched chats)
            if (abortController.signal.aborted) {
                return
            }

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}))
                const errorMessage = errorData.detail || `HTTP error! status: ${response.status}`
                const errorMessages = [...updatedMessages, { text: `Error: ${errorMessage}`, sender: 'bot' }]

                // Only update local state if we're still on the same chat
                if (currentChatIdRef.current === sendingChatId) {
                    setMessages(errorMessages)
                }

                // Always update parent with the correct chatId
                if (onUpdateMessages && sendingChatId) {
                    onUpdateMessages(sendingChatId, errorMessages)
                }
                return
            }

            const data = await response.json()
            const botResponse = data.response || "Error: Model response key not found."

            // Check again if request was aborted before updating
            if (abortController.signal.aborted) {
                return
            }

            setTimeout(() => {
                // Check if request was aborted
                if (abortController.signal.aborted) {
                    return
                }

                const finalMessages = [...updatedMessages, { text: botResponse, sender: 'bot' }]

                // Always update parent first (this ensures response is saved to correct chat)
                // This is the source of truth
                if (onUpdateMessages && sendingChatId) {
                    onUpdateMessages(sendingChatId, finalMessages)
                }

                // Update local state if we're still on the same chat
                // Check currentChatIdRef instead of chatId prop to avoid stale closures
                if (currentChatIdRef.current === sendingChatId) {
                    setMessages(finalMessages)
                }
            }, 300)
        } catch (error) {
            // Ignore abort errors
            if (error.name === 'AbortError') {
                return
            }

            console.error('Error fetching model response:', error)
            const errorMessages = [...updatedMessages, {
                text: `Error: Could not connect to the chatbot API. Please make sure the server is running.`,
                sender: 'bot'
            }]

            // Only update local state if we're still on the same chat
            if (currentChatIdRef.current === sendingChatId) {
                setMessages(errorMessages)
            }

            // Always update parent with the correct chatId
            if (onUpdateMessages && sendingChatId) {
                onUpdateMessages(sendingChatId, errorMessages)
            }
        } finally {
            // Clear abort controller if this was the active request
            if (abortControllerRef.current === abortController) {
                abortControllerRef.current = null
            }
        }
    }

    return (
        <div className="chat-page">
            <div id="main-chat-container">
                <MessageList messages={messages} messageEndRef={messageEndRef} />
                <InputArea onSendMessage={sendMessage} />
            </div>
        </div>
    )
}

export default ChatPage

