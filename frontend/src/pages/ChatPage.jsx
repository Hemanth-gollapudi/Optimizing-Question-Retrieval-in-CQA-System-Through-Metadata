import React, { useState, useRef, useEffect } from 'react'
import MessageList from '../components/MessageList'
import InputArea from '../components/InputArea'
import { API_ENDPOINT } from '../config'
import './ChatPage.css'

function ChatPage({ chatId, messages: initialMessages, onUpdateMessages }) {
    // Initialize with initialMessages if provided, otherwise empty array
    const [messages, setMessages] = useState(() => {
        // Always use initialMessages if provided, even if empty array
        return initialMessages || []
    })

    // Ensure messages are set on first render if initialMessages is provided
    useEffect(() => {
        if (initialMessages && initialMessages.length > 0) {
            // Check if we need to initialize
            if (messages.length === 0 || JSON.stringify(messages) !== JSON.stringify(initialMessages)) {
                setMessages(initialMessages)
                lastInitialMessagesRef.current = initialMessages
                messagesLengthRef.current = initialMessages.length
            }
        }
    }, [initialMessages]) // Run when initialMessages changes
    const [isLoading, setIsLoading] = useState(false)
    const messageEndRef = useRef(null)
    const currentChatIdRef = useRef(chatId)
    const abortControllerRef = useRef(null)
    const isMountedRef = useRef(true)
    const skipUpdateRef = useRef(false)
    const lastInitialMessagesRef = useRef(initialMessages)
    const messagesLengthRef = useRef(messages.length)
    const lastSentMessagesRef = useRef(null)

    // Initialize messages on mount and when chatId changes
    useEffect(() => {
        // Initialize on first mount or when chatId changes
        if (currentChatIdRef.current !== chatId || messages.length === 0) {
            const newMessages = initialMessages || []
            if (newMessages.length > 0 || currentChatIdRef.current !== chatId) {
                setMessages(newMessages)
                currentChatIdRef.current = chatId
                lastInitialMessagesRef.current = newMessages
                messagesLengthRef.current = newMessages.length
            }

            // Cancel any pending requests when switching chats
            if (abortControllerRef.current && currentChatIdRef.current !== chatId) {
                abortControllerRef.current.abort()
                abortControllerRef.current = null
            }
        }
    }, [chatId, initialMessages]) // Include initialMessages to ensure we get the latest

    // Sync messages from parent when initialMessages changes (but only if chatId matches)
    // This handles cases where parent was updated from another source
    useEffect(() => {
        if (currentChatIdRef.current === chatId && initialMessages) {
            // Use ref to compare, not state (avoids dependency on messages state)
            const lastInitial = lastInitialMessagesRef.current
            const lastInitialStr = JSON.stringify(lastInitial)
            const currentInitialStr = JSON.stringify(initialMessages)

            // Only sync if initialMessages actually changed from what we last saw
            if (lastInitialStr !== currentInitialStr) {
                // Get current messages from state using a function to avoid dependency
                setMessages(currentMessages => {
                    const currentMessagesStr = JSON.stringify(currentMessages)
                    const currentLength = messagesLengthRef.current
                    const hasMoreMessages = initialMessages.length > currentLength
                    const messagesAreDifferent = currentMessagesStr !== currentInitialStr

                    // Only update if:
                    // 1. Parent has more messages (response came from elsewhere), OR
                    // 2. Messages are actually different (not just same content, new reference)
                    if (hasMoreMessages || messagesAreDifferent) {
                        lastInitialMessagesRef.current = initialMessages
                        skipUpdateRef.current = true // Skip parent update to prevent loop
                        messagesLengthRef.current = initialMessages.length
                        return initialMessages
                    }
                    // No change needed
                    return currentMessages
                })

                // Always update the ref to prevent re-checking the same initialMessages
                if (lastInitialStr !== currentInitialStr) {
                    lastInitialMessagesRef.current = initialMessages
                }
            }
        }
    }, [initialMessages, chatId]) // No messages dependency - use functional setState instead

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

    // Update messages length ref when messages change
    useEffect(() => {
        messagesLengthRef.current = messages.length
    }, [messages])

    // Notify parent when messages change (but only if we're still on the same chat)
    useEffect(() => {
        // Skip update if we're syncing from parent
        if (skipUpdateRef.current) {
            skipUpdateRef.current = false
            return
        }

        // Only update parent if messages actually changed (deep comparison)
        const messagesStr = JSON.stringify(messages)
        const lastSentStr = JSON.stringify(lastSentMessagesRef.current)

        if (messagesStr !== lastSentStr && onUpdateMessages && chatId && currentChatIdRef.current === chatId) {
            lastSentMessagesRef.current = messages
            onUpdateMessages(chatId, messages)
        }
    }, [messages, chatId, onUpdateMessages])

    const scrollToBottom = () => {
        messageEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    const handleAbandon = () => {
        // Abort the current request
        if (abortControllerRef.current) {
            abortControllerRef.current.abort()
            abortControllerRef.current = null
        }

        // Stop loading
        setIsLoading(false)

        // Remove any pending bot message (keep only user messages)
        // Find the last user message and keep everything up to and including it
        const lastUserMessageIndex = messages.length - 1
        if (lastUserMessageIndex >= 0 && messages[lastUserMessageIndex].sender === 'user') {
            // Keep all messages as is (user message is already there, no bot response to remove)
            // Just ensure we're not showing loading state
        } else {
            // If somehow there's a bot message, remove it
            const userMessagesOnly = messages.filter(msg => msg.sender === 'user')
            setMessages(userMessagesOnly)
            if (onUpdateMessages && chatId) {
                onUpdateMessages(chatId, userMessagesOnly)
            }
        }
    }

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

        // Show loading indicator
        if (currentChatIdRef.current === sendingChatId) {
            setIsLoading(true)
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
                if (currentChatIdRef.current === sendingChatId) {
                    setIsLoading(false)
                }
                return
            }

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}))
                const errorMessage = errorData.detail || `HTTP error! status: ${response.status}`
                const errorMessages = [...updatedMessages, { text: `Error: ${errorMessage}`, sender: 'bot' }]

                // Only update local state if we're still on the same chat
                if (currentChatIdRef.current === sendingChatId) {
                    setMessages(errorMessages)
                    setIsLoading(false)
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
                if (currentChatIdRef.current === sendingChatId) {
                    setIsLoading(false)
                }
                return
            }

            setTimeout(() => {
                // Check if request was aborted
                if (abortController.signal.aborted) {
                    if (currentChatIdRef.current === sendingChatId) {
                        setIsLoading(false)
                    }
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
                    setIsLoading(false)
                }
            }, 300)
        } catch (error) {
            // Ignore abort errors
            if (error.name === 'AbortError') {
                if (currentChatIdRef.current === sendingChatId) {
                    setIsLoading(false)
                }
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
                setIsLoading(false)
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
                <MessageList messages={messages} messageEndRef={messageEndRef} isLoading={isLoading} />
                <InputArea
                    onSendMessage={sendMessage}
                    disabled={isLoading}
                    isLoading={isLoading}
                    onAbandon={handleAbandon}
                />
            </div>
        </div>
    )
}

export default ChatPage

