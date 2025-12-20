import React, { useState, useRef, useEffect } from 'react'
import './Navbar.css'

function Navbar({ sidebarCollapsed, onToggleSidebar, currentPage, onNavigate }) {
    const [dropdownOpen, setDropdownOpen] = useState(false)
    const dropdownRef = useRef(null)

    // Close dropdown when clicking outside
    useEffect(() => {
        function handleClickOutside(event) {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setDropdownOpen(false)
            }
        }

        if (dropdownOpen) {
            document.addEventListener('mousedown', handleClickOutside)
        }

        return () => {
            document.removeEventListener('mousedown', handleClickOutside)
        }
    }, [dropdownOpen])

    const handleProfileClick = () => {
        setDropdownOpen(!dropdownOpen)
    }

    const handleMenuItemClick = (page) => {
        onNavigate(page)
        setDropdownOpen(false)
    }

    return (
        <div id="navbar">
            <button
                id="sidebar-toggle"
                aria-label="Toggle Sidebar"
                onClick={onToggleSidebar}
            >
                ☰
            </button>
            <span id="navbar-title">CQA</span>
            <div id="nav-buttons">
                <button
                    className={`nav-button ${currentPage === 'chat' ? 'active' : ''}`}
                    data-page="chat"
                    onClick={() => onNavigate('chat')}
                >
                    Chat
                </button>
            </div>
            <div id="profile-area" ref={dropdownRef}>
                <span
                    id="profile-icon"
                    title="User Profile"
                    onClick={handleProfileClick}
                >
                    👤
                </span>
                {dropdownOpen && (
                    <div className="profile-dropdown">
                        <div
                            className="dropdown-item"
                            onClick={() => handleMenuItemClick('performance')}
                        >
                            📊 Model Performance
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}

export default Navbar

