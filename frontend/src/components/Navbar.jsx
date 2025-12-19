import React from 'react'
import './Navbar.css'

function Navbar({ sidebarCollapsed, onToggleSidebar, currentPage, onNavigate }) {
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
            <div id="profile-area">
                <span id="profile-icon" title="User Profile">👤</span>
            </div>
        </div>
    )
}

export default Navbar

