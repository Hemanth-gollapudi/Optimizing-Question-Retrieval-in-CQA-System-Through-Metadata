import React, { useState, useEffect, useRef } from 'react'
import { API_CONFIG } from '../config'
import './ModelPerformance.css'

function ModelPerformance() {
    const [testResults, setTestResults] = useState(null)
    const [loading, setLoading] = useState(true)
    const [runningTests, setRunningTests] = useState(false)
    const [error, setError] = useState(null)

    useEffect(() => {
        loadTestResults()
    }, [])

    const loadTestResults = async () => {
        setLoading(true)
        setError(null)
        try {
            const response = await fetch(`${API_CONFIG.BASE_URL}/api/test-results`)
            const data = await response.json()

            if (data.error) {
                setError(data.error)
                setTestResults(null)
            } else {
                setTestResults(data)
                setError(null)
            }
        } catch (err) {
            setError('Failed to load test results')
            setTestResults(null)
        } finally {
            setLoading(false)
        }
    }

    const runTests = async () => {
        setRunningTests(true)
        setError(null)
        try {
            const response = await fetch(`${API_CONFIG.BASE_URL}/api/run-tests`, {
                method: 'POST'
            })
            const data = await response.json()

            if (data.status === 'success' && data.results) {
                setTestResults(data.results)
                setError(null)
            } else {
                setError(data.message || 'Tests failed to complete')
            }
        } catch (err) {
            setError('Failed to run tests: ' + err.message)
        } finally {
            setRunningTests(false)
        }
    }

    if (loading) {
        return (
            <div className="performance-page">
                <div className="loading-container">
                    <div className="loading-content">
                        <div className="spinner-wrapper">
                            <div className="spinner"></div>
                            <div className="spinner-ring"></div>
                        </div>
                        <h2>Loading Performance Data</h2>
                        <p>Fetching test results...</p>
                    </div>
                </div>
            </div>
        )
    }

    if (error && !testResults) {
        return (
            <div className="performance-page">
                <div className="performance-container">
                    <div className="empty-state">
                        <div className="empty-state-icon error-icon">⚠️</div>
                        <h1>📊 Model Performance</h1>
                        <div className="empty-state-content">
                            <h2>Unable to Load Results</h2>
                            <p className="error-text">{error}</p>
                            <button
                                className="run-tests-button large-button"
                                onClick={runTests}
                                disabled={runningTests}
                            >
                                {runningTests ? (
                                    <>
                                        <span className="button-spinner"></span>
                                        Running Tests...
                                    </>
                                ) : (
                                    <>
                                        🚀 Run Tests
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        )
    }

    if (!testResults || !testResults.metrics) {
        return (
            <div className="performance-page">
                <div className="performance-container">
                    <div className="empty-state">
                        <div className="empty-state-icon">📊</div>
                        <h1>Model Performance Dashboard</h1>
                        <div className="empty-state-content">
                            <h2>No Test Results Available</h2>
                            <p>Get started by running comprehensive tests to analyze your model's performance across different metrics.</p>
                            <div className="features-list">
                                <div className="feature-item">
                                    <span className="feature-icon">✅</span>
                                    <span>Success Rate Analysis</span>
                                </div>
                                <div className="feature-item">
                                    <span className="feature-icon">🎯</span>
                                    <span>Dataset Hit Rate Metrics</span>
                                </div>
                                <div className="feature-item">
                                    <span className="feature-icon">⏱️</span>
                                    <span>Response Time Statistics</span>
                                </div>
                                <div className="feature-item">
                                    <span className="feature-icon">📈</span>
                                    <span>Category & Topic Breakdowns</span>
                                </div>
                            </div>
                            <button
                                className="run-tests-button large-button"
                                onClick={runTests}
                                disabled={runningTests}
                            >
                                {runningTests ? (
                                    <>
                                        <span className="button-spinner"></span>
                                        Running Tests...
                                    </>
                                ) : (
                                    <>
                                        🚀 Run Tests Now
                                    </>
                                )}
                            </button>
                            {runningTests && (
                                <div className="test-progress">
                                    <p>⏳ This may take a few minutes. Please wait...</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        )
    }

    // Show fancy loader when tests are running
    if (runningTests) {
        return <FancyLoader />
    }

    const { metrics } = testResults

    return (
        <div className="performance-page">
            <div className="performance-container">
                <div className="performance-header">
                    <h1>📊 Model Performance</h1>
                    <div className="header-actions">
                        <button
                            className="refresh-button"
                            onClick={loadTestResults}
                            disabled={loading}
                        >
                            🔄 Refresh
                        </button>
                        <button
                            className="run-tests-button"
                            onClick={runTests}
                            disabled={runningTests}
                        >
                            {runningTests ? (
                                <>
                                    <span className="button-spinner"></span>
                                    Running...
                                </>
                            ) : (
                                '🚀 Run Tests'
                            )}
                        </button>
                    </div>
                </div>

                {error && (
                    <div className="error-banner">
                        {error}
                    </div>
                )}

                {/* Overall Metrics */}
                <div className="metrics-grid">
                    <MetricCard
                        title="Success Rate"
                        value={`${(metrics.success_rate * 100).toFixed(1)}%`}
                        color="#667eea"
                        icon="✅"
                    />
                    <MetricCard
                        title="Dataset Hit Rate"
                        value={`${(metrics.dataset_hit_rate * 100).toFixed(1)}%`}
                        color="#4facfe"
                        icon="🎯"
                    />
                    <MetricCard
                        title="LLM Fallback Rate"
                        value={`${(metrics.llm_fallback_rate * 100).toFixed(1)}%`}
                        color="#f093fb"
                        icon="🤖"
                    />
                    <MetricCard
                        title="Total Questions"
                        value={metrics.total_questions}
                        color="#764ba2"
                        icon="❓"
                    />
                </div>

                {/* Similarity Score Distribution */}
                <div className="chart-section">
                    <h2>📈 Similarity Score Distribution</h2>
                    <div className="similarity-stats">
                        <AnimatedStatItem label="Mean:" value={metrics.similarity_stats.mean.toFixed(3)} delay={0.1} />
                        <AnimatedStatItem label="Min:" value={metrics.similarity_stats.min.toFixed(3)} delay={0.2} />
                        <AnimatedStatItem label="Max:" value={metrics.similarity_stats.max.toFixed(3)} delay={0.3} />
                        <AnimatedStatItem
                            label="Above Threshold:"
                            value={`${metrics.similarity_stats.above_threshold}/${metrics.similarity_stats.count}`}
                            delay={0.4}
                        />
                    </div>
                    <SimilarityChart similarityStats={metrics.similarity_stats} />
                </div>

                {/* Response Time Stats */}
                <div className="chart-section">
                    <h2>⏱️ Response Time Statistics</h2>
                    <div className="response-time-stats">
                        <AnimatedStatItem label="Mean:" value={`${metrics.response_time_stats.mean.toFixed(2)}s`} delay={0.1} />
                        <AnimatedStatItem label="Min:" value={`${metrics.response_time_stats.min.toFixed(2)}s`} delay={0.2} />
                        <AnimatedStatItem label="Max:" value={`${metrics.response_time_stats.max.toFixed(2)}s`} delay={0.3} />
                    </div>
                </div>

                {/* Category Breakdown */}
                <div className="chart-section">
                    <h2>📂 Performance by Category</h2>
                    <CategoryBreakdown categories={metrics.by_category} />
                </div>

                {/* Topic Breakdown */}
                <div className="chart-section">
                    <h2>🏷️ Performance by Topic</h2>
                    <TopicBreakdown topics={metrics.by_topic} />
                </div>
            </div>
        </div>
    )
}

function MetricCard({ title, value, color, icon }) {
    const [displayValue, setDisplayValue] = useState('0')
    const [isVisible, setIsVisible] = useState(false)
    const cardRef = useRef(null)

    useEffect(() => {
        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting && !isVisible) {
                    setIsVisible(true)
                    animateValue(value)
                }
            },
            { threshold: 0.1 }
        )

        if (cardRef.current) {
            observer.observe(cardRef.current)
        }

        return () => {
            if (cardRef.current) {
                observer.unobserve(cardRef.current)
            }
        }
    }, [value, isVisible])

    const animateValue = (targetValue) => {
        // Convert to string to handle both numbers and strings
        const valueStr = String(targetValue)
        // Extract number from value (handles percentages and plain numbers)
        const numMatch = valueStr.match(/[\d.]+/)
        if (!numMatch) {
            setDisplayValue(valueStr)
            return
        }

        const targetNum = parseFloat(numMatch[0])
        const suffix = valueStr.replace(/[\d.]+/, '')
        const duration = 1500
        const startTime = Date.now()
        const startValue = 0

        const animate = () => {
            const now = Date.now()
            const elapsed = now - startTime
            const progress = Math.min(elapsed / duration, 1)

            // Easing function for smooth animation
            const easeOutQuart = 1 - Math.pow(1 - progress, 4)
            const currentValue = startValue + (targetNum - startValue) * easeOutQuart

            setDisplayValue(currentValue.toFixed(suffix.includes('%') ? 1 : 0) + suffix)

            if (progress < 1) {
                requestAnimationFrame(animate)
            } else {
                setDisplayValue(targetValue)
            }
        }

        requestAnimationFrame(animate)
    }

    return (
        <div className="metric-card" style={{ borderTopColor: color }} ref={cardRef}>
            <div className="metric-icon">{icon}</div>
            <div className="metric-content">
                <div className="metric-title">{title}</div>
                <div className="metric-value" style={{ color }}>
                    <span className="metric-value-text">{displayValue}</span>
                </div>
            </div>
        </div>
    )
}

function SimilarityChart({ similarityStats }) {
    const [animatedMean, setAnimatedMean] = useState(0)
    const [animatedThreshold, setAnimatedThreshold] = useState(0)
    const threshold = 0.55
    const maxScore = 1.0
    const meanPercent = (similarityStats.mean / maxScore) * 100
    const thresholdPercent = (threshold / maxScore) * 100
    const hasAnimated = useRef(false)

    useEffect(() => {
        if (!hasAnimated.current) {
            hasAnimated.current = true
            // Animate mean
            animateValue(0, meanPercent, 1000, setAnimatedMean)
            // Animate threshold with delay
            setTimeout(() => {
                animateValue(0, thresholdPercent, 800, setAnimatedThreshold)
            }, 300)
        }
    }, [meanPercent, thresholdPercent])

    const animateValue = (start, end, duration, setter) => {
        const startTime = Date.now()
        const animate = () => {
            const now = Date.now()
            const elapsed = now - startTime
            const progress = Math.min(elapsed / duration, 1)
            const easeOutCubic = 1 - Math.pow(1 - progress, 3)
            const current = start + (end - start) * easeOutCubic
            setter(current)
            if (progress < 1) {
                requestAnimationFrame(animate)
            }
        }
        requestAnimationFrame(animate)
    }

    return (
        <div className="similarity-chart">
            <div className="chart-bar-container">
                <div className="chart-bar">
                    <div
                        className="chart-fill mean-fill"
                        style={{ width: `${animatedMean}%` }}
                    >
                        <span className="chart-label">Mean: {similarityStats.mean.toFixed(3)}</span>
                    </div>
                </div>
                <div className="chart-bar">
                    <div
                        className="chart-fill threshold-fill"
                        style={{ width: `${animatedThreshold}%` }}
                    >
                        <span className="chart-label">Threshold: {threshold}</span>
                    </div>
                </div>
            </div>
            <div className="chart-legend">
                <div className="legend-item">
                    <span className="legend-color" style={{ background: '#667eea' }}></span>
                    <span>Mean Similarity</span>
                </div>
                <div className="legend-item">
                    <span className="legend-color" style={{ background: '#f093fb' }}></span>
                    <span>Threshold (0.55)</span>
                </div>
            </div>
        </div>
    )
}

function CategoryBreakdown({ categories }) {
    const [animatedBars, setAnimatedBars] = useState({})
    const hasAnimated = useRef(false)

    useEffect(() => {
        if (!hasAnimated.current) {
            hasAnimated.current = true
            const newAnimated = {}
            Object.entries(categories).forEach(([category, stats], index) => {
                setTimeout(() => {
                    const datasetPercent = (stats.dataset / stats.total) * 100
                    const llmPercent = (stats.llm / stats.total) * 100
                    animateBar(category, 'dataset', datasetPercent, 800)
                    setTimeout(() => {
                        animateBar(category, 'llm', llmPercent, 800)
                    }, 200)
                }, index * 150)
            })
        }
    }, [categories])

    const animateBar = (category, type, targetPercent, duration) => {
        const startTime = Date.now()
        const animate = () => {
            const now = Date.now()
            const elapsed = now - startTime
            const progress = Math.min(elapsed / duration, 1)
            const easeOutCubic = 1 - Math.pow(1 - progress, 3)
            const current = targetPercent * easeOutCubic

            setAnimatedBars(prev => ({
                ...prev,
                [`${category}-${type}`]: current
            }))

            if (progress < 1) {
                requestAnimationFrame(animate)
            } else {
                setAnimatedBars(prev => ({
                    ...prev,
                    [`${category}-${type}`]: targetPercent
                }))
            }
        }
        requestAnimationFrame(animate)
    }

    return (
        <div className="category-breakdown">
            {Object.entries(categories).map(([category, stats]) => {
                const datasetPercent = animatedBars[`${category}-dataset`] ?? 0
                const llmPercent = animatedBars[`${category}-llm`] ?? 0
                return (
                    <div key={category} className="category-item">
                        <div className="category-header">
                            <span className="category-name">{category}</span>
                            <span className="category-count">{stats.total} questions</span>
                        </div>
                        <div className="category-bars">
                            <div className="bar-group">
                                <div className="bar-label">Dataset</div>
                                <div className="bar-container">
                                    <div
                                        className="bar-fill dataset-bar"
                                        style={{ width: `${datasetPercent}%` }}
                                    >
                                        {datasetPercent > 5 ? stats.dataset : ''}
                                    </div>
                                </div>
                            </div>
                            <div className="bar-group">
                                <div className="bar-label">LLM</div>
                                <div className="bar-container">
                                    <div
                                        className="bar-fill llm-bar"
                                        style={{ width: `${llmPercent}%` }}
                                    >
                                        {llmPercent > 5 ? stats.llm : ''}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )
            })}
        </div>
    )
}

function TopicBreakdown({ topics }) {
    const sortedTopics = Object.entries(topics).sort((a, b) => b[1].avg_similarity - a[1].avg_similarity)
    const [animatedWidths, setAnimatedWidths] = useState({})
    const hasAnimated = useRef(false)

    useEffect(() => {
        if (!hasAnimated.current) {
            hasAnimated.current = true
            sortedTopics.forEach(([topic, stats], index) => {
                setTimeout(() => {
                    const targetWidth = stats.avg_similarity * 100
                    animateTopicBar(topic, targetWidth, 1000)
                }, index * 100)
            })
        }
    }, [topics])

    const animateTopicBar = (topic, targetWidth, duration) => {
        const startTime = Date.now()
        const animate = () => {
            const now = Date.now()
            const elapsed = now - startTime
            const progress = Math.min(elapsed / duration, 1)
            const easeOutCubic = 1 - Math.pow(1 - progress, 3)
            const current = targetWidth * easeOutCubic

            setAnimatedWidths(prev => ({
                ...prev,
                [topic]: current
            }))

            if (progress < 1) {
                requestAnimationFrame(animate)
            } else {
                setAnimatedWidths(prev => ({
                    ...prev,
                    [topic]: targetWidth
                }))
            }
        }
        requestAnimationFrame(animate)
    }

    return (
        <div className="topic-breakdown">
            {sortedTopics.map(([topic, stats]) => {
                const animatedWidth = animatedWidths[topic] ?? 0
                return (
                    <div key={topic} className="topic-item">
                        <div className="topic-header">
                            <span className="topic-name">{topic}</span>
                            <span className="topic-stats">
                                {stats.total} questions • Avg Similarity: {stats.avg_similarity.toFixed(3)}
                            </span>
                        </div>
                        <div className="topic-bar">
                            <div
                                className="topic-fill"
                                style={{ width: `${animatedWidth}%` }}
                            ></div>
                        </div>
                    </div>
                )
            })}
        </div>
    )
}

function FancyLoader() {
    const [currentStep, setCurrentStep] = useState(0)
    const steps = [
        { icon: '✓', text: 'Initializing test suite', completed: true },
        { icon: '⏳', text: 'Processing test questions', completed: false },
        { icon: '⏳', text: 'Calculating metrics', completed: false },
        { icon: '○', text: 'Generating reports', completed: false },
    ]

    useEffect(() => {
        const interval = setInterval(() => {
            setCurrentStep(prev => {
                if (prev < steps.length - 1) {
                    return prev + 1
                }
                return prev
            })
        }, 3000) // Move to next step every 3 seconds

        return () => clearInterval(interval)
    }, [])

    return (
        <div className="performance-page">
            <div className="fancy-loader-container">
                <div className="fancy-loader-content">
                    <div className="loader-orb-container">
                        <div className="loader-orb orb-1"></div>
                        <div className="loader-orb orb-2"></div>
                        <div className="loader-orb orb-3"></div>
                        <div className="loader-orb orb-4"></div>
                        <div className="loader-center">
                            <div className="loader-icon">🚀</div>
                        </div>
                    </div>
                    <h2 className="loader-title">Running Performance Tests</h2>
                    <p className="loader-subtitle">Analyzing model performance across all metrics...</p>
                    <div className="loader-progress">
                        <div className="loader-progress-bar">
                            <div className="loader-progress-fill"></div>
                        </div>
                    </div>
                    <div className="loader-steps">
                        {steps.map((step, index) => (
                            <div
                                key={index}
                                className={`loader-step ${index <= currentStep ? 'active' : ''} ${index < currentStep ? 'completed' : ''}`}
                            >
                                <span className="step-icon">
                                    {index < currentStep ? '✓' : index === currentStep ? '⏳' : '○'}
                                </span>
                                <span>{step.text}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
}

function AnimatedStatItem({ label, value, delay = 0 }) {
    const [displayValue, setDisplayValue] = useState('0')
    const [isVisible, setIsVisible] = useState(false)
    const itemRef = useRef(null)

    useEffect(() => {
        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting && !isVisible) {
                    setIsVisible(true)
                    setTimeout(() => {
                        animateValue(value)
                    }, delay * 1000)
                }
            },
            { threshold: 0.1 }
        )

        if (itemRef.current) {
            observer.observe(itemRef.current)
        }

        return () => {
            if (itemRef.current) {
                observer.unobserve(itemRef.current)
            }
        }
    }, [value, delay, isVisible])

    const animateValue = (targetValue) => {
        const numMatch = targetValue.match(/[\d.]+/)
        if (!numMatch) {
            setDisplayValue(targetValue)
            return
        }

        const targetNum = parseFloat(numMatch[0])
        const suffix = targetValue.replace(/[\d.]+/, '')
        const duration = 1000
        const startTime = Date.now()
        const startValue = 0

        const animate = () => {
            const now = Date.now()
            const elapsed = now - startTime
            const progress = Math.min(elapsed / duration, 1)
            const easeOutQuart = 1 - Math.pow(1 - progress, 4)
            const currentValue = startValue + (targetNum - startValue) * easeOutQuart

            const decimals = targetValue.includes('.') ? (targetValue.split('.')[1]?.length || 2) : 0
            setDisplayValue(currentValue.toFixed(decimals) + suffix)

            if (progress < 1) {
                requestAnimationFrame(animate)
            } else {
                setDisplayValue(targetValue)
            }
        }

        requestAnimationFrame(animate)
    }

    return (
        <div className="stat-item" ref={itemRef}>
            <span className="stat-label">{label}</span>
            <span className="stat-value">{displayValue}</span>
        </div>
    )
}

export default ModelPerformance

