class Chatbot {
    constructor() {
        this.messages = [];
        this.isLoading = false;
        this.currentTypingInterval = null;
        this.sessionId = this.generateSessionId();
        this.conversationStats = null;
        this.initializeElements();
        this.setupEventListeners();
        this.autoResizeTextarea();
        this.checkServerStatus();
        this.setupSessionControls();
        
        // Load initial conversation stats
        this.loadInitialStats();
        
        // Set initial status color after a short delay to ensure elements are loaded
        setTimeout(() => {
            if (this.statusPingOuter && this.statusPingInner) {
                this.updateStatus('Ready', '#4ade80');
            }
        }, 100);
    }

    generateSessionId() {
        // Check for existing session in localStorage first
        const existingSessionId = localStorage.getItem('chatbot_session_id');
        
        if (existingSessionId) {
            console.log('Resuming existing session:', existingSessionId);
            return existingSessionId;
        }
        
        // Generate a unique session ID for conversation persistence
        const newSessionId = 'session_' + Date.now() + '_' + this._secureRandomString(9);
        
        // Store the new session ID in localStorage
        localStorage.setItem('chatbot_session_id', newSessionId);
        console.log('Created new session:', newSessionId);
        
        return newSessionId;
    }
    
    createNewSession() {
        // Force create a new session and store it
        const newSessionId = 'session_' + Date.now() + '_' + this._secureRandomString(9);
        localStorage.setItem('chatbot_session_id', newSessionId);
        this.sessionId = newSessionId;
        
        console.log('Forced new session creation:', newSessionId);
        
        // Clear conversation and reset stats
        this.clearConversation();
        
        return newSessionId;
    }

    /**
     * Generate a cryptographically secure random string
     * @param {number} length - number of bytes, not final string length
     */
    _secureRandomString(length) {
        const array = new Uint8Array(length);
        window.crypto.getRandomValues(array);
        // Convert bytes to hex string
        return Array.from(array).map(b => ('0' + b.toString(16)).slice(-2)).join('');
    }

    async getConversationStats() {
        try {
            const servers = window.location.hostname === 'antonjijo.github.io' 
                ? ['https://nvidia-nim-bot.onrender.com', 'https://Nvidia.pythonanywhere.com']
                : ['http://localhost:5000'];

            for (const serverURL of servers) {
                try {
                    const response = await fetch(`${serverURL}/api/conversation/stats?session_id=${this.sessionId}`);
                    if (response.ok) {
                        this.conversationStats = await response.json();
                        return this.conversationStats;
                    }
                } catch (error) {
                    console.warn(`Failed to get stats from ${serverURL}:`, error);
                }
            }
        } catch (error) {
            console.error('Failed to get conversation stats:', error);
        }
        return null;
    }

    async clearConversation() {
        try {
            const servers = window.location.hostname === 'antonjijo.github.io' 
                ? ['https://nvidia-nim-bot.onrender.com', 'https://Nvidia.pythonanywhere.com']
                : ['http://localhost:5000'];

            for (const serverURL of servers) {
                try {
                    const response = await fetch(`${serverURL}/api/conversation/clear`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            session_id: this.sessionId,
                            keep_system_prompt: true
                        })
                    });
                    
                    if (response.ok) {
                        // Clear the UI messages as well
                        this.messages = [];
                        this.chatMessages.innerHTML = '';
                        
                        // Add welcome message back
                        this.addWelcomeMessage();
                        
                        // Update conversation stats
                        const result = await response.json();
                        this.conversationStats = result.stats;
                        this.updateConversationStatsUI();
                        
                        console.log('Conversation cleared successfully');
                        return true;
                    }
                } catch (error) {
                    console.warn(`Failed to clear conversation on ${serverURL}:`, error);
                }
            }
        } catch (error) {
            console.error('Failed to clear conversation:', error);
        }
        return false;
    }

    addWelcomeMessage() {
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'message bot-message';
        welcomeDiv.innerHTML = `
            <div class="message-avatar">
                <img src="Main_Logo.svg" alt="NVIDIA" class="nvidia-logo" />
            </div>
            <div class="message-content">
                <p>Hello! I'm your NVIDIA-powered chatbot with advanced capabilities:</p>
                <p><strong>Multiple AI Models available:</strong></p>
                <p><strong>Meta:</strong> Llama 4 Maverick 17B</p>
                <p><strong>DeepSeek:</strong> R1</p>
                <p><strong>Qwen:</strong> Qwen2.5 Coder 32B</p>
                <p><strong>Qwen:</strong> Qwen3 Coder 480B</p>
                <p><strong>DeepSeek:</strong> V3.1</p>
                <p><strong>OpenAI:</strong> GPT-OSS 120B</p>
                <p><strong>Qwen:</strong> Qwen3 235B A22B</p>
                <p><strong>Google:</strong> Gemma 3 27B IT</p>
                <p><strong>Select</strong> your preferred model above and start chatting!</p>
            </div>
        `;
        this.chatMessages.appendChild(welcomeDiv);
        this.attachCopyListeners(welcomeDiv.querySelector('.message-content'));
    }

    initializeElements() {
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.charCount = document.getElementById('charCount');
        this.statusPing = document.getElementById('statusPing');
        this.statusPingOuter = document.getElementById('statusPingOuter');
        this.statusPingInner = document.getElementById('statusPingInner');
        this.modelSelect = document.getElementById('modelSelect');
        
        // Custom dropdown elements
        this.customSelectTrigger = document.getElementById('customSelectTrigger');
        this.customSelectDropdown = document.getElementById('customSelectDropdown');
        this.selectedOption = document.getElementById('selectedOption');
        
        this.initializeCustomDropdown();
        this.setupGlobalScrolling();
    }

    setupEventListeners() {
        this.sendButton.addEventListener('click', (e) => {
            if (this.sendButton.disabled) {
                e.preventDefault();
                this.showErrorPopup('Maximum character limit (5000) reached!');
                return;
            }
            if (this.isLoading) {
                this.stopGeneration();
            } else {
                this.sendMessage();
            }
        });

        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (this.sendButton.disabled) {
                    this.showErrorPopup('Maximum character limit (5000) reached!');
                    return;
                }
                this.sendMessage();
            }
        });

        this.messageInput.addEventListener('input', (e) => {
            if (this.messageInput.value.includes("Hello! I'm your NVIDIA-powered chatbot with advanced capabilities")) {
                this.messageInput.value = this.messageInput.value.replace(/Hello! I'm your NVIDIA-powered chatbot with advanced capabilities[\s\S]*/, '');
            }
            if (this.messageInput.value.length >= 5000) {
                this.messageInput.value = this.messageInput.value.substring(0, 5000);
                this.showErrorPopup('Maximum character limit (5000) reached!');
                e.preventDefault();
            }
            this.updateCharCount();
            this.autoResizeTextarea(); // keep resizing on input
        });

        // Simple model dropdown change handler
        this.modelSelect.addEventListener('change', (e) => {
            this.handleModelChange(e.target.value);
        });
    }
    
    async handleModelChange(newModel) {
        // If this is the first model selection or no conversation exists, just switch
        if (!this.conversationStats || this.conversationStats.total_messages <= 1) {
            this.switchModel(newModel);
            return;
        }
        
        // Show confirmation dialog for model change
        this.showModelChangeConfirmation(newModel);
    }
    
    showModelChangeConfirmation(newModel) {
        const stats = this.conversationStats;
        const currentModel = stats.current_model || 'Unknown';
        
        // Create modal overlay
        const modalOverlay = document.createElement('div');
        modalOverlay.className = 'modal-overlay';
        modalOverlay.innerHTML = `
            <div class="confirmation-modal">
                <div class="modal-header">
                    <h3>Change Model & Start New Session?</h3>
                    <button class="modal-close" aria-label="Close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-content">
                    <div class="model-change-info">
                        <div class="current-session-stats">
                            <h4>Current Session</h4>
                            <div class="stats-grid">
                                <div class="stat-item">
                                    <span class="stat-label">Model:</span>
                                    <span class="stat-value">${this.getModelDisplayName(currentModel)}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Messages:</span>
                                    <span class="stat-value">${stats.total_messages}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Tokens:</span>
                                    <span class="stat-value">${stats.total_tokens.toLocaleString()}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Utilization:</span>
                                    <span class="stat-value">${stats.utilization_percent}%</span>
                                </div>
                            </div>
                        </div>
                        <div class="arrow-divider">
                            <i class="fas fa-arrow-down"></i>
                        </div>
                        <div class="new-session-info">
                            <h4>New Session</h4>
                            <div class="new-model-display">
                                <span class="new-model-name">${this.getModelDisplayName(newModel)}</span>
                                <span class="fresh-start">Fresh conversation start</span>
                            </div>
                        </div>
                    </div>
                    <div class="warning-notice">
                        <i class="fas fa-exclamation-triangle"></i>
                        <span>Changing models will start a new session. Your current conversation will be cleared.</span>
                    </div>
                </div>
                <div class="modal-actions">
                    <button class="cancel-btn" data-action="cancel">
                        <i class="fas fa-times"></i>
                        Keep Current Session
                    </button>
                    <button class="confirm-btn" data-action="confirm">
                        <i class="fas fa-check"></i>
                        Change Model & Start New
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modalOverlay);
        
        // Add event listeners
        const handleAction = (action) => {
            if (action === 'confirm') {
                this.switchModel(newModel);
                this.createNewSession();
            } else {
                // Reset the dropdown to the current model
                this.modelSelect.value = currentModel;
                this.updateCustomDropdownDisplay(currentModel);
            }
            document.body.removeChild(modalOverlay);
        };
        
        modalOverlay.querySelector('.cancel-btn').addEventListener('click', () => handleAction('cancel'));
        modalOverlay.querySelector('.confirm-btn').addEventListener('click', () => handleAction('confirm'));
        modalOverlay.querySelector('.modal-close').addEventListener('click', () => handleAction('cancel'));
        
        // Close on overlay click
        modalOverlay.addEventListener('click', (e) => {
            if (e.target === modalOverlay) {
                handleAction('cancel');
            }
        });
        
        // Close on Escape key
        const handleKeyDown = (e) => {
            if (e.key === 'Escape') {
                handleAction('cancel');
                document.removeEventListener('keydown', handleKeyDown);
            }
        };
        document.addEventListener('keydown', handleKeyDown);
    }
    
    switchModel(newModel) {
        console.log('Model changed to:', newModel);
        // Update the model select value
        this.modelSelect.value = newModel;
        this.updateCustomDropdownDisplay(newModel);
        
        // Hide loader until actual conversation starts
        const statsElement = document.getElementById('conversationStats');
        if (statsElement) {
            statsElement.style.display = 'none';
        }
        
        // Reset conversation stats - loader will only show after real usage
        this.conversationStats = null;
    }
    
    getModelDisplayName(modelValue) {
        const modelNames = {
            'meta/llama-4-maverick-17b-128e-instruct': 'Llama 4 Maverick 17B',
            'deepseek-ai/deepseek-r1': 'DeepSeek R1',
            'qwen/qwen2.5-coder-32b-instruct': 'Qwen2.5 Coder 32B',
            'qwen/qwen3-coder-480b-a35b-instruct': 'Qwen3 Coder 480B',
            'deepseek-ai/deepseek-v3.1': 'DeepSeek V3.1',
            'openai/gpt-oss-120b': 'GPT-OSS 120B',
            'qwen/qwen3-235b-a22b:free': 'Qwen3 235B A22B',
            'google/gemma-3-27b-it:free': 'Gemma 3 27B IT'
        };
        return modelNames[modelValue] || modelValue;
    }
    
    updateCustomDropdownDisplay(modelValue) {
        const displayName = this.getModelDisplayName(modelValue);
        if (this.selectedOption) {
            this.selectedOption.textContent = displayName;
        }
        
        // Update selected state in dropdown options
        const options = document.querySelectorAll('.select-option');
        options.forEach(option => {
            option.classList.remove('selected');
            if (option.dataset.value === modelValue) {
                option.classList.add('selected');
            }
        });
    }    
    
    setupSessionControls() {
        // Add session management controls to the UI
        const navbar = document.querySelector('.navbar-right');
        if (navbar) {
            // Create session controls container
            const sessionControls = document.createElement('div');
            sessionControls.className = 'navbar-item session-controls';
            sessionControls.innerHTML = `
                <button id="newSessionBtn" class="session-btn" title="Start New Session">
                    <i class="fas fa-plus"></i>
                </button>
            `;
            
            // Insert before the status indicator
            const statusContainer = navbar.querySelector('.status-container');
            if (statusContainer) {
                navbar.insertBefore(sessionControls, statusContainer);
            }
            
            // Add event listeners
            document.getElementById('newSessionBtn')?.addEventListener('click', () => {
                this.showNewSessionDialog();
            });
        }
    }
    
    async loadInitialStats() {
        // Load conversation stats on page load
        const stats = await this.getConversationStats();
        if (stats) {
            this.updateConversationStatsUI();
        }
    }
    
    showNewSessionDialog() {
        const currentStats = this.conversationStats;
        const currentModel = this.modelSelect.value;
        
        // Create modal overlay
        const modalOverlay = document.createElement('div');
        modalOverlay.className = 'modal-overlay';
        modalOverlay.innerHTML = `
            <div class="confirmation-modal">
                <div class="modal-header">
                    <h3>Start New Conversation Session</h3>
                    <button class="modal-close" aria-label="Close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-content">
                    <div class="new-session-config">
                        <div class="model-selection-section">
                            <h4>Select AI Model</h4>
                            <div class="model-grid">
                                <div class="model-option" data-model="meta/llama-4-maverick-17b-128e-instruct">
                                    <div class="model-name">Llama 4 Maverick 17B</div>
                                    <div class="model-description">Latest Meta model with enhanced reasoning</div>
                                </div>
                                <div class="model-option" data-model="deepseek-ai/deepseek-r1">
                                    <div class="model-name">DeepSeek R1</div>
                                    <div class="model-description">Advanced reasoning and problem-solving</div>
                                </div>
                                <div class="model-option" data-model="qwen/qwen2.5-coder-32b-instruct">
                                    <div class="model-name">Qwen2.5 Coder 32B</div>
                                    <div class="model-description">Specialized for coding tasks</div>
                                </div>
                                <div class="model-option" data-model="qwen/qwen3-coder-480b-a35b-instruct">
                                    <div class="model-name">Qwen3 Coder 480B</div>
                                    <div class="model-description">Advanced coding and development</div>
                                </div>
                                <div class="model-option" data-model="deepseek-ai/deepseek-v3.1">
                                    <div class="model-name">DeepSeek V3.1</div>
                                    <div class="model-description">General purpose AI assistant</div>
                                </div>
                                <div class="model-option" data-model="openai/gpt-oss-120b">
                                    <div class="model-name">GPT-OSS 120B</div>
                                    <div class="model-description">Open source GPT variant</div>
                                </div>
                                <div class="model-option" data-model="qwen/qwen3-235b-a22b:free">
                                    <div class="model-name">Qwen3 235B A22B</div>
                                    <div class="model-description">Free tier Qwen model</div>
                                </div>
                                <div class="model-option" data-model="google/gemma-3-27b-it:free">
                                    <div class="model-name">Gemma 3 27B IT</div>
                                    <div class="model-description">Google's instruction-tuned model</div>
                                </div>
                            </div>
                        </div>
                        ${currentStats && currentStats.total_tokens > 0 && currentStats.current_model ? `
                        <div class="current-session-info">
                            <h4>Current Session</h4>
                            <div class="session-summary">
                                <div class="summary-item">
                                    <span class="summary-label">Model:</span>
                                    <span class="summary-value">${this.getModelDisplayName(currentStats.current_model || currentModel)}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">Tokens Used:</span>
                                    <span class="summary-value">${currentStats.total_tokens.toLocaleString()}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">Utilization:</span>
                                    <span class="summary-value">${currentStats.utilization_percent}%</span>
                                </div>
                            </div>
                            <div class="warning-notice">
                                <i class="fas fa-exclamation-triangle"></i>
                                <span>Starting a new session will clear your current conversation.</span>
                            </div>
                        </div>
                        ` : ''}
                    </div>
                </div>
                <div class="modal-actions">
                    <button class="cancel-btn" data-action="cancel">
                        <i class="fas fa-times"></i>
                        Cancel
                    </button>
                    <button class="confirm-btn" data-action="confirm">
                        <i class="fas fa-plus"></i>
                        Start New Session
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modalOverlay);
        
        // Mark current model as selected
        const modelOptions = modalOverlay.querySelectorAll('.model-option');
        modelOptions.forEach(option => {
            if (option.dataset.model === currentModel) {
                option.classList.add('selected');
            }
            
            // Add click handler for model selection
            option.addEventListener('click', () => {
                modelOptions.forEach(opt => opt.classList.remove('selected'));
                option.classList.add('selected');
            });
        });
        
        // Add event listeners
        const handleAction = (action) => {
            if (action === 'confirm') {
                const selectedModelOption = modalOverlay.querySelector('.model-option.selected');
                const selectedModel = selectedModelOption ? selectedModelOption.dataset.model : currentModel;
                
                // Switch to selected model and create new session
                this.switchModel(selectedModel);
                this.createNewSession();
            }
            document.body.removeChild(modalOverlay);
        };
        
        modalOverlay.querySelector('.cancel-btn').addEventListener('click', () => handleAction('cancel'));
        modalOverlay.querySelector('.confirm-btn').addEventListener('click', () => handleAction('confirm'));
        modalOverlay.querySelector('.modal-close').addEventListener('click', () => handleAction('cancel'));
        
        // Close on overlay click
        modalOverlay.addEventListener('click', (e) => {
            if (e.target === modalOverlay) {
                handleAction('cancel');
            }
        });
        
        // Close on Escape key
        const handleKeyDown = (e) => {
            if (e.key === 'Escape') {
                handleAction('cancel');
                document.removeEventListener('keydown', handleKeyDown);
            }
        };
        document.addEventListener('keydown', handleKeyDown);
    }


    autoResizeTextarea() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height =
            Math.min(this.messageInput.scrollHeight, window.innerHeight * 0.4) + 'px';
    }

    updateCharCount() {
        const count = this.messageInput.value.length;
        this.charCount.textContent = `${count}/5000`;

        if (count > 4500) {
            this.charCount.style.color = '#ef4444';
        } else if (count > 3500) {
            this.charCount.style.color = '#f59e0b';
        } else {
            this.charCount.style.color = '#6b7280';
        }

        if (count >= 5000) {
            this.sendButton.disabled = true;
            this.sendButton.style.opacity = '0.5';
            this.sendButton.style.cursor = 'not-allowed';
        } else {
            this.sendButton.disabled = false;
            this.sendButton.style.opacity = '1';
            this.sendButton.style.cursor = 'pointer';
        }
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isLoading) return;

        if (this.messageInput.value.length >= 5000) {
            this.showErrorPopup('Maximum character limit (5000) reached!');
            return;
        }

        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.updateCharCount();
        this.autoResizeTextarea();

        this.showTypingIndicator();

        try {
            const response = await this.callNvidiaAPI(message);
            
            // Always display the response, even if generation was stopped
            // Don't hide typing indicator yet - let addMessageWithTyping handle it
            this.addMessageWithTyping(response, 'bot');
        } catch (error) {
            // Only show error if generation wasn't manually stopped
            if (this.isLoading) {
                this.hideTypingIndicator();
                this.updateStatus('Error', '#ef4444');

                let errorMessage = 'Sorry, I encountered an error. Please try again.';
                if (error.message.includes('Failed to fetch')) {
                    errorMessage = 'Unable to connect to server. Please make sure the backend is running.';
                } else if (error.message.includes('HTTP error')) {
                    errorMessage = 'Server error occurred. Please try again.';
                } else if (error.message.includes('NVIDIA API returned unexpected response')) {
                    errorMessage = 'NVIDIA API configuration issue. Please check the API key and try again.';
                } else if (error.message.includes('Server is not processing user messages correctly')) {
                    errorMessage = 'Server configuration issue. Please restart the backend server.';
                }

                this.addMessage(errorMessage, 'bot');
            }
        }
    }

    async callNvidiaAPI(message) {
        const selectedModel = this.modelSelect.value;

        // Define server URLs with primary and failover
        const servers = window.location.hostname === 'antonjijo.github.io' 
            ? [
                'https://nvidia-nim-bot.onrender.com',  // Primary server (Render)
                'https://Nvidia.pythonanywhere.com'     // Failover server (PythonAnywhere)
              ]
            : ['http://localhost:5000'];  // Development uses localhost only

        let lastError = null;
        
        // Try each server in sequence until one works
        for (let i = 0; i < servers.length; i++) {
            const serverURL = servers[i];
            
            try {
                console.log(`Attempting to connect to server ${i + 1}/${servers.length}: ${serverURL}`);
                
                const response = await fetch(`${serverURL}/api/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message,
                        model: selectedModel,
                        session_id: this.sessionId,
                        max_tokens: 1024,
                        temperature: 0.7,
                        top_p: 0.9,
                        frequency_penalty: 0.0,
                        presence_penalty: 0.0
                    }),
                    timeout: 10000  // 10 second timeout
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                if (!data.response && !data.message) {
                    if (data.error) throw new Error(data.error);
                    throw new Error('Invalid response from server');
                }

                const botResponse = data.response || data.message;
                if (botResponse.includes("Hello! I'm your NVIDIA-powered chatbot with advanced capabilities")) {
                    throw new Error('NVIDIA API returned unexpected response. Please check API configuration.');
                }

                // Update conversation stats if available
                if (data.conversation_stats) {
                    this.conversationStats = data.conversation_stats;
                    this.updateConversationStatsUI();
                }

                // Success! Update status and return response
                const serverName = serverURL.includes('render.com') ? 'Primary' : 
                                 serverURL.includes('pythonanywhere.com') ? 'Failover' : 'Local';
                this.updateStatus(`Connected (${serverName})`, '#4ade80');
                console.log(`Successfully connected to ${serverName} server: ${serverURL}`);
                
                return botResponse;
                
            } catch (error) {
                console.warn(`Server ${i + 1} failed (${serverURL}):`, error.message);
                lastError = error;
                
                // If this isn't the last server, continue to next one
                if (i < servers.length - 1) {
                    console.log(`Trying next server...`);
                    continue;
                }
            }
        }
        
        // All servers failed
        this.updateStatus('Connection Failed', '#ef4444');
        console.error('All servers failed. Last error:', lastError);
        throw lastError || new Error('All servers are unavailable');
    }

    updateModelInfo() {
        const selectedModel = this.modelSelect.value;
        const modelNames = {
            'meta/llama-4-maverick-17b-128e-instruct': 'Llama 4 Maverick',
            'deepseek-ai/deepseek-r1': 'DeepSeek R1'
        };
    }

    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        if (sender === 'user') {
            const userIcon = document.createElement('img');
            userIcon.src = 'https://img.icons8.com/fluency-systems-regular/48/user--v1.png';
            userIcon.alt = 'User';
            userIcon.className = 'user-avatar-icon';
            avatar.appendChild(userIcon);
        } else {
            const nvidiaLogo = document.createElement('img');
            nvidiaLogo.src = 'Main_Logo.svg';
            nvidiaLogo.alt = 'NVIDIA';
            nvidiaLogo.className = 'nvidia-logo';
            avatar.appendChild(nvidiaLogo);
        }

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = this.processMessageContent(content);

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        this.attachCopyListeners(messageContent);

        this.messages.push({ role: sender === 'user' ? 'user' : 'assistant', content });
    }

    addMessageWithTyping(content, sender) {
        // Always add the message properly, whether loading or stopped
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        if (sender === 'user') {
            const userIcon = document.createElement('img');
            userIcon.src = 'https://img.icons8.com/fluency-systems-regular/48/user--v1.png';
            userIcon.alt = 'User';
            userIcon.className = 'user-avatar-icon';
            avatar.appendChild(userIcon);
        } else {
            const nvidiaLogo = document.createElement('img');
            nvidiaLogo.src = 'Main_Logo.svg';
            nvidiaLogo.alt = 'NVIDIA';
            nvidiaLogo.className = 'nvidia-logo';
            avatar.appendChild(nvidiaLogo);
        }

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = '<p></p>';

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();

        // If generation was stopped, show content immediately without typing
        if (!this.isLoading) {
            this.displayMessageInstantly(content, messageContent);
        } else {
            // Otherwise, use typing animation
            this.typeMessage(content, messageContent, messageDiv);
        }
    }
    
    displayMessageInstantly(content, messageContent) {
        // Display message content immediately with proper processing
        const processedContent = this.processMessageContent(content);
        messageContent.innerHTML = processedContent;
        
        // Apply syntax highlighting to ALL code blocks that were created
        const codeContainers = messageContent.querySelectorAll('.code-content');
        
        // Extract ALL code blocks from content
        const codeBlockRegex = /```(\w+)?\s*\n?([\s\S]*?)```/g;
        const codeBlocks = [];
        let match;
        
        while ((match = codeBlockRegex.exec(content)) !== null) {
            const language = match[1] || 'text';
            const code = match[2].trim();
            codeBlocks.push({ language, code });
        }
        
        // Apply highlighting to each container with its corresponding code block
        codeContainers.forEach((container, index) => {
            if (index < codeBlocks.length) {
                const codeBlock = codeBlocks[index];
                const codeBlockElement = container.closest('.code-block');
                
                if (codeBlockElement) {
                    // Decode the HTML-escaped code and apply it as text WITHOUT re-escaping
                    const tempDiv = document.createElement('div');
                    tempDiv.innerHTML = codeBlock.code;
                    const decodedCode = tempDiv.textContent || tempDiv.innerText || '';
                    container.textContent = decodedCode;
                    
                    // Apply syntax highlighting with the new safe approach
                    this.applySafeHighlighting(container, codeBlock.language);
                }
            }
        });
        
        this.attachCopyListeners(messageContent);
        this.scrollToBottom();
        this.messages.push({ role: 'assistant', content });
    }

    async typeMessage(content, messageContent, messageDiv) {
        const processedContent = this.processMessageContent(content);
        
        // Check if this contains code blocks or terminal output
        const hasCodeBlock = processedContent.includes('code-block');
        const hasTerminal = processedContent.includes('terminal-output');
        
        if (hasCodeBlock || hasTerminal) {
            // For code/terminal: Show structure immediately and type content
            this.typeCodeBlockContent(content, messageContent, messageDiv, processedContent);
        } else {
            // For regular text: Use standard typing
            this.typeRegularText(content, messageContent, messageDiv, processedContent);
        }
    }
    
    async typeCodeBlockContent(content, messageContent, messageDiv, processedContent) {
        console.log('Processing code block:', { content: content.substring(0, 50) + '...', processedContentLength: processedContent.length });
        
        // Show the complete structure immediately (header, etc.)
        messageContent.innerHTML = processedContent;
        
        console.log('Message content after setting HTML:', messageContent.innerHTML.substring(0, 200) + '...');
        
        // Extract ALL code blocks from content
        const codeBlockRegex = /```(\w+)?\s*\n?([\s\S]*?)```/g;
        const codeBlocks = [];
        let match;
        
        while ((match = codeBlockRegex.exec(content)) !== null) {
            const language = match[1] || 'text';
            const code = match[2].trim();
            codeBlocks.push({ language, code });
        }
        
        if (codeBlocks.length === 0) {
            console.log('No code blocks found, falling back to regular text');
            this.typeRegularText(content, messageContent, messageDiv, processedContent);
            return;
        }
        
        console.log('Found', codeBlocks.length, 'code blocks');
        
        // Find all content containers to type into
        const contentContainers = messageContent.querySelectorAll('.code-content');
        
        console.log('Content containers found:', { count: contentContainers.length });
        
        if (contentContainers.length === 0) {
            console.log('No containers found, keeping as-is');
            this.attachCopyListeners(messageContent);
            this.hideTypingIndicator();
            this.messages.push({ role: 'assistant', content });
            return;
        }
        
        // Clear all containers and prepare for typing
        contentContainers.forEach(container => {
            container.textContent = '';
        });
        
        // Type into each code block sequentially
        await this.typeMultipleCodeBlocks(codeBlocks, contentContainers, messageContent, content);
    }
    
    async typeMultipleCodeBlocks(codeBlocks, contentContainers, messageContent, originalContent) {
        let currentBlockIndex = 0;
        
        const typeNextBlock = () => {
            if (currentBlockIndex >= codeBlocks.length || currentBlockIndex >= contentContainers.length) {
                // All blocks completed
                console.log('All code blocks completed');
                this.attachCopyListeners(messageContent);
                this.scrollToBottom();
                this.hideTypingIndicator();
                this.messages.push({ role: 'assistant', content: originalContent });
                return;
            }
            
            const codeBlock = codeBlocks[currentBlockIndex];
            const container = contentContainers[currentBlockIndex];
            
            console.log(`Typing block ${currentBlockIndex + 1}/${codeBlocks.length}:`, {
                language: codeBlock.language,
                codeLength: codeBlock.code.length
            });
            
            // Type this block
            let currentText = '';
            let currentIndex = 0;
            
            this.currentTypingInterval = setInterval(() => {
                if (!this.isLoading) {
                    clearInterval(this.currentTypingInterval);
                    this.currentTypingInterval = null;
                    return;
                }
                
                if (currentIndex < codeBlock.code.length) {
                    currentText += codeBlock.code[currentIndex];
                    currentIndex++;
                    container.textContent = currentText;
                    this.scrollToBottom();
                } else {
                    clearInterval(this.currentTypingInterval);
                    this.currentTypingInterval = null;
                    
                    // Apply syntax highlighting when this block is complete
                    console.log(`Applying syntax highlighting for block ${currentBlockIndex + 1}, language:`, codeBlock.language);
                    this.applySafeHighlighting(container, codeBlock.language);
                    
                    // Move to next block
                    currentBlockIndex++;
                    
                    // Small delay before starting next block
                    setTimeout(() => {
                        typeNextBlock();
                    }, 100);
                }
            }, 3);
        };
        
        // Start typing the first block
        typeNextBlock();
    }
    
    async typeRegularText(content, messageContent, messageDiv, processedContent) {
        // For regular text (no code blocks), just type the text content without re-processing
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = processedContent;
        const textContent = tempDiv.textContent || tempDiv.innerText || '';

        messageContent.innerHTML = '<p></p>';

        let currentText = '';
        let currentIndex = 0;

        this.currentTypingInterval = setInterval(() => {
            if (!this.isLoading) {
                clearInterval(this.currentTypingInterval);
                this.currentTypingInterval = null;
                if (messageDiv && messageDiv.parentNode) {
                    messageDiv.remove();
                }
                return;
            }
            
            if (currentIndex < textContent.length) {
                currentText += textContent[currentIndex];
                currentIndex++;
                // Simply update text content, don't reprocess HTML
                messageContent.innerHTML = `<p>${this.escapeHtml(currentText)}</p>`;
                this.scrollToBottom();
            } else {
                clearInterval(this.currentTypingInterval);
                this.currentTypingInterval = null;
                // Show the final processed content
                messageContent.innerHTML = processedContent;
                this.attachCopyListeners(messageContent);
                this.scrollToBottom();
                
                this.hideTypingIndicator();
                this.messages.push({ role: 'assistant', content });
            }
        }, 3);
    }
    
    getTypingDelay(char) {
        // Realistic typing delays
        switch (char) {
            case '\n': return 200; // Pause at line breaks
            case '.': case '!': case '?': return 150; // Pause at sentence endings
            case ',': case ';': case ':': return 100; // Pause at punctuation
            case ' ': return 50; // Slight pause at spaces
            case '(': case ')': case '[': case ']': case '{': case '}': return 80;
            default: return Math.random() < 0.1 ? 60 : 15; // Occasional hesitation
        }
    }
    
    applySafeHighlighting(container, language) {
        // Safe syntax highlighting approach that completely avoids HTML conflicts
        if (!container || !language) return;
        
        const content = container.textContent;
        if (!content) return;
        
        // Prevent double-highlighting
        if (container.querySelector('.syntax-keyword, .syntax-string, .syntax-comment')) {
            console.log('Container already highlighted, skipping');
            return;
        }
        
        console.log('Safe highlighting for language:', language);
        
        // Apply highlighting based on language with completely safe approach
        let highlighted = '';
        switch (language.toLowerCase()) {
            case 'python':
            case 'py':
                highlighted = this.highlightPythonSafer(content);
                break;
            case 'javascript':
            case 'js':
            case 'typescript':
            case 'ts':
                highlighted = this.highlightJavaScriptSafer(content);
                break;
            default:
                highlighted = this.highlightGenericSafer(content);
        }
        
        container.innerHTML = highlighted;
    }
    
    highlightPythonSafer(code) {
        // Use a completely different approach that builds tokens and then renders them
        // This eliminates any possibility of HTML corruption
        
        const tokens = [];
        let position = 0;
        
        // Helper function to add token
        const addToken = (type, value, start, end) => {
            tokens.push({ type, value, start, end });
        };
        
        // Helper function to get remaining code
        const remaining = () => code.slice(position);
        
        // Helper function to advance position
        const advance = (count) => {
            position += count;
        };
        
        // Pattern matching functions
        const tryMatch = (pattern) => {
            const match = remaining().match(pattern);
            if (match && match.index === 0) {
                return match;
            }
            return null;
        };
        
        while (position < code.length) {
            const char = code[position];
            const rem = remaining();
            
            // Skip whitespace but keep track
            if (/\s/.test(char)) {
                const wsMatch = tryMatch(/^\s+/);
                if (wsMatch) {
                    addToken('whitespace', wsMatch[0], position, position + wsMatch[0].length);
                    advance(wsMatch[0].length);
                }
                continue;
            }
            
            // Comments
            const commentMatch = tryMatch(/^#.*$/);
            if (commentMatch) {
                addToken('comment', commentMatch[0], position, position + commentMatch[0].length);
                advance(commentMatch[0].length);
                continue;
            }
            
            // Triple-quoted strings
            const tripleMatch = tryMatch(/^(["']{3})[\s\S]*?\1/);
            if (tripleMatch) {
                addToken('string', tripleMatch[0], position, position + tripleMatch[0].length);
                advance(tripleMatch[0].length);
                continue;
            }
            
            // Regular strings
            const stringMatch = tryMatch(/^(f?["'])([^"'\n]*?)\1/);
            if (stringMatch) {
                addToken('string', stringMatch[0], position, position + stringMatch[0].length);
                advance(stringMatch[0].length);
                continue;
            }
            
            // Numbers
            const numberMatch = tryMatch(/^\b(0x[0-9a-fA-F]+|0b[01]+|\d+\.\d+|\d+)\b/);
            if (numberMatch) {
                addToken('number', numberMatch[0], position, position + numberMatch[0].length);
                advance(numberMatch[0].length);
                continue;
            }
            
            // Keywords
            const keywordMatch = tryMatch(/^\b(def|class|if|else|elif|for|while|return|import|from|as|try|except|finally|with|lambda|yield|async|await|and|or|not|in|is|pass|break|continue|global|nonlocal|assert|del|raise)\b/);
            if (keywordMatch) {
                addToken('keyword', keywordMatch[0], position, position + keywordMatch[0].length);
                advance(keywordMatch[0].length);
                continue;
            }
            
            // Constants
            const constantMatch = tryMatch(/^\b(None|True|False)\b/);
            if (constantMatch) {
                addToken('constant', constantMatch[0], position, position + constantMatch[0].length);
                advance(constantMatch[0].length);
                continue;
            }
            
            // Built-in types
            const typeMatch = tryMatch(/^\b(int|str|float|bool|list|dict|tuple|set|type|object)\b/);
            if (typeMatch) {
                addToken('type', typeMatch[0], position, position + typeMatch[0].length);
                advance(typeMatch[0].length);
                continue;
            }
            
            // Built-in functions
            const builtinMatch = tryMatch(/^\b(print|len|range|enumerate|zip|map|filter|sorted|reversed|any|all|sum|min|max|abs|round|input|open)\b/);
            if (builtinMatch) {
                addToken('builtin', builtinMatch[0], position, position + builtinMatch[0].length);
                advance(builtinMatch[0].length);
                continue;
            }
            
            // Decorators
            const decoratorMatch = tryMatch(/^@\w+/);
            if (decoratorMatch) {
                addToken('decorator', decoratorMatch[0], position, position + decoratorMatch[0].length);
                advance(decoratorMatch[0].length);
                continue;
            }
            
            // Function/method calls
            const methodMatch = tryMatch(/^\b(\w+)(?=\s*\()/);
            if (methodMatch) {
                addToken('method', methodMatch[1], position, position + methodMatch[1].length);
                advance(methodMatch[1].length);
                continue;
            }
            
            // Class names (PascalCase)
            const classMatch = tryMatch(/^\b[A-Z][a-zA-Z0-9_]*\b/);
            if (classMatch) {
                addToken('class', classMatch[0], position, position + classMatch[0].length);
                advance(classMatch[0].length);
                continue;
            }
            
            // Self keyword
            const selfMatch = tryMatch(/^\bself\b/);
            if (selfMatch) {
                addToken('self', selfMatch[0], position, position + selfMatch[0].length);
                advance(selfMatch[0].length);
                continue;
            }
            
            // Magic methods
            const magicMatch = tryMatch(/^\b__\w+__\b/);
            if (magicMatch) {
                addToken('magic', magicMatch[0], position, position + magicMatch[0].length);
                advance(magicMatch[0].length);
                continue;
            }
            
            // Variables
            const variableMatch = tryMatch(/^\b[a-z_][a-zA-Z0-9_]*\b/);
            if (variableMatch) {
                addToken('variable', variableMatch[0], position, position + variableMatch[0].length);
                advance(variableMatch[0].length);
                continue;
            }
            
            // Operators
            const operatorMatch = tryMatch(/^(\*\*|\/\/|<<|>>|==|!=|<=|>=|[+\-*\/%=<>!&|^~])/);
            if (operatorMatch) {
                addToken('operator', operatorMatch[0], position, position + operatorMatch[0].length);
                advance(operatorMatch[0].length);
                continue;
            }
            
            // Punctuation
            const punctMatch = tryMatch(/^([\[\]{}(),.:;])/);
            if (punctMatch) {
                addToken('punctuation', punctMatch[0], position, position + punctMatch[0].length);
                advance(punctMatch[0].length);
                continue;
            }
            
            // Default: single character
            addToken('text', char, position, position + 1);
            advance(1);
        }
        
        // Now render tokens as HTML
        let result = '';
        for (const token of tokens) {
            const escapedValue = this.escapeHtml(token.value);
            
            switch (token.type) {
                case 'keyword':
                    result += `<span class="syntax-keyword">${escapedValue}</span>`;
                    break;
                case 'string':
                    result += `<span class="syntax-string">${escapedValue}</span>`;
                    break;
                case 'comment':
                    result += `<span class="syntax-comment">${escapedValue}</span>`;
                    break;
                case 'number':
                    result += `<span class="syntax-number">${escapedValue}</span>`;
                    break;
                case 'constant':
                    result += `<span class="syntax-constant">${escapedValue}</span>`;
                    break;
                case 'type':
                    result += `<span class="syntax-type">${escapedValue}</span>`;
                    break;
                case 'builtin':
                    result += `<span class="syntax-builtin">${escapedValue}</span>`;
                    break;
                case 'decorator':
                    result += `<span class="syntax-decorator">${escapedValue}</span>`;
                    break;
                case 'method':
                    result += `<span class="syntax-method">${escapedValue}</span>`;
                    break;
                case 'class':
                    result += `<span class="syntax-class">${escapedValue}</span>`;
                    break;
                case 'self':
                    result += `<span class="syntax-self">${escapedValue}</span>`;
                    break;
                case 'magic':
                    result += `<span class="syntax-magic">${escapedValue}</span>`;
                    break;
                case 'variable':
                    result += `<span class="syntax-variable">${escapedValue}</span>`;
                    break;
                case 'operator':
                    result += `<span class="syntax-operator">${escapedValue}</span>`;
                    break;
                case 'punctuation':
                    result += `<span class="syntax-punctuation">${escapedValue}</span>`;
                    break;
                default:
                    result += escapedValue;
                    break;
            }
        }
        
        return result;
    }
    
    highlightJavaScriptSafer(code) {
        // Use token-based approach to eliminate HTML corruption
        const tokens = [];
        let position = 0;
        
        const addToken = (type, value, start, end) => {
            tokens.push({ type, value, start, end });
        };
        
        const remaining = () => code.slice(position);
        const advance = (count) => { position += count; };
        
        const tryMatch = (pattern) => {
            const match = remaining().match(pattern);
            return (match && match.index === 0) ? match : null;
        };
        
        while (position < code.length) {
            const char = code[position];
            
            // Whitespace
            if (/\s/.test(char)) {
                const wsMatch = tryMatch(/^\s+/);
                if (wsMatch) {
                    addToken('whitespace', wsMatch[0], position, position + wsMatch[0].length);
                    advance(wsMatch[0].length);
                }
                continue;
            }
            
            // Block comments
            const blockCommentMatch = tryMatch(/^\/\*[\s\S]*?\*\//);
            if (blockCommentMatch) {
                addToken('comment', blockCommentMatch[0], position, position + blockCommentMatch[0].length);
                advance(blockCommentMatch[0].length);
                continue;
            }
            
            // Line comments
            const lineCommentMatch = tryMatch(/^\/\/.*$/);
            if (lineCommentMatch) {
                addToken('comment', lineCommentMatch[0], position, position + lineCommentMatch[0].length);
                advance(lineCommentMatch[0].length);
                continue;
            }
            
            // Template literals
            const templateMatch = tryMatch(/^`[^`]*`/);
            if (templateMatch) {
                addToken('template', templateMatch[0], position, position + templateMatch[0].length);
                advance(templateMatch[0].length);
                continue;
            }
            
            // Regular strings
            const stringMatch = tryMatch(/^(["'])(?:(?!\1)[^\\\n]|\\.)*\1/);
            if (stringMatch) {
                addToken('string', stringMatch[0], position, position + stringMatch[0].length);
                advance(stringMatch[0].length);
                continue;
            }
            
            // Regular expressions
            const regexMatch = tryMatch(/^\/(?:[^\/\\\n]|\\.)+\/[gimuy]*/);
            if (regexMatch) {
                addToken('regex', regexMatch[0], position, position + regexMatch[0].length);
                advance(regexMatch[0].length);
                continue;
            }
            
            // Numbers
            const numberMatch = tryMatch(/^\b(0x[0-9a-fA-F]+|0b[01]+|\d+\.\d+|\d+)\b/);
            if (numberMatch) {
                addToken('number', numberMatch[0], position, position + numberMatch[0].length);
                advance(numberMatch[0].length);
                continue;
            }
            
            // Keywords
            const keywordMatch = tryMatch(/^\b(function|const|let|var|if|else|for|while|return|class|extends|import|export|from|default|async|await|try|catch|finally|throw|new|this|super|typeof|instanceof|in|of|break|continue|switch|case|do|with)\b/);
            if (keywordMatch) {
                addToken('keyword', keywordMatch[0], position, position + keywordMatch[0].length);
                advance(keywordMatch[0].length);
                continue;
            }
            
            // Constants
            const constantMatch = tryMatch(/^\b(true|false|null|undefined|NaN|Infinity)\b/);
            if (constantMatch) {
                addToken('constant', constantMatch[0], position, position + constantMatch[0].length);
                advance(constantMatch[0].length);
                continue;
            }
            
            // Built-in objects
            const builtinMatch = tryMatch(/^\b(Array|Object|String|Number|Boolean|Date|Math|JSON|Promise|RegExp|Error|console|window|document)\b/);
            if (builtinMatch) {
                addToken('builtin', builtinMatch[0], position, position + builtinMatch[0].length);
                advance(builtinMatch[0].length);
                continue;
            }
            
            // Function/method calls
            const methodMatch = tryMatch(/^\b(\w+)(?=\s*\()/);
            if (methodMatch) {
                addToken('method', methodMatch[1], position, position + methodMatch[1].length);
                advance(methodMatch[1].length);
                continue;
            }
            
            // Class names (PascalCase)
            const classMatch = tryMatch(/^\b[A-Z][a-zA-Z0-9_]*\b/);
            if (classMatch) {
                addToken('class', classMatch[0], position, position + classMatch[0].length);
                advance(classMatch[0].length);
                continue;
            }
            
            // Properties (after dot)
            const propertyMatch = tryMatch(/^\.(\w+)/);
            if (propertyMatch) {
                addToken('text', '.', position, position + 1);
                addToken('property', propertyMatch[1], position + 1, position + propertyMatch[0].length);
                advance(propertyMatch[0].length);
                continue;
            }
            
            // Variables
            const variableMatch = tryMatch(/^\b[a-z_$][a-zA-Z0-9_$]*\b/);
            if (variableMatch) {
                addToken('variable', variableMatch[0], position, position + variableMatch[0].length);
                advance(variableMatch[0].length);
                continue;
            }
            
            // Operators
            const operatorMatch = tryMatch(/^(\+\+|--|&&|\|\||===|!==|==|!=|<=|>=|[+\-*\/%=<>!&|^~?:])/);
            if (operatorMatch) {
                addToken('operator', operatorMatch[0], position, position + operatorMatch[0].length);
                advance(operatorMatch[0].length);
                continue;
            }
            
            // Punctuation
            const punctMatch = tryMatch(/^([\[\]{}(),;])/);
            if (punctMatch) {
                addToken('punctuation', punctMatch[0], position, position + punctMatch[0].length);
                advance(punctMatch[0].length);
                continue;
            }
            
            // Default: single character
            addToken('text', char, position, position + 1);
            advance(1);
        }
        
        // Render tokens as HTML
        let result = '';
        for (const token of tokens) {
            const escapedValue = this.escapeHtml(token.value);
            
            switch (token.type) {
                case 'keyword':
                    result += `<span class="syntax-keyword">${escapedValue}</span>`;
                    break;
                case 'string':
                    result += `<span class="syntax-string">${escapedValue}</span>`;
                    break;
                case 'template':
                    result += `<span class="syntax-template">${escapedValue}</span>`;
                    break;
                case 'regex':
                    result += `<span class="syntax-regex">${escapedValue}</span>`;
                    break;
                case 'comment':
                    result += `<span class="syntax-comment">${escapedValue}</span>`;
                    break;
                case 'number':
                    result += `<span class="syntax-number">${escapedValue}</span>`;
                    break;
                case 'constant':
                    result += `<span class="syntax-constant">${escapedValue}</span>`;
                    break;
                case 'builtin':
                    result += `<span class="syntax-builtin">${escapedValue}</span>`;
                    break;
                case 'method':
                    result += `<span class="syntax-method">${escapedValue}</span>`;
                    break;
                case 'class':
                    result += `<span class="syntax-class">${escapedValue}</span>`;
                    break;
                case 'property':
                    result += `<span class="syntax-property">${escapedValue}</span>`;
                    break;
                case 'variable':
                    result += `<span class="syntax-variable">${escapedValue}</span>`;
                    break;
                case 'operator':
                    result += `<span class="syntax-operator">${escapedValue}</span>`;
                    break;
                case 'punctuation':
                    result += `<span class="syntax-punctuation">${escapedValue}</span>`;
                    break;
                default:
                    result += escapedValue;
                    break;
            }
        }
        
        return result;
    }
    
    highlightGenericSafer(code) {
        // Token-based approach for generic languages
        const tokens = [];
        let position = 0;
        
        const addToken = (type, value, start, end) => {
            tokens.push({ type, value, start, end });
        };
        
        const remaining = () => code.slice(position);
        const advance = (count) => { position += count; };
        
        const tryMatch = (pattern) => {
            const match = remaining().match(pattern);
            return (match && match.index === 0) ? match : null;
        };
        
        while (position < code.length) {
            const char = code[position];
            
            // Whitespace
            if (/\s/.test(char)) {
                const wsMatch = tryMatch(/^\s+/);
                if (wsMatch) {
                    addToken('whitespace', wsMatch[0], position, position + wsMatch[0].length);
                    advance(wsMatch[0].length);
                }
                continue;
            }
            
            // Block comments (C-style)
            const blockCommentMatch = tryMatch(/^\/\*[\s\S]*?\*\//);
            if (blockCommentMatch) {
                addToken('comment', blockCommentMatch[0], position, position + blockCommentMatch[0].length);
                advance(blockCommentMatch[0].length);
                continue;
            }
            
            // Single line comments (multiple styles)
            const commentMatch = tryMatch(/^(\/\/.*$|#.*$|--.*$|;.*$|'.*$|REM.*$)/i);
            if (commentMatch) {
                addToken('comment', commentMatch[0], position, position + commentMatch[0].length);
                advance(commentMatch[0].length);
                continue;
            }
            
            // Template literals and backtick strings
            const templateMatch = tryMatch(/^`[^`]*`/);
            if (templateMatch) {
                addToken('template', templateMatch[0], position, position + templateMatch[0].length);
                advance(templateMatch[0].length);
                continue;
            }
            
            // Strings (various quote styles)
            const stringMatch = tryMatch(/^(["'])(?:(?!\1)[^\\\n]|\\.)*\1/);
            if (stringMatch) {
                addToken('string', stringMatch[0], position, position + stringMatch[0].length);
                advance(stringMatch[0].length);
                continue;
            }
            
            // Numbers (hex, binary, decimal)
            const numberMatch = tryMatch(/^\b(0x[0-9a-fA-F]+|0b[01]+|\d+\.\d+|\d+)\b/);
            if (numberMatch) {
                addToken('number', numberMatch[0], position, position + numberMatch[0].length);
                advance(numberMatch[0].length);
                continue;
            }
            
            // Common keywords across languages
            const keywordMatch = tryMatch(/^\b(if|else|for|while|do|return|function|def|class|import|export|from|as|try|catch|finally|throw|new|this|self|super|true|false|null|nil|undefined|var|let|const|int|string|bool|float|double|char|void|public|private|protected|static|final|abstract|interface|extends|implements|namespace|using|include|package)\b/);
            if (keywordMatch) {
                addToken('keyword', keywordMatch[0], position, position + keywordMatch[0].length);
                advance(keywordMatch[0].length);
                continue;
            }
            
            // Constants
            const constantMatch = tryMatch(/^\b(true|false|null|nil|undefined|True|False|None|NULL|YES|NO)\b/);
            if (constantMatch) {
                addToken('constant', constantMatch[0], position, position + constantMatch[0].length);
                advance(constantMatch[0].length);
                continue;
            }
            
            // Types
            const typeMatch = tryMatch(/^\b(int|float|double|char|string|bool|boolean|void|object|array|list|dict|map|set|uint|long|short|byte)\b/);
            if (typeMatch) {
                addToken('type', typeMatch[0], position, position + typeMatch[0].length);
                advance(typeMatch[0].length);
                continue;
            }
            
            // Function/method calls
            const methodMatch = tryMatch(/^\b(\w+)(?=\s*\()/);
            if (methodMatch) {
                addToken('method', methodMatch[1], position, position + methodMatch[1].length);
                advance(methodMatch[1].length);
                continue;
            }
            
            // Class names (PascalCase)
            const classMatch = tryMatch(/^\b[A-Z][a-zA-Z0-9_]*\b/);
            if (classMatch) {
                addToken('class', classMatch[0], position, position + classMatch[0].length);
                advance(classMatch[0].length);
                continue;
            }
            
            // Properties (after dot)
            const propertyMatch = tryMatch(/^\.(\w+)/);
            if (propertyMatch) {
                addToken('text', '.', position, position + 1);
                addToken('property', propertyMatch[1], position + 1, position + propertyMatch[0].length);
                advance(propertyMatch[0].length);
                continue;
            }
            
            // Variables
            const variableMatch = tryMatch(/^\b[a-z_][a-zA-Z0-9_]*\b/);
            if (variableMatch) {
                addToken('variable', variableMatch[0], position, position + variableMatch[0].length);
                advance(variableMatch[0].length);
                continue;
            }
            
            // Operators
            const operatorMatch = tryMatch(/^(==|!=|<=|>=|&&|\|\||[+\-*\/%=<>!&|^~])/);
            if (operatorMatch) {
                addToken('operator', operatorMatch[0], position, position + operatorMatch[0].length);
                advance(operatorMatch[0].length);
                continue;
            }
            
            // Punctuation
            const punctMatch = tryMatch(/^([\[\]{}(),;.])/);
            if (punctMatch) {
                addToken('punctuation', punctMatch[0], position, position + punctMatch[0].length);
                advance(punctMatch[0].length);
                continue;
            }
            
            // Default: single character
            addToken('text', char, position, position + 1);
            advance(1);
        }
        
        // Render tokens as HTML
        let result = '';
        for (const token of tokens) {
            const escapedValue = this.escapeHtml(token.value);
            
            switch (token.type) {
                case 'keyword':
                    result += `<span class="syntax-keyword">${escapedValue}</span>`;
                    break;
                case 'string':
                    result += `<span class="syntax-string">${escapedValue}</span>`;
                    break;
                case 'template':
                    result += `<span class="syntax-template">${escapedValue}</span>`;
                    break;
                case 'comment':
                    result += `<span class="syntax-comment">${escapedValue}</span>`;
                    break;
                case 'number':
                    result += `<span class="syntax-number">${escapedValue}</span>`;
                    break;
                case 'constant':
                    result += `<span class="syntax-constant">${escapedValue}</span>`;
                    break;
                case 'type':
                    result += `<span class="syntax-type">${escapedValue}</span>`;
                    break;
                case 'method':
                    result += `<span class="syntax-method">${escapedValue}</span>`;
                    break;
                case 'class':
                    result += `<span class="syntax-class">${escapedValue}</span>`;
                    break;
                case 'property':
                    result += `<span class="syntax-property">${escapedValue}</span>`;
                    break;
                case 'variable':
                    result += `<span class="syntax-variable">${escapedValue}</span>`;
                    break;
                case 'operator':
                    result += `<span class="syntax-operator">${escapedValue}</span>`;
                    break;
                case 'punctuation':
                    result += `<span class="syntax-punctuation">${escapedValue}</span>`;
                    break;
                default:
                    result += escapedValue;
                    break;
            }
        }
        
        return result;
    }
    
    highlightPython(code) {
        // First escape HTML to prevent injection and ensure clean highlighting
        code = this.escapeHtml(code);
        code = code.replace(/\b(def|class|if|else|elif|for|while|return|import|from|as|try|except|finally|with|lambda|yield|async|await|and|or|not|in|is|pass|break|continue|global|nonlocal|assert|del|raise)\b/g, '<span class="syntax-keyword">$1</span>');
        
        // Python constants
        code = code.replace(/\b(None|True|False|__name__|__main__|__file__|__doc__)\b/g, '<span class="syntax-constant">$1</span>');
        
        // Python built-ins
        code = code.replace(/\b(print|len|str|int|float|list|dict|tuple|set|range|enumerate|zip|map|filter|sorted|reversed|any|all|sum|min|max|abs|round|type|isinstance|hasattr|getattr|setattr|delattr|open|input|format|super|property|staticmethod|classmethod)\b/g, '<span class="syntax-builtin">$1</span>');
        
        // Python decorators
        code = code.replace(/@\w+/g, '<span class="syntax-decorator">$&</span>');
        
        // Python class names (capitalized words)
        code = code.replace(/\b[A-Z][a-zA-Z0-9_]*\b/g, '<span class="syntax-class">$&</span>');
        
        // Function definitions
        code = code.replace(/def\s+(\w+)/g, 'def <span class="syntax-function">$1</span>');
        
        // Strings (including f-strings)
        code = code.replace(/(f?["'])(?:(?!\1)[^\\]|\\.)*\1/g, '<span class="syntax-string">$&</span>');
        code = code.replace(/(f?["]{3})[\s\S]*?\1/g, '<span class="syntax-string">$&</span>');
        
        // Comments
        code = code.replace(/#.*/g, '<span class="syntax-comment">$&</span>');
        
        // Numbers
        code = code.replace(/\b\d+\.?\d*[eE]?[+-]?\d*\b/g, '<span class="syntax-number">$&</span>');
        
        // Operators
        code = code.replace(/([+\-*\/=%<>!&|^~]|\*\*|\/\/|<<|>>|==|!=|<=|>=|and|or|not|in|is)/g, '<span class="syntax-operator">$1</span>');
        
        return code;
    }
    
    highlightJavaScript(code) {
        // First escape HTML to prevent injection and ensure clean highlighting
        code = this.escapeHtml(code);
        
        // Simple and reliable approach - process in order of priority
        
        // 1. Comments first (highest priority)
        code = code.replace(/\/\*[\s\S]*?\*\//g, '<span class="syntax-comment">$&</span>');
        code = code.replace(/\/\/.*/g, '<span class="syntax-comment">$&</span>');
        
        // 2. Strings (second highest priority)
        code = code.replace(/(["'`])(?:(?!\1)[^\\\n]|\\.)*\1/g, '<span class="syntax-string">$&</span>');
        
        // 3. Numbers
        code = code.replace(/\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/g, '<span class="syntax-number">$&</span>');
        
        // 4. Keywords (avoid matching inside existing spans)
        const keywords = ['function', 'const', 'let', 'var', 'if', 'else', 'for', 'while', 'return', 'class', 'extends', 'import', 'export', 'from', 'default', 'async', 'await', 'try', 'catch', 'finally', 'throw', 'new', 'this', 'super', 'typeof', 'instanceof', 'in', 'of', 'break', 'continue', 'switch', 'case', 'do'];
        for (const keyword of keywords) {
            const regex = new RegExp(`\\b${keyword}\\b(?![^<]*<\/span>)`, 'g');
            code = code.replace(regex, `<span class="syntax-keyword">${keyword}</span>`);
        }
        
        // 5. Constants
        const constants = ['true', 'false', 'null', 'undefined', 'NaN', 'Infinity'];
        for (const constant of constants) {
            const regex = new RegExp(`\\b${constant}\\b(?![^<]*<\/span>)`, 'g');
            code = code.replace(regex, `<span class="syntax-constant">${constant}</span>`);
        }
        
        // 6. Built-ins
        const builtins = ['console', 'document', 'window', 'Array', 'Object', 'String', 'Number', 'Boolean', 'Date', 'Math', 'JSON', 'Promise', 'setTimeout', 'setInterval'];
        for (const builtin of builtins) {
            const regex = new RegExp(`\\b${builtin}\\b(?![^<]*<\/span>)`, 'g');
            code = code.replace(regex, `<span class="syntax-builtin">${builtin}</span>`);
        }
        
        // 7. Method calls (simple pattern, avoid keywords)
        code = code.replace(/\b(\w+)(?=\s*\()(?![^<]*<\/span>)/g, function(match, methodName) {
            // Don't highlight if it's already a keyword, constant, or builtin
            if (keywords.includes(methodName) || constants.includes(methodName) || builtins.includes(methodName)) {
                return match;
            }
            return `<span class="syntax-method">${methodName}</span>`;
        });
        
        return code;
    }
    
    isInsideToken(tokens, position) {
        return tokens.some(token => position >= token.start && position < token.end);
    }
    
    highlightJava(code) {
        // First escape HTML to prevent injection
        code = this.escapeHtml(code);
        
        // Java keywords
        code = code.replace(/\b(public|private|protected|static|final|abstract|synchronized|volatile|transient|native|strictfp|class|interface|extends|implements|import|package|if|else|for|while|do|return|try|catch|finally|throw|throws|new|this|super|instanceof|break|continue|switch|case|default|assert)\b/g, '<span class="syntax-keyword">$1</span>');
        
        // Java constants
        code = code.replace(/\b(true|false|null)\b/g, '<span class="syntax-constant">$1</span>');
        
        // Java primitive types
        code = code.replace(/\b(void|int|long|double|float|boolean|char|byte|short)\b/g, '<span class="syntax-type">$1</span>');
        
        // Java built-in classes
        code = code.replace(/\b(String|Integer|Double|Float|Boolean|Character|Long|Short|Byte|Object|Class|System|Math|ArrayList|HashMap|HashSet|LinkedList|Vector|Scanner|BufferedReader|FileReader|PrintWriter|Exception|RuntimeException|IOException|NullPointerException)\b/g, '<span class="syntax-builtin">$1</span>');
        
        // Annotations
        code = code.replace(/@\w+/g, '<span class="syntax-annotation">$&</span>');
        
        // Class names (capitalized)
        code = code.replace(/\b[A-Z][a-zA-Z0-9_]*\b/g, '<span class="syntax-class">$&</span>');
        
        // Method definitions and calls
        code = code.replace(/(\w+)\s*(?=\()/g, '<span class="syntax-method">$1</span>');
        
        // Strings
        code = code.replace(/"(?:[^"\\]|\\.)*"/g, '<span class="syntax-string">$&</span>');
        code = code.replace(/'(?:[^'\\]|\\.)*'/g, '<span class="syntax-string">$&</span>');
        
        // Comments
        code = code.replace(/\/\*[\s\S]*?\*\//g, '<span class="syntax-comment">$&</span>');
        code = code.replace(/\/\/.*/g, '<span class="syntax-comment">$&</span>');
        
        // Numbers
        code = code.replace(/\b\d+\.?\d*[fFlLdD]?\b/g, '<span class="syntax-number">$&</span>');
        
        // Operators
        code = code.replace(/([+\-*\/=%<>!&|^~]|\+\+|--|<<|>>|>>>|==|!=|<=|>=|&&|\|\|)/g, '<span class="syntax-operator">$1</span>');
        
        return code;
    }
    
    highlightCpp(code) {
        // First escape HTML to prevent injection
        code = this.escapeHtml(code);
        
        // C++ keywords
        code = code.replace(/\b(int|char|float|double|bool|void|auto|const|static|extern|inline|virtual|override|final|explicit|mutable|constexpr|decltype|nullptr|public|private|protected|class|struct|enum|union|namespace|using|typedef|template|typename|if|else|for|while|do|return|break|continue|switch|case|default|try|catch|throw|new|delete|this|sizeof|alignof|noexcept|static_assert)\b/g, '<span class="syntax-keyword">$1</span>');
        
        // C++ constants
        code = code.replace(/\b(true|false|nullptr|NULL)\b/g, '<span class="syntax-constant">$1</span>');
        
        // C++ built-ins and STL
        code = code.replace(/\b(std|cout|cin|endl|cerr|vector|string|map|set|list|queue|stack|array|deque|pair|tuple|iostream|fstream|sstream|algorithm|iterator|memory|thread|mutex|unique_ptr|shared_ptr|weak_ptr|make_unique|make_shared)\b/g, '<span class="syntax-builtin">$1</span>');
        
        // C++ types
        code = code.replace(/\b(size_t|ptrdiff_t|uint8_t|uint16_t|uint32_t|uint64_t|int8_t|int16_t|int32_t|int64_t)\b/g, '<span class="syntax-type">$1</span>');
        
        // Function and method calls
        code = code.replace(/(\w+)\s*(?=\()/g, '<span class="syntax-method">$1</span>');
        
        // Class names (capitalized)
        code = code.replace(/\b[A-Z][a-zA-Z0-9_]*\b/g, '<span class="syntax-class">$&</span>');
        
        // Strings
        code = code.replace(/"(?:[^"\\]|\\.)*"/g, '<span class="syntax-string">$&</span>');
        code = code.replace(/'(?:[^'\\]|\\.)'/g, '<span class="syntax-string">$&</span>');
        
        // Raw strings
        code = code.replace(/R"\([\s\S]*?\)"/, '<span class="syntax-string">$&</span>');
        
        // Comments
        code = code.replace(/\/\*[\s\S]*?\*\//g, '<span class="syntax-comment">$&</span>');
        code = code.replace(/\/\/.*/g, '<span class="syntax-comment">$&</span>');
        
        // Preprocessor directives
        code = code.replace(/#\w+.*$/gm, '<span class="syntax-preprocessor">$&</span>');
        
        // Numbers
        code = code.replace(/\b\d+\.?\d*[fFlLuU]*\b/g, '<span class="syntax-number">$&</span>');
        
        // Operators
        code = code.replace(/([+\-*\/=%<>!&|^~]|\+\+|--|<<|>>|==|!=|<=|>=|&&|\|\||->|::|\.\*|->\*)/g, '<span class="syntax-operator">$1</span>');
        
        return code;
    }
    
    highlightHtml(code) {
        // HTML tags
        code = code.replace(/&lt;\/?([a-zA-Z][a-zA-Z0-9-]*)([^&]*?)&gt;/g, function(match, tagName, attributes) {
            let highlighted = '&lt;<span class="syntax-tag">/' + tagName + '</span>';
            if (attributes) {
                // Highlight attributes
                attributes = attributes.replace(/(\w+)=/g, '<span class="syntax-attribute">$1</span>=');
                // Highlight attribute values
                attributes = attributes.replace(/=\"([^\"]*)\"/g, '=<span class="syntax-string">"$1"</span>');
                attributes = attributes.replace(/=\'([^\']*)\'/, '=<span class="syntax-string">\"$1\"</span>');
                highlighted = '&lt;<span class="syntax-tag">' + tagName + '</span>' + attributes;
            } else {
                highlighted = match.replace(tagName, '<span class="syntax-tag">' + tagName + '</span>');
            }
            return highlighted + '&gt;';
        });
        
        // HTML entities
        code = code.replace(/&[a-zA-Z][a-zA-Z0-9]*;/g, '<span class="syntax-escape">$&</span>');
        
        // HTML comments
        code = code.replace(/&lt;!--[\s\S]*?--&gt;/g, '<span class="syntax-comment">$&</span>');
        
        return code;
    }
    
    highlightCss(code) {
        // CSS selectors
        code = code.replace(/^\s*([.#]?[\w-]+(?:[.#][\w-]+)*(?:\s*[>+~]\s*[.#]?[\w-]+(?:[.#][\w-]+)*)*)\s*(?=\{)/gm, '<span class="syntax-selector">$1</span>');
        
        // CSS at-rules
        code = code.replace(/@[\w-]+/g, '<span class="syntax-keyword">$&</span>');
        
        // CSS properties
        code = code.replace(/([\w-]+)\s*:/g, '<span class="syntax-property">$1</span>:');
        
        // CSS values
        code = code.replace(/:\s*([^;{]+)(?=;|\})/g, function(match, value) {
            // Highlight different types of values
            value = value.replace(/\b(\d+(?:\.\d+)?(?:px|em|rem|%|vh|vw|pt|pc|in|mm|cm|ex|ch|fr|vmin|vmax)?)/g, '<span class="syntax-number">$1</span>');
            value = value.replace(/\b(inherit|initial|unset|auto|none|normal|bold|italic|underline|center|left|right|block|inline|flex|grid|absolute|relative|fixed|static|transparent|currentColor)\b/g, '<span class="syntax-keyword">$1</span>');
            value = value.replace(/#[0-9a-fA-F]{3,6}\b/g, '<span class="syntax-number">$&</span>');
            value = value.replace(/\b(rgb|rgba|hsl|hsla|url|calc|var|min|max|clamp)\(/g, '<span class="syntax-function">$1</span>(');
            value = value.replace(/"[^"]*"|'[^']*'/g, '<span class="syntax-string">$&</span>');
            return ': ' + value;
        });
        
        // CSS important
        code = code.replace(/!important/g, '<span class="syntax-important">$&</span>');
        
        // CSS comments
        code = code.replace(/\/\*[\s\S]*?\*\//g, '<span class="syntax-comment">$&</span>');
        
        return code;
    }
    
    highlightGeneric(code) {
        // First escape HTML to prevent injection
        code = this.escapeHtml(code);
        
        // Generic highlighting for unknown languages
        
        // Common keywords across many languages
        code = code.replace(/\b(if|else|for|while|do|return|function|def|class|import|export|from|as|try|catch|finally|throw|new|this|self|super|true|false|null|nil|undefined|var|let|const|int|string|bool|float|double|char|void|public|private|protected|static|final|abstract)\b/g, '<span class="syntax-keyword">$1</span>');
        
        // Strings (various quote styles)
        code = code.replace(/(["'])(?:(?!\1)[^\\]|\\.)*\1/g, '<span class="syntax-string">$&</span>');
        code = code.replace(/`[^`]*`/g, '<span class="syntax-string">$&</span>');
        
        // Numbers (integers, floats, hex, binary)
        code = code.replace(/\b0x[0-9a-fA-F]+\b/g, '<span class="syntax-number">$&</span>');
        code = code.replace(/\b0b[01]+\b/g, '<span class="syntax-number">$&</span>');
        code = code.replace(/\b\d+\.?\d*[eE]?[+-]?\d*[fFlLdD]?\b/g, '<span class="syntax-number">$&</span>');
        
        // Function calls (word followed by parentheses)
        code = code.replace(/(\w+)\s*(?=\()/g, '<span class="syntax-method">$1</span>');
        
        // Class names (capitalized words)
        code = code.replace(/\b[A-Z][a-zA-Z0-9_]*\b/g, '<span class="syntax-class">$&</span>');
        
        // Constants (ALL_CAPS)
        code = code.replace(/\b[A-Z][A-Z0-9_]*\b/g, '<span class="syntax-constant">$&</span>');
        
        // Single line comments (various styles)
        code = code.replace(/\/\/.*/g, '<span class="syntax-comment">$&</span>');
        code = code.replace(/#.*/g, '<span class="syntax-comment">$&</span>');
        code = code.replace(/;.*/g, '<span class="syntax-comment">$&</span>'); // Lisp-style
        code = code.replace(/%.*/g, '<span class="syntax-comment">$&</span>'); // Erlang/Prolog-style
        code = code.replace(/\'.*$/gm, '<span class="syntax-comment">$&</span>'); // VB-style
        code = code.replace(/REM.*/gi, '<span class="syntax-comment">$&</span>'); // BASIC-style
        
        // Multi-line comments
        code = code.replace(/\/\*[\s\S]*?\*\//g, '<span class="syntax-comment">$&</span>');
        code = code.replace(/\(\*[\s\S]*?\*\)/g, '<span class="syntax-comment">$&</span>'); // Pascal/ML-style
        
        // Common operators
        code = code.replace(/([+\-*\/=%<>!&|^~]|==|!=|<=|>=|&&|\|\||\+\+|--|<<|>>|->|=>|::|\.\.|\?\?)/g, '<span class="syntax-operator">$1</span>');
        
        // Preprocessor directives
        code = code.replace(/#\w+.*$/gm, '<span class="syntax-preprocessor">$&</span>');
        
        // Annotations/decorators
        code = code.replace(/@\w+/g, '<span class="syntax-annotation">$&</span>');
        
        return code;
    }
    
    highlightShell(code) {
        // Shell commands and built-ins
        code = code.replace(/\b(echo|cd|ls|pwd|mkdir|rmdir|rm|cp|mv|cat|grep|find|awk|sed|sort|uniq|head|tail|wc|chmod|chown|ps|kill|top|df|du|mount|umount|tar|zip|unzip|wget|curl|ssh|scp|rsync|git|npm|pip|sudo|su|exit|which|whereis|alias|history|export|source|bash|sh|zsh|fish)\b/g, '<span class="syntax-builtin">$1</span>');
        
        // Shell keywords
        code = code.replace(/\b(if|then|else|elif|fi|for|do|done|while|until|case|esac|function|return|local|readonly|declare|unset|shift|break|continue|test|true|false)\b/g, '<span class="syntax-keyword">$1</span>');
        
        // Environment variables
        code = code.replace(/\$[{]?[A-Za-z_][A-Za-z0-9_]*[}]?/g, '<span class="syntax-variable">$&</span>');
        
        // Command substitution
        code = code.replace(/\$\([^)]*\)/g, '<span class="syntax-function">$&</span>');
        
        // Pipes and redirections
        code = code.replace(/[|>&<]+/g, '<span class="syntax-operator">$&</span>');
        
        // Strings
        code = code.replace(/(["'])(?:(?!\1)[^\\]|\\.)*\1/g, '<span class="syntax-string">$&</span>');
        
        // Comments
        code = code.replace(/#.*/g, '<span class="syntax-comment">$&</span>');
        
        return code;
    }
    
    highlightJson(code) {
        // JSON keys
        code = code.replace(/"([^"]+)"\s*:/g, '<span class="syntax-property">"$1"</span>:');
        
        // JSON values - strings
        code = code.replace(/:\s*"([^"]*)"/g, ': <span class="syntax-string">"$1"</span>');
        
        // JSON values - numbers
        code = code.replace(/:\s*(-?\d+\.?\d*)/g, ': <span class="syntax-number">$1</span>');
        
        // JSON values - booleans and null
        code = code.replace(/\b(true|false|null)\b/g, '<span class="syntax-constant">$1</span>');
        
        return code;
    }
    
    getLanguageKeywords(language) {
        const keywordMap = {
            'javascript': ['function', 'const', 'let', 'var', 'if', 'else', 'for', 'while', 'return', 'class', 'extends'],
            'python': ['def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return', 'import', 'from', 'as'],
            'java': ['public', 'private', 'class', 'interface', 'extends', 'implements', 'if', 'else', 'for', 'while', 'return'],
            'cpp': ['int', 'char', 'float', 'double', 'if', 'else', 'for', 'while', 'return', 'class', 'public', 'private'],
            'csharp': ['public', 'private', 'class', 'interface', 'if', 'else', 'for', 'while', 'return', 'string', 'int'],
            'go': ['func', 'var', 'const', 'if', 'else', 'for', 'return', 'package', 'import', 'struct'],
            'rust': ['fn', 'let', 'mut', 'if', 'else', 'for', 'while', 'return', 'struct', 'impl', 'pub']
        };
        
        return keywordMap[language.toLowerCase()] || [];
    }

    processMessageContent(content) {
        console.log('Processing message content:', content.substring(0, 100) + '...');
        
        // Handle code blocks first - separate from regular text
        const codeBlockRegex = /```(\w+)?\n?([\s\S]*?)```/g;
        const parts = [];
        let lastIndex = 0;
        let match;
        
        while ((match = codeBlockRegex.exec(content)) !== null) {
            // Add text before code block
            if (match.index > lastIndex) {
                const textBefore = content.slice(lastIndex, match.index).trim();
                if (textBefore) {
                    parts.push({ type: 'text', content: textBefore });
                }
            }
            
            // Add code block
            const language = match[1] || 'text';
            const code = match[2];
            parts.push({ type: 'code', language, code });
            lastIndex = match.index + match[0].length;
        }
        
        // Add remaining text
        if (lastIndex < content.length) {
            const remainingText = content.slice(lastIndex).trim();
            if (remainingText) {
                parts.push({ type: 'text', content: remainingText });
            }
        }
        
        // If no code blocks found, treat as text
        if (parts.length === 0) {
            parts.push({ type: 'text', content });
        }
        
        // Process each part
        let processedContent = '';
        for (const part of parts) {
            if (part.type === 'code') {
                const lang = part.language;
                const code = part.code.trim();
                
                // All code blocks (including shell/bash) are treated as regular code blocks
                const id = 'code-' + Math.random().toString(36).substr(2, 9);
                const langIcon = this.getLanguageIcon(lang);
                processedContent += `
                    <div class="code-block" data-language="${lang}">
                        <div class="code-header">
                            <div class="code-language-info">
                                ${langIcon}
                                <span class="code-language">${this.getLanguageDisplayName(lang)}</span>
                            </div>
                            <button class="copy-button" data-code-id="${id}" title="Copy code">
                                <i class="fas fa-copy"></i>
                                <span class="copy-text">Copy</span>
                            </button>
                        </div>
                        <div class="code-content" id="${id}" data-code="${this.escapeHtml(code)}"></div>
                    </div>
                `;
            } else {
                // Process text content - handle inline code and paragraphs
                let textContent = part.content;
                
                // Handle inline code
                textContent = textContent.replace(/`([^`\n]+)`/g, '<span class="inline-code">$1</span>');
                
                // Split into paragraphs
                const paragraphs = textContent.split('\n\n').filter(p => p.trim());
                for (const paragraph of paragraphs) {
                    if (paragraph.trim()) {
                        processedContent += `<p>${paragraph.trim()}</p>`;
                    }
                }
            }
        }
        
        console.log('Final processed content length:', processedContent.length);
        console.log('First 200 chars:', processedContent.substring(0, 200));
        
        return processedContent;
    }

    // Get language icon from code-icons folder
    getLanguageIcon(language) {
        const iconMap = {
            'javascript': 'javascript.svg',
            'js': 'javascript.svg',
            'typescript': 'typescript.svg',
            'ts': 'typescript.svg',
            'python': 'python.svg',
            'py': 'python.svg',
            'java': 'java.svg',
            'cpp': 'cpp.svg',
            'c++': 'cpp.svg',
            'c': 'c.svg',
            'csharp': 'csharp.svg',
            'cs': 'csharp.svg',
            'php': 'php.svg',
            'ruby': 'ruby.svg',
            'rb': 'ruby.svg',
            'go': 'go.svg',
            'rust': 'rust.svg',
            'rs': 'rust.svg',
            'html': 'html.svg',
            'css': 'css.svg',
            'scss': 'sass.svg',
            'sass': 'sass.svg',
            'less': 'less.svg',
            'json': 'json.svg',
            'xml': 'xml.svg',
            'yaml': 'yaml.svg',
            'yml': 'yaml.svg',
            'md': 'md',
            'shell': 'console.svg',
            'bash': 'console.svg',
            'sh': 'console.svg',
            'sql': 'database.svg',
            'dockerfile': 'docker.svg',
            'docker': 'docker.svg',
            'vue': 'vue.svg',
            'react': 'react.svg',
            'jsx': 'react.svg',
            'tsx': 'react.svg',
            'angular': 'angular.svg',
            'svelte': 'svelte.svg',
            'swift': 'swift.svg',
            'kotlin': 'kotlin.svg',
            'dart': 'dart.svg',
            'r': 'r.svg',
            'matlab': 'matlab.svg',
            'lua': 'lua.svg',
            'perl': 'perl.svg',
            'scala': 'scala.svg',
            'haskell': 'haskell.svg',
            'elixir': 'elixir.svg',
            'erlang': 'erlang.svg',
            'clojure': 'clojure.svg',
            'vim': 'vim.svg',
            'makefile': 'makefile.svg',
            'gradle': 'gradle.svg',
            'cmake': 'cmake.svg',
            'text': 'document.svg',
            'txt': 'txt'
        };
        
        const iconFile = iconMap[language.toLowerCase()] || 'document.svg';
        return `<img src="code-icons/${iconFile}" alt="${language}" class="language-icon" />`;
    }

    // Get display name for language
    getLanguageDisplayName(language) {
        const displayNames = {
            'js': 'JavaScript',
            'ts': 'TypeScript',
            'py': 'Python',
            'cpp': 'C++',
            'cs': 'C#',
            'rb': 'Ruby',
            'rs': 'Rust',
            'sh': 'Shell',
            'yml': 'YAML',
            'md': 'Markdown',
            'jsx': 'React JSX',
            'tsx': 'React TSX'
        };
        
        return displayNames[language.toLowerCase()] || language.charAt(0).toUpperCase() + language.slice(1);
    }

    attachCopyListeners(messageContent) {
        const copyButtons = messageContent.querySelectorAll('.copy-button');
        copyButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                
                // Handle both button and icon clicks
                const actualButton = e.target.closest('.copy-button');
                const codeId = actualButton.getAttribute('data-code-id');
                const codeElement = document.getElementById(codeId);
                
                if (codeElement) {
                    this.copyToClipboard(codeElement.textContent, actualButton);
                }
            });
        });
    }

    async copyToClipboard(text, button) {
        try {
            await navigator.clipboard.writeText(text);
            const copyText = button.querySelector('.copy-text');
            const copyIcon = button.querySelector('i');
            
            // Update button appearance
            if (copyText) copyText.textContent = 'Copied!';
            if (copyIcon) {
                copyIcon.className = 'fas fa-check';
            }
            button.classList.add('copied');
            
            setTimeout(() => {
                if (copyText) copyText.textContent = 'Copy';
                if (copyIcon) {
                    copyIcon.className = 'fas fa-copy';
                }
                button.classList.remove('copied');
            }, 2000);
        } catch (err) {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            const copyText = button.querySelector('.copy-text');
            const copyIcon = button.querySelector('i');
            
            if (copyText) copyText.textContent = 'Copied!';
            if (copyIcon) {
                copyIcon.className = 'fas fa-check';
            }
            button.classList.add('copied');
            
            setTimeout(() => {
                if (copyText) copyText.textContent = 'Copy';
                if (copyIcon) {
                    copyIcon.className = 'fas fa-copy';
                }
                button.classList.remove('copied');
            }, 2000);
        }
    }

    showTypingIndicator() {
        this.isLoading = true;
        this.sendButton.disabled = false;
        this.updateStatus('Typing...', '#f59e0b');
        this.updateSendButtonToStop();
        this.disableModelSelection();

        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message typing-indicator';
        typingDiv.id = 'typingIndicator';

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'typing-dot';
            typingDiv.appendChild(dot);
        }

        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.isLoading = false;
        this.sendButton.disabled = false;
        this.updateStatus('Ready', '#4ade80');
        this.updateSendButtonToNormal();
        this.enableModelSelection();

        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) typingIndicator.remove();
    }

    updateConversationStatsUI() {
        if (!this.conversationStats) {
            // Hide stats display if no real conversation stats exist
            const statsElement = document.getElementById('conversationStats');
            if (statsElement) {
                statsElement.style.display = 'none';
            }
            return;
        }
        
        const { total_tokens, max_tokens, utilization_percent, current_model } = this.conversationStats;
        
        // Only show loader if there are actual tokens being used
        if (total_tokens === 0 || !current_model) {
            const statsElement = document.getElementById('conversationStats');
            if (statsElement) {
                statsElement.style.display = 'none';
            }
            return;
        }
        
        // Create or update stats display
        let statsElement = document.getElementById('conversationStats');
        if (!statsElement) {
            statsElement = document.createElement('div');
            statsElement.id = 'conversationStats';
            statsElement.className = 'conversation-stats';
            
            // Insert after the model info in the footer
            const modelInfo = document.getElementById('modelInfo');
            if (modelInfo) {
                modelInfo.parentNode.insertBefore(statsElement, modelInfo.nextSibling);
            }
        }
        
        // Show the stats element
        statsElement.style.display = 'block';
        
        // Determine colors based on utilization
        const utilizationColor = utilization_percent > 80 ? '#ef4444' : 
                                utilization_percent > 60 ? '#f59e0b' : '#4ade80';
        
        const tokenDisplay = total_tokens > 999 ? 
            `${(total_tokens / 1000).toFixed(1)}K` : 
            total_tokens.toLocaleString();
            
        const maxTokenDisplay = max_tokens > 999 ? 
            `${(max_tokens / 1000).toFixed(0)}K` : 
            max_tokens.toLocaleString();
        
        // Create small loader with percentage on the right
        statsElement.innerHTML = `
            <div class="context-loader-container">
                <div class="small-loader" data-tooltip="Tokens: ${tokenDisplay}/${maxTokenDisplay} | Model: ${this.getModelDisplayName(current_model)}">
                    <svg class="loader-circle" viewBox="0 0 20 20">
                        <circle class="loader-bg" cx="10" cy="10" r="8" fill="none" stroke="rgba(76, 175, 80, 0.2)" stroke-width="2"/>
                        <circle class="loader-progress" 
                                cx="10" cy="10" r="8" 
                                fill="none" 
                                stroke="${utilizationColor}" 
                                stroke-width="2" 
                                stroke-linecap="round"
                                style="stroke-dasharray: ${2 * Math.PI * 8}; stroke-dashoffset: ${2 * Math.PI * 8 * (1 - utilization_percent / 100)}; transform: rotate(-90deg); transform-origin: center;"/>
                    </svg>
                </div>
                <span class="loader-percentage" style="color: ${utilizationColor};">${utilization_percent}%</span>
            </div>
        `;
        
        // Apply enhanced styling for the loader
        statsElement.style.cssText = `
            margin-top: 0.5rem;
            font-size: 0.8rem;
            color: #9ca3af;
            font-family: 'JetBrains Mono', monospace;
            background: rgba(76, 175, 80, 0.05);
            border: 1px solid rgba(76, 175, 80, 0.2);
            border-radius: 8px;
            padding: 8px 12px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            transition: all 0.3s ease;
            position: relative;
            display: block;
        `;
        
        // Add hover effect for glassmorphism enhancement
        statsElement.addEventListener('mouseenter', () => {
            statsElement.style.borderColor = 'rgba(76, 175, 80, 0.4)';
            statsElement.style.background = 'rgba(76, 175, 80, 0.08)';
        });
        
        statsElement.addEventListener('mouseleave', () => {
            statsElement.style.borderColor = 'rgba(76, 175, 80, 0.2)';
            statsElement.style.background = 'rgba(76, 175, 80, 0.05)';
        });
    }

    updateStatus(text, color) {
        // map color codes to tailwind classes
        const isGreen = color === '#4ade80'; // emerald-400/500
        const isAmber = color === '#f59e0b'; // amber-400/500
        const isRed = color === '#ef4444';   // red-400/500

        // update hover tooltip/title with current status
        if (this.statusPing) {
            this.statusPing.title = text;
            this.statusPing.setAttribute('aria-label', text);
        }

        const classSets = {
            greenOuter: ['bg-emerald-400'],
            greenInner: ['bg-emerald-500'],
            amberOuter: ['bg-amber-400'],
            amberInner: ['bg-amber-500'],
            redOuter: ['bg-red-400'],
            redInner: ['bg-red-500']
        };

        // clear previous color classes
        const allOuter = [...classSets.greenOuter, ...classSets.amberOuter, ...classSets.redOuter];
        const allInner = [...classSets.greenInner, ...classSets.amberInner, ...classSets.redInner];
        
        if (this.statusPingOuter && this.statusPingInner) {
            this.statusPingOuter.classList.remove(...allOuter);
            this.statusPingInner.classList.remove(...allInner);

            if (isGreen) {
                this.statusPingOuter.classList.add(...classSets.greenOuter);
                this.statusPingInner.classList.add(...classSets.greenInner);
            } else if (isAmber) {
                this.statusPingOuter.classList.add(...classSets.amberOuter);
                this.statusPingInner.classList.add(...classSets.amberInner);
            } else if (isRed) {
                this.statusPingOuter.classList.add(...classSets.redOuter);
                this.statusPingInner.classList.add(...classSets.redInner);
            } else {
                // default to green
                this.statusPingOuter.classList.add(...classSets.greenOuter);
                this.statusPingInner.classList.add(...classSets.greenInner);
            }
        }
    }

    updateSendButtonToStop() {
        const icon = document.getElementById('sendIcon');
        if (icon) {
            icon.className = 'fas fa-stop';
        }
        this.sendButton.classList.add('stop-mode');
        this.sendButton.setAttribute('data-mode', 'stop');
        this.sendButton.title = 'Stop generating';
    }

    updateSendButtonToNormal() {
        const icon = document.getElementById('sendIcon');
        if (icon) {
            icon.className = 'fas fa-paper-plane';
        }
        this.sendButton.classList.remove('stop-mode');
        this.sendButton.setAttribute('data-mode', 'send');
        this.sendButton.title = 'Send message';
    }

    stopGeneration() {
        // Stop any ongoing typing animation immediately
        if (this.currentTypingInterval) {
            clearInterval(this.currentTypingInterval);
            this.currentTypingInterval = null;
        }
        
        // Remove typing indicator and reset UI state
        this.hideTypingIndicator();
        
        // Set status to ready instead of stopped
        this.updateStatus('Ready', '#4ade80');
        
        // Don't add any message to chat - just stop silently
    }

    disableModelSelection() {
        const selectContainer = document.querySelector('.custom-select-container');
        if (selectContainer) {
            selectContainer.classList.add('disabled');
        }
        
        // Close dropdown if open
        this.closeDropdown();
    }

    enableModelSelection() {
        const selectContainer = document.querySelector('.custom-select-container');
        if (selectContainer) {
            selectContainer.classList.remove('disabled');
        }
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showErrorPopup(message) {
        const existingPopup = document.querySelector('.error-popup');
        if (existingPopup) existingPopup.remove();

        const popup = document.createElement('div');
        popup.className = 'error-popup';
        popup.textContent = message;

        const inputContainer = document.querySelector('.chat-input-container');
        inputContainer.style.position = 'relative';
        inputContainer.appendChild(popup);

        setTimeout(() => {
            if (popup.parentNode) popup.remove();
        }, 3000);
    }

    async checkServerStatus() {
        try {
            // Define server URLs with primary and failover
            const servers = window.location.hostname === 'antonjijo.github.io' 
                ? [
                    'https://nvidia-nim-bot.onrender.com',  // Primary server (Render)
                    'https://Nvidia.pythonanywhere.com'     // Failover server (PythonAnywhere)
                  ]
                : ['http://localhost:5000'];  // Development uses localhost only

            let connected = false;
            let connectedServer = '';
            
            // Try each server in sequence until one works
            for (let i = 0; i < servers.length; i++) {
                const serverURL = servers[i];
                
                try {
                    console.log(`Checking server ${i + 1}/${servers.length}: ${serverURL}`);
                    
                    const response = await fetch(`${serverURL}/health`, {
                        timeout: 5000  // 5 second timeout for health check
                    });
                    
                    if (response.ok) {
                        connected = true;
                        const serverHost = (new URL(serverURL)).hostname;
                        const serverName = serverHost === 'nvidia-nim-bot.onrender.com' ? 'Primary' :
                            (serverHost === 'nvidia.pythonanywhere.com' ||
                             serverHost.endsWith('.pythonanywhere.com')) ? 'Failover' : 'Local';
                        connectedServer = serverName;
                        console.log(`Server ${serverName} is online: ${serverURL}`);
                        break; // Found working server, stop checking
                    }
                } catch (error) {
                    console.warn(`Server ${i + 1} health check failed (${serverURL}):`, error.message);
                    // Continue to next server
                }
            }
            
            if (connected) {
                this.updateStatus(`Ready (${connectedServer})`, '#4ade80');
            } else {
                this.updateStatus('All Servers Offline', '#ef4444');
            }
            
        } catch (error) {
            console.error('Health check error:', error);
            this.updateStatus('Connection Error', '#ef4444');
        }
    }
    
    initializeCustomDropdown() {
        if (!this.customSelectTrigger || !this.customSelectDropdown) return;
        
        // Toggle dropdown on trigger click
        this.customSelectTrigger.addEventListener('click', (e) => {
            e.stopPropagation();
            
            // Don't open dropdown if disabled
            const selectContainer = document.querySelector('.custom-select-container');
            if (selectContainer && selectContainer.classList.contains('disabled')) {
                return;
            }
            
            this.toggleDropdown();
        });
        
        // Handle option selection
        const options = this.customSelectDropdown.querySelectorAll('.select-option');
        options.forEach(option => {
            option.addEventListener('click', (e) => {
                // Don't allow selection if disabled
                const selectContainer = document.querySelector('.custom-select-container');
                if (selectContainer && selectContainer.classList.contains('disabled')) {
                    return;
                }
                
                this.selectOption(e.target);
            });
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!this.customSelectTrigger.contains(e.target) && !this.customSelectDropdown.contains(e.target)) {
                this.closeDropdown();
            }
        });
        
        // Mark the default option as selected
        const firstOption = options[0];
        if (firstOption) {
            firstOption.classList.add('selected');
        }
    }
    
    toggleDropdown() {
        const isActive = this.customSelectDropdown.classList.contains('active');
        if (isActive) {
            this.closeDropdown();
        } else {
            this.openDropdown();
        }
    }
    
    openDropdown() {
        this.customSelectTrigger.classList.add('active');
        this.customSelectDropdown.classList.add('active');
    }
    
    closeDropdown() {
        this.customSelectTrigger.classList.remove('active');
        this.customSelectDropdown.classList.remove('active');
    }
    
    selectOption(optionElement) {
        // Get the value before making changes
        const value = optionElement.getAttribute('data-value');
        
        // Close the dropdown first
        this.closeDropdown();
        
        // Handle model change with confirmation
        this.handleModelChange(value);
    }
    
    setupGlobalScrolling() {
        // Enable scrolling of chat messages from anywhere in the chat container
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.addEventListener('wheel', (e) => {
                // Check if the scroll event should be handled by the chat messages
                const isInputFocused = document.activeElement === this.messageInput;
                const isDropdownOpen = this.customSelectDropdown && this.customSelectDropdown.classList.contains('active');
                
                // Only scroll chat messages if input is not focused and dropdown is not open
                if (!isInputFocused && !isDropdownOpen) {
                    e.preventDefault();
                    this.chatMessages.scrollTop += e.deltaY;
                }
            }, { passive: false });
            
            // Also enable touch scrolling for mobile devices
            let startY = 0;
            let scrollTop = 0;
            
            chatContainer.addEventListener('touchstart', (e) => {
                startY = e.touches[0].clientY;
                scrollTop = this.chatMessages.scrollTop;
            }, { passive: true });
            
            chatContainer.addEventListener('touchmove', (e) => {
                const isInputFocused = document.activeElement === this.messageInput;
                const isDropdownOpen = this.customSelectDropdown && this.customSelectDropdown.classList.contains('active');
                
                if (!isInputFocused && !isDropdownOpen) {
                    const currentY = e.touches[0].clientY;
                    const deltaY = startY - currentY;
                    this.chatMessages.scrollTop = scrollTop + deltaY;
                }
            }, { passive: true });
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new Chatbot();
});
