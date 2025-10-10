class Chatbot {
    // Escape HTML meta-characters from a string
    escapeHTML(str) {
        if (typeof str !== 'string') return '';
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    // Safe DOM manipulation to prevent XSS
    safeSetHTML(element, content) {
        if (!element || typeof content !== 'string') return;
        
        // Clear existing content safely
        while (element.firstChild) {
            element.removeChild(element.firstChild);
        }
        
        // Create a temporary container to parse HTML safely
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = content;
        
        // Move all child nodes safely
        while (tempDiv.firstChild) {
            element.appendChild(tempDiv.firstChild);
        }
        
        // Initialize DataTables for any tables in the content
        this.initializeDataTables(element);
    }
    
    initializeDataTables(container) {
        // Wait for DOM to be ready and jQuery to be available
        setTimeout(() => {
            if (typeof $ !== 'undefined' && $.fn.DataTable) {
                const tables = container.querySelectorAll('table');
                tables.forEach(table => {
                    // Only initialize if not already initialized
                    if (!$(table).hasClass('dataTable')) {
                        $(table).DataTable({
                            paging: false,
                            searching: false,
                            info: false,
                            ordering: true,
                            responsive: true,
                            language: {
                                emptyTable: "No data available in table"
                            }
                        });
                    }
                });
            }
        }, 100);
    }

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
        
        if (existingSessionId && this._validateSessionId(existingSessionId)) {
            console.log('Resuming existing session:', existingSessionId);
            return existingSessionId;
        }
        
        // Generate a unique session ID for conversation persistence
        const newSessionId = 'session_' + Date.now() + '_' + this._secureRandomString(16);
        
        // Store the new session ID in localStorage
        localStorage.setItem('chatbot_session_id', newSessionId);
        console.log('Created new session:', newSessionId);
        
        return newSessionId;
    }

    _validateSessionId(sessionId) {
        // Validate session ID format
        if (!sessionId || typeof sessionId !== 'string') {
            return false;
        }
        
        // Check format: session_timestamp_randomstring
        const sessionPattern = /^session_\d+_[a-f0-9]{16}$/;
        return sessionPattern.test(sessionId);
    }
    
    createNewSession() {
        // Force create a new session and store it
        const newSessionId = 'session_' + Date.now() + '_' + this._secureRandomString(16);
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
                    console.log(`Fetching stats from: ${serverURL}/api/conversation/stats?session_id=${this.sessionId}`);
                    const response = await fetch(`${serverURL}/api/conversation/stats?session_id=${this.sessionId}`);
                    if (response.ok) {
                        this.conversationStats = await response.json();
                        console.log('Stats fetched successfully:', this.conversationStats);
                        return this.conversationStats;
                    } else {
                        console.warn(`Failed to fetch stats from ${serverURL}: ${response.status}`);
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
                        // Clear messages safely
                        while (this.chatMessages.firstChild) {
                            this.chatMessages.removeChild(this.chatMessages.firstChild);
                        }
                        
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
        // Create welcome message content safely
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        const logoImg = document.createElement('img');
        logoImg.src = 'Main_Logo.svg';
        logoImg.alt = 'NVIDIA';
        logoImg.className = 'nvidia-logo';
        avatarDiv.appendChild(logoImg);
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const messages = [
            "Hello! I'm your NVIDIA-powered chatbot with advanced capabilities:",
            "Multiple AI Models available:",
            "Meta: Llama 4 Maverick 17B",
            "DeepSeek: R1",
            "Qwen: Qwen2.5 Coder 32B",
            "Qwen: Qwen3 Coder 480B",
            "DeepSeek: V3.1",
            "OpenAI: GPT-OSS 120B",
            "Qwen: Qwen3 235B A22B",
            "Google: Gemma 3 27B IT",
            "Select your preferred model above and start chatting!"
        ];
        
        messages.forEach((text, index) => {
            const p = document.createElement('p');
            if (index === 1) {
                const strong = document.createElement('strong');
                strong.textContent = text;
                p.appendChild(strong);
            } else if (index > 1 && index < 10) {
                const parts = text.split(':');
                const strong = document.createElement('strong');
                strong.textContent = parts[0] + ':';
                p.appendChild(strong);
                p.appendChild(document.createTextNode(' ' + parts[1]));
            } else if (index === 10) {
                const strong = document.createElement('strong');
                strong.textContent = 'Select';
                p.appendChild(strong);
                p.appendChild(document.createTextNode(' your preferred model above and start chatting!'));
            } else {
                p.textContent = text;
            }
            contentDiv.appendChild(p);
        });
        
        welcomeDiv.appendChild(avatarDiv);
        welcomeDiv.appendChild(contentDiv);
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
        // Create modal content safely
        const modal = document.createElement('div');
        modal.className = 'confirmation-modal';
        
        // Header
        const header = document.createElement('div');
        header.className = 'modal-header';
        const h3 = document.createElement('h3');
        h3.textContent = 'Change Model & Start New Session?';
        const closeBtn = document.createElement('button');
        closeBtn.className = 'modal-close';
        closeBtn.setAttribute('aria-label', 'Close');
        const closeIcon = document.createElement('i');
        closeIcon.className = 'fas fa-times';
        closeBtn.appendChild(closeIcon);
        header.appendChild(h3);
        header.appendChild(closeBtn);
        
        // Content
        const content = document.createElement('div');
        content.className = 'modal-content';
        
        const modelChangeInfo = document.createElement('div');
        modelChangeInfo.className = 'model-change-info';
        
        // Current session stats
        const currentStats = document.createElement('div');
        currentStats.className = 'current-session-stats';
        const currentH4 = document.createElement('h4');
        currentH4.textContent = 'Current Session';
        currentStats.appendChild(currentH4);
        
        const statsGrid = document.createElement('div');
        statsGrid.className = 'stats-grid';
        
        const statData = [
            ['Model:', this.escapeHTML(this.getModelDisplayName(currentModel))],
            ['Messages:', stats.total_messages.toString()],
            ['Tokens:', stats.total_tokens.toLocaleString()],
            ['Utilization:', stats.utilization_percent + '%']
        ];
        
        statData.forEach(([label, value]) => {
            const statItem = document.createElement('div');
            statItem.className = 'stat-item';
            
            const statLabel = document.createElement('span');
            statLabel.className = 'stat-label';
            statLabel.textContent = label;
            
            const statValue = document.createElement('span');
            statValue.className = 'stat-value';
            statValue.textContent = value;
            
            statItem.appendChild(statLabel);
            statItem.appendChild(statValue);
            statsGrid.appendChild(statItem);
        });
        
        currentStats.appendChild(statsGrid);
        modelChangeInfo.appendChild(currentStats);
        
        // Arrow divider
        const arrowDiv = document.createElement('div');
        arrowDiv.className = 'arrow-divider';
        const arrowIcon = document.createElement('i');
        arrowIcon.className = 'fas fa-arrow-down';
        arrowDiv.appendChild(arrowIcon);
        modelChangeInfo.appendChild(arrowDiv);
        
        // New session info
        const newSessionInfo = document.createElement('div');
        newSessionInfo.className = 'new-session-info';
        const newH4 = document.createElement('h4');
        newH4.textContent = 'New Session';
        newSessionInfo.appendChild(newH4);
        
        const newModelDisplay = document.createElement('div');
        newModelDisplay.className = 'new-model-display';
        const newModelName = document.createElement('span');
        newModelName.className = 'new-model-name';
        newModelName.textContent = this.escapeHTML(this.getModelDisplayName(newModel));
        const freshStart = document.createElement('span');
        freshStart.className = 'fresh-start';
        freshStart.textContent = 'Fresh conversation start';
        newModelDisplay.appendChild(newModelName);
        newModelDisplay.appendChild(freshStart);
        newSessionInfo.appendChild(newModelDisplay);
        modelChangeInfo.appendChild(newSessionInfo);
        
        content.appendChild(modelChangeInfo);
        
        // Warning notice
        const warning = document.createElement('div');
        warning.className = 'warning-notice';
        const warningIcon = document.createElement('i');
        warningIcon.className = 'fas fa-exclamation-triangle';
        const warningText = document.createElement('span');
        warningText.textContent = 'Changing models will start a new session. Your current conversation will be cleared.';
        warning.appendChild(warningIcon);
        warning.appendChild(warningText);
        content.appendChild(warning);
        
        // Actions
        const actions = document.createElement('div');
        actions.className = 'modal-actions';
        
        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'cancel-btn';
        cancelBtn.setAttribute('data-action', 'cancel');
        const cancelIcon = document.createElement('i');
        cancelIcon.className = 'fas fa-times';
        cancelBtn.appendChild(cancelIcon);
        cancelBtn.appendChild(document.createTextNode(' Keep Current Session'));
        
        const confirmBtn = document.createElement('button');
        confirmBtn.className = 'confirm-btn';
        confirmBtn.setAttribute('data-action', 'confirm');
        const confirmIcon = document.createElement('i');
        confirmIcon.className = 'fas fa-check';
        confirmBtn.appendChild(confirmIcon);
        confirmBtn.appendChild(document.createTextNode(' Change Model & Start New'));
        
        actions.appendChild(cancelBtn);
        actions.appendChild(confirmBtn);
        
        modal.appendChild(header);
        modal.appendChild(content);
        modal.appendChild(actions);
        modalOverlay.appendChild(modal);
        
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
            // Create session controls safely
            const newSessionBtn = document.createElement('button');
            newSessionBtn.id = 'newSessionBtn';
            newSessionBtn.className = 'session-btn';
            newSessionBtn.title = 'Start New Session';
            const plusIcon = document.createElement('i');
            plusIcon.className = 'fas fa-plus';
            newSessionBtn.appendChild(plusIcon);
            sessionControls.appendChild(newSessionBtn);
            
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
        console.log('Loading initial conversation stats...');
        // Load conversation stats on page load
        const stats = await this.getConversationStats();
        console.log('Initial stats loaded:', stats);
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
        // Create modal content safely
        const modal = document.createElement('div');
        modal.className = 'confirmation-modal';
        
        // Header
        const header = document.createElement('div');
        header.className = 'modal-header';
        const h3 = document.createElement('h3');
        h3.textContent = 'Start New Conversation Session';
        const closeBtn = document.createElement('button');
        closeBtn.className = 'modal-close';
        closeBtn.setAttribute('aria-label', 'Close');
        const closeIcon = document.createElement('i');
        closeIcon.className = 'fas fa-times';
        closeBtn.appendChild(closeIcon);
        header.appendChild(h3);
        header.appendChild(closeBtn);
        
        // Content
        const content = document.createElement('div');
        content.className = 'modal-content';
        
        const newSessionConfig = document.createElement('div');
        newSessionConfig.className = 'new-session-config';
        
        // Model selection section
        const modelSection = document.createElement('div');
        modelSection.className = 'model-selection-section';
        const modelH4 = document.createElement('h4');
        modelH4.textContent = 'Select AI Model';
        modelSection.appendChild(modelH4);
        
        const modelGrid = document.createElement('div');
        modelGrid.className = 'model-grid';
        
        const models = [
            { value: 'meta/llama-4-maverick-17b-128e-instruct', name: 'Llama 4 Maverick 17B', desc: 'Latest Meta model with enhanced reasoning' },
            { value: 'deepseek-ai/deepseek-r1', name: 'DeepSeek R1', desc: 'Advanced reasoning and problem-solving' },
            { value: 'qwen/qwen2.5-coder-32b-instruct', name: 'Qwen2.5 Coder 32B', desc: 'Specialized for coding tasks' },
            { value: 'qwen/qwen3-coder-480b-a35b-instruct', name: 'Qwen3 Coder 480B', desc: 'Advanced coding and development' },
            { value: 'deepseek-ai/deepseek-v3.1', name: 'DeepSeek V3.1', desc: 'General purpose AI assistant' },
            { value: 'openai/gpt-oss-120b', name: 'GPT-OSS 120B', desc: 'Open source GPT variant' },
            { value: 'qwen/qwen3-235b-a22b:free', name: 'Qwen3 235B A22B', desc: 'Free tier Qwen model' },
            { value: 'google/gemma-3-27b-it:free', name: 'Gemma 3 27B IT', desc: 'Google\'s instruction-tuned model' }
        ];
        
        models.forEach(model => {
            const option = document.createElement('div');
            option.className = 'model-option';
            option.setAttribute('data-model', model.value);
            
            const nameDiv = document.createElement('div');
            nameDiv.className = 'model-name';
            nameDiv.textContent = model.name;
            
            const descDiv = document.createElement('div');
            descDiv.className = 'model-description';
            descDiv.textContent = model.desc;
            
            option.appendChild(nameDiv);
            option.appendChild(descDiv);
            modelGrid.appendChild(option);
        });
        
        modelSection.appendChild(modelGrid);
        newSessionConfig.appendChild(modelSection);
        
        // Current session info (conditional)
        if (currentStats && currentStats.total_tokens > 0 && currentStats.current_model) {
            const currentInfo = document.createElement('div');
            currentInfo.className = 'current-session-info';
            const currentH4 = document.createElement('h4');
            currentH4.textContent = 'Current Session';
            currentInfo.appendChild(currentH4);
            
            const sessionSummary = document.createElement('div');
            sessionSummary.className = 'session-summary';
            
            const summaryData = [
                ['Model:', this.escapeHTML(this.getModelDisplayName(currentStats.current_model || currentModel))],
                ['Tokens Used:', currentStats.total_tokens.toLocaleString()],
                ['Utilization:', currentStats.utilization_percent + '%']
            ];
            
            summaryData.forEach(([label, value]) => {
                const item = document.createElement('div');
                item.className = 'summary-item';
                
                const labelSpan = document.createElement('span');
                labelSpan.className = 'summary-label';
                labelSpan.textContent = label;
                
                const valueSpan = document.createElement('span');
                valueSpan.className = 'summary-value';
                valueSpan.textContent = value;
                
                item.appendChild(labelSpan);
                item.appendChild(valueSpan);
                sessionSummary.appendChild(item);
            });
            
            currentInfo.appendChild(sessionSummary);
            
            const warning = document.createElement('div');
            warning.className = 'warning-notice';
            const warningIcon = document.createElement('i');
            warningIcon.className = 'fas fa-exclamation-triangle';
            const warningText = document.createElement('span');
            warningText.textContent = 'Starting a new session will clear your current conversation.';
            warning.appendChild(warningIcon);
            warning.appendChild(warningText);
            currentInfo.appendChild(warning);
            
            newSessionConfig.appendChild(currentInfo);
        }
        
        content.appendChild(newSessionConfig);
        
        // Actions
        const actions = document.createElement('div');
        actions.className = 'modal-actions';
        
        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'cancel-btn';
        cancelBtn.setAttribute('data-action', 'cancel');
        const cancelIcon = document.createElement('i');
        cancelIcon.className = 'fas fa-times';
        cancelBtn.appendChild(cancelIcon);
        cancelBtn.appendChild(document.createTextNode(' Cancel'));
        
        const confirmBtn = document.createElement('button');
        confirmBtn.className = 'confirm-btn';
        confirmBtn.setAttribute('data-action', 'confirm');
        const confirmIcon = document.createElement('i');
        confirmIcon.className = 'fas fa-plus';
        confirmBtn.appendChild(confirmIcon);
        confirmBtn.appendChild(document.createTextNode(' Start New Session'));
        
        actions.appendChild(cancelBtn);
        actions.appendChild(confirmBtn);
        
        modal.appendChild(header);
        modal.appendChild(content);
        modal.appendChild(actions);
        modalOverlay.appendChild(modal);
        
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

        // Live token estimation for user feedback
        const estimatedTokens = this.estimateTokens(this.messageInput.value);
        
        // Update character count with token estimate
        if (estimatedTokens > 0) {
            this.charCount.textContent = `${count}/5000 (~${estimatedTokens} tokens)`;
        }

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

    // Add client-side token estimation for live feedback
    estimateTokens(text) {
        if (!text || text.trim().length === 0) return 0;
        
        const words = text.split(/\s+/).filter(word => word.length > 0);
        const wordCount = words.length;
        const charCount = text.length;
        
        // Match backend estimation logic
        let estimatedTokens = Math.floor(wordCount * 2.2);
        
        // Special characters
        const specialChars = (text.match(/[^\w\s]/g) || []).length;
        estimatedTokens += Math.floor(specialChars / 2);
        
        // Long words
        const longWords = words.filter(word => word.length > 6).length;
        estimatedTokens += longWords;
        
        // Long text penalty
        if (charCount > 1000) {
            estimatedTokens += Math.floor(charCount / 40);
        }
        
        return Math.max(1, estimatedTokens);
    }

    validateUserInput(message) {
        // Check length
        if (message.length > 10000) {
            return { isValid: false, error: 'Message too long (max 10,000 characters)' };
        }

        if (message.length === 0) {
            return { isValid: false, error: 'Message cannot be empty' };
        }

        // Check for suspicious patterns
        const suspiciousPatterns = [
            /<script[^>]*>/i,
            /javascript:/i,
            /vbscript:/i,
            /data:text\/html/i,
            /<iframe[^>]*>/i,
            /<object[^>]*>/i,
            /<embed[^>]*>/i,
            /<link[^>]*>/i,
            /<meta[^>]*>/i,
            /<style[^>]*>/i
        ];

        for (const pattern of suspiciousPatterns) {
            if (pattern.test(message)) {
                return { isValid: false, error: 'Message contains potentially dangerous content' };
            }
        }

        // Check for excessive special characters (potential injection)
        const specialCharCount = (message.match(/[<>'"&]/g) || []).length;
        if (specialCharCount > message.length * 0.1) {
            return { isValid: false, error: 'Message contains too many special characters' };
        }

        return { isValid: true, error: null };
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isLoading) return;

        // Client-side input validation
        const validationResult = this.validateUserInput(message);
        if (!validationResult.isValid) {
            this.showErrorPopup(validationResult.error);
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
                    console.log('Received conversation stats:', data.conversation_stats);
                    this.conversationStats = data.conversation_stats;
                    
                    // Immediate UI update with animation
                    this.updateConversationStatsUI(true);
                    
                    // Force refresh stats after a short delay to ensure accuracy
                    setTimeout(() => {
                        this.refreshConversationStats();
                    }, 500);
                } else {
                    console.warn('No conversation stats received from server');
                }

                // Success! Update status and return response
                let serverName = 'Local';
                try {
                    const urlObj = new URL(serverURL);
                    const hostname = urlObj.hostname;
                    if (hostname === 'render.com' || hostname.endsWith('.render.com')) {
                        serverName = 'Primary';
                    } else if (hostname === 'pythonanywhere.com' || hostname.endsWith('.pythonanywhere.com')) {
                        serverName = 'Failover';
                    }
                } catch(e) {
                    // If invalid URL, keep serverName as 'Local'
                }
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
            userIcon.src = 'https://img.icons8.com/puffy-filled/48/FFFFFF/user.png';
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
        this.safeSetHTML(messageContent, this.processMessageContent(content));

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
            userIcon.src = 'https://img.icons8.com/puffy-filled/48/FFFFFF/user.png';
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
        this.safeSetHTML(messageContent, processedContent);
        
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
        this.safeSetHTML(messageContent, processedContent);
        
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
                const p = document.createElement('p');
                p.textContent = currentText;
                messageContent.innerHTML = '';
                messageContent.appendChild(p);
                this.scrollToBottom();
            } else {
                clearInterval(this.currentTypingInterval);
                this.currentTypingInterval = null;
                // Show the final processed content
                this.safeSetHTML(messageContent, processedContent);
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
                // Process text content with Markdown formatting
                let textContent = part.content;
                
                // Apply Markdown formatting
                textContent = this.applyMarkdownFormatting(textContent);
                
                // Split into paragraphs, but don't wrap tables in <p> tags
                const paragraphs = textContent.split('\n\n').filter(p => p.trim());
                for (const paragraph of paragraphs) {
                    if (paragraph.trim()) {
                        // Check if this paragraph contains a table
                        if (paragraph.includes('<table>')) {
                            processedContent += paragraph.trim();
                        } else {
                            processedContent += `<p>${paragraph.trim()}</p>`;
                        }
                    }
                }
            }
        }
        
        console.log('Final processed content length:', processedContent.length);
        console.log('First 200 chars:', processedContent.substring(0, 200));
        
        return processedContent;
    }

    // Function to apply Markdown formatting to text
    applyMarkdownFormatting(text) {
        // Process inline code (must be done first to avoid conflicts)
        text = text.replace(/`([^`\n]+)`/g, '<span class="inline-code">$1</span>');
        
        // Process bold text (**text** or __text__)
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/__(.*?)__/g, '<strong>$1</strong>');
        
        // Process italic text (*text* or _text_)
        text = text.replace(/\*([^\*\n]+?)\*/g, '<em>$1</em>');
        text = text.replace(/_([^_\n]+?)_/g, '<em>$1</em>');
        
        // Process strikethrough text (~~text~~)
        text = text.replace(/~~(.*?)~~/g, '<del>$1</del>');
        
        // Process links ([text](url))
        text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
        
        // Process headers (# Header)
        text = text.replace(/^######\s+(.+)$/gm, '<h6>$1</h6>');
        text = text.replace(/^#####\s+(.+)$/gm, '<h5>$1</h5>');
        text = text.replace(/^####\s+(.+)$/gm, '<h4>$1</h4>');
        text = text.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
        text = text.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>');
        text = text.replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');
        
        // Process horizontal rules
        text = text.replace(/^---$/gm, '<hr>');
        
        // Process blockquotes
        text = text.replace(/^>\s+(.+)$/gm, '<blockquote>$1</blockquote>');
        
        // Process tables
        text = this.processTables(text);
        
        // Process unordered lists
        text = text.replace(/^(\s*[-*+]\s.+)$/gm, '<li>$1</li>');
        text = text.replace(/(<li>.*<\/li>)+/gs, '<ul>$&</ul>');
        text = text.replace(/<li>\s*[-*+]\s/g, '<li>');
        
        // Process ordered lists with proper numbering
        text = text.replace(/^(\s*\d+\.\s.+)$/gm, '<li>$1</li>');
        text = text.replace(/(<li>.*<\/li>)+/gs, '<ol>$&</ol>');
        text = text.replace(/<li>\s*\d+\.\s/g, '<li>');
        
        // Process line breaks
        text = text.replace(/\n/g, '<br>');
        
        return text;
    }

    // Function to process Markdown tables
    processTables(text) {
        const lines = text.split('\n');
        let inTable = false;
        let tableLines = [];
        let result = [];
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            
            // Check if this line starts a table row (has | characters and proper table structure)
            if (this.isTableRow(line)) {
                if (!inTable) {
                    inTable = true;
                    tableLines = [];
                }
                tableLines.push(line);
            } else {
                if (inTable) {
                    // Process the collected table lines
                    result.push(this.convertTableToHTML(tableLines));
                    inTable = false;
                    tableLines = [];
                }
                result.push(line);
            }
        }
        
        // Handle case where table is at the end of the text
        if (inTable) {
            result.push(this.convertTableToHTML(tableLines));
        }
        
        return result.join('\n');
    }

    // Helper function to check if a line is a table row
    isTableRow(line) {
        // Trim the line and check if it starts and ends with |
        const trimmed = line.trim();
        if (!trimmed.startsWith('|') || !trimmed.endsWith('|')) {
            return false;
        }
        
        // Check if it has at least 3 | characters (minimum for a table row)
        const pipeCount = (trimmed.match(/\|/g) || []).length;
        return pipeCount >= 3;
    }

    // Function to convert Markdown table lines to HTML table
    convertTableToHTML(tableLines) {
        if (tableLines.length < 2) return tableLines.join('\n');
        
        // Parse header
        const headerLine = tableLines[0];
        const separatorLine = tableLines[1];
        const dataLines = tableLines.slice(2);
        
        // Parse header cells
        const headers = headerLine.split('|')
            .map(cell => cell.trim())
            .filter(cell => cell.length > 0);
        
        // Start building HTML table
        let html = '<table>\n';
        
        // Add header row
        html += '  <thead>\n    <tr>\n';
        headers.forEach(header => {
            // Apply markdown formatting to headers instead of escaping HTML
            const formattedHeader = this.applyMarkdownFormatting(header);
            html += `      <th>${formattedHeader}</th>\n`;
        });
        html += '    </tr>\n  </thead>\n';
        
        // Add body rows
        html += '  <tbody>\n';
        dataLines.forEach(line => {
            // Only process lines that are actual data rows
            if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
                const cells = line.split('|')
                    .map(cell => cell.trim())
                    .filter(cell => cell.length > 0);
                if (cells.length > 0) {
                    html += '    <tr>\n';
                    cells.forEach(cell => {
                        html += `      <td>${this.escapeHtml(cell)}</td>\n`;
                    });
                    html += '    </tr>\n';
                }
            }
        });
        html += '  </tbody>\n';
        html += '</table>';
        
        return html;
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

    updateConversationStatsUI(animate = false) {
        console.log('updateConversationStatsUI called with stats:', this.conversationStats);
        
        if (!this.conversationStats) {
            console.log('No conversation stats available, hiding display');
            const statsElement = document.getElementById('conversationStats');
            if (statsElement) {
                statsElement.style.display = 'none';
            }
            return;
        }
        
        const { total_tokens, max_tokens, utilization_percent, current_model } = this.conversationStats;
        console.log('Stats details:', { total_tokens, max_tokens, utilization_percent, current_model });
        
        // Show loader even with 0 tokens if we have valid stats and a model
        // This allows users to see that stats tracking is working
        if (!current_model) {
            console.log('No current model, hiding stats');
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
        
        // Create small loader with percentage on the right safely
        const container = document.createElement('div');
        container.className = 'context-loader-container';
        
        const loader = document.createElement('div');
        loader.className = 'small-loader';
        loader.setAttribute('data-tooltip', `Tokens: ${tokenDisplay}/${maxTokenDisplay} | Model: ${this.getModelDisplayName(current_model)}`);
        
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('class', 'loader-circle');
        svg.setAttribute('viewBox', '0 0 20 20');
        
        const bgCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        bgCircle.setAttribute('class', 'loader-bg');
        bgCircle.setAttribute('cx', '10');
        bgCircle.setAttribute('cy', '10');
        bgCircle.setAttribute('r', '8');
        bgCircle.setAttribute('fill', 'none');
        bgCircle.setAttribute('stroke', 'rgba(76, 175, 80, 0.2)');
        bgCircle.setAttribute('stroke-width', '2');
        
        const progressCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        progressCircle.setAttribute('class', 'loader-progress');
        progressCircle.setAttribute('cx', '10');
        progressCircle.setAttribute('cy', '10');
        progressCircle.setAttribute('r', '8');
        progressCircle.setAttribute('fill', 'none');
        progressCircle.setAttribute('stroke', utilizationColor);
        progressCircle.setAttribute('stroke-width', '2');
        progressCircle.setAttribute('stroke-linecap', 'round');
        progressCircle.style.strokeDasharray = 2 * Math.PI * 8;
        progressCircle.style.strokeDashoffset = 2 * Math.PI * 8 * (1 - utilization_percent / 100);
        progressCircle.style.transform = 'rotate(-90deg)';
        progressCircle.style.transformOrigin = 'center';
        if (animate) {
            progressCircle.style.transition = 'stroke-dashoffset 0.8s ease-in-out';
        }
        
        svg.appendChild(bgCircle);
        svg.appendChild(progressCircle);
        loader.appendChild(svg);
        
        const percentage = document.createElement('span');
        percentage.className = 'loader-percentage';
        percentage.style.color = utilizationColor;
        if (animate) {
            percentage.style.transition = 'color 0.3s ease';
        }
        percentage.textContent = utilization_percent + '%';
        
        container.appendChild(loader);
        container.appendChild(percentage);
        
        // Clear and append safely
        while (statsElement.firstChild) {
            statsElement.removeChild(statsElement.firstChild);
        }
        statsElement.appendChild(container);
        
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

    // Force refresh conversation stats from server
    async refreshConversationStats() {
        const stats = await this.getConversationStats();
        if (stats) {
            console.log('Refreshed stats from server:', stats);
            this.updateConversationStatsUI(true);
        }
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
