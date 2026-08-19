document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
    const startBtn = document.getElementById('start-btn');
    const videoFeed = document.getElementById('video-feed');
    const logTerminal = document.getElementById('log-terminal');
    
    const stopBtn = document.getElementById('stop-btn');
    const resetBtn = document.getElementById('reset-btn');
    
    const MAX_LOG_LINES = 200;
    let isTesting = false;

    startBtn.addEventListener('click', () => {
        isTesting = true;
        const env = document.getElementById('env-select').value;
        socket.emit('start_testing', { env: env });
        startBtn.disabled = true;
        stopBtn.disabled = false;
        resetBtn.disabled = false;
        startBtn.textContent = 'Testing...';
        
        const bugList = document.getElementById('bug-list');
        if (bugList && bugList.innerHTML.includes('start testing...')) {
            bugList.innerHTML = '<li id="bug-placeholder" style="color: #666; font-style: italic;">No bugs found yet...</li>';
        }
        
        if (logTerminal.innerHTML.includes('start testing...')) {
            logTerminal.innerHTML = '';
        }
    });

    stopBtn.addEventListener('click', () => {
        isTesting = false;
        socket.emit('stop_testing');
        stopBtn.disabled = true;
        startBtn.disabled = false;
        startBtn.textContent = 'START TESTING';
    });

    resetBtn.addEventListener('click', () => {
        isTesting = false;
        socket.emit('stop_testing');
        socket.emit('reset_game');
        
        logTerminal.innerHTML = '<p id="log-placeholder" style="color: #666; font-style: italic;">start testing...</p>';
        const bugList = document.getElementById('bug-list');
        if (bugList) {
            bugList.innerHTML = '<li id="bug-placeholder" style="color: #666; font-style: italic;">start testing...</li>';
        }
        
        videoFeed.removeAttribute('src');
        
        startBtn.disabled = false;
        stopBtn.disabled = true;
        resetBtn.disabled = true;
        startBtn.textContent = 'START TESTING';
    });

    socket.on('video_frame', (data) => {
        if (!isTesting) return;
        if (data.frame) {
            videoFeed.src = 'data:image/jpeg;base64,' + data.frame;
        }
    });

    socket.on('agent_log', (data) => {
        if (!isTesting) return;
        
        // Handle Bug Tracking UI
        if (data.log && data.log.includes('🚨 BUG FOUND:')) {
            const bugList = document.getElementById('bug-list');
            if (bugList) {
                // Remove the placeholder if it's there
                if (bugList.children[0] && bugList.children[0].textContent.includes('No bugs')) {
                    bugList.innerHTML = '';
                }
                
                // Extract the bug reason and step
                const stepMatch = data.log.match(/Step (\d+):/);
                const bugMatch = data.log.split('🚨 BUG FOUND: ')[1];
                
                const li = document.createElement('li');
                li.style.color = '#ff4444';
                li.style.marginBottom = '8px';
                li.innerHTML = `<strong>Step ${stepMatch ? stepMatch[1] : '?'}:</strong> ${bugMatch}`;
                bugList.appendChild(li);
                
                // Auto-scroll bug tracker
                bugList.parentElement.scrollTop = bugList.parentElement.scrollHeight;
            }
        }

        const p = document.createElement('p');
        p.textContent = data.log;
        logTerminal.appendChild(p);
        
        // Cap the log terminal
        while (logTerminal.children.length > MAX_LOG_LINES) {
            logTerminal.removeChild(logTerminal.firstChild);
        }
        
        // Auto-scroll
        logTerminal.scrollTop = logTerminal.scrollHeight;
    });

    // Info Modal Logic
    const infoBtn = document.getElementById('info-btn');
    const infoModal = document.getElementById('info-modal');
    const closeModal = document.querySelector('.close-modal');

    if (infoBtn && infoModal && closeModal) {
        infoBtn.addEventListener('click', () => {
            infoModal.classList.remove('hidden');
        });

        closeModal.addEventListener('click', () => {
            infoModal.classList.add('hidden');
        });

        // Close when clicking outside
        infoModal.addEventListener('click', (e) => {
            if (e.target === infoModal) {
                infoModal.classList.add('hidden');
            }
        });
    }
});
