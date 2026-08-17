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
        socket.emit('start_testing');
        startBtn.disabled = true;
        stopBtn.disabled = false;
        resetBtn.disabled = false;
        startBtn.textContent = 'Testing...';
    });

    stopBtn.addEventListener('click', () => {
        isTesting = false;
        socket.emit('stop_testing');
        stopBtn.disabled = true;
        startBtn.disabled = false;
        startBtn.textContent = 'Start Autonomous Test';
    });

    resetBtn.addEventListener('click', () => {
        isTesting = false;
        socket.emit('stop_testing');
        logTerminal.innerHTML = '';
        videoFeed.removeAttribute('src');
        
        startBtn.disabled = false;
        stopBtn.disabled = true;
        resetBtn.disabled = true;
        startBtn.textContent = 'Start Autonomous Test';
    });

    socket.on('video_frame', (data) => {
        if (!isTesting) return;
        if (data.frame) {
            videoFeed.src = 'data:image/jpeg;base64,' + data.frame;
        }
    });

    socket.on('agent_log', (data) => {
        if (!isTesting) return;
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
});
