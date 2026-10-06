function sendCommand(cmd) {
    fetch('/command', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({command: cmd})
    }).then(res => {
        console.log("Command sent: " + cmd);
    });
}

function toggleMute() {
    fetch('/toggle_mute', { method: 'POST' });
}

function updateStatus() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            // Update FPS
            document.getElementById('fps-counter').innerText = `FPS: ${data.fps}`;
            
            // Update Mute Button
            const muteBtn = document.getElementById('mute-btn');
            if (muteBtn) {
                if (data.is_muted) {
                    muteBtn.innerText = "Unmute Auto-Alerts";
                    muteBtn.style.background = "#ef4444";
                } else {
                    muteBtn.innerText = "Mute Auto-Alerts";
                    muteBtn.style.background = "#22c55e";
                }
            }
            
            // Update Objects List
            const objList = document.getElementById('objects-list');
            document.getElementById('obj-count').innerText = data.objects.length;
            objList.innerHTML = '';
            
            data.objects.forEach(obj => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <span class="obj-name">ID:${obj.id} ${obj.class}</span>
                    <span class="obj-dist">${obj.distance}m</span>
                `;
                objList.appendChild(li);
            });

            // Update Logs
            const logsContainer = document.getElementById('logs-container');
            logsContainer.innerHTML = '';
            data.logs.forEach(log => {
                const div = document.createElement('div');
                div.className = 'log-entry ' + (log.includes('YOU:') ? 'you' : 'sys');
                div.innerText = log;
                logsContainer.appendChild(div);
            });
        })
        .catch(error => console.error('Error fetching status:', error));
}

// Poll status every 500ms
setInterval(updateStatus, 500);
