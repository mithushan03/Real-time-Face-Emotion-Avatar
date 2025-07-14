document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam-feed');
    const emotionLabel = document.getElementById('emotion-label');
    const emojiDisplay = document.getElementById('emoji-display');
    const meshPreview = document.getElementById('mesh-preview');
    const videoContainer = document.getElementById('video-container');

    let stream;
    let mediaRecorder;
    let intervalId;

    // Get webcam feed
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((s) => {
            stream = s;
            video.srcObject = stream;
            video.play();

            video.onloadedmetadata = () => {
                // Start sending frames after video metadata is loaded
                startSendingFrames();
            };
        })
        .catch((err) => {
            console.error("Error accessing webcam: ", err);
            emotionLabel.textContent = "Error: Could not access webcam.";
        });

    function startSendingFrames() {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        intervalId = setInterval(() => {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/jpeg', 0.7); // Convert frame to base64 JPEG

            fetch('/video_feed', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData }),
            })
            .then(response => response.json())
            .then(data => {
                emotionLabel.textContent = data.emotion;
                emojiDisplay.innerHTML = data.emoji; // Use innerHTML to allow for potential HTML in emoji display

                // Remove funnyQuotes and related logic
                const emotion = data.emotion;
                
                console.log("Received emotion:", emotion);

                // Display emoji only
                emojiDisplay.innerHTML = data.emoji;

                if (data.mesh_image) {
                    meshPreview.src = 'data:image/jpeg;base64,' + data.mesh_image;
                    meshPreview.style.display = 'block';
                } else {
                    meshPreview.style.display = 'none';
                }
            })
            .catch(error => {
                console.error('Error sending frame to backend:', error);
                emotionLabel.textContent = "Error: Backend communication failed.";
            });
        }, 100); // Send frame every 100ms (10 FPS)
    }

    // Stop sending frames when the page is unloaded
    window.addEventListener('beforeunload', () => {
        if (intervalId) {
            clearInterval(intervalId);
        }
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });
});