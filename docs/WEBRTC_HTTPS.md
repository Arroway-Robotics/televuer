# Using WebRTC with HTTPS in TeleVuer

## Overview

TeleVuer supports WebRTC for low-latency video streaming from robots to VR headsets. However, modern browsers require HTTPS for WebRTC connections, which can cause issues when the robot's WebRTC server uses HTTP.

## The Mixed Content Problem

When using TeleVuer with WebRTC, you may encounter a "mixed content" error if:
- Your Vuer server is running on HTTPS (required for VR/XR features)
- Your robot's WebRTC endpoint is HTTP (e.g., `http://192.168.123.161:8081/offer`)

Browsers block this combination for security reasons.

## Solution: HTTPS Reverse Proxy

The recommended solution is to set up an HTTPS reverse proxy on the robot that forwards requests to the local HTTP WebRTC server.

### Architecture

```
TeleVuer (HTTPS) → HTTPS Proxy (Robot) → WebRTC Server (HTTP, localhost)
```

## Implementation

### Step 1: Set Up Proxy on Robot

Choose either Caddy (easier) or Nginx (more control):

#### Option A: Caddy

```bash
# Install Caddy
sudo apt install caddy

# Create /etc/caddy/Caddyfile
<robot-ip>:8443 {
    tls internal
    reverse_proxy localhost:8081
}

# Start Caddy
sudo systemctl enable caddy
sudo systemctl start caddy
```

#### Option B: Nginx

```bash
# Install Nginx
sudo apt install nginx

# Generate SSL certificate
sudo mkdir -p /etc/nginx/ssl
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/robot.key \
  -out /etc/nginx/ssl/robot.crt \
  -subj "/C=US/ST=State/L=City/O=Robotics/CN=<robot-ip>"

# Create /etc/nginx/sites-available/webrtc-proxy
server {
    listen 8443 ssl;
    server_name <robot-ip>;
    
    ssl_certificate /etc/nginx/ssl/robot.crt;
    ssl_certificate_key /etc/nginx/ssl/robot.key;
    
    location / {
        proxy_pass http://localhost:8081;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}

# Enable and start
sudo ln -s /etc/nginx/sites-available/webrtc-proxy /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

### Step 2: Update TeleVuer Code

```python
from televuer import TeleVuerWrapper

# Initialize with WebRTC enabled
tv_wrapper = TeleVuerWrapper(
    use_hand_tracking=False,
    pass_through=False,
    binocular=True,
    img_shape=(480, 1280),
    cert_file="/path/to/vuer/cert.pem",  # SSL cert for Vuer server
    key_file="/path/to/vuer/key.pem",     # SSL key for Vuer server
    webrtc=True,                          # Enable WebRTC
    webrtc_url="https://192.168.123.161:8443/offer"  # HTTPS endpoint
)
```

### Step 3: Handle Certificate Warning

Since you're using a self-signed certificate, browsers will show a security warning. You can:

1. **Accept the warning**: Click "Advanced" → "Proceed" (quick fix)
2. **Add to trust store**: Install the certificate on your client machine (better for development)

## Usage Examples

### Basic WebRTC Streaming

```python
from televuer import TeleVuer
import asyncio

# Initialize TeleVuer with WebRTC
tv = TeleVuer(
    use_hand_tracking=True,
    binocular=True,
    img_shape=(480, 1280),
    webrtc=True,
    webrtc_url="https://192.168.123.161:8443/offer"
)

# Start the Vuer server
tv.run()
```

### With TeleVuerWrapper

```python
from televuer import TeleVuerWrapper

tv_wrapper = TeleVuerWrapper(
    use_hand_tracking=True,
    binocular=True,
    img_shape=(480, 1280),
    webrtc=True,
    webrtc_url="https://192.168.123.161:8443/offer"
)

# Get teleoperation data
while True:
    tele_data = tv_wrapper.get_tele_data()
    
    # Access pose data
    head_pose = tele_data['head_pose']
    left_hand_pose = tele_data['left_hand_pose']
    right_hand_pose = tele_data['right_hand_pose']
    
    # Video streaming is handled automatically by WebRTC
    # No need to call render_to_xr() manually
```

### Fallback to ImageBackground

If WebRTC is not available, you can fall back to ImageBackground mode:

```python
tv_wrapper = TeleVuerWrapper(
    use_hand_tracking=True,
    binocular=True,
    img_shape=(480, 1280),
    webrtc=False  # Disable WebRTC
)

# Manually send images
while True:
    tele_data = tv_wrapper.get_tele_data()
    
    # Get image from robot camera
    image = get_robot_camera_image()
    
    # Send to VR headset
    tv_wrapper.render_to_xr(image)
```

## Troubleshooting

### "Mixed Content" Error

**Symptom**: Browser console shows "Mixed content blocked"

**Solution**: 
- Verify `webrtc_url` uses `https://` (not `http://`)
- Check that proxy is running: `curl -k https://<robot-ip>:8443/offer`

### WebRTC Connection Fails

**Symptom**: No video stream in VR headset

**Solution**:
- Check WebRTC server is running: `netstat -tulpn | grep 8081`
- Check proxy logs:
  - Caddy: `sudo journalctl -u caddy -f`
  - Nginx: `sudo tail -f /var/log/nginx/error.log`
- Verify firewall allows port 8443

### Certificate Warning

**Symptom**: Browser shows "Your connection is not private"

**Solution**: This is normal for self-signed certificates. Click "Advanced" and "Proceed" to continue.

### High Latency

**Symptom**: Video stream has noticeable delay

**Solution**:
- Check network connection quality
- Reduce video resolution or frame rate
- Ensure proxy buffering is disabled (should be by default in provided configs)

## Testing

```bash
# 1. Test WebRTC server (on robot)
curl http://localhost:8081/offer

# 2. Test HTTPS proxy (on robot)
curl -k https://localhost:8443/offer

# 3. Test from client machine
curl -k https://192.168.123.161:8443/offer

# 4. Check browser console (F12) for WebRTC connection status
```

## Additional Resources

For complete setup scripts and detailed documentation, see the [g1-teleoperate repository](https://github.com/Arroway-Robotics/g1-teleoperate/tree/main/docs/webrtc_https).

## Parameters Reference

### TeleVuer.__init__()

```python
def __init__(
    self,
    use_hand_tracking: bool,
    pass_through: bool = False,
    binocular: bool = True,
    img_shape: tuple = None,
    cert_file: str = None,
    key_file: str = None,
    webrtc: bool = False,
    webrtc_url: str = None
)
```

**Parameters:**
- `use_hand_tracking`: Whether to use hand tracking or controller tracking
- `pass_through`: Enable AR passthrough mode
- `binocular`: True for stereo video, False for mono
- `img_shape`: Image dimensions (height, width)
- `cert_file`: Path to SSL certificate for Vuer server
- `key_file`: Path to SSL key for Vuer server
- `webrtc`: Enable WebRTC mode for video streaming
- `webrtc_url`: URL for WebRTC offer endpoint (must be HTTPS if Vuer uses HTTPS)

## Notes

- WebRTC provides lower latency than ImageBackground mode
- HTTPS is required for WebRTC in modern browsers
- The proxy solution adds minimal latency (<1ms typically)
- Self-signed certificates are fine for development but use proper certificates for production
- ICE servers can be configured via `iceServers` parameter in WebRTCVideoPlane/WebRTCStereoVideoPlane if needed for NAT traversal
