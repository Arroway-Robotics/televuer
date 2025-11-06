# Same-Origin Solution for WebRTC

## Overview

The **same-origin solution** is the cleanest way to enable WebRTC between Vuer and your robot. Instead of dealing with CORS headers, you serve both Vuer and the WebRTC endpoint from the same origin (domain and port), then use a reverse proxy to forward WebRTC requests to the robot.

## Why Same-Origin is Better

### Compared to CORS Approach

| Aspect | CORS Approach | Same-Origin Approach |
|--------|--------------|---------------------|
| **Configuration** | Need CORS headers on robot | Simple proxy route |
| **Security** | Wildcard origins (`*`) | No cross-origin requests |
| **Complexity** | Handle preflight requests | Just proxy the path |
| **Browser Support** | May have edge cases | Works everywhere |
| **Debugging** | CORS errors can be cryptic | Standard HTTP proxying |

### Architecture Comparison

**CORS Approach:**
```
Browser → https://10.1.10.187:8012 (Vuer)
Browser → https://192.168.123.161:8443/offer (WebRTC) ← Different origin!
          ↑ Requires CORS headers
```

**Same-Origin Approach:**
```
Browser → https://10.1.10.187:8012 (Vuer)
Browser → https://10.1.10.187:8012/offer (WebRTC) ← Same origin!
          ↓ Caddy proxies internally
          http://192.168.123.161:8081 (Robot)
```

---

## Implementation

### Step 1: Configure Caddy on Vuer Server (10.1.10.187)

Edit `/etc/caddy/Caddyfile` on your Vuer server:

```caddy
10.1.10.187:8012 {
    tls internal  # Or your real certificate
    
    # Proxy /offer to robot's WebRTC server
    handle_path /offer* {
        reverse_proxy http://192.168.123.161:8081 {
            header_up Host {upstream_hostport}
            header_up X-Real-IP {remote_host}
            header_up X-Forwarded-For {remote_host}
            header_up X-Forwarded-Proto {scheme}
        }
    }
    
    # Your existing Vuer routes
    handle /* {
        # Your Vuer app configuration
        reverse_proxy localhost:8000
        # Or: file_server, etc.
    }
}
```

### Step 2: Restart Caddy

```bash
# On Vuer server (10.1.10.187)
sudo systemctl restart caddy
sudo systemctl status caddy
```

### Step 3: Update Your Code

Change the `webrtc_url` to use the same origin as Vuer:

```python
from televuer import TeleVuerWrapper

tv_wrapper = TeleVuerWrapper(
    use_hand_tracking=False,
    pass_through=False,
    binocular=True,
    img_shape=(480, 1280),
    cert_file="/path/to/cert.pem",
    key_file="/path/to/key.pem",
    webrtc=True,
    webrtc_url="https://10.1.10.187:8012/offer"  # Same origin as Vuer!
)
```

### Step 4: Test

```bash
# From any machine that can reach 10.1.10.187
curl -k https://10.1.10.187:8012/offer

# Should proxy to robot and return WebRTC response
```

---

## Complete Configuration Examples

### Example 1: Basic Setup

**Caddyfile on 10.1.10.187:**
```caddy
10.1.10.187:8012 {
    tls internal
    
    # WebRTC proxy
    handle_path /offer* {
        reverse_proxy http://192.168.123.161:8081
    }
    
    # Vuer static files
    handle /* {
        root * /var/www/vuer
        file_server
    }
}
```

**Python code:**
```python
tv_wrapper = TeleVuerWrapper(
    webrtc=True,
    webrtc_url="https://10.1.10.187:8012/offer"
)
```

### Example 2: Multiple Robots

**Caddyfile on 10.1.10.187:**
```caddy
10.1.10.187:8012 {
    tls internal
    
    # Robot 1
    handle_path /offer/robot1* {
        reverse_proxy http://192.168.123.161:8081
    }
    
    # Robot 2
    handle_path /offer/robot2* {
        reverse_proxy http://192.168.123.162:8081
    }
    
    # Robot 3 (alternative IP)
    handle_path /offer/robot3* {
        reverse_proxy http://10.1.10.48:8081
    }
    
    # Vuer app
    handle /* {
        reverse_proxy localhost:8000
    }
}
```

**Python code:**
```python
# Connect to different robots
robot1 = TeleVuerWrapper(
    webrtc=True,
    webrtc_url="https://10.1.10.187:8012/offer/robot1"
)

robot2 = TeleVuerWrapper(
    webrtc=True,
    webrtc_url="https://10.1.10.187:8012/offer/robot2"
)

robot3 = TeleVuerWrapper(
    webrtc=True,
    webrtc_url="https://10.1.10.187:8012/offer/robot3"
)
```

### Example 3: Robot with HTTPS

If your robot already has HTTPS with a self-signed certificate:

**Caddyfile on 10.1.10.187:**
```caddy
10.1.10.187:8012 {
    tls internal
    
    handle_path /offer* {
        reverse_proxy https://192.168.123.161:8443 {
            header_up Host {upstream_hostport}
            header_up X-Real-IP {remote_host}
            header_up X-Forwarded-For {remote_host}
            header_up X-Forwarded-Proto {scheme}
            
            transport http {
                tls_insecure_skip_verify  # For self-signed certs
            }
        }
    }
    
    handle /* {
        reverse_proxy localhost:8000
    }
}
```

### Example 4: With Load Balancing

If you have multiple WebRTC servers for redundancy:

**Caddyfile on 10.1.10.187:**
```caddy
10.1.10.187:8012 {
    tls internal
    
    handle_path /offer* {
        reverse_proxy {
            to http://192.168.123.161:8081
            to http://192.168.123.162:8081
            to http://10.1.10.48:8081
            
            lb_policy round_robin
            health_uri /health
            health_interval 10s
        }
    }
    
    handle /* {
        reverse_proxy localhost:8000
    }
}
```

---

## Testing

### Test 1: Verify Caddy Configuration

```bash
# On Vuer server (10.1.10.187)
sudo caddy validate --config /etc/caddy/Caddyfile
```

### Test 2: Test Proxy Route

```bash
# From any machine
curl -k https://10.1.10.187:8012/offer

# Should return WebRTC offer from robot
```

### Test 3: Test from Browser Console

Open browser console (F12) on Vuer page:

```javascript
fetch('https://10.1.10.187:8012/offer', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({test: 'data'})
})
.then(response => response.json())
.then(data => console.log('Success:', data))
.catch(error => console.error('Error:', error));
```

Should complete without CORS errors!

### Test 4: Full WebRTC Connection

1. Run your teleoperation code
2. Open browser console (F12)
3. Check for:
   - ✅ No CORS errors
   - ✅ WebRTC connection established
   - ✅ Video stream visible in VR headset

---

## Troubleshooting

### Issue: "Connection refused" to /offer

**Symptoms:**
```
curl: (7) Failed to connect to 10.1.10.187 port 8012: Connection refused
```

**Solutions:**
1. Check Caddy is running:
   ```bash
   sudo systemctl status caddy
   ```

2. Check Caddy is listening on port 8012:
   ```bash
   sudo netstat -tulpn | grep 8012
   ```

3. Check firewall:
   ```bash
   sudo ufw status
   sudo ufw allow 8012/tcp
   ```

### Issue: "502 Bad Gateway"

**Symptoms:**
```
HTTP/2 502
```

**Solutions:**
1. Robot WebRTC server is not running:
   ```bash
   ssh robot@192.168.123.161
   netstat -tulpn | grep 8081
   ```

2. Network connectivity issue:
   ```bash
   # From Vuer server
   ping 192.168.123.161
   curl http://192.168.123.161:8081/offer
   ```

3. Firewall on robot blocking connections:
   ```bash
   # On robot
   sudo ufw allow from 10.1.10.187 to any port 8081
   ```

### Issue: "404 Not Found" on /offer

**Symptoms:**
```
HTTP/2 404
```

**Solutions:**
1. Check Caddyfile syntax:
   ```bash
   sudo caddy validate --config /etc/caddy/Caddyfile
   ```

2. Verify `handle_path` is correct:
   ```caddy
   handle_path /offer* {  # Note the * for wildcard
       reverse_proxy http://192.168.123.161:8081
   }
   ```

3. Check Caddy logs:
   ```bash
   sudo journalctl -u caddy -n 50
   ```

### Issue: WebRTC negotiation fails

**Symptoms:**
- Connection to /offer succeeds
- But WebRTC video doesn't appear

**Solutions:**
1. Check browser console for specific WebRTC errors

2. Verify ICE candidates are being exchanged:
   ```javascript
   // In browser console
   console.log('ICE candidates:', peerConnection.localDescription);
   ```

3. Check if STUN/TURN servers are needed for NAT traversal

4. Verify robot's WebRTC server is responding correctly:
   ```bash
   curl -X POST http://192.168.123.161:8081/offer \
     -H "Content-Type: application/json" \
     -d '{"type":"offer","sdp":"..."}'
   ```

### Issue: Slow video streaming

**Symptoms:**
- WebRTC connects but video is laggy

**Solutions:**
1. Check network latency:
   ```bash
   ping 192.168.123.161
   ```

2. Check bandwidth:
   ```bash
   iperf3 -c 192.168.123.161
   ```

3. Reduce video resolution or frame rate in robot configuration

4. Check CPU usage on robot:
   ```bash
   ssh robot@192.168.123.161
   top
   ```

---

## Advanced Configuration

### Adding Authentication

Protect your /offer endpoint with basic auth:

```caddy
10.1.10.187:8012 {
    tls internal
    
    handle_path /offer* {
        basicauth {
            user $2a$14$hashed_password_here
        }
        reverse_proxy http://192.168.123.161:8081
    }
}
```

### Adding Rate Limiting

Prevent abuse:

```caddy
10.1.10.187:8012 {
    tls internal
    
    handle_path /offer* {
        rate_limit {
            zone webrtc {
                key {remote_host}
                events 10
                window 1m
            }
        }
        reverse_proxy http://192.168.123.161:8081
    }
}
```

### Custom Logging

Log WebRTC requests separately:

```caddy
10.1.10.187:8012 {
    tls internal
    
    handle_path /offer* {
        log {
            output file /var/log/caddy/webrtc-offer.log
            format json
        }
        reverse_proxy http://192.168.123.161:8081
    }
}
```

---

## Migration from CORS Approach

If you're currently using the CORS approach, here's how to migrate:

### Before (CORS on Robot)

**Robot Caddyfile (192.168.123.161):**
```caddy
192.168.123.161:8443 {
    tls internal
    header {
        Access-Control-Allow-Origin *
        # ... more CORS headers
    }
    reverse_proxy localhost:8081
}
```

**Python code:**
```python
webrtc_url="https://192.168.123.161:8443/offer"
```

### After (Same-Origin on Vuer Server)

**Vuer Server Caddyfile (10.1.10.187):**
```caddy
10.1.10.187:8012 {
    tls internal
    handle_path /offer* {
        reverse_proxy http://192.168.123.161:8081
    }
    # ... Vuer routes
}
```

**Python code:**
```python
webrtc_url="https://10.1.10.187:8012/offer"
```

**Robot:** No changes needed! Remove the CORS Caddy config if you want.

---

## Summary

The same-origin solution:
- ✅ Eliminates CORS complexity
- ✅ Improves security (no wildcard origins)
- ✅ Simplifies configuration
- ✅ Works reliably across all browsers
- ✅ Easier to debug

It's the **recommended approach** for production deployments.
