# Fixing CORS Errors for WebRTC

## The Problem

You're seeing these errors in the browser console:

```
Cross-Origin Request Blocked: The Same Origin Policy disallows reading the remote resource at https://192.168.123.164:8443/offer. 
(Reason: CORS header 'Access-Control-Allow-Origin' missing). Status code: 204.

Cross-Origin Request Blocked: The Same Origin Policy disallows reading the remote resource at https://192.168.123.164:8443/offer. 
(Reason: CORS request did not succeed). Status code: (null).

WebRTC negotiation error: TypeError: NetworkError when attempting to fetch resource.
```

This happens because the WebRTC endpoint doesn't include the required **CORS (Cross-Origin Resource Sharing)** headers that allow Vuer (running on a different origin) to access it.

## The Solution

You need to add CORS headers to your reverse proxy configuration.

---

## Quick Fix for Caddy

### Step 1: Update Caddyfile

Replace your `/etc/caddy/Caddyfile` with this:

```caddy
192.168.123.164:8443 {
    # Use internal (self-signed) TLS certificate
    tls internal
    
    # Add CORS headers to all responses
    header {
        Access-Control-Allow-Origin *
        Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
        Access-Control-Allow-Headers "Content-Type, Authorization, X-Requested-With"
        Access-Control-Allow-Credentials true
        Access-Control-Max-Age 3600
    }
    
    # Handle OPTIONS preflight requests
    @options {
        method OPTIONS
    }
    handle @options {
        respond 204
    }
    
    # Reverse proxy to local WebRTC server
    reverse_proxy localhost:8081 {
        header_up Host {host}
        header_up X-Real-IP {remote}
        header_up X-Forwarded-For {remote}
        header_up X-Forwarded-Proto {scheme}
    }
    
    # Logging
    log {
        output file /var/log/caddy/webrtc-proxy.log
        format console
    }
}
```

### Step 2: Restart Caddy

```bash
sudo systemctl restart caddy
sudo systemctl status caddy
```

### Step 3: Test

```bash
# Test CORS headers are present
curl -I -k https://192.168.123.164:8443/offer
```

You should see headers like:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
```

---

## Quick Fix for Nginx

### Step 1: Update Nginx Configuration

Edit `/etc/nginx/sites-available/webrtc-proxy`:

```nginx
server {
    listen 8443 ssl http2;
    server_name 192.168.123.164;

    ssl_certificate /etc/nginx/ssl/robot.crt;
    ssl_certificate_key /etc/nginx/ssl/robot.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    access_log /var/log/nginx/webrtc-proxy-access.log;
    error_log /var/log/nginx/webrtc-proxy-error.log;

    location / {
        # Handle OPTIONS preflight requests
        if ($request_method = 'OPTIONS') {
            add_header 'Access-Control-Allow-Origin' '*' always;
            add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
            add_header 'Access-Control-Allow-Headers' 'Content-Type, Authorization, X-Requested-With' always;
            add_header 'Access-Control-Allow-Credentials' 'true' always;
            add_header 'Access-Control-Max-Age' 3600 always;
            add_header 'Content-Type' 'text/plain charset=UTF-8' always;
            add_header 'Content-Length' 0 always;
            return 204;
        }
        
        proxy_pass http://localhost:8081;
        proxy_http_version 1.1;
        
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        proxy_buffering off;
        proxy_request_buffering off;
        
        # CORS headers for all responses
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'Content-Type, Authorization, X-Requested-With' always;
        add_header 'Access-Control-Allow-Credentials' 'true' always;
    }
}
```

### Step 2: Test and Restart Nginx

```bash
# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
sudo systemctl status nginx
```

### Step 3: Test

```bash
curl -I -k https://192.168.123.164:8443/offer
```

---

## Understanding CORS

### What is CORS?

CORS (Cross-Origin Resource Sharing) is a security mechanism that allows a web page from one origin (domain) to access resources from a different origin.

### Why is it needed?

When Vuer (running on one domain/port) tries to connect to your WebRTC endpoint (on a different domain/port), the browser blocks the request unless the server explicitly allows it via CORS headers.

### Key CORS Headers

1. **Access-Control-Allow-Origin**: Specifies which origins can access the resource
   - `*` = allow all origins (fine for development)
   - For production, specify exact origin: `https://your-vuer-domain.com`

2. **Access-Control-Allow-Methods**: Which HTTP methods are allowed
   - WebRTC typically needs: `GET, POST, OPTIONS`

3. **Access-Control-Allow-Headers**: Which headers the client can send
   - Common: `Content-Type, Authorization`

4. **Access-Control-Allow-Credentials**: Whether cookies/auth can be sent
   - Set to `true` if you need authentication

5. **Access-Control-Max-Age**: How long browsers can cache the preflight response
   - `3600` = 1 hour

### OPTIONS Preflight Request

Before making the actual request, browsers send an OPTIONS request to check if CORS is allowed. Your server must:
1. Respond to OPTIONS with status 204 (No Content)
2. Include all CORS headers in the response

---

## Testing CORS

### Test 1: Check Headers

```bash
curl -I -k https://192.168.123.164:8443/offer
```

Expected output should include:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization, X-Requested-With
```

### Test 2: Test OPTIONS Request

```bash
curl -X OPTIONS -k https://192.168.123.164:8443/offer -v
```

Should return status 204 with CORS headers.

### Test 3: Test from Browser Console

Open browser console (F12) and run:

```javascript
fetch('https://192.168.123.164:8443/offer', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({test: 'data'})
})
.then(response => console.log('Success:', response))
.catch(error => console.error('Error:', error));
```

Should not show CORS errors.

---

## Troubleshooting

### Still Getting CORS Errors?

1. **Clear browser cache**: Hard refresh (Ctrl+Shift+R or Cmd+Shift+R)

2. **Check proxy is running**:
   ```bash
   # Caddy
   sudo systemctl status caddy
   
   # Nginx
   sudo systemctl status nginx
   ```

3. **Check logs for errors**:
   ```bash
   # Caddy
   sudo journalctl -u caddy -n 50
   
   # Nginx
   sudo tail -50 /var/log/nginx/webrtc-proxy-error.log
   ```

4. **Verify CORS headers are being sent**:
   ```bash
   curl -I -k https://192.168.123.164:8443/offer | grep -i access-control
   ```

5. **Check if WebRTC server is responding**:
   ```bash
   curl http://localhost:8081/offer
   ```

### Different Origin in Production?

If you want to restrict CORS to specific origins instead of `*`, replace:

```
Access-Control-Allow-Origin *
```

With:

```
Access-Control-Allow-Origin https://your-vuer-domain.com
```

---

## Summary

The CORS error occurs because browsers block cross-origin requests by default. By adding CORS headers to your reverse proxy configuration, you tell the browser it's safe to allow Vuer to connect to your WebRTC endpoint.

After updating your configuration and restarting the proxy, the CORS errors should disappear and WebRTC should connect successfully.
