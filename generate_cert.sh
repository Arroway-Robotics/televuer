#!/bin/bash
# Script to generate a self-signed SSL certificate with Subject Alternative Names (SAN)
# This certificate will work for localhost and LAN IP addresses

CERT_DIR="/home/unitree/g1-teleoperate/teleop/televuer"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"

# Get the LAN IP addresses
LAN_IP1=$(hostname -I | awk '{print $1}')
# You can add additional IP addresses here if needed
ADDITIONAL_IP="10.1.10.187"

echo "Generating SSL certificate with SAN for:"
echo "  - localhost"
echo "  - 127.0.0.1"
echo "  - $LAN_IP1"
if [ -n "$ADDITIONAL_IP" ]; then
    echo "  - $ADDITIONAL_IP"
fi
echo ""

# Create a temporary config file for OpenSSL
CONFIG_FILE=$(mktemp)
cat > "$CONFIG_FILE" <<EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = v3_req

[dn]
C=US
ST=CA
L=Palo Alto
O=Unitree
CN=localhost

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.local
IP.1 = 127.0.0.1
IP.2 = ::1
IP.3 = $LAN_IP1
EOF

# Add additional IP if specified
if [ -n "$ADDITIONAL_IP" ]; then
    echo "IP.4 = $ADDITIONAL_IP" >> "$CONFIG_FILE"
fi

# Generate the certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "$KEY_FILE" \
    -out "$CERT_FILE" \
    -config "$CONFIG_FILE" \
    -extensions v3_req

# Set proper permissions
chmod 600 "$KEY_FILE"
chmod 644 "$CERT_FILE"

# Clean up
rm "$CONFIG_FILE"

echo ""
echo "Certificate generated successfully!"
echo "  Certificate: $CERT_FILE"
echo "  Private Key: $KEY_FILE"
echo ""
echo "To verify the certificate:"
echo "  openssl x509 -in $CERT_FILE -text -noout | grep -A 5 'Subject Alternative Name'"
echo ""
echo "Note: This is a self-signed certificate. Browsers will still show a warning,"
echo "      but you can accept it once and it will work for your local network."

