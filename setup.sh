mkdir -p ~/.streamlit/

echo "
[server]\n
port = $PORT\n
enableCORS = false\n
headless = true\n
\n
" > ~/.streamlit/config.toml

mkdir ~/credentials/
echo ${GOOGLE_CREDENTIALS} > ~/credentials/gcp_credentials.json
