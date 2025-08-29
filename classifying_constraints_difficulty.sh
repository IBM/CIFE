
set -e
source ./env


HEADER=$(echo -n '{"alg":"HS256","typ":"JWT"}' | openssl base64 -A | tr '+/' '-_' | tr -d '=')
PAYLOAD=$(jq -n \
    --arg sub "user123" \
    --arg namespace "granite-build" \
    --arg cluster "vela" \
    --argjson exp $(($(date +%s) + 300)) \
    '{sub: $sub, namespace: $namespace, cluster: $cluster, exp: $exp}' | \
    openssl base64 -A | tr '+/' '-_' | tr -d '=')

SECRET=$JWT_SECRET
SIGNATURE=$(echo -n "$HEADER.$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" -binary | openssl base64 -A | tr '+/' '-_' | tr -d '=')

TOKEN="$HEADER.$PAYLOAD.$SIGNATURE"
echo "Calling proxy to retrieve API key..."

RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" \
    http://gb-api-proxy.granite-build.svc.cluster.local:8000/get-api-key/azure_openai_api)

export AZURE_OPENAI_API_KEY=$(echo "$RESPONSE" | jq -r .api_key)
export AZURE_ENDPOINT=$(echo "$RESPONSE" | jq -r .endpoint_url)

echo "AZURE_ENDPOINT is ${AZURE_ENDPOINT}"

python classifying_constraints_difficulty.py \
  --input_path=benchmark_dataset/benchmark_v10.jsonl \
  --output_filename=benchmark_v10_v2.jsonl
