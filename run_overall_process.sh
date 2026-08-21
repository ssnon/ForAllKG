set -a
source .env
set +a

CORPUS=dac_her_expanded_v2

python -m scripts.discovery.run_dac_discovery_e2e \
  --corpus-id "$CORPUS" \
  --domain-profile "$DOMAIN" \
  --run-dir "runs/e2e/v290_integrated_her_002" \
  --source "metal-pair identity and nitrogen coordination" \
  --stop "charge redistribution and metal-metal geometry" \
  --target "hydrogen evolution activity" \
  --question "What combination of metal-pair identity, nitrogen coordination, charge redistribution, and metal-metal geometry places a dual-atom catalyst in the optimal regime for hydrogen adsorption and HER activity?" \
  --objective "explain_connection" \
  --grounding-policy "semantic_stop_fallback_top_n" \
  --model "$OPENROUTER_AGENT_MODEL" \
  --critic-model "$OPENROUTER_CRITIC_MODEL" \
  --base-url "$BASE_URL" \
  --api-key-env "OPENROUTER_API_KEY"
  --node-map-k 20 \
  --waypoint-k 12 \
  --endpoint-pair-k 12 \
  --max-depth 12 \
  --top-k 8 \
  --discovery-top-k 8 \
  --max-axes 5

python -m scripts.discovery.run_dac_discovery_e2e \
  --corpus-id "$CORPUS" \
  --domain-profile "$DOMAIN" \
  --run-dir "runs/e2e/v290_charge_transfer_her_002" \
  --source "metal-pair identity" \
  --stop "charge transfer" \
  --target "hydrogen evolution activity" \
  --question "How does metal-pair identity control the magnitude and direction of charge redistribution in dual-atom catalysts, and when does stronger charge transfer lead to more favorable hydrogen adsorption and HER activity?" \
  --objective "explain_connection" \
  --grounding-policy "semantic_stop_fallback_top_n" \
  --model "$OPENROUTER_AGENT_MODEL" \
  --critic-model "$OPENROUTER_CRITIC_MODEL" \
  --base-url "$BASE_URL" \
  --api-key-env "OPENROUTER_API_KEY"
  --node-map-k 20 \
  --waypoint-k 12 \
  --endpoint-pair-k 12 \
  --max-depth 12 \
  --top-k 8 \
  --discovery-top-k 8 \
  --max-axes 5

python -m scripts.discovery.run_dac_discovery_e2e \
  --corpus-id "$CORPUS" \
  --domain-profile "$DOMAIN" \
  --run-dir "runs/e2e/v290_ndoped_mm_distance_002" \
  --source "metal-pair identity and nitrogen coordination geometry" \
  --stop "metal-metal distance" \
  --target "dual-atom structural stability" \
  --question "How do metal-pair identity and nitrogen coordination geometry determine the optimized metal-metal distance and structural stability of dual-atom sites on N-doped graphene, and what does this imply for choosing initial metal-metal separations in DFT geometry optimization?" \
  --objective "explain_connection" \
  --grounding-policy "semantic_stop_fallback_top_n" \
  --model "$OPENROUTER_AGENT_MODEL" \
  --critic-model "$OPENROUTER_CRITIC_MODEL" \
  --base-url "$BASE_URL" \
  --api-key-env "OPENROUTER_API_KEY"
  --node-map-k 20 \
  --waypoint-k 12 \
  --endpoint-pair-k 12 \
  --max-depth 12 \
  --top-k 8 \
  --discovery-top-k 8 \
  --max-axes 5

python -m scripts.discovery.run_dac_discovery_e2e \
  --corpus-id "$CORPUS" \
  --domain-profile "$DOMAIN" \
  --run-dir "runs/e2e/v290_low_overpotential_002" \
  --source "dual-atom catalyst identity and coordination environment" \
  --stop "HER overpotential" \
  --target "hydrogen evolution activity" \
  --question "Which dual-atom catalysts exhibit the lowest reported HER overpotentials under comparable measurement conditions, and what combination of metal-pair identity, coordination environment, and hydrogen adsorption energetics distinguishes the best-performing catalysts?" \
  --objective "explain_connection" \
  --grounding-policy "semantic_stop_fallback_top_n" \
  --model "$OPENROUTER_AGENT_MODEL" \
  --critic-model "$OPENROUTER_CRITIC_MODEL" \
  --base-url "$BASE_URL" \
  --api-key-env "OPENROUTER_API_KEY"
  --node-map-k 20 \
  --waypoint-k 12 \
  --endpoint-pair-k 12 \
  --max-depth 12 \
  --top-k 8 \
  --discovery-top-k 8 \
  --max-axes 5