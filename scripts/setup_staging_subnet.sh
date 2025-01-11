#!/usr/bin/env bash
set -euxo pipefail

echo "y" | btcli config set --subtensor.network ws://127.0.0.1:9946 --wallet.path ~/.bittensor/wallets

# Create subnet and owner
btcli wallet faucet --wallet.name owner --max-successes 5 --no_prompt
btcli subnet create --wallet.name owner --no_prompt

# Initialize miner
miner_address="$(jq -r ".ss58Address" ~/.bittensor/wallets/miner/coldkeypub.txt)"
btcli wallet transfer --wallet.name owner --dest $miner_address --amount 1000 --no_prompt
btcli subnet register --wallet.name miner --wallet.hotkey default --netuid 1 --no_prompt

# Initialize validator
validator_address="$(jq -r ".ss58Address" ~/.bittensor/wallets/validator/coldkeypub.txt)"
btcli wallet transfer --wallet.name owner --dest $validator_address --amount 1000 --no_prompt
btcli subnet register --wallet.name validator --netuid 1 --wallet.hotkey default --no_prompt
btcli stake add --wallet.name validator --wallet.hotkey default --amount 300 --no-prompt

# Verify initialization was successful
btcli subnet list  # Should have N=2 for NETUID=1 row
btcli wallet overview --wallet.name validator  # verify stake
btcli wallet overview --wallet.name miner
