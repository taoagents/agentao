#!/usr/bin/env bash
set -euxo pipefail

sleep 5  # Allows this to be run at the same time as subtensor/scripts/localnet.sh

chain_address="ws://127.0.0.1:9944"
wallet_path=~/.bittensor/wallets

echo "y" | btcli config set --subtensor.network "$chain_address" --wallet.path "$wallet_path"

# Wallets created (coldkey, hotkey):
# - (owner, default)
# - (miner, default)
# - (validator, default)
wallet_names=("owner" "miner" "miner2" "validator")

function create-wallet() {
  local wallet_name="$1"

  btcli wallet create \
    --n-words 12 \
    --no-use-password \
    --wallet-name "$wallet_name" \
    --hotkey default \
    --wallet-path "$wallet_path"
}

# Create wallets if they don't exist
for wallet_name in "${wallet_names[@]}"; do
  if ! [ -e "$wallet_path"/"$wallet_name"/hotkeys/default ]; then
    echo "Wallet $wallet_name and hotkey 'default' not found, creating it..."
    create-wallet "$wallet_name"
  else
    echo "Wallet $wallet_name and hotkey 'default found."
  fi
done


# Create subnet and owner
btcli wallet faucet --wallet.name owner --max-successes 6 --no_prompt
btcli subnet create --wallet.name owner --no_prompt

# Initialize miner
miner_address="$(jq -r ".ss58Address" ~/.bittensor/wallets/miner/coldkeypub.txt)"
btcli wallet transfer --wallet.name owner --dest $miner_address --amount 1000 --no_prompt
btcli subnet register --wallet.name miner --wallet.hotkey default --netuid 1 --no_prompt

# Initialize miner2
miner2_address="$(jq -r ".ss58Address" ~/.bittensor/wallets/miner2/coldkeypub.txt)"
btcli wallet transfer --wallet.name owner --dest $miner2_address --amount 1000 --no_prompt
btcli subnet register --wallet.name miner2 --wallet.hotkey default --netuid 1 --no_prompt

# Initialize validator
validator_address="$(jq -r ".ss58Address" ~/.bittensor/wallets/validator/coldkeypub.txt)"
btcli wallet transfer --wallet.name owner --dest "$validator_address" --amount 1000 --no_prompt
btcli subnet register --wallet.name validator --netuid 1 --wallet.hotkey default --no_prompt
btcli stake add --wallet.name validator --wallet.hotkey default --amount 300 --no-prompt

# Verify initialization was successful
btcli subnet list  # Should have N=2 for NETUID=1 row
btcli wallet overview --wallet.name validator  # verify stake
btcli wallet overview --wallet.name miner
