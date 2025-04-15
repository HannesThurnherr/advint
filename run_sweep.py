import wandb
import argparse
import subprocess
import yaml # Import the YAML library

# --- Configuration ---
SWEEP_CONFIG_FILE = "sweep.yaml"
WANDB_PROJECT = "sae_adv_lambda_sweep" # Must match the project in sweep.yaml
WANDB_ENTITY = None # Optional: Your wandb username or team name, must match sweep.yaml if set there
NUM_AGENTS = 4 # Number of parallel agents to run

def main():
    parser = argparse.ArgumentParser(description="Run wandb sweep agents.")
    parser.add_argument('--sweep_id', type=str, default=None, help='Provide an existing sweep ID to join.')
    parser.add_argument('--num_agents', type=int, default=NUM_AGENTS, help='Number of agents to run in parallel.')
    args = parser.parse_args()

    sweep_id = args.sweep_id
    num_agents = args.num_agents

    # 1. Create the sweep if no ID is provided
    if not sweep_id:
        print(f"Loading sweep config from {SWEEP_CONFIG_FILE}...")
        try:
            with open(SWEEP_CONFIG_FILE, 'r') as f:
                sweep_config = yaml.safe_load(f) # Load YAML into a dictionary
            print("Sweep config loaded successfully.")
        except Exception as e:
            print(f"Error loading sweep YAML file '{SWEEP_CONFIG_FILE}': {e}")
            return # Exit if YAML loading fails

        print(f"Creating sweep from loaded config...")
        # Pass the loaded dictionary instead of the filename
        sweep_id = wandb.sweep(sweep=sweep_config, project=WANDB_PROJECT, entity=WANDB_ENTITY)
        print(f"Sweep created with ID: {sweep_id}")
    else:
        print(f"Joining existing sweep with ID: {sweep_id}")

    # 2. Run the agents
    print(f"Starting {num_agents} agents for sweep {sweep_id}...")

    # Construct the command for a single agent
    agent_command = ["wandb", "agent"]
    if WANDB_ENTITY:
        agent_command.append(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}")
    else:
         agent_command.append(f"{WANDB_PROJECT}/{sweep_id}") # Assumes default entity

    # Run multiple agents in parallel (simple approach using subprocess)
    processes = []
    for i in range(num_agents):
        print(f"Starting agent {i+1}/{num_agents}...")
        # Run each agent in its own process
        # Ensure the environment (CUDA_VISIBLE_DEVICES, etc.) is handled correctly
        # if running multiple agents on the same multi-GPU machine.
        # You might need more sophisticated process/GPU management here.
        proc = subprocess.Popen(agent_command)
        processes.append(proc)
        print(f"Agent {i+1} started with PID {proc.pid}.")

    # Wait for all agent processes to complete (optional)
    for i, proc in enumerate(processes):
        proc.wait()
        print(f"Agent {i+1} (PID {proc.pid}) finished with return code {proc.returncode}.")

    print("All agents finished or script interrupted.")


if __name__ == "__main__":
    main() 