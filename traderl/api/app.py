from flask import Flask, request, jsonify
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from traderl.agent import agent_dict
from traderl.model.build import NetworkKeyNotFoundError

import logging

app = Flask(__name__)

# Initialize global variables
agent_instance = None
agent_name: str = None
model_name: str = None
args: dict = None

# Set up logging
logging.basicConfig(level=logging.INFO)


@app.route("/get_config", methods=["GET"])
def get_config():
    """
    Response example
    ----------------
    ::

        {
            "Content-Type": "application/json",
            "body": {
                "agent": "DQN",
                "model": "CNN",
                "args": {
                    "learning_rate": 0.001,
                    "gamma": 0.99,
                    "epsilon": 1.0,
                    "batch_size": 64,
                    "memory_size": 10000
                }
            }
        }
    """
    return jsonify({
        "agent": agent_name,
        "model": model_name,
        "args": args
    })


@app.route("/start_training", methods=["POST"])
def start_training():
    """
    Request example
    ----------------
    ::

        {
            "Content-Type": "application/json",
            "body": {
                "agent": "DQN",
                "model": "CNN",
                "args": {
                    "learning_rate": 0.001,
                    "gamma": 0.99,
                    "epsilon": 1.0,
                    "batch_size": 64,
                    "memory_size": 10000
                }
            }
        }
    """
    # Validate incoming request
    if not request.json or "body" not in request.json:
        return jsonify({"error": "Invalid request", "message": "Request body is missing or not a valid JSON"}), 400

    body = request.json["body"]
    if "agent" not in body or "model" not in body or "args" not in body:
        return jsonify({"error": "Invalid request", "message": "Required fields: agent, model, args are missing"}), 400

    # Extract parameters from request and update global variables
    global agent_name, model_name, args, AGENT
    agent_name = body["agent"].upper()
    model_name = body["model"]
    args = body["args"]

    try:
        # Initialize agent with provided parameters
        agent_instance = agent_dict[agent_name](model_name, **args)
    except NetworkKeyNotFoundError:
        return jsonify({
            "error": "NetworkKeyNotFoundError",
            "message": f"Network {model_name} not found in the all_models dictionary"
        }), 400
    except KeyError:
        return jsonify({
            "error": "KeyError",
            "message": f"Agent {agent_name} not found in the agent_dict dictionary"
        }), 400
    except TypeError:
        return jsonify({
            "error": "TypeError",
            "message": f"An unspecified argument was provided"
        }), 400
    except Exception as e:
        return jsonify({
            "error": "UnknownError",
            "message": f"An unknown error occurred: {e}"
        }), 400

    # Start training in a separate thread
    def train():
        try:
            with ThreadPoolExecutor() as executor:
                executor.submit(agent_instance.train)
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")

    Thread(target=train).start()

    return jsonify({"message": "Training started"}), 200


if __name__ == "__main__":
    app.run()
