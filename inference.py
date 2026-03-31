from flask import Flask, jsonify, request
from env import IndianTrafficNavigationEnv

app = Flask(__name__)
env = IndianTrafficNavigationEnv()

@app.route("/reset", methods=["POST"])
def reset():
    obs, info = env.reset()
    return jsonify({
        "observation": obs.tolist(),
        "info": info
    })

@app.route("/step", methods=["POST"])
def step():
    data = request.get_json()
    action = data["action"]
    obs, reward, done, truncated, info = env.step(action)
    return jsonify({
        "observation": obs.tolist(),
        "reward": float(reward),
        "done": bool(done),
        "truncated": bool(truncated),
        "info": info
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
