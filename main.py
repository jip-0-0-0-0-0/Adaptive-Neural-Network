import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import platform
import requests
import json
import os
import ast
import random
import time
import numpy as np
import psutil
import socket
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup

from core.transformer import TransformerModel
from core.memory_module import MemoryNetwork
from core.liquid_network import LiquidNeuralNetwork
from core.pinn import PhysicsInformedNeuralNetwork
from core.nas import NeuralArchitectureSearch
from explainable_ai.lrp import LayerwiseRelevancePropagation
from explainable_ai.shap import SHAPExplainer
from neuro_symbolic.symbolic_logic import SymbolicLogic
from utils.data_loader import create_data_loader
from utils.visualization import plot_loss_curve, plot_feature_importance

def detect_system():
    system_info = {
        "os": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "ip_address": socket.gethostbyname(socket.gethostname()),
        "active_processes": [proc.info for proc in psutil.process_iter(["pid", "name", "username"])],
        "memory_info": psutil.virtual_memory()._asdict(),
        "disk_info": psutil.disk_usage("/")._asdict(),
        "network_info": psutil.net_if_addrs()
    }
    return system_info

def scan_for_vulnerabilities(system_info):
    vulnerabilities = []

    if system_info["os"] == "Windows":
        vulnerabilities.append("Check for outdated SMB protocol versions.")
    elif system_info["os"] == "Linux":
        vulnerabilities.append("Ensure SSH is configured with strong encryption.")

    if "127.0.0.1" in system_info["ip_address"]:
        vulnerabilities.append("Potential misconfiguration in network settings.")

    for process in system_info["active_processes"]:
        if "python" in process["name"]:
            vulnerabilities.append(f"Python process detected: {process['name']} (PID: {process['pid']}).")

    return vulnerabilities

def fetch_external_data(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return {}

def discover_apis(api_directory_url="https://api.publicapis.org/entries"):
    api_list = []
    try:
        response = requests.get(api_directory_url)
        response.raise_for_status()
        entries = response.json().get("entries", [])
        for entry in entries:
            api_list.append(entry.get("Link"))
        print(f"Discovered {len(api_list)} APIs.")
    except Exception as e:
        print(f"Error discovering APIs: {e}")
    return api_list

def scrape_web_for_apis(search_url="https://www.google.com/search?q=free+public+apis"):
    apis = []
    try:
        response = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a")
        for link in links:
            href = link.get("href")
            if "http" in href:
                apis.append(href)
        print(f"Scraped {len(apis)} potential APIs from the web.")
    except Exception as e:
        print(f"Error scraping web for APIs: {e}")
    return apis

def log_self_learning(log_file, data):
    os.makedirs("logs", exist_ok=True)
    with open(log_file, "a") as f:
        f.write(json.dumps(data, indent=4) + "\n")

def decompile_file(file_path):
    try:
        with open(file_path, "r") as f:
            code = f.read()
        tree = ast.parse(code)
        return ast.dump(tree)
    except Exception as e:
        return f"Error decompiling {file_path}: {e}"

def scan_file_system_for_vulnerabilities(root_path):
    vulnerabilities = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if filename.endswith(('.py', '.exe', '.dll', '.so')):
                vulnerabilities.append({
                    "file": full_path,
                    "analysis": decompile_file(full_path) if filename.endswith('.py') else "Binary file. Needs external tools for detailed analysis."
                })
    return vulnerabilities

class DeepQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DeepQLearningAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.q_network = DeepQNetwork(state_dim, action_dim)
        self.target_network = DeepQNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = []
        self.batch_size = 32

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def choose_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

class SelfModifyingSystem:
    def __init__(self):
        self.system_info = detect_system()
        self.vulnerabilities = scan_for_vulnerabilities(self.system_info)
        self.file_vulnerabilities = scan_file_system_for_vulnerabilities(".")
        self.learning_logs = []
        self.symbolic_logic = SymbolicLogic()
        self.trained_model = None
        self.state_dim = 10
        self.action_dim = 3
        self.rl_agent = DeepQLearningAgent(state_dim=self.state_dim, action_dim=self.action_dim)

    def learn_from_environment(self):
        external_data_sources = [
            "https://api.publicapis.org/entries",
            "https://catfact.ninja/fact",
            "https://api.github.com",
            "https://en.wikipedia.org/api/rest_v1/page/random/summary",
            "https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY",
            "https://www.virustotal.com/api/v3/files",
            "https://api.shodan.io/shodan/host/search?key=DEMO_KEY",
            "https://jsonplaceholder.typicode.com/posts",
            "https://jsonplaceholder.typicode.com/users",
            "https://api.agify.io?name=michael",
            "https://api.genderize.io?name=alex",
            "https://api.nationalize.io?name=john",
            "https://datausa.io/api/data?drilldowns=Nation&measures=Population",
            "https://api.coincap.io/v2/assets",
            "https://randomuser.me/api/"
        ]
        for api_url in external_data_sources:
            external_data = fetch_external_data(api_url)
            if external_data:
                print(f"Fetched External Data from {api_url}")
                self.learning_logs.append({"type": "external_data", "source": api_url, "data": external_data})
                log_self_learning("logs/self_learning.json", {"type": "external_data", "source": api_url, "data": external_data})

    def discover_and_learn_apis(self):
        discovered_apis = discover_apis()
        scraped_apis = scrape_web_for_apis()
        all_apis = set(discovered_apis + scraped_apis)
        for api in list(all_apis)[:10]:  # Limit to avoid overload
            data = fetch_external_data(api)
            if data:
                print(f"Learning from discovered API: {api}")
                self.learning_logs.append({"type": "discovered_api", "source": api, "data": data})
                log_self_learning("logs/discovered_apis.json", {"type": "discovered_api", "source": api, "data": data})

    def perform_action(self, state):
        action = self.rl_agent.choose_action(state)
        print(f"Performing action {action}")
        reward = random.uniform(-1, 1)
        next_state = np.random.rand(self.state_dim)
        done = random.choice([True, False])
        self.rl_agent.store_transition(state, action, reward, next_state, done)
        self.rl_agent.train()
        return next_state

    def train_model(self):
        input_dim = 10
        hidden_dim = 128
        output_dim = 1
        num_heads = 8
        num_layers = 6
        batch_size = 16
        epochs = 10

        x_data = torch.rand(500, input_dim)
        y_data = torch.rand(500, output_dim)
        train_loader = create_data_loader(x_data, y_data, batch_size)

        model = TransformerModel(input_dim, hidden_dim, num_heads, num_layers, output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        train_losses = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)
            train_losses.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        plot_loss_curve(train_losses)
        self.trained_model = model
        return model

    def run(self, steps=10):
        state = np.random.rand(self.state_dim)
        for step in range(steps):
            state = self.perform_action(state)
            if step % 5 == 0:
                self.rl_agent.update_target_network()


def main():
    system = SelfModifyingSystem()
    print("System Information:", system.system_info)
    print("Detected Vulnerabilities:", system.vulnerabilities)
    print("File System Vulnerabilities:", system.file_vulnerabilities)

    system.learn_from_environment()
    system.discover_and_learn_apis()
    system.train_model()
    system.run(steps=20)

if __name__ == "__main__":
    main()