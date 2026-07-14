"""
TopicGraph: Concept dependency graph with root-cause tracing.
Predicts cross-topic transfer strength based on prerequisite mastery.
"""

from dataclasses import dataclass, field


# Default prerequisite graph for common CS/academic topics
DEFAULT_GRAPH = {
    "Red Black Tree":       ["BST", "Recursion"],
    "BST":                  ["Recursion", "Pointers"],
    "Graph Algorithms":     ["Recursion", "Trees", "Arrays"],
    "Dynamic Programming":  ["Recursion", "Arrays"],
    "Trees":                ["Recursion", "Pointers"],
    "Recursion":            ["Functions", "Stack"],
    "Pointers":             ["Memory Management"],
    "Operating Systems":    ["Processes", "Memory Management", "CPU Scheduling"],
    "CPU Scheduling":       ["Processes", "Queues"],
    "Deadlock":             ["Processes", "Semaphores"],
    "Semaphores":           ["Processes", "Synchronization"],
    "Neural Networks":      ["Linear Algebra", "Calculus", "Probability"],
    "Backpropagation":      ["Neural Networks", "Calculus", "Chain Rule"],
    "Transformers":         ["Attention Mechanism", "Neural Networks", "Linear Algebra"],
}

# Transfer prediction: strong in X → likely strong in Y
TRANSFER_MAP = {
    "Recursion":        ["Trees", "Dynamic Programming", "Graph Algorithms"],
    "Linear Algebra":   ["Neural Networks", "Transformers", "PCA"],
    "Calculus":         ["Backpropagation", "Optimization"],
    "Probability":      ["Bayesian Networks", "Neural Networks"],
    "Processes":        ["Deadlock", "CPU Scheduling", "Semaphores"],
}


@dataclass
class TopicGraph:
    graph: dict = field(default_factory=lambda: dict(DEFAULT_GRAPH))

    def add_dependency(self, topic: str, prerequisite: str, weight: float = 1.0):
        if topic not in self.graph:
            self.graph[topic] = []
        if prerequisite not in self.graph[topic]:
            self.graph[topic].append(prerequisite)

    def get_prerequisites(self, topic: str) -> list[str]:
        return self.graph.get(topic, [])

    def trace_root_cause(self, failed_topic: str, mastery_map: dict) -> dict:
        """
        Trace the weakest prerequisite for a failed topic.
        mastery_map: {topic_id: mastery_score}
        """
        prereqs = self.get_prerequisites(failed_topic)
        if not prereqs:
            return {"failed_topic": failed_topic, "root_cause": failed_topic,
                    "confidence": 0.5, "path": []}

        # Find weakest prerequisite recursively
        weakest = min(prereqs, key=lambda p: mastery_map.get(p, 0.5))
        weakest_score = mastery_map.get(weakest, 0.5)

        # Recurse into weakest
        sub_result = self.trace_root_cause(weakest, mastery_map)
        if sub_result["root_cause"] != weakest and mastery_map.get(sub_result["root_cause"], 0.5) < weakest_score:
            root = sub_result["root_cause"]
        else:
            root = weakest

        confidence = round(1.0 - mastery_map.get(root, 0.5), 3)
        path = [failed_topic] + [p for p in prereqs] + [root]

        return {
            "failed_topic": failed_topic,
            "root_cause": root,
            "confidence": confidence,
            "path": list(dict.fromkeys(path)),  # deduplicate preserving order
        }

    def predict_transfer(self, strong_topics: list[str]) -> list[str]:
        """Predict topics the student is likely strong in based on transfer."""
        predicted = set()
        for topic in strong_topics:
            for transfer_target in TRANSFER_MAP.get(topic, []):
                predicted.add(transfer_target)
        return list(predicted)

    def get_all_nodes_edges(self) -> tuple[list[str], list[tuple]]:
        nodes = set()
        edges = []
        for topic, prereqs in self.graph.items():
            nodes.add(topic)
            for p in prereqs:
                nodes.add(p)
                edges.append((p, topic))
        return list(nodes), edges
