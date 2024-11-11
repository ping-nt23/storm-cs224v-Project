import sys
import json
import networkx as nx
from together import Together
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import Field, BaseModel
import os

class WeightOutput(BaseModel):
    score: float = Field(description="A score from 0 to 1, 0 meaning the topics are not correlated at all and 1 being the topics are tightly correlated")

class MindmapGraph:
    def __init__(self, method="tfidf"):
        # Initialize Together API
        self.together = Together()

        # Initialize two empty graphs
        self.G = nx.Graph()  # Complete graph with full passages
        self.G_summaries = nx.Graph()  # Graph for LLM with summaries

        # Initialize an empty list for topics and an empty TfidfVectorizer
        self.topics_list = []
        self.vectorizer = TfidfVectorizer()
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Load sentence transformer for embeddings

        # Cache for tf-idf vectors of topics and embeddings
        self.tfidf_cache = {}
        self.embedding_cache = {}

        # Method for weight computation
        self.method = method

        # TODO: Pann Remove , use for debug graphs
        self.count = 0

    def refit_vectorizer(self):
        if self.topics_list:
            self.vectorizer.fit(self.topics_list)
            self.tfidf_cache = {topic: self.vectorizer.transform([topic]) for topic in self.topics_list}

    def update_embedding_cache(self):
        if self.topics_list:
            self.embedding_cache = {topic: self.embedding_model.encode([topic])[0] for topic in self.topics_list}

    def add_to_graph(self, graph, topic_summary, connections, full_passage=None):
        if topic_summary not in self.tfidf_cache:
            graph.add_node(topic_summary, passages=[])
            self.topics_list.append(topic_summary)
            self.refit_vectorizer()
            self.update_embedding_cache()

        for conn in connections:
            graph.add_edge(topic_summary, conn['topic'], weight=conn['weight'])

        if full_passage:
            graph.nodes[topic_summary]['passages'].append(full_passage)

    def summarize_passage(self, passage):
        summary_response = self.together.chat.completions.create(
            messages=[
                {"role": "system", "content": "Summarize the passage into a key topic or concept."},
                {"role": "user", "content": passage},
            ],
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        )
        return summary_response.choices[0].message.content.strip() if summary_response.choices else None

    def compute_weight(self, passage_summary, topic):
        if self.method == "tfidf":
            passage_vector = self.vectorizer.transform([passage_summary])
            weight = float(cosine_similarity(passage_vector, self.tfidf_cache[topic])[0][0])

        elif self.method == "embedding":
            passage_embedding = self.embedding_model.encode([passage_summary])[0]
            topic_embedding = self.embedding_cache[topic]
            weight = float(np.dot(passage_embedding, topic_embedding) / (np.linalg.norm(passage_embedding) * np.linalg.norm(topic_embedding)))

        elif self.method == "llm":
            score_response = self.together.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Evaluate how related the two topics are on a scale of 0 to 1."},
                    {"role": "user", "content": f"Topic 1: {passage_summary}\nTopic 2: {topic}"},
                ],
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                response_format={
                    "type": "json_object",
                    "schema": WeightOutput.model_json_schema(),
                },
            )
            if score_response.choices:
                response_data = json.loads(score_response.choices[0].message.content.strip())
                weight = float(response_data["score"])
            else:
                weight = -1
        else:
            weight = -1
        return weight

    def relate_passage_to_topics(self, passage_summary, threshold=0.3):
        if not self.topics_list:
            return []
        related_topics = []
        for topic in self.topics_list:
            weight = self.compute_weight(passage_summary, topic)
            if weight >= threshold:
                related_topics.append({"topic": topic, "weight": weight})
        return related_topics

    def process_passage(self, passage, topic):
        print("COUNT IS", self.count)
        topic_summary = self.summarize_passage(passage)
        if topic_summary:
            connections = self.relate_passage_to_topics(topic_summary)
            self.add_to_graph(self.G, topic_summary, connections, full_passage=passage)
            self.add_to_graph(self.G_summaries, topic_summary, connections)
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=True, node_color="lightblue", edge_color="gray", font_size=10)
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_color="red")
        plt.savefig(f"graph-{self.count}.png")
        self.count += 1
        plt.close()
        graph_data = nx.readwrite.json_graph.node_link_data(self.G_summaries, edges="edges")
        # Ensure the folder exists, create it if it doesn't
        os.makedirs('llm', exist_ok=True)
        
        with open(f'llm/graph_data_{topic}.json', 'w') as f:
            json.dump(graph_data, f)
        graph_mindmap = json.dumps(graph_data, indent=2)
        return graph_mindmap


    # PANN: TODO REMOVE
    def generate_article(self):
        graph_data = nx.readwrite.json_graph.node_link_data(self.G, edges="edges")
        graph_json = json.dumps(graph_data)
        article_response = self.together.chat.completions.create(
            messages=[
                {"role": "system", "content": "Use the provided mindmap JSON to generate a coherent article."},
                {"role": "user", "content": graph_json},
            ],
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        )
        return article_response.choices[0].message.content.strip() if article_response.choices else "Error generating article."