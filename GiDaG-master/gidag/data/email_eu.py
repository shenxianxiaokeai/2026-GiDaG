from gidag.data.graph_pt import load_graph_pt
from gidag.types import GraphData


def load_email_eu_pt(dataset_path: str) -> GraphData:
    return load_graph_pt(dataset_path=dataset_path, dataset_name="EmailEU")
