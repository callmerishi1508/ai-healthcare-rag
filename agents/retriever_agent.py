import faiss
import json
import numpy as np
from openai import OpenAI

client = OpenAI()

class RetrieverAgent:
    def decompose_query(self, query):
        # For now, skip OpenAI and just use the query as is
        return [query]