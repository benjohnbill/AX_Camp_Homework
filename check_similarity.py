# check_similarity.py (임시 확인용)
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text1 = "인간은 서로를 끊임없이 원하면서도, 밀어내는 것 같다."
text2 = "밥을 먹다가 문득 주변 친구들이 취업하는 소식이 떠올라서..." # (사용자의 긴 글)

def get_vec(text):
    return client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding

vec1 = np.array(get_vec(text1)).reshape(1, -1)
vec2 = np.array(get_vec(text2)).reshape(1, -1)

score = cosine_similarity(vec1, vec2)[0][0]
print(f"------------\n두 글의 유사도 점수: {score:.4f}\n------------")