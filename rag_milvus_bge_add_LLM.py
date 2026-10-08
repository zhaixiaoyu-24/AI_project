import os

# 清空代理
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "*"

from dotenv import load_dotenv

load_dotenv()

import httpx
from openai import OpenAI
from pymilvus import DataType, MilvusClient
from sentence_transformers import SentenceTransformer

# ====================== 配置区 ======================
COLLECTION_NAME = "rag_bge_m3_demo"
DIM = 1024
TOP_K = 2
MILVUS_DB_FILE = "milvus_rag.db"
# Kimi 配置
API_KEY = os.getenv("API_KEY")  # 从环境变量读取，不要直接写key到代码
BASE_URL = os.getenv("BASE_URL")
AI_MODEL = os.getenv("AI_MODEL")

# 加载BGE-M3，适配pytorch_model.bin
model_path = "/home/administrator/rag_project/bge-m3"
embed_model = SentenceTransformer(
    model_path,
    device="cpu",
    model_kwargs={"low_cpu_mem_usage": True, "use_safetensors": False},
)

# Kimi客户端初始化

client_kimi = OpenAI(
    api_key=API_KEY, base_url=BASE_URL, http_client=httpx.Client(timeout=30.0)
)


# 【替换为真实Kimi API调用】
def llm_answer(prompt: str):
    resp = client_kimi.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一个专业助手，严格根据参考资料回答问题，不能编造信息。",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


# 初始化Milvus Lite
client = MilvusClient(MILVUS_DB_FILE)

# 测试用：每次运行清空旧集合，生产环境务必删除这一段
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)

# ========== 正确构建schema ==========
schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIM)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2000)

# 创建集合
client.create_collection(
    collection_name=COLLECTION_NAME, schema=schema, metric_type="IP"
)

# 知识库文档
raw_docs = [
    "RAG全称检索增强生成，用来缓解大模型幻觉问题，不会修改大模型权重。",
    "RAG分为文档加载、文本切片、向量化、向量检索、LLM生成这五个核心步骤。",
    "向量是文本转换成的一串浮点数，语义相近的文本，向量在高维空间距离更近。",
    "模型微调是更新大模型权重，训练成本高；RAG不用训练模型，只检索外部文档。",
    "BGE-M3是智源开源多语言嵌入模型，支持稠密向量、稀疏向量，中文RAG表现优秀。",
    "Milvus是开源向量数据库，专门用来存储海量向量，支持HNSW索引，毫秒级语义检索。",
]

# BGE-M3批量向量化，开启归一化适配IP点积
embeddings = embed_model.encode(raw_docs, normalize_embeddings=True)
# 插入数据，带上id主键
insert_data = []
for i in range(len(raw_docs)):
    insert_data.append({"id": i, "vector": embeddings[i].tolist(), "text": raw_docs[i]})

res = client.insert(collection_name=COLLECTION_NAME, data=insert_data)
print(f"Milvus入库成功，写入数量：{res['insert_count']}")


# 检索函数
def milvus_retrieve(query: str, top_k=TOP_K):
    q_emb = embed_model.encode(query, normalize_embeddings=True).tolist()
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=[q_emb],
        limit=top_k,
        output_fields=["text"],
        search_params={"metric_type": "IP", "params": {"ef": 10}},
    )
    retrieved_texts = []
    for hit in search_res[0]:
        retrieved_texts.append(hit["entity"]["text"])
    return retrieved_texts


# RAG完整流水线
def rag_pipeline(user_question: str):
    context_list = milvus_retrieve(user_question)
    context_block = "\n".join(context_list)
    prompt = f"""
# 参考资料
{context_block}
要求：
1. 严格基于上面【参考资料】回答用户问题；
2. 如果资料没有对应信息，直接回答「参考资料中无相关内容」，禁止编造；
3. 回答简洁准确。
用户提问：{user_question}
"""
    print("\n===== 送入Kimi完整Prompt =====")
    print(prompt)
    return llm_answer(prompt)


if __name__ == "__main__":
    question = "Milvus是什么，BGE-M3在RAG里作用是什么？"
    answer = rag_pipeline(question)
    print("\n===== Kimi最终回答 =====", answer)
