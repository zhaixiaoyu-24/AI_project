from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

print("正在下载/加载 BGE-M3 模型(国内Modelscope镜像)...")
# model_dir = snapshot_download("BAAI/bge-m3")
# embed_model = SentenceTransformer(model_dir)
# 改成直接读取本地已经下载好的模型
model_path = "/home/administrator/rag_project/bge-m3"

embed_model = SentenceTransformer(model_path, device="cpu")


vec = embed_model.encode("WSL环境测试文本", batch_size=1)
print(f"✅ BGE-M3 加载成功，向量维度：{vec.shape[0]}")

client = MilvusClient("test_milvus.db")
print("✅ Milvus Lite 初始化成功")
client.close()
print("\n🎉 全部环境就绪！")
