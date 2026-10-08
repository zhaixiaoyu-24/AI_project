from sentence_transformers import SentenceTransformer

model_path = "/home/administrator/rag_project/bge-m3"
model = SentenceTransformer(
    model_path,
    device="cpu",
    model_kwargs={
        "low_cpu_mem_usage": True,
        "use_safetensors": False,  # 改成False，读取pytorch_model.bin
    },
)
print("✅模型加载完成")
emb = model.encode("测试文本", batch_size=1)
print(f"向量正常，维度：{emb.shape}")
