# RAG 机制原理与代码使用指南

> 本文配套代码：`rag_milvus_bge_add_LLM.py`（BGE-M3 + Milvus Lite + Kimi 的最小 RAG 流水线）
>
> 技术栈：SentenceTransformer(BGE-M3) · Milvus Lite · OpenAI SDK(兼容 Kimi API)

---

## 目录

1. [RAG 是什么：开卷考试 vs 闭卷考试](#一rag-是什么)
2. [RAG 解决的三个痛点](#二rag-解决的三个痛点)
3. [原理与机制：五幕剧](#三原理与机制脚本的五幕剧)
4. [为什么这套机制有效（以及能力边界）](#四为什么这套机制有效以及能力边界)
5. [怎么跑起来（使用前准备）](#五怎么跑起来使用前准备)
6. [代码逐段讲解](#六代码逐段讲解七个部分)
7. [常见改法：如何换成自己的用法](#七常见改法如何换成自己的用法)
8. [生产环境注意事项](#八生产环境注意事项)
9. [一张图记住数据流](#九一张图记住数据流)

---

## 一、RAG 是什么

RAG（Retrieval-Augmented Generation，检索增强生成），可以理解为：

- **纯大模型（裸调 Kimi/GPT）= 闭卷考试**——只能凭训练时背下来的知识答题，背错了就硬编（即「幻觉」）；
- **RAG = 开卷考试**——先翻书查到相关资料，再照着资料作答。

RAG 的思路是：**模型记不住的知识，不要硬塞进它脑子里（微调），而是放在它手边的书架上，用时现查。**

---

## 二、RAG 解决的三个痛点

1. **幻觉**：模型训练没见过的内容，会一本正经地编造。代码中 system prompt 那句「严格根据参考资料回答问题，不能编造信息」就是在压制这个本能。
2. **知识截止**：模型训练完那一刻，世界就冻结了，之后发生的事它不可能知道。
3. **私有/领域数据**：公司内部文档、个人笔记，从来没进过任何模型的训练集。

---

## 三、原理与机制：脚本的五幕剧

知识库文档里那句「RAG 分为文档加载、文本切片、向量化、向量检索、LLM 生成这五个核心步骤」，恰好就是脚本本身的流程。

### 第 1 幕：建库（文档加载）

`raw_docs` 里的 6 条文档，就是「教科书」。真实场景里这里是加载 PDF、Word、网页，并切成小块（chunking），demo 直接用了现成句子。

### 第 2 幕：向量化（嵌入）—— 最魔法的一幕

```python
embeddings = embed_model.encode(raw_docs, normalize_embeddings=True)
```

BGE-M3 把每句话翻译成一串 1024 个浮点数（向量）。原理：模型在海量文本上做过对比学习训练，语义相近的句子，被映射到高维空间后**距离就近**。

关键细节：`normalize_embeddings=True` 把向量长度归一化为 1 后，**内积（IP）就等价于余弦相似度**——两个向量方向越一致，语义越近，打分越高。这是后面建集合时 `metric_type="IP"` 的原因，两者配套使用。

### 第 3 幕：入库（向量数据库）

Milvus 就是这个「书架」。向量库和普通数据库的区别：

- 普通数据库只能精确匹配；
- 向量库存的是语义，支持「语义相近搜索」。

Milvus 内部用 HNSW 索引（分层图结构的近似最近邻算法），在百万、亿级向量里也能毫秒级捞出最相近的 top_k 条——用一点点精度换取秒杀级的速度。

### 第 4 幕：检索（查书架）

用户提问时，`milvus_retrieve` 做两件事：

1. 把**问题本身**也用 BGE-M3 向量化——问题和文档在同一个向量空间里，「距离」才有意义；
2. 在 Milvus 里搜最相近的 top_k 条，把对应的 `text` 拿回来。

`top_k` 是个权衡：拿太少可能信息不够，拿太多塞爆上下文、引入噪音、还费 token。

### 第 5 幕：生成（作答）

`rag_pipeline` 把检索结果拼进 prompt：

```text
# 参考资料
（检索回来的原文）
要求：1. 严格基于资料回答 2. 资料没有就说「无相关内容」3. 简洁准确
用户提问：……
```

然后交给 Kimi 生成最终答案。本质是**用 prompt 给模型戴上镣铐**：参考资料是地板也是天花板——模型只能基于它回答，资料里没有的必须承认不知道，禁止发挥。

---

## 四、为什么这套机制有效（以及能力边界）

**「不改模型权重」是 RAG 的灵魂**。所有新知识都以向量形式存在 Milvus 里，模型本体纹丝不动。好处：

- **知识可实时更新**：新增一篇文档 = 向量化 + insert 一行，毫秒级完成；换成微调得重新训练，又贵又慢。
- **可追溯**：答案有据可查，错了能定位是哪条文档的问题。
- **隐私友好**：模型可以调公有云 API（如 Kimi），但敏感文档只存在本地 `milvus_rag.db` 里，不出本机。
- **成本低**：对比动辄上千元一次的微调，RAG 几乎零成本。

**能力边界**（RAG 不是万能的）：

- 检索质量决定一切——查错资料，后面全错；
- 文档里根本没有的知识，它也变不出来（所以 prompt 要求「直接回答无相关内容」）；
- 复杂推理、跨多文档归纳偏弱。

一句话：它是「给模型配了个图书馆」，而不是「给模型换了个大脑」。

---

## 五、怎么跑起来（使用前准备）

### 第 1 步：确认 `.env` 文件配好三个变量

脚本从环境变量读取，不写硬编码：

```bash
API_KEY=sk-你的key
BASE_URL=https://api.moonshot.cn/v1
AI_MODEL=kimi-latest        # 注意：推理模型（如 kimi-thinking）只支持 temperature=1
```

### 第 2 步：确认模型文件在本地

`rag_milvus_bge_add_LLM.py:28` 写死了模型路径，需存在：

```text
/home/administrator/rag_project/bge-m3
```

### 第 3 步：运行

```bash
rag-env/bin/python rag_milvus_bge_add_LLM.py
```

成功输出顺序：`Milvus入库成功，写入数量：6` → 打印送入 Kimi 的完整 Prompt → Kimi 的最终回答。

---

## 六、代码逐段讲解（七个部分）

### ① 环境准备（1–15 行）

```python
os.environ["HTTP_PROXY"] = ""      # 清空代理，防止系统代理干扰直连
load_dotenv()                      # 加载 .env 文件里的环境变量
```

### ② 配置区（17–25 行）

| 变量                | 作用                    | 什么时候改                             |
| ------------------- | ----------------------- | -------------------------------------- |
| `COLLECTION_NAME` | Milvus 里的表名         | 想换库时改                             |
| `DIM = 1024`      | 向量维度                | **别改**，必须和 BGE-M3 输出一致 |
| `TOP_K = 2`       | 检索返回几条资料        | 答不全调大，噪音多调小                 |
| `MILVUS_DB_FILE`  | 本地向量库文件名        | 想换库存储就改                         |
| 三个`KIMI_*`      | Kimi 的密钥/地址/模型名 | 在`.env` 里改                        |

### ③ 加载嵌入模型（27–33 行）

```python
embed_model = SentenceTransformer(model_path, device="cpu", ...)
```

BGE-M3 加载到内存（CPU 上约几秒）。`use_safetensors=False` 是因为本地模型是 `pytorch_model.bin` 格式。

**注意**：整个程序里所有文本向量化都靠这一个对象——入库时用它，检索时也用它。必须用同一个模型，否则两个向量不在同一空间，检索失效。

### ④ Kimi 客户端（35–54 行）

`client_kimi` 用 OpenAI 官方 SDK 创建，`base_url` 指向 Kimi 的兼容接口，调用写法与调 GPT 完全一致。

`llm_answer(prompt)` 是最底层函数：接收 prompt，带上 system 设定，返回 Kimi 生成的字符串。**它本身不知道任何 RAG 的事**，只负责"问 Kimi 一个问题"。

### ⑤ 建库 + 入库（56–93 行）

「准备书架」阶段，只在入库时执行一次：

```python
client = MilvusClient(MILVUS_DB_FILE)        # 打开本地向量库文件
client.drop_collection(...)                  # ⚠️ 测试用：每次清空重来，生产环境删掉
schema.add_field(...)                        # 表结构：id主键 / 1024维向量 / 原文文本
client.create_collection(..., metric_type="IP")   # 建表，距离度量用内积
embeddings = embed_model.encode(raw_docs, normalize_embeddings=True)
client.insert(...)                           # 写入
```

schema 三张字段各司其职：`vector` 负责被检索（Milvus 只对向量做相似度搜索），`text` 负责被 LLM 阅读，`id` 负责标识和删改。

### ⑥ 检索函数（96–109 行）

```python
def milvus_retrieve(query: str, top_k=TOP_K):
    q_emb = embed_model.encode(query, normalize_embeddings=True).tolist()  # 问题→向量
    search_res = client.search(..., limit=top_k, output_fields=["text"], ...)
    return [hit["entity"]["text"] for hit in search_res[0]]                # 返回原文列表
```

输入问句，输出最相关的 top_k 条原文。`ef: 10` 是 HNSW 搜索精度参数，越大越准越慢，一般 10~64。

### ⑦ RAG 流水线 + 主程序（112–133 行）

```python
def rag_pipeline(user_question: str):
    context_list = milvus_retrieve(user_question)     # 1. 检索
    prompt = f"# 参考资料\n{context_block}\n...用户提问：{user_question}"  # 2. 拼prompt
    return llm_answer(prompt)                          # 3. 生成
```

主程序里改 `question` 那一行就能换问题。

---

## 七、常见改法：如何换成自己的用法

| 需求       | 改哪里                                                                                                            |
| ---------- | ----------------------------------------------------------------------------------------------------------------- |
| 换知识库   | 改 76–83 行的`raw_docs`。真实项目通常是从 PDF/网页加载后切片的文本块，每条不超过 `text` 字段的 2000 字符上限 |
| 换问题     | 改 131 行的`question`                                                                                           |
| 调检索数量 | 改 20 行`TOP_K`                                                                                                 |
| 多轮问答   | 把主程序改成循环；`import` 后会执行整段入库流程（含清空重建），更适合当独立脚本用                               |

---

## 八、生产环境注意事项

1. **删掉测试清空逻辑**：`rag_milvus_bge_add_LLM.py:61-62` 的 `drop_collection`，否则每次运行知识库都被清空；
2. **入库与查询拆分**：入库逻辑跑一次，查询逻辑跑无数次，应拆成两个脚本；
3. **密钥不进代码**：通过 `.env` + `os.getenv` 读取，`.env` 加入 `.gitignore`。

---

## 九、一张图记住数据流

```text
raw_docs ──BGE-M3向量化──┐
                          ├→ Milvus (id, vector, text)
你的问题 ──BGE-M3向量化──┘        ↓ 检索top_k条原文
                               拼prompt(资料+问题+要求)
                                    ↓
                              Kimi生成答案
```

一句话总结：**BGE-M3 管"把文字变成向量"，Milvus 管"存和查向量"，Kimi 管"看着资料说话"**；三个函数 `milvus_retrieve` → `rag_pipeline` → `llm_answer` 就是这条链路的代码实现。
