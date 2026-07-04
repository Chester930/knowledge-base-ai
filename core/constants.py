INTEREST_INIT = 0.5
PROFESSIONAL_INIT = 0.5
SCORE_REMOVE_THRESHOLD = 0.01
INERTIA_K = 2.0
VECTOR_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2

# KG 路由門檻
KG_ROUTE_THRESHOLD = 0.05      # Agent 問答：低於此分數的 KG 不召回
MAX_KG_PER_QUERY = 5           # Agent 問答：最多召回幾個 KG

# 文件分配門檻
CLASSIFY_AUTO_THRESHOLD = 0.30  # 自動分配：top score 需超過此值才自動移動
CLASSIFY_MIN_THRESHOLD = 0.05   # 低於此值視為完全無相關，留 _staging/ 等待

# 兩階段向量粗精篩（Two-Stage Retrieval）
# Stage-1：用 Neo4j Vector Index 對每個 query concept 取 Top-K 候選 ConceptNode
# Stage-2：僅對候選節點做 Python 端的對齊/強度精篩，取代全表掃描
CONCEPT_COARSE_TOP_K = 100
