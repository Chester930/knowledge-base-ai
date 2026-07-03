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

# 二階段粗篩-精篩檢索（THEORETICAL_ARCHITECTURE.md 第9節⑧）
TWO_STAGE_COARSE_TOP_K = 100    # Stage-1：每個 query concept 用 Neo4j Vector Index 取回的候選數上限

# 時序知識圖譜衰減（THEORETICAL_ARCHITECTURE.md 第9節⑥）
TEMPORAL_DECAY_RATE = 0.005     # 每日衰減率：decay = exp(-rate * delta_days)，SVO 邊 created_at 缺失時視為不衰減
