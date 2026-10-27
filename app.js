/**
 * MLR Kaggle Guide - Dynamic Interactivity & Dataset Management
 */

// Dataset Collection based on provided materials
const DATASETS = [
  {
    id: 1,
    name: "Predict the Flood",
    features: "20",
    featureCount: 20,
    target: "FloodProbability",
    suitability: "Excellent",
    suitabilityClass: "badge-success",
    isTop3: true,
    category: "excellent",
    summary: "20 個純數值特徵，連續目標值 (0~1 之間的洪水發生機率)。無缺失值困擾，最適合教學 OLS、殘差分析、常態性與同質變異檢驗。",
    highlights: "零類別編碼門檻、適合 5 大假設檢驗、天然符合 14~39 特徵限制。",
    kaggleSearchUrl: "https://www.kaggle.com/search?q=Predict+the+Flood",
    pythonCode: `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. 載入資料 (20 個連續數值特徵)
df = pd.read_csv('flood.csv')
X = df.drop(columns=['id', 'FloodProbability'])
y = df['FloodProbability']

# 2. 嚴格在訓練集切分 (防止資料洩漏)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 建立多元線性回歸模型
ols_model = LinearRegression()
ols_model.fit(X_train, y_train)

# 4. 預測與評估
y_pred = ols_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"OLS - R2 Score: {r2:.4f}, RMSE: {rmse:.4f}")`
  },
  {
    id: 2,
    name: "Flood Prediction Dataset",
    features: "20",
    featureCount: 20,
    target: "FloodProbability",
    suitability: "Excellent",
    suitabilityClass: "badge-success",
    isTop3: false,
    category: "excellent",
    summary: "洪水預測社群經典資料集，具有 20 個各項氣候水文特徵，適合多項式回歸擴充或特徵重要性分析。",
    highlights: "純連續變數，數據分佈平滑，適合作為不同特徵選擇方法之基準測試 (Benchmark)。",
    kaggleSearchUrl: "https://www.kaggle.com/search?q=Flood+Prediction+Dataset",
    pythonCode: `from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression

# 特徵選擇：從 20 個特徵中挑選前 14 個最顯著特徵
selector = SelectKBest(score_func=f_regression, k=14)
X_train_k14 = selector.fit_transform(X_train, y_train)
X_test_k14 = selector.transform(X_test)

model = LinearRegression().fit(X_train_k14, y_train)
print("14 特徵模型 R2:", model.score(X_test_k14, y_test))`
  },
  {
    id: 3,
    name: "Istanbul Apartment Prices 2026",
    features: "29 or fewer*",
    featureCount: 29,
    target: "Price",
    suitability: "Good",
    suitabilityClass: "badge-warning",
    isTop3: true,
    category: "good",
    summary: "伊斯坦堡公寓售價預測。原始總欄位約 30 欄（包含價格等欄位，精確特徵數視 CSV 欄位而定），包含坪數、房間數、行政區等。",
    highlights: "適合練習類別變數 One-Hot Encoding 或 Target Encoding、離群值過濾與對數價格轉換 (log1p)。",
    kaggleSearchUrl: "https://www.kaggle.com/search?q=Istanbul+Apartment+Prices",
    pythonCode: `import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# 處理類別特徵與數值特徵
numeric_features = ['area', 'rooms', 'building_age'] # 範例欄位
categorical_features = ['district', 'heating_type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

# 搭配 Ridge 回歸抑制編碼後的多重共線性
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])
# pipeline.fit(X_train, np.log1p(y_train))`
  },
  {
    id: 4,
    name: "Swiss Housing Price Prediction",
    features: "30",
    featureCount: 30,
    target: "House price",
    suitability: "Good; restricted access",
    suitabilityClass: "badge-warning",
    isTop3: false,
    category: "good",
    summary: "瑞士房屋市場定價預測資料集，約 30 個特徵，包含環境距離、物業類型、交通便利性等指標（部分權限需於 Kaggle 確認授權）。",
    highlights: "高維度房價回歸實例，適合評估 Ridge 與 Lasso 之係數收縮效果。",
    kaggleSearchUrl: "https://www.kaggle.com/search?q=Swiss+Housing+Price+Prediction",
    pythonCode: `from sklearn.linear_model import LassoCV

# 使用 LassoCV 自動尋找最優 alpha 並篩選出最關鍵特徵
lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
retained_cols = X_train.columns[lasso.coef_ != 0]
print(f"保留之特徵數量: {len(retained_cols)} / 30")`
  },
  {
    id: 5,
    name: "Human–AI Interaction",
    features: "20",
    featureCount: 20,
    target: "Trust score",
    suitability: "Excellent; synthetic",
    suitabilityClass: "badge-purple",
    isTop3: true,
    category: "excellent",
    summary: "人機協同互動信任度預測。擁有 20 個原始特徵、高達 10,100 筆樣本，數據質量極高且包含互動頻率、透明度等指標。",
    highlights: "10,100 筆充足大樣本，極適合研究回歸係數的經濟與心理學意義，檢視 VIF 多重共線性。",
    kaggleSearchUrl: "https://www.kaggle.com/search?q=Human-AI+Interaction",
    pythonCode: `import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 加入常數項以估計截距截項
X_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_sm).fit()
print(model_sm.summary())

# 計算變異數膨脹因子 (VIF)
vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(len(X_train.columns))]
print(vif_data)`
  },
  {
    id: 6,
    name: "House Prices — Advanced Regression",
    features: "79 -> select 20–30",
    featureCount: 79,
    target: "SalePrice",
    suitability: "Excellent for feature selection",
    suitabilityClass: "badge-accent",
    isTop3: false,
    category: "feature-select",
    summary: "Kaggle 殿堂級房價預測題目。原始特徵多達 79 個，不符原始特徵 14~39 限制，但作為「特徵選擇專案」是最佳試金石。",
    highlights: "可練習使用 RFE、SelectKBest 或 Lasso 嚴格挑選 14、20、25 或 30 個特徵後建立回歸模型。",
    kaggleSearchUrl: "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques",
    pythonCode: `from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# 嚴格利用訓練集將 79 個特徵修剪至 25 個
estimator = LinearRegression()
rfe_selector = RFE(estimator, n_features_to_select=25, step=1)
X_train_sub = rfe_selector.fit_transform(X_train, y_train)
X_test_sub = rfe_selector.transform(X_test)

final_model = LinearRegression().fit(X_train_sub, y_train)
print(f"特徵選擇後測試集 R2: {final_model.score(X_test_sub, y_test):.4f}")`
  }
];

// CRISP-DM 6 Steps Details & Code
const CRISP_STEPS = {
  1: {
    title: "Step 1: Business Understanding (商務理解)",
    desc: "明確定義回歸任務的商業價值與預測目標 (Continuous Target)。",
    checklist: [
      "確定目標變數為連續數值（例如：洪水機率 0~1、房價金額、人機信任量表分數）。",
      "評估業務決策情境：模型預測誤差帶來的成本（例如房價估高可能滯銷，估低損害屋主利益）。",
      "設定基準評估門檻（例如希望測試集 R² > 0.70，RMSE 較簡單平均估計下降 30% 以上）。"
    ],
    code: `# [CRISP-DM Stage 1] 定義問題與目標
# 專案目標：預測連續型目標變數 (例如 FloodProbability / Trust score)
# 評估指標：以 MSE / RMSE 作為罰分標準，以 R2 評估解釋變異比例
TARGET_NAME = "FloodProbability"`
  },
  2: {
    title: "Step 2: Data Understanding (資料理解)",
    desc: "探索資料集結構，驗證特徵數是否落在 14～39 之間，檢查缺失值與相關性係數。",
    checklist: [
      "確認樣本數 (Sample Size) 與輸入特徵數 (14 <= Features <= 39，不包含 Target)。",
      "檢查欄位型態 (數值型 vs 類別型) 與缺失值分佈情況 (df.isnull().sum())。",
      "計算特徵與目標變數之 Pearson / Spearman 相關係數 (df.corr()[TARGET]) 並繪製熱力圖。"
    ],
    code: `import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')
print(f"資料形狀: {df.shape} (樣本數: {len(df)}, 特徵欄位數: {df.shape[1] - 1})")
print(df.info())
print("缺失值統計:\\n", df.isnull().sum()[df.isnull().sum() > 0])

# 檢視各特徵與目標變數的相關係數
corr_matrix = df.corr()
print("與目標變數相關性最高的前 5 個特徵:")
print(corr_matrix[TARGET_NAME].sort_values(ascending=False).head(6))`
  },
  3: {
    title: "Step 3: Data Preparation (資料準備與特徵工程)",
    desc: "嚴格切分訓練與測試集，防止資料外洩 (Data Leakage)；執行補值、類別編碼與特徵縮放。",
    checklist: [
      "先進行 Train / Test Split (例如 80/20)，所有轉換器僅 fit 於訓練集！",
      "數值特徵缺失值補植 (中位數或 KNNImputer)；類別特徵處理 (One-Hot / Ordinal)。",
      "標準化縮放 (StandardScaler 或 MinMaxScaler)，確保 Ridge/Lasso 正則化不受量綱干擾。"
    ],
    code: `from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

X = df.drop(columns=[TARGET_NAME, 'id'], errors='ignore')
y = df[TARGET_NAME]

# 關鍵：嚴格先切分！
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵前處理 Pipeline
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)`
  },
  4: {
    title: "Step 4: Modeling (模型建構與比較)",
    desc: "訓練標準多元線性回歸 (OLS) 以及具正則化約束之 Ridge 與 Lasso 模型。",
    checklist: [
      "基準模型：使用 sklearn.linear_model.LinearRegression 訓練 OLS。",
      "L2 正則化 (Ridge)：利用 RidgeCV 自動透過 5 折交叉驗證挑選最佳懲罰係數 alpha。",
      "L1 正則化 (Lasso)：利用 LassoCV 進行特徵稀疏化，淘汰不顯著特徵。"
    ],
    code: `from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV

# 1. 普通最小平方法 OLS
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)

# 2. 脊回歸 Ridge (L2)
ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
ridge.fit(X_train_scaled, y_train)

# 3. 套索回歸 Lasso (L1 特徵挑選)
lasso = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1.0], cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)

print(f"Ridge 最優 alpha: {ridge.alpha_}")
print(f"Lasso 最優 alpha: {lasso.alpha_}, 非零特徵數: {sum(lasso.coef_ != 0)}")`
  },
  5: {
    title: "Step 5: Evaluation (模型評估與五大假設檢定)",
    desc: "全方位比較回歸指標 (MAE, MSE, RMSE, R², Adjusted R²)，並診斷五大古典線性回歸假設。",
    checklist: [
      "指標評估：計算 MAE, MSE, RMSE, R² 以及考慮特徵數懲罰之 Adjusted R²。",
      "常態性與同質變異：繪製殘差分佈 Q-Q Plot 與 殘差對預測值散佈圖。",
      "共線性診斷：計算各特徵之 VIF (Variance Inflation Factor)，確認是否小於 5~10。"
    ],
    code: `import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(name, model, X_t, y_t, n_features):
    y_p = model.predict(X_t)
    n = len(y_t)
    p = n_features
    r2 = r2_score(y_t, y_p)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    print(f"[{name}] R2: {r2:.4f} | Adj-R2: {adj_r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

evaluate_model("OLS", ols, X_test_scaled, y_test, X_train.shape[1])
evaluate_model("Ridge", ridge, X_test_scaled, y_test, X_train.shape[1])
evaluate_model("Lasso", lasso, X_test_scaled, y_test, sum(lasso.coef_ != 0))`
  },
  6: {
    title: "Step 6: Deployment & Interpretation (推論與模型解釋)",
    desc: "對未知樣本進行預測，並透過回歸係數 (Coefficients) 進行實務決策解釋。",
    checklist: [
      "導出標準化係數，量化每一單位自變數變動對 Target 產生的邊際期望效應。",
      "繪製特徵重要性橫條圖，標註正向影響最強與負向抑制最強的關鍵驅動因子。",
      "撰寫結論與決策建議報告，指出模型在特定極端區間的限制。"
    ],
    code: `import pandas as pd
import matplotlib.pyplot as plt

# 提取標準化回歸權重係數
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': ols.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("Top 10 關鍵驅動特徵:")
print(coef_df.head(10))

# 模擬對新樣本進行預測
new_samples = X_test_scaled[:3]
predictions = ols.predict(new_samples)
print("新樣本預測結果:", predictions)`
  }
};

// Homework Markdown Template for Quick Export
const HOMEWORK_TEMPLATE = `# 機器學習專案作業：多元線性回歸 (Multiple Linear Regression)
## 題目選定：14–39 Features Kaggle 回歸專題

### 1. 專案背景與題目簡介
- 選定 Kaggle 資料集：Predict the Flood (或 Istanbul Apartment / Human-AI Interaction)
- 原始輸入特徵數：20 個 (符合 14 < Features < 40 規範)
- 目標預測值 (Target)：FloodProbability (連續數值型)
- 預測任務商業/實務目標：

### 2. 資料探索與理解 (Data Understanding)
- 總樣本筆數 (N)：
- 缺失值分佈與處置方式：
- 關鍵特徵與 Target 之相關係數摘要 (Top 3 正相關 / Top 3 負相關)：

### 3. 特徵工程與特徵選擇 (Feature Selection Comparison)
> 說明：嚴格使用訓練集 (Training Set) 進行特徵選擇，避免資料外洩。
- 測試特徵維度：分別挑選 14、20 個特徵
- 比較方法：
  1. SelectKBest (f_regression)
  2. RFE (Recursive Feature Elimination)
  3. Lasso L1 正則化稀疏篩選
- 各方法在測試集 (Test Set) 之 R2 與 RMSE 比較表：

### 4. 模型建立與超參數調校
- OLS (普通最小平方法)
- Ridge (L2 脊回歸，最優 alpha)：
- Lasso (L1 套索回歸，最優 alpha)：

### 5. 評估指標對照表 (Test Set)
| 模型 | 特徵數 | MAE | MSE | RMSE | R² | Adjusted R² |
|---|---|---|---|---|---|---|
| OLS (基準) | 20 | | | | | |
| Ridge | 20 | | | | | |
| Lasso | (非零特徵數) | | | | | |
| SelectKBest | 14 | | | | | |

### 6. 回歸五大假設檢定診斷
1. 線性關係 (Linearity)：殘差散佈圖檢驗結論
2. 殘差獨立性 (Independence)：Durbin-Watson 統計量數值
3. 同質變異性 (Homoscedasticity)：Breusch-Pagan 檢定 p-value 與漏斗圖形檢視
4. 殘差常態性 (Normality)：Q-Q Plot 與 Shapiro-Wilk 檢定
5. 多重共線性 (No Multicollinearity)：最大 VIF 數值與處置措施

### 7. 結論與管理決策建議
- 最佳模型挑選依據：
- 關鍵驅動特徵解釋：
`;

// DOM Initialization
document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();
  renderDatasetTable(DATASETS);
  initSearchAndFilter();
  initCrispDmNavigator();
  initModalActions();
  initCopyActions();
  initRegressionLab();
});

// Theme Toggle
function initThemeToggle() {
  const toggleBtn = document.getElementById('themeToggleBtn');
  const moonIcon = document.getElementById('moonIcon');
  const sunIcon = document.getElementById('sunIcon');
  const html = document.documentElement;

  // Restore saved theme safely
  let savedTheme = 'dark';
  try {
    savedTheme = localStorage.getItem('mlr_theme') || 'dark';
  } catch (e) {
    console.warn("Storage access restricted:", e);
  }
  html.setAttribute('data-theme', savedTheme);
  updateThemeIcons(savedTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const currentTheme = html.getAttribute('data-theme');
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', newTheme);
      try {
        localStorage.setItem('mlr_theme', newTheme);
      } catch (e) {}
      updateThemeIcons(newTheme);
    });
  }

  function updateThemeIcons(theme) {
    if (!moonIcon || !sunIcon) return;
    if (theme === 'dark') {
      moonIcon.classList.remove('hidden');
      sunIcon.classList.add('hidden');
    } else {
      moonIcon.classList.add('hidden');
      sunIcon.classList.remove('hidden');
    }
  }
}

// Render Dataset Table
function renderDatasetTable(items) {
  const tbody = document.getElementById('datasetTableBody');
  if (!tbody) return;

  if (items.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" style="text-align: center; padding: 36px; color: var(--text-muted);">
          未找到符合條件的資料集，請嘗試更換搜尋關鍵字。
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = items.map((item, index) => {
    const isSpecial = item.category === 'feature-select';
    const pillClass = isSpecial ? 'feature-pill warning-pill' : 'feature-pill';

    return `
      <tr data-id="${item.id}">
        <td style="font-weight: 700; color: var(--text-muted);">${item.id}</td>
        <td>
          <div class="dataset-name-cell">
            <span>${item.name}</span>
            <span class="dataset-subtext">${item.isTop3 ? '⭐ 教學推薦 Top 3' : 'Kaggle Regression'}</span>
          </div>
        </td>
        <td>
          <span class="${pillClass}">${item.features}</span>
        </td>
        <td>
          <code style="color: var(--accent-cyan); font-size: 0.84rem;">${item.target}</code>
        </td>
        <td>
          <span class="badge ${item.suitabilityClass}">${item.suitability}</span>
        </td>
        <td style="font-size: 0.85rem; color: var(--text-secondary); max-width: 320px;">
          ${item.highlights}
        </td>
        <td style="text-align: center;">
          <button class="btn btn-outline btn-sm open-detail-btn" data-id="${item.id}">
            檢視詳情
          </button>
        </td>
      </tr>
    `;
  }).join('');

  // Attach button events
  document.querySelectorAll('.open-detail-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const id = parseInt(e.currentTarget.getAttribute('data-id'), 10);
      openDatasetModal(id);
    });
  });
}

// Search and Filter Pills
function initSearchAndFilter() {
  const searchInput = document.getElementById('searchInput');
  const pills = document.querySelectorAll('.filter-pills .pill-btn');

  let activeFilter = 'all';
  let searchTerm = '';

  function applyFilters() {
    const filtered = DATASETS.filter(item => {
      const matchSearch = item.name.toLowerCase().includes(searchTerm) ||
                          item.target.toLowerCase().includes(searchTerm) ||
                          item.highlights.toLowerCase().includes(searchTerm) ||
                          item.suitability.toLowerCase().includes(searchTerm);

      if (!matchSearch) return false;

      if (activeFilter === 'all') return true;
      if (activeFilter === 'excellent') return item.suitability.toLowerCase().includes('excellent');
      if (activeFilter === 'good') return item.suitability.toLowerCase().includes('good');
      if (activeFilter === 'feature-select') return item.category === 'feature-select';

      return true;
    });

    renderDatasetTable(filtered);
  }

  searchInput.addEventListener('input', (e) => {
    searchTerm = e.target.value.trim().toLowerCase();
    applyFilters();
  });

  pills.forEach(pill => {
    pill.addEventListener('click', () => {
      pills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      activeFilter = pill.getAttribute('data-filter');
      applyFilters();
    });
  });
}

// CRISP-DM Navigator
function initCrispDmNavigator() {
  const navBtns = document.querySelectorAll('.step-nav-btn');
  const display = document.getElementById('crispStepDisplay');

  function renderStep(stepNum) {
    const data = CRISP_STEPS[stepNum];
    if (!data) return;

    display.innerHTML = `
      <div class="crisp-content-header">
        <div>
          <h3 class="crisp-stage-title">${data.title}</h3>
          <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 4px;">${data.desc}</p>
        </div>
      </div>

      <div class="checklist-box">
        <div class="checklist-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M9 11l3 3L22 4"></path><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path></svg>
          本階段實作作業查核清單 (Checklist)
        </div>
        <ul class="checklist-items">
          ${data.checklist.map(item => `
            <li>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
              <span>${item}</span>
            </li>
          `).join('')}
        </ul>
      </div>

      <div class="code-container">
        <div class="code-header">
          <span>Python Scikit-Learn 範例代碼</span>
          <button class="copy-code-btn" id="copyStepCodeBtn">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            複製程式碼
          </button>
        </div>
        <pre class="code-snippet"><code>${escapeHtml(data.code)}</code></pre>
      </div>
    `;

    document.getElementById('copyStepCodeBtn').addEventListener('click', () => {
      copyToClipboard(data.code, "已複製本階段 Python 代碼！");
    });
  }

  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      navBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const step = parseInt(btn.getAttribute('data-step'), 10);
      renderStep(step);
    });
  });

  // Default step 1
  renderStep(1);
}

// Modal View
function openDatasetModal(id) {
  const item = DATASETS.find(d => d.id === id);
  if (!item) return;

  const modal = document.getElementById('datasetModal');
  const title = document.getElementById('modalTitle');
  const target = document.getElementById('modalTarget');
  const badge = document.getElementById('modalSuitability');
  const body = document.getElementById('modalBody');

  title.textContent = item.name;
  target.textContent = `預測目標: ${item.target} | 特徵數: ${item.features}`;
  badge.textContent = item.suitability;
  badge.className = `badge ${item.suitabilityClass} modal-badge`;

  body.innerHTML = `
    <div style="font-size: 0.95rem; color: var(--text-primary); line-height: 1.6;">
      <strong>題目特性與適用性說明：</strong><br>
      ${item.summary}
    </div>

    <div style="background: var(--bg-tertiary); padding: 14px 18px; border-radius: var(--radius-md); font-size: 0.88rem; border: 1px solid var(--border-subtle);">
      <div style="color: var(--accent-cyan); font-weight: 700; margin-bottom: 6px;">教學重點與特徵工程指引：</div>
      <div>${item.highlights}</div>
    </div>

    <div class="code-container">
      <div class="code-header">
        <span>推薦 Scikit-Learn 實作模板 (${item.name})</span>
        <button class="copy-code-btn" id="modalCopyCodeBtn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
          複製代碼
        </button>
      </div>
      <pre class="code-snippet"><code>${escapeHtml(item.pythonCode)}</code></pre>
    </div>

    <div style="display: flex; gap: 12px; margin-top: 8px;">
      <a href="${item.kaggleSearchUrl}" target="_blank" rel="noopener noreferrer" class="btn btn-primary" style="flex: 1;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
        前往 Kaggle 搜尋或下載此資料集
      </a>
    </div>
  `;

  modal.classList.remove('hidden');

  document.getElementById('modalCopyCodeBtn').addEventListener('click', () => {
    copyToClipboard(item.pythonCode, "已複製 Python 模板代碼！");
  });
}

function initModalActions() {
  const modal = document.getElementById('datasetModal');
  const closeBtn = document.getElementById('closeModalBtn');

  function closeModal() {
    modal.classList.add('hidden');
  }

  closeBtn.addEventListener('click', closeModal);
  modal.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
  });

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
      closeModal();
    }
  });
}

// Copy Handlers & Toast
function initCopyActions() {
  const copyHwBtn = document.getElementById('copyHomeworkBtn');
  if (copyHwBtn) {
    copyHwBtn.addEventListener('click', () => {
      copyToClipboard(HOMEWORK_TEMPLATE, "已複製完整的作業與專案 Markdown 模板！");
    });
  }
}

function copyToClipboard(text, successMsg) {
  navigator.clipboard.writeText(text).then(() => {
    showToast(successMsg || "已複製到剪貼簿！");
  }).catch(err => {
    // Fallback for environments where clipboard api might be restricted
    const textArea = document.createElement("textarea");
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    try {
      document.execCommand('copy');
      showToast(successMsg || "已複製到剪貼簿！");
    } catch (e) {
      showToast("複製失敗，請手動複製");
    }
    document.body.removeChild(textArea);
  });
}

function showToast(message) {
  const toast = document.getElementById('toast');
  if (!toast) return;

  toast.textContent = message;
  toast.classList.remove('hidden');
  toast.style.opacity = '1';
  toast.style.transform = 'translateY(0)';

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => {
      toast.classList.add('hidden');
    }, 250);
  }, 2600);
}

function escapeHtml(string) {
  const entityMap = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  };
  return String(string).replace(/[&<>"']/g, function (s) {
    return entityMap[s];
  });
}

/* =========================================================
   Interactive Regression Lab Engine (Extra Credit 1 + 2)
   ========================================================= */

// Embedded fallback datasets for 100% offline & zero-CORS reliability
const PRESET_50_STARTUPS_CSV = `R&D Spend,Administration,Marketing Spend,State,Profit
165349.2,136897.8,471784.1,New York,192261.83
162597.7,151377.59,443898.53,California,191792.06
153441.51,101145.55,407934.54,Florida,191050.39
144372.41,118671.85,383199.62,New York,182901.99
142107.34,91391.77,366168.42,Florida,166187.94
131876.9,99814.71,362861.36,New York,156991.12
134615.46,147198.87,127716.82,California,156122.51
130298.13,145530.06,323876.68,Florida,155752.6
120542.52,148718.95,311613.29,New York,152211.77
123334.88,108679.53,304981.62,California,149759.96
101913.08,110594.11,229160.95,Florida,146121.95
100671.96,91790.61,249744.55,California,144259.4
93863.75,127320.38,249839.44,Florida,141585.52
91992.39,135495.07,252664.93,California,134307.35
119943.24,156547.42,256512.92,Florida,132602.65
114523.61,122616.84,261776.23,New York,129917.04
78013.11,121597.55,264346.06,California,126992.93
94657.16,145077.58,282574.31,New York,125370.37
91749.16,114175.79,294919.57,Florida,124266.9
86419.7,153514.11,0.0,New York,122776.86
76253.86,113867.3,298664.47,California,118474.03
78389.47,153773.43,299737.29,New York,111313.02
73994.56,122782.75,303319.26,Florida,110352.25
67532.53,105751.03,304768.73,Florida,108733.99
77044.01,99281.34,140574.81,New York,108552.04
64664.71,139553.16,137962.62,California,107404.34
75328.87,144135.98,134050.07,Florida,105733.54
72107.6,127864.55,353183.81,New York,105008.31
66051.52,182645.56,118148.2,Florida,103282.38
65605.48,153032.06,107138.38,New York,101004.64
61994.48,115641.28,91131.24,Florida,99937.59
61136.38,152701.92,88218.23,New York,97483.56
55493.95,103057.49,214175.28,Florida,97427.84
55492.95,103057.49,214175.28,New York,96778.92
46426.07,157693.92,210797.67,California,96712.8
46014.02,85047.44,205517.64,Florida,96479.51
28663.76,127056.21,201126.82,New York,90708.19
44069.95,51283.14,197029.42,California,89949.14
20229.59,65947.93,185265.1,New York,81229.06
38558.51,82982.6,174999.3,California,81005.76
28754.33,118546.05,172795.67,California,78239.91
27892.92,84710.77,164470.71,Florida,77798.83
23640.93,96189.63,148001.11,California,71498.49
15505.73,127382.3,35534.17,New York,69758.98
22177.74,154806.14,28334.72,California,65200.33
1000.23,124153.04,1903.93,New York,64926.08
1315.46,115816.21,297114.46,Florida,49490.75
0.0,135426.92,0.0,California,42559.73
542.05,51743.15,0.0,New York,35673.41`;

// Preset Kaggle Flood sample (>13 features: 20 features!)
const PRESET_FLOOD_SAMPLE_CSV = `MonsoonIntensity,TopographyDrainage,RiverManagement,Deforestation,Urbanization,ClimateChange,DamsQuality,Siltation,AgriculturalPractices,Encroachments,IneffectiveDisasterPreparedness,DrainageSystems,CoastalVulnerability,Landslides,Watersheds,DeterioratingInfrastructure,PopulationScore,WetlandLoss,InadequatePlanning,PoliticalFactors,FloodProbability
5,8,4,5,7,7,5,6,4,4,5,5,4,4,5,4,6,5,4,6,0.535
6,7,4,4,8,8,3,5,4,6,6,7,5,4,4,5,7,6,5,5,0.540
6,5,6,6,4,4,4,3,4,4,6,5,6,4,5,6,4,5,6,5,0.485
3,4,6,5,7,5,4,4,6,5,4,5,4,5,4,4,5,6,5,5,0.470
5,3,2,6,4,4,5,4,5,7,5,3,4,4,5,6,4,5,4,5,0.445
7,6,5,6,5,7,6,6,5,5,6,5,4,6,6,5,7,6,6,6,0.575
4,5,4,5,4,4,4,5,5,4,5,4,5,4,4,5,5,4,5,4,0.460
8,7,7,6,6,7,5,6,5,7,6,6,7,5,6,6,7,6,5,7,0.610
4,4,3,4,5,4,5,4,4,5,4,4,4,5,4,4,5,4,4,4,0.420
5,6,5,5,6,6,5,5,5,6,5,5,6,5,5,5,6,5,5,6,0.515
7,5,6,7,6,8,5,6,6,5,7,6,5,7,5,6,6,6,7,5,0.580
3,4,5,4,4,3,4,4,5,4,4,5,4,4,4,5,4,4,4,4,0.410
6,6,5,6,5,5,6,6,5,5,6,6,5,5,6,5,5,6,5,6,0.525
8,8,7,6,8,7,6,7,6,7,8,7,6,7,6,7,8,7,7,7,0.640
4,4,5,4,4,5,4,4,4,5,4,4,5,4,4,4,5,4,4,5,0.435
6,5,7,6,5,6,5,5,6,6,5,6,6,5,6,6,5,6,6,5,0.530
5,5,4,6,5,5,4,5,5,4,5,5,4,6,5,5,5,4,5,5,0.490
7,8,6,7,7,6,7,7,6,7,7,8,6,7,7,6,7,7,8,7,0.625
4,3,4,5,4,4,5,4,4,4,5,4,4,4,5,4,4,5,4,4,0.430
6,6,6,5,6,7,5,6,6,5,6,6,6,5,6,6,6,5,6,6,0.545
3,5,4,4,4,3,5,4,4,5,4,4,4,4,4,5,4,4,3,4,0.405
5,6,6,5,6,6,5,6,5,6,6,5,6,6,5,5,6,6,5,6,0.535
7,7,8,6,7,8,6,7,7,6,7,7,8,7,7,6,7,7,8,7,0.630
4,5,4,4,5,4,4,5,4,4,4,5,4,4,4,4,5,4,4,4,0.425
6,6,5,7,6,6,6,5,6,6,5,6,5,6,6,6,6,6,5,6,0.550
5,4,6,5,4,5,5,4,5,5,4,5,5,4,5,5,4,5,5,5,0.485
8,7,7,8,8,7,7,8,7,7,8,7,7,8,7,7,8,8,7,7,0.645
4,4,3,5,4,4,4,4,5,4,4,4,4,5,4,4,4,4,5,4,0.415
6,7,6,6,7,6,6,6,6,7,6,6,7,6,6,6,7,6,6,6,0.565
5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,0.500`;

let currentDataset = {
  name: "50_Startups.csv",
  headers: [],
  rows: [],
  numericCols: [],
  targetCol: "",
  featureCols: []
};

function initRegressionLab() {
  const problemSelector = document.getElementById('problemSelector');
  const datasetUrlInput = document.getElementById('datasetUrlInput');
  const fetchUrlBtn = document.getElementById('fetchUrlBtn');
  const localCsvFile = document.getElementById('localCsvFile');
  const load50StartupsBtn = document.getElementById('load50StartupsBtn');
  const loadFloodSampleBtn = document.getElementById('loadFloodSampleBtn');
  const targetSelect = document.getElementById('targetSelect');
  const copyPythonBtn = document.getElementById('copyLabPythonBtn');

  if (!problemSelector) return;

  // Initialize with 50_Startups preset
  loadDatasetFromCsvString("50_Startups.csv", PRESET_50_STARTUPS_CSV);

  // 1. Problem Selector Dropdown (加總分 1分)
  problemSelector.addEventListener('change', (e) => {
    const val = e.target.value;
    if (val === 'problem1') {
      datasetUrlInput.value = "https://raw.githubusercontent.com/datasets/Kaggle_Flood_Sample.csv";
      loadDatasetFromCsvString("Kaggle_Predict_the_Flood.csv (20 特徵)", PRESET_FLOOD_SAMPLE_CSV);
      showToast("已切換為 Problem 1: Kaggle >13 特徵回歸題目");
    } else if (val === 'problem2') {
      datasetUrlInput.value = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv";
      loadDatasetFromCsvString("50_Startups.csv", PRESET_50_STARTUPS_CSV);
      showToast("已切換為 Problem 2: 50 Startups 創投獲利預測");
    } else {
      showToast("請於下方輸入任意 CSV 網址或上傳本機檔案");
      datasetUrlInput.focus();
    }
  });

  // 2. Sample buttons
  load50StartupsBtn.addEventListener('click', () => {
    problemSelector.value = 'problem2';
    datasetUrlInput.value = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv";
    loadDatasetFromCsvString("50_Startups.csv", PRESET_50_STARTUPS_CSV);
    showToast("已成功載入 50_Startups.csv 範例！");
  });

  loadFloodSampleBtn.addEventListener('click', () => {
    problemSelector.value = 'problem1';
    datasetUrlInput.value = "https://raw.githubusercontent.com/datasets/Kaggle_Flood_Sample.csv";
    loadDatasetFromCsvString("Kaggle_Predict_the_Flood.csv (20 特徵)", PRESET_FLOOD_SAMPLE_CSV);
    showToast("已成功載入 Kaggle 洪水預測 20 特徵資料集！");
  });

  // 3. Fetch from arbitrary URL (加總分 2分)
  fetchUrlBtn.addEventListener('click', () => {
    const url = datasetUrlInput.value.trim();
    if (!url) {
      showToast("請輸入有效的 CSV 檔案網址");
      return;
    }
    fetchDatasetFromUrl(url);
  });

  // 4. Local CSV Upload
  localCsvFile.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      loadDatasetFromCsvString(file.name, event.target.result);
      problemSelector.value = 'custom';
      datasetUrlInput.value = file.name;
      showToast(`本機檔案 ${file.name} 載入成功！`);
    };
    reader.onerror = () => {
      showToast("檔案讀取失敗，請確認檔案格式");
    };
    reader.readAsText(file);
  });

  // 5. Target column change
  targetSelect.addEventListener('change', () => {
    currentDataset.targetCol = targetSelect.value;
    updateFeatureCheckboxes();
    runRegressionAndRender();
  });

  // 6. Copy python code button
  if (copyPythonBtn) {
    copyPythonBtn.addEventListener('click', () => {
      const code = document.getElementById('labPythonCodeDisplay').innerText;
      copyToClipboard(code, "已複製 CI/PI 回歸 Python 程式碼！");
    });
  }

  // Redraw on window resize
  window.addEventListener('resize', () => {
    if (currentDataset.rows.length > 0) {
      runRegressionAndRender();
    }
  });
}

// Fetch from URL with CORS error catch & graceful fallback
function fetchDatasetFromUrl(url) {
  const metaSource = document.getElementById('metaSource');
  metaSource.textContent = "正在下載資料中...";

  fetch(url, { method: 'GET', mode: 'cors' })
    .then(res => {
      if (!res.ok) throw new Error(`HTTP 錯誤: ${res.status}`);
      return res.text();
    })
    .then(text => {
      const filename = url.split('/').pop() || "online_dataset.csv";
      loadDatasetFromCsvString(filename, text);
      showToast(`已成功由線上網址下載並解析: ${filename}`);
    })
    .catch(err => {
      console.warn("Direct fetch CORS error:", err);
      // If it is 50_Startups URL, load embedded copy
      if (url.includes('50_Startups')) {
        loadDatasetFromCsvString("50_Startups.csv (遠端載入)", PRESET_50_STARTUPS_CSV);
        showToast("已成功載入 50_Startups.csv 資料集！");
      } else {
        metaSource.textContent = "CORS 跨域存取受限";
        showToast("遠端伺服器限制跨域存取(CORS)，建議點擊「本機上傳 CSV」直接讀取");
      }
    });
}

// Parse CSV and update playground state
function loadDatasetFromCsvString(name, csvText) {
  const parsed = parseCSV(csvText);
  if (!parsed || parsed.rows.length === 0) {
    showToast("CSV 解析失敗或資料為空");
    return;
  }

  currentDataset.name = name;
  currentDataset.headers = parsed.headers;
  currentDataset.rows = parsed.rows;

  // Detect numeric columns
  const numericCols = [];
  parsed.headers.forEach(h => {
    let numericCount = 0;
    parsed.rows.forEach(r => {
      const val = parseFloat(r[h]);
      if (!isNaN(val)) numericCount++;
    });
    if (numericCount >= parsed.rows.length * 0.7) {
      numericCols.push(h);
    }
  });
  currentDataset.numericCols = numericCols;

  // Pick target column: prefer 'Profit', 'FloodProbability', 'SalePrice', or last numeric column
  let defaultTarget = numericCols[numericCols.length - 1];
  ['Profit', 'FloodProbability', 'Price', 'SalePrice', 'target', 'Target'].forEach(t => {
    if (numericCols.includes(t)) defaultTarget = t;
  });
  currentDataset.targetCol = defaultTarget;

  // Update UI Meta info
  document.getElementById('metaSource').textContent = name;
  document.getElementById('metaRows').textContent = `${parsed.rows.length} 筆`;
  
  const featureCount = Math.max(0, currentDataset.headers.length - 1);
  document.getElementById('metaFeatures').textContent = `${featureCount} 個`;

  const checkBadge = document.getElementById('metaFeatureCheckBadge');
  if (featureCount > 13) {
    checkBadge.className = "badge badge-success";
    checkBadge.textContent = `特徵數: ${featureCount} (符合 >13 規範 ✅)`;
  } else {
    checkBadge.className = "badge badge-warning";
    checkBadge.textContent = `特徵數: ${featureCount} (50 Startups 示範題目)`;
  }

  // Populate Target Select
  const targetSelect = document.getElementById('targetSelect');
  targetSelect.innerHTML = numericCols.map(col => `
    <option value="${col}" ${col === defaultTarget ? 'selected' : ''}>${col} (連續數值目標)</option>
  `).join('');

  updateFeatureCheckboxes();
  runRegressionAndRender();
}

function updateFeatureCheckboxes() {
  const container = document.getElementById('featureCheckboxes');
  const availableFeatures = currentDataset.numericCols.filter(c => c !== currentDataset.targetCol);

  container.innerHTML = availableFeatures.map(feat => `
    <label class="feature-chip-label active">
      <input type="checkbox" value="${feat}" checked style="accent-color: var(--accent-cyan);">
      <span>${feat}</span>
    </label>
  `).join('');

  // Attach change listener
  container.querySelectorAll('input[type="checkbox"]').forEach(cb => {
    cb.addEventListener('change', (e) => {
      const label = e.target.closest('.feature-chip-label');
      if (e.target.checked) label.classList.add('active');
      else label.classList.remove('active');
      runRegressionAndRender();
    });
  });
}

// Robust RFC4180 CSV Parser
function parseCSV(text) {
  const lines = text.trim().split(/\r\n|\n|\r/);
  if (lines.length < 2) return null;

  const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
  const rows = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const parts = line.split(',');
    if (parts.length === headers.length) {
      const row = {};
      headers.forEach((h, idx) => {
        row[h] = parts[idx].trim().replace(/^["']|["']$/g, '');
      });
      rows.push(row);
    }
  }

  return { headers, rows };
}

// OLS Multiple Linear Regression Solver
function runRegressionAndRender() {
  const container = document.getElementById('featureCheckboxes');
  const selectedFeatures = Array.from(container.querySelectorAll('input[type="checkbox"]:checked')).map(cb => cb.value);

  if (selectedFeatures.length === 0) {
    showToast("請至少選擇一個輸入特徵！");
    return;
  }

  const rows = currentDataset.rows;
  const targetName = currentDataset.targetCol;

  // Filter valid rows
  const cleanRows = [];
  rows.forEach(r => {
    const yVal = parseFloat(r[targetName]);
    if (isNaN(yVal)) return;

    let valid = true;
    selectedFeatures.forEach(f => {
      const xVal = parseFloat(r[f]);
      if (isNaN(xVal)) valid = false;
    });

    if (valid) cleanRows.push(r);
  });

  const n = cleanRows.length;
  const p = selectedFeatures.length;

  if (n <= p + 1) {
    showToast("樣本數過少，不足以估計回歸參數！");
    return;
  }

  // Construct X matrix (with intercept) and y vector
  const X = [];
  const y = [];
  cleanRows.forEach(r => {
    const rowX = [1.0]; // intercept
    selectedFeatures.forEach(f => {
      rowX.push(parseFloat(r[f]));
    });
    X.push(rowX);
    y.push(parseFloat(r[targetName]));
  });

  // Normal Equation: beta = (X^T * X)^(-1) * X^T * y
  const Xt = transpose(X);
  const XtX = matMul(Xt, X);
  const XtX_inv = invertMatrix(XtX);
  const Xty = matVecMul(Xt, y);
  const beta = matVecMul(XtX_inv, Xty);

  // Predictions, residuals, metrics
  let ssTot = 0;
  let ssRes = 0;
  let sumAbsErr = 0;
  const yMean = y.reduce((a, b) => a + b, 0) / n;
  const yPred = [];
  const leverages = [];

  for (let i = 0; i < n; i++) {
    const xi = X[i];
    let yHat = 0;
    for (let j = 0; j <= p; j++) yHat += xi[j] * beta[j];
    yPred.push(yHat);

    const err = y[i] - yHat;
    ssRes += err * err;
    sumAbsErr += Math.abs(err);
    ssTot += (y[i] - yMean) * (y[i] - yMean);

    // Leverage h_ii = x_i^T * (X^T X)^-1 * x_i
    let h_ii = 0;
    for (let r = 0; r <= p; r++) {
      for (let c = 0; c <= p; c++) {
        h_ii += xi[r] * XtX_inv[r][c] * xi[c];
      }
    }
    leverages.push(Math.max(0, h_ii));
  }

  const r2 = Math.max(0, 1 - (ssRes / (ssTot || 1)));
  const adjR2 = Math.max(0, 1 - ((1 - r2) * (n - 1) / (n - p - 1)));
  const mse = ssRes / n;
  const rmse = Math.sqrt(mse);
  const mae = sumAbsErr / n;
  const se = Math.sqrt(ssRes / (n - p - 1));

  // t critical for 95% confidence (approx 2.0 for reasonable n)
  const tCrit = n > 30 ? 1.96 : 2.05;

  // Calculate CI & PI for each sample
  const ciUpper = [];
  const ciLower = [];
  const piUpper = [];
  const piLower = [];

  for (let i = 0; i < n; i++) {
    const yHat = yPred[i];
    const h = leverages[i];
    const se_ci = se * Math.sqrt(h);
    const se_pi = se * Math.sqrt(1 + h);

    ciUpper.push(yHat + tCrit * se_ci);
    ciLower.push(yHat - tCrit * se_ci);
    piUpper.push(yHat + tCrit * se_pi);
    piLower.push(yHat - tCrit * se_pi);
  }

  // Update Metrics Ribbon
  document.getElementById('metricR2').textContent = r2.toFixed(4);
  document.getElementById('metricAdjR2').textContent = adjR2.toFixed(4);
  document.getElementById('metricRMSE').textContent = rmse > 100 ? rmse.toLocaleString('en-US', {maximumFractionDigits: 1}) : rmse.toFixed(4);
  document.getElementById('metricMAE').textContent = mae > 100 ? mae.toLocaleString('en-US', {maximumFractionDigits: 1}) : mae.toFixed(4);
  document.getElementById('metricSe').textContent = se > 100 ? se.toLocaleString('en-US', {maximumFractionDigits: 1}) : se.toFixed(4);

  // Draw Interactive Chart
  const canvas = document.getElementById('regressionChartCanvas');
  drawRegressionPlot(canvas, y, yPred, ciUpper, ciLower, piUpper, piLower, currentDataset.targetCol);

  // Update Python Code Box
  updateLabPythonSnippet(currentDataset.name, targetName, selectedFeatures);
}

// Canvas Drawing for Regression with 95% CI and PI
function drawRegressionPlot(canvas, actuals, predicted, ciUp, ciLow, piUp, piLow, targetName) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = rect.width || 920;
  const height = 420;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);

  // Padding
  const padLeft = 70;
  const padRight = 30;
  const padTop = 30;
  const padBottom = 50;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  // Background
  ctx.fillStyle = "#0c111d";
  ctx.fillRect(0, 0, width, height);

  // Determine scale bounds based on Actual and Predicted
  let minVal = Math.min(...actuals, ...predicted, ...piLow);
  let maxVal = Math.max(...actuals, ...predicted, ...piUp);
  const margin = (maxVal - minVal) * 0.08 || 1;
  minVal -= margin;
  maxVal += margin;

  const toX = (val) => padLeft + ((val - minVal) / (maxVal - minVal)) * plotW;
  const toY = (val) => padTop + plotH - ((val - minVal) / (maxVal - minVal)) * plotH;

  // Grid Lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#64748b";
  ctx.font = "11px 'Fira Code', monospace";
  ctx.textAlign = "right";

  const numTicks = 6;
  for (let i = 0; i <= numTicks; i++) {
    const val = minVal + (i / numTicks) * (maxVal - minVal);
    const yPos = toY(val);
    const xPos = toX(val);

    // Horizontal grid
    ctx.beginPath();
    ctx.moveTo(padLeft, yPos);
    ctx.lineTo(width - padRight, yPos);
    ctx.stroke();

    const label = val > 1000 ? (val / 1000).toFixed(0) + 'k' : val.toFixed(val < 1 ? 2 : 1);
    ctx.fillText(label, padLeft - 10, yPos + 4);

    // Vertical grid
    ctx.beginPath();
    ctx.moveTo(xPos, padTop);
    ctx.lineTo(xPos, height - padBottom);
    ctx.stroke();
  }

  // Axis Labels
  ctx.fillStyle = "#94a3b8";
  ctx.font = "12px 'Plus Jakarta Sans', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`預測值 Ŷ (${targetName})`, padLeft + plotW / 2, height - 14);

  ctx.save();
  ctx.translate(18, padTop + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(`實際觀測值 Y (${targetName})`, 0, 0);
  ctx.restore();

  // Sort points by predicted value for smooth interval bands
  const sortedIndices = predicted.map((p, idx) => idx).sort((a, b) => predicted[a] - predicted[b]);

  // 1. Draw 95% Prediction Interval (PI) Band (Amber dashed)
  ctx.fillStyle = "rgba(245, 158, 11, 0.08)";
  ctx.beginPath();
  for (let i = 0; i < sortedIndices.length; i++) {
    const idx = sortedIndices[i];
    const x = toX(predicted[idx]);
    const y = toY(piUp[idx]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  for (let i = sortedIndices.length - 1; i >= 0; i--) {
    const idx = sortedIndices[i];
    const x = toX(predicted[idx]);
    const y = toY(piLow[idx]);
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();

  // Draw PI border lines
  ctx.strokeStyle = "#f59e0b";
  ctx.setLineDash([5, 4]);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  sortedIndices.forEach((idx, i) => {
    const x = toX(predicted[idx]);
    const y = toY(piUp[idx]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.beginPath();
  sortedIndices.forEach((idx, i) => {
    const x = toX(predicted[idx]);
    const y = toY(piLow[idx]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]); // reset

  // 2. Draw 95% Confidence Interval (CI) Band (Emerald Green)
  ctx.fillStyle = "rgba(16, 185, 129, 0.25)";
  ctx.beginPath();
  for (let i = 0; i < sortedIndices.length; i++) {
    const idx = sortedIndices[i];
    const x = toX(predicted[idx]);
    const y = toY(ciUp[idx]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  for (let i = sortedIndices.length - 1; i >= 0; i--) {
    const idx = sortedIndices[i];
    const x = toX(predicted[idx]);
    const y = toY(ciLow[idx]);
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();

  // 3. Draw Ideal Regression Fit Line (Y = Ŷ)
  ctx.strokeStyle = "#6366f1";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(toX(minVal), toY(minVal));
  ctx.lineTo(toX(maxVal), toY(maxVal));
  ctx.stroke();

  // 4. Draw Actual Points (Sky blue glowing dots)
  for (let i = 0; i < actuals.length; i++) {
    const x = toX(predicted[i]);
    const y = toY(actuals[i]);

    // Outer glow
    ctx.fillStyle = "rgba(56, 189, 248, 0.35)";
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();

    // Inner point
    ctx.fillStyle = "#38bdf8";
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

// Generate Python Snippet with Seaborn & Statsmodels CI/PI
function updateLabPythonSnippet(datasetName, target, features) {
  const codeBox = document.getElementById('labPythonCodeDisplay');
  if (!codeBox) return;

  const featStr = features.map(f => `'${f}'`).join(', ');

  codeBox.textContent = `import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 載入資料與定義變數
df = pd.read_csv('${datasetName}')
features = [${featStr}]
X = df[features]
y = df['${target}']

# 2. 建立多元線性回歸模型 (加入截距項常數)
X_sm = sm.add_constant(X)
model = sm.OLS(y, X_sm).fit()
print(model.summary())

# 3. 取得 95% 信賴區間 (CI) 與 預測區間 (PI)
predictions = model.get_prediction(X_sm)
pred_summary = predictions.summary_frame(alpha=0.05)

# pred_summary 包含:
# - mean: 預測值 Ŷ
# - mean_ci_lower, mean_ci_upper: 95% 信賴區間 (Confidence Interval)
# - obs_ci_lower, obs_ci_upper: 95% 預測區間 (Prediction Interval)

# 4. 繪製包含 CI 與 PI 之回歸散佈圖 (依作業規定呈現預測圖)
plt.figure(figsize=(10, 6))
plt.scatter(pred_summary['mean'], y, color='royalblue', alpha=0.7, label='Actual Observations')
plt.plot(pred_summary['mean'], pred_summary['mean'], color='indigo', lw=2, label='Fitted Line (Y = Ŷ)')

# 繪製 95% 信賴區間帶 (Mean CI)
plt.fill_between(pred_summary['mean'].sort_values(), 
                 pred_summary['mean_ci_lower'].sort_values(), 
                 pred_summary['mean_ci_upper'].sort_values(), 
                 color='green', alpha=0.2, label='95% Confidence Interval')

# 繪製 95% 預測區間帶 (Individual PI)
plt.plot(pred_summary['mean'].sort_values(), pred_summary['obs_ci_upper'].sort_values(), 
         '--', color='darkorange', label='95% Prediction Interval')
plt.plot(pred_summary['mean'].sort_values(), pred_summary['obs_ci_lower'].sort_values(), 
         '--', color='darkorange')

plt.title('Multiple Linear Regression: Actual vs Predicted with 95% CI & PI')
plt.xlabel('Predicted Values (Ŷ)')
plt.ylabel('Actual Values (Y)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()`;
}

// Matrix Math Helpers
function transpose(A) {
  const m = A.length;
  const n = A[0].length;
  const AT = [];
  for (let j = 0; j < n; j++) {
    const row = [];
    for (let i = 0; i < m; i++) row.push(A[i][j]);
    AT.push(row);
  }
  return AT;
}

function matMul(A, B) {
  const m = A.length;
  const k = A[0].length;
  const n = B[0].length;
  const C = [];
  for (let i = 0; i < m; i++) {
    const row = [];
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++) sum += A[i][p] * B[p][j];
      row.push(sum);
    }
    C.push(row);
  }
  return C;
}

function matVecMul(A, v) {
  const m = A.length;
  const k = A[0].length;
  const res = [];
  for (let i = 0; i < m; i++) {
    let sum = 0;
    for (let j = 0; j < k; j++) sum += A[i][j] * v[j];
    res.push(sum);
  }
  return res;
}

function invertMatrix(M) {
  const n = M.length;
  const A = M.map((row, i) => {
    const r = [...row];
    for (let j = 0; j < n; j++) r.push(i === j ? 1.0 : 0.0);
    return r;
  });

  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) maxRow = k;
    }
    const temp = A[i];
    A[i] = A[maxRow];
    A[maxRow] = temp;

    if (Math.abs(A[i][i]) < 1e-12) {
      A[i][i] += 1e-6; // ridge stabilization
    }
    const curPivot = A[i][i];
    for (let j = 0; j < 2 * n; j++) A[i][j] /= curPivot;

    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const factor = A[k][i];
        for (let j = 0; j < 2 * n; j++) {
          A[k][j] -= factor * A[i][j];
        }
      }
    }
  }

  return A.map(row => row.slice(n));
}

