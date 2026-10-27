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
});

// Theme Toggle
function initThemeToggle() {
  const toggleBtn = document.getElementById('themeToggleBtn');
  const moonIcon = document.getElementById('moonIcon');
  const sunIcon = document.getElementById('sunIcon');
  const html = document.documentElement;

  // Restore saved theme
  const savedTheme = localStorage.getItem('mlr_theme') || 'dark';
  html.setAttribute('data-theme', savedTheme);
  updateThemeIcons(savedTheme);

  toggleBtn.addEventListener('click', () => {
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('mlr_theme', newTheme);
    updateThemeIcons(newTheme);
  });

  function updateThemeIcons(theme) {
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
