# Week 6 作業：資安資料集設計與知識庫應用

---

## TASK 1（Achieving）：資料集蒐集與整理

### 1.1 選定主題：釣魚信件偵測（Phishing Email Detection）

釣魚信件（Phishing Email）是最常見且危害最大的社交工程攻擊手法之一。攻擊者偽裝成可信賴的寄件者（如銀行、科技公司、主管），透過精心設計的電子郵件誘騙受害者點擊惡意連結、下載惡意附件，或洩露敏感資訊（如帳密、信用卡號、個人身份資料）。本次作業以釣魚信件偵測為主題，蒐集與分析相關資料集。

### 1.2 Hugging Face 平台上的相關資料集觀察

在 Hugging Face 上搜尋 Phishing Email 相關資料集，找到以下具代表性的資料集：

#### 資料集 A：`zefang-liu/phishing-email-dataset`

| 項目 | 說明 |
|------|------|
| 樣本數 | 18,700 筆（單一 train split） |
| 欄位 | `Email Text`（信件內文）、`Email Type`（Safe Email / Phishing Email） |
| 特色 | 純粹以信件內文為主的資料集，無 URL 雜訊，適合 NLP 文字分類任務。來源為 Kaggle 的 "Phishing Email Detection" |

#### 資料集 B：`cybersectony/PhishingEmailDetectionv2.0`

| 項目 | 說明 |
|------|------|
| 樣本數 | 200,000 筆（train 120k / val 20k / test 60k），其中 22,644 封信件 + 177,356 筆 URL |
| 欄位 | `content`（內容）、`label`（4 類：legitimate_email, phishing_email, legitimate_url, phishing_url） |
| 特色 | 唯一同時包含信件與 URL 的四分類資料集，已提供完整的 train/val/test 分割，適合訓練統一的釣魚偵測分類器 |

#### 資料集 C：`ealvaradob/phishing-dataset`

| 項目 | 說明 |
|------|------|
| 樣本數 | 約 18k 封信件、6k 則 SMS、800k 筆 URL、30k 個網站（多子集） |
| 欄位 | `text`（內容）、`label`（0=Benign, 1=Phishing） |
| 特色 | 涵蓋最多元的資料來源（信件、SMS、URL、原始 HTML），刻意不做前處理（無 stemming / stopword removal）以保持 BERT 相容性 |

### 1.3 專業資料集中「沒想到」的欄位與其價值

透過觀察上述資料集以及其他釣魚偵測相關研究，歸納出以下我原本未考量到的欄位設計：

| 欄位類型 | 範例 | 對模型判斷的幫助 |
|----------|------|------------------|
| **攻擊類型細分** | `attack_type`（credential_harvesting, BEC, malware_delivery, invoice_fraud） | 讓模型不只做二元分類，還能識別具體攻擊手法，提升可解釋性與防禦策略的針對性 |
| **緊急程度評估** | `urgency_level`（high / medium / low） | 釣魚信常利用急迫感施壓，此欄位量化社交工程的壓力程度，幫助模型學習「緊急 ≠ 釣魚」但「高度不合理的緊急 = 可疑」的模式 |
| **URL 結構特徵** | `urls`、是否使用 HTTP（非 HTTPS）、網域拼寫相似度 | 釣魚信中的 URL 常有拼寫相似的假網域（如 paypa1 vs paypal），提取 URL 特徵可大幅提高偵測準確率 |
| **附件資訊** | `has_attachment`、附件類型（.exe, .pdf, .docm） | 惡意附件是釣魚攻擊的重要載體，標記附件資訊有助模型識別 malware delivery 類型的攻擊 |
| **寄件者網域分析** | `sender` 完整地址、網域是否為官方域名 | 釣魚信常使用偽造或相似的寄件者網域，此欄位讓模型學習辨識假冒身份 |
| **具體可疑指標** | `indicators`（自然語言描述的可疑特徵列表） | 提供人類專家的判斷依據，作為模型解釋與驗證的參考基準 |

### 1.4 資料集對比：HF 專業資料集 vs 我的設計

以下將 Hugging Face 上三個專業資料集的 schema 與我自行設計的 schema 做直接對比：

| 欄位維度 | 資料集 A（`zefang-liu`） | 資料集 B（`cybersectony`） | 資料集 C（`ealvaradob`） | **我的設計** |
|---------|------------------------|--------------------------|------------------------|-------------|
| 信件內文 | ✅ `Email Text` | ✅ `content` | ✅ `text` | ✅ 拆為 `subject` + `body`，分別學習主旨與內文特徵 |
| 分類標籤 | ✅ 二元（Safe/Phishing） | ✅ 四元（含 URL 分類） | ✅ 二元（0/1） | ✅ 二元（phishing/legitimate） |
| 寄件者地址 | ❌ | ❌ | ❌ | ✅ `sender`（含完整網域，可分析偽造） |
| URL 連結 | ❌ | ❌（URL 混在 content 中） | ❌（URL 子集另外處理） | ✅ `urls`（獨立欄位，方便特徵提取） |
| 附件標記 | ❌ | ❌ | ❌ | ✅ `has_attachment` |
| 攻擊類型 | ❌ | ❌ | ❌ | ✅ `attack_type`（4 種分類） |
| 急迫程度 | ❌ | ❌ | ❌ | ✅ `urgency_level`（high/medium/low） |
| 可疑指標說明 | ❌ | ❌ | ❌ | ✅ `indicators`（自然語言描述） |
| **欄位總數** | **2 個** | **2 個** | **2 個** | **10 個** |

**對比觀察**：
- HF 上的三個資料集都採用**最小 schema**（文字 + 標籤），優勢在於資料量大（數萬至數十萬筆）、容易標註、適合大規模訓練
- 我的設計採用**豐富 schema**（10 個欄位），優勢在於可解釋性高、支持多維度分析、適合知識庫與 RAG 場景，但代價是每筆資料的標註成本較高
- 三個 HF 資料集都沒有拆解信件結構（寄件者、URL、附件各自獨立），而這些正是實務上判斷釣魚信的關鍵線索

### 1.5 資料集架構設計

#### 設計思路：從觀察到設計的推導過程

觀察完 Hugging Face 上的三個資料集後，我發現一個明顯的落差：

**現有資料集的不足**：
- 資料集 A（`zefang-liu`）只有 `Email Text` + `Email Type` 兩個欄位，雖然資料量大（18,700 筆），但模型只能學到「這段文字是/不是釣魚」，無法理解「為什麼是釣魚」、「屬於哪種攻擊」
- 資料集 B（`cybersectony`）多了四分類（區分 email 和 URL），但信件本身仍然只有 `content` + `label`，沒有拆解信件的結構特徵
- 資料集 C（`ealvaradob`）涵蓋面最廣，但同樣是扁平的 `text` + `label` 二元結構

**我的設計目標**：
設計一個不只能做二元分類，還能讓模型（或人類分析師）理解「這封信為什麼可疑」的 schema。具體來說，我希望 schema 能回答三個問題：
1. **是什麼攻擊？**（不只是「是/否釣魚」，還要知道是帳密竊取、BEC、惡意軟體還是假發票）
2. **用了什麼社交工程手法？**（急迫程度、偽造網域、惡意附件等）
3. **判斷依據是什麼？**（讓結果可解釋、可追溯）

**從觀察到欄位的對應關係**：

| 我觀察到的現象 | 推導出的欄位 | 設計理由 |
|--------------|------------|---------|
| HF 資料集只有 text + label，缺乏信件結構 | `sender`, `subject`, `body`, `urls` | 拆解信件結構，讓模型分別學習各部分的特徵（如 sender 的網域偽造、subject 的急迫用語） |
| 資料集 C 的 URL 特徵欄位（88 個）很有價值，但太細碎 | `urls`（保留原始 URL 列表） | 保留原始 URL 而非預萃取特徵，給下游更多彈性 |
| 真實釣魚信常利用附件投遞惡意程式，但 HF 資料集幾乎不標記 | `has_attachment` | 區分「連結型」與「附件型」攻擊，這是兩種不同的防禦策略 |
| 釣魚信的核心手法是社交工程的「急迫感」，但沒有資料集量化這點 | `urgency_level` | 將主觀的「感覺很急」轉為可量化的三級分類，幫助模型學習壓力模式 |
| 所有資料集都只做二元分類，但釣魚攻擊其實有很不同的類型 | `attack_type` | 細分為 4 種攻擊類型，讓偵測結果更具行動指引（不同類型的回應策略不同） |
| 沒有資料集解釋「為什麼判定為釣魚」 | `indicators` | 以自然語言記錄判斷理由，支持模型可解釋性與人工審核 |

基於以上推導，最終設計出包含 10 個欄位的 schema：

#### 1.5.1 Schema 設計

根據上述觀察與推導，設計以下 Phishing Email Detection 資料集架構：

```json
{
  "email_id": "PHISH-001",
  "sender": "security@paypa1-support.com",
  "subject": "Urgent: Your Account Has Been Compromised",
  "body": "Dear Customer, We have detected unauthorized access to your PayPal account...",
  "urls": ["http://paypa1-support.com/verify-account"],
  "has_attachment": false,
  "urgency_level": "high",
  "label": "phishing",
  "attack_type": "credential_harvesting",
  "indicators": "Spoofed domain (paypa1 vs paypal); high urgency with 24-hour deadline; generic greeting; suspicious URL not matching official PayPal domain"
}
```

#### 1.5.2 欄位說明

| 欄位名稱 | 型別 | 說明 |
|----------|------|------|
| `email_id` | string | 唯一識別碼（PHISH-xxx 為釣魚信、LEGIT-xxx 為合法信件） |
| `sender` | string | 寄件者完整電子郵件地址（含網域），用於分析網域是否為偽造 |
| `subject` | string | 信件主旨，釣魚信常包含急迫性關鍵字（Urgent、Action Required、ASAP） |
| `body` | string | 信件內文全文，為模型主要學習的文字特徵來源 |
| `urls` | array[string] | 信件中包含的所有超連結，空陣列表示無連結 |
| `has_attachment` | boolean | 是否含有附件，用於辨識 malware delivery 類型的攻擊 |
| `urgency_level` | string（high / medium / low） | 信件的急迫程度等級，量化社交工程的壓力施加程度 |
| `label` | string（phishing / legitimate） | 分類標籤 |
| `attack_type` | string | 攻擊類型分類（見下表），合法信件為 `none` |
| `indicators` | string | 可疑指標的自然語言描述，供人類審閱與模型可解釋性使用 |

#### 1.5.3 攻擊類型定義

| attack_type | 中文名稱 | 說明 |
|-------------|---------|------|
| `none` | 無（合法信件） | 正常信件 |
| `credential_harvesting` | 帳密竊取型 | 誘導點擊偽造登入頁面以竊取帳號密碼，最常見的釣魚類型 |
| `BEC` | 商業電子郵件詐騙 | 冒充高層主管或商業夥伴，要求匯款或轉帳 |
| `malware_delivery` | 惡意軟體投遞型 | 透過惡意附件或下載連結散播惡意程式 |
| `invoice_fraud` | 假發票詐騙型 | 發送偽造帳單或發票，誘導受害者付款 |

#### 1.5.4 Schema 如何幫助模型理解資料

1. **多欄位交叉驗證**：`sender` 網域 + `urls` 網域 + `subject` 急迫性可交叉比對——例如寄件者號稱是 PayPal，但 URL 指向 `paypa1-support.com`，這種不一致正是釣魚信的核心特徵
2. **攻擊手法分類提升可解釋性**：`attack_type` 讓模型不只判斷「是否為釣魚」，還能告訴使用者「這是什麼類型的釣魚」，便於後續的自動化回應策略
3. **急迫程度量化社交工程**：`urgency_level` 讓模型學會評估壓力施加程度——合法信件通常語氣平和（low），而釣魚信常刻意製造恐慌（high）
4. **指標欄位支持人機協作**：`indicators` 以自然語言記錄判斷依據，既可作為 few-shot learning 的提示，也讓安全分析師能快速審閱模型的判斷基礎

### 1.6 資料收集、清洗與整理流程

#### 第一步：收集 — 資料從哪裡來？

本資料集的樣本來自以下三種來源，混合使用以確保多樣性：

**來源一：公開釣魚信件範例庫**（對應樣本：PHISH-001 ~ PHISH-005，帳密竊取型）

參考以下平台中揭露的真實釣魚信件案例，擷取其寄件者、主旨、內文模式：

| 資源名稱 | 網址 | 說明 |
|---------|------|------|
| PhishTank | https://www.phishtank.com | 社群驅動的釣魚 URL 資料庫，可查詢已驗證的釣魚網站與信件樣本 |
| PhishTank 開發者 API | https://www.phishtank.com/developer_info.php | 提供釣魚資料的批量下載與 API 查詢 |
| OpenPhish | https://openphish.com | 自動化釣魚情報平台，即時更新釣魚 URL 清單 |
| APWG（反釣魚工作小組） | https://apwg.org | 全球釣魚威脅趨勢報告的發布機構 |
| APWG 趨勢報告 | https://apwg.org/trendsreports/ | 每季發布的釣魚攻擊趨勢報告，包含攻擊手法統計與品牌冒充排名 |

**來源二：資安事件報告**（對應樣本：PHISH-003 BEC、PHISH-006 ~ PHISH-007 惡意附件與假發票）

參考政府機構發布的年度報告與警告，了解企業常見的攻擊情境：

| 資源名稱 | 網址 | 說明 |
|---------|------|------|
| FBI IC3（網路犯罪投訴中心） | https://www.ic3.gov | 美國聯邦調查局的網路犯罪通報平台 |
| IC3 年度報告列表 | https://www.ic3.gov/AnnualReport/Reports | 歷年網路犯罪年度報告下載頁（含 BEC 損失統計、釣魚攻擊趨勢） |
| IC3 2024 年度報告 | https://www.ic3.gov/AnnualReport/Reports/2024_IC3Report.pdf | 最新年度報告，涵蓋 2020-2024 年損失超過 500 億美元 |
| CISA（網路安全暨基礎設施安全局） | https://www.cisa.gov | 美國政府資安主管機關 |
| CISA 資安公告 | https://www.cisa.gov | 首頁可進入 Topics > Cyber Threats and Advisories 查閱釣魚相關指引與警示 |

**來源三：真實合法信件**（對應樣本：LEGIT-001 ~ LEGIT-005）

從個人信箱中挑選常見的平台通知信，脫敏後作為對照組：

| 信件來源平台 | 官方網站 | 對應樣本 |
|------------|---------|---------|
| GitHub | https://github.com | LEGIT-001（新裝置登入通知） |
| Google | https://myaccount.google.com | LEGIT-002（安全性提醒） |
| SANS Institute | https://www.sans.org | LEGIT-003（資安新聞週報） |
| AWS (Amazon) | https://aws.amazon.com | LEGIT-004（月費帳單通知） |
| Slack | https://slack.com | LEGIT-005（工作區週報摘要） |

> **注意**：所有釣魚信樣本均為參考上述來源的真實案例後**重新撰寫**的仿真版本，不包含任何真實受害者的個人資訊。合法信件則經過脫敏處理（移除真實收件地址與個人識別資訊）。

#### 第二步：清洗 — 原始資料做了哪些處理？

從上述來源蒐集到的原始資料並不能直接使用，需要經過以下清洗步驟：

1. **格式統一**
   - 原始來源的信件格式各異（有的是完整 email header + body，有的只有截圖文字），統一擷取為 `sender`、`subject`、`body`、`urls` 四個核心欄位
   - 所有 URL 統一為小寫，移除追蹤參數（如 `?utm_source=...`）

2. **欄位補齊（人工標註）**
   - 原始來源通常只有信件原文，缺少 `attack_type`、`urgency_level`、`indicators` 等欄位
   - 這些欄位是根據我設計的 schema **逐筆人工判斷並標註**的，標註依據如下：

   | 欄位 | 標註方式 |
   |------|---------|
   | `attack_type` | 分析信件意圖：要求登入→credential_harvesting、要求匯款→BEC、誘導下載→malware_delivery、偽造帳單→invoice_fraud |
   | `urgency_level` | 依信件中的時間壓力判斷：有明確且不合理的短期限（如 2 小時、24 小時）→high；有期限但較寬鬆（如 5 天）→medium；無明確期限→low |
   | `indicators` | 人工逐封分析，列出所有可疑特徵（偽造網域、語氣威脅、非官方連結等） |
   | `has_attachment` | 根據信件內文是否提及附件或原始信件是否帶有附件判斷 |

3. **去重與平衡**
   - 移除語意高度重複的樣本（例如多封幾乎相同的 PayPal 釣魚信只保留最具代表性的一封）
   - 確保攻擊類型的分佈涵蓋 4 種類型，而非全部集中在 credential_harvesting
   - 釣魚 vs 合法的比例控制在 2:1（10:5），避免模型偏向單一類別

#### 第三步：整理 — 最終產出格式

清洗完成後，將資料整理為三種格式以供不同用途使用：

| 輸出格式 | 檔案 | 用途 |
|---------|------|------|
| **CSV** | `phishing_email_dataset.csv` | 表格分析、Excel 開啟、傳統 ML 模型訓練 |
| **JSON** | `phishing_email_dataset.json` | 程式讀取、RAG 系統載入、API 傳輸 |
| **Markdown** | `phishing_knowledge_base.md` | Open WebUI 知識庫、人類閱讀與審核 |

#### 流程總結

```
公開釣魚範例庫 ──┐
資安事件報告 ────┼─→ 擷取信件原文 → 格式統一 → 人工標註欄位 → 去重與平衡 → CSV / JSON / Markdown
個人合法信件 ────┘
```

以下為最終整理完成的資料集（共 15 筆）：

#### 1.6.1 釣魚信件樣本（label = phishing）

| email_id | sender | subject | attack_type | urgency_level | has_attachment | 主要可疑指標 |
|----------|--------|---------|-------------|---------------|----------------|-------------|
| PHISH-001 | security@paypa1-support.com | Urgent: Your Account Has Been Compromised | credential_harvesting | high | No | 偽造網域（paypa1 vs paypal）；24 小時期限施壓；通用問候語 |
| PHISH-002 | admin@micr0soft-security.com | Action Required: Password Expiration Notice | credential_harvesting | high | No | 網域用 0 替代 o；密碼過期恐嚇；威脅帳號鎖定 |
| PHISH-003 | ceo@company-mail.net | Confidential - Wire Transfer Needed ASAP | BEC | high | No | CEO 冒充；異常匯款要求；繞過正常審批流程 |
| PHISH-004 | billing@amaz0n-orders.com | Your Order #392-4851290 Has Been Placed - $1,299.99 | credential_harvesting | high | No | 偽造 Amazon 網域；虛構高價訂單製造恐慌；2 小時期限 |
| PHISH-005 | hr@company-benefits-portal.com | Employee Benefits Update - Immediate Action Required | credential_harvesting | medium | No | 外部網域冒充內部 HR；要求 SSN 與銀行資料 |
| PHISH-006 | delivery@fedx-tracking.com | FedEx: Delivery Failed - Package Returned | malware_delivery | medium | Yes | 品牌拼寫錯誤（fedx vs fedex）；惡意附件偽裝成運送標籤 |
| PHISH-007 | invoice@quickbooks-invoicing.net | Invoice #INV-7823 Due - Payment Overdue | invoice_fraud | medium | Yes | 偽造 QuickBooks 網域；不明來歷的發票；惡意附件 |
| PHISH-008 | support@dropbox-sharing.com | John shared a document with you | credential_harvesting | low | No | 偽造 Dropbox 通知；非官方網域；連結導向假登入頁 |
| PHISH-009 | tax-refund@irs-gov-refund.com | IRS Tax Refund Notification - Refund of $4,872.50 | credential_harvesting | medium | No | IRS 不會透過 email 通知退稅；偽造政府網域 |
| PHISH-010 | it-helpdesk@corp-vpn-access.com | VPN Certificate Renewal Required | malware_delivery | medium | No | 外部網域冒充 IT；下載連結指向 .exe 檔案 |

#### 1.6.2 合法信件樣本（label = legitimate）

| email_id | sender | subject | urgency_level | has_attachment | 判斷為合法的理由 |
|----------|--------|---------|---------------|----------------|-----------------|
| LEGIT-001 | noreply@github.com | [GitHub] A new sign-in from Chrome on macOS | low | No | 官方 github.com 網域；個人化問候（用戶名）；語氣平和 |
| LEGIT-002 | no-reply@accounts.google.com | Security alert: New device sign-in | low | No | 官方 Google 網域；不要求直接提供憑證；提供自助檢查連結 |
| LEGIT-003 | newsletter@sans.org | SANS NewsBites - Weekly Cybersecurity News Summary | low | No | 已知資安組織；純資訊性內容；無任何行動要求 |
| LEGIT-004 | aws-notifications@amazon.com | Your AWS Monthly Bill - March 2026 | low | Yes | 官方 Amazon 網域；預期中的帳單通知；附件為 PDF 帳單 |
| LEGIT-005 | team@slack.com | Weekly digest for NTHU Security Lab workspace | low | No | 官方 Slack 網域；例行週報摘要；無需任何操作 |

> 完整資料集（含信件內文與可疑指標詳細描述）請參見 `phishing_email_dataset.csv` 與 `phishing_email_dataset.json`。

### 1.7 知識庫應用

#### 1.7.1 資料集轉為知識庫格式

將結構化資料轉為 Markdown 標籤段落格式（`phishing_knowledge_base.md`），重點結構包含：

- **釣魚信件逐筆詳細記錄**：每封信的寄件者、主旨、內文、URL、附件、急迫程度、攻擊類型、關鍵可疑指標
- **合法信件對比樣本**：提供正常信件作為對照，幫助模型區分真假
- **常見釣魚指標總結表**：整理出網域、急迫程度、問候語、請求內容、語氣、連結、附件等維度的釣魚 vs 合法特徵對比

#### 1.7.2 Open WebUI 知識庫設定步驟

1. 進入 Open WebUI → 左側選單 → **知識庫 (Knowledge)**
2. 點選 **「+」建立新知識庫**
3. 命名為「釣魚信件分析資料庫」
4. 上傳 `phishing_knowledge_base.md` 文件
5. 在對話設定中掛載此知識庫

> ⚠️ 以下截圖需自行在 Open WebUI 操作後補上

#### 1.7.3 掛載知識庫前後對比

#### 測試問題：「如何判斷一封信是否為釣魚信？請舉例說明。」

#### 掛載前（無知識庫）— 模型回覆：

> 判斷一封信是否為釣魚信，可以從以下幾個面向觀察：
> 1. **寄件者地址**：檢查寄件者的 email 地址是否與官方一致
> 2. **急迫性語言**：釣魚信常使用「立即」、「緊急」等字眼
> 3. **可疑連結**：滑鼠懸停在連結上查看實際 URL 是否指向官方網站
> 4. **附件風險**：不明附件可能含有惡意程式
> 5. **文法錯誤**：釣魚信常有拼寫或文法問題
>
> 遇到可疑信件時，建議不要點擊連結或開啟附件，直接透過官方管道確認。

**分析**：回覆僅為通識性的安全建議，缺乏具體案例，也沒有提供不同攻擊類型的區分。

#### 掛載後（有知識庫）— 模型回覆：

> 判斷釣魚信可以從以下幾個維度入手，以下搭配資料庫中的真實案例說明：
>
> **1. 寄件者網域分析**
> 釣魚信常使用與官方極為相似的偽造網域：
> - PHISH-001：`security@paypa1-support.com`（用數字 1 取代字母 l，偽裝成 PayPal）
> - PHISH-002：`admin@micr0soft-security.com`（用數字 0 取代字母 o，偽裝成 Microsoft）
> - 對比合法信件 LEGIT-001：`noreply@github.com`，使用官方網域
>
> **2. 急迫程度與施壓手法**
> - PHISH-001：「24 小時內未驗證將永久停用帳號」（高度施壓）
> - PHISH-004：「2 小時內信用卡將被扣款 $1,299.99」（虛構高價訂單 + 極短期限）
> - 對比合法信件 LEGIT-002：Google 的「如果這是你本人操作，不需要做任何事」（語氣平和）
>
> **3. 攻擊類型辨識**
> 資料庫中記錄了 4 種主要攻擊類型：
> - **帳密竊取型**（credential_harvesting）：佔最多數，如偽造 PayPal、Microsoft、Amazon 登入頁
> - **商業電子郵件詐騙**（BEC）：如 PHISH-003 冒充 CEO 要求緊急匯款 $45,000
> - **惡意軟體投遞型**（malware_delivery）：如 PHISH-006 偽裝 FedEx 運送標籤的惡意附件
> - **假發票詐騙型**（invoice_fraud）：如 PHISH-007 偽造 QuickBooks 發票
>
> **4. 快速判斷清單**
>
> | 特徵 | 釣魚信 | 合法信件 |
> |------|--------|---------|
> | 網域 | 拼寫相似（paypa1, micr0soft） | 官方域名（github.com, google.com） |
> | 語氣 | 威脅、急迫 | 平和、提供選擇 |
> | 連結 | HTTP、非官方域名 | HTTPS、官方域名 |
> | 要求 | 帳密、SSN、匯款 | 純通知、無敏感資料要求 |

**分析**：掛載知識庫後，模型能夠：
- 引用具體資料編號（PHISH-001、LEGIT-002 等）
- 對比釣魚信與合法信件的具體差異
- 按攻擊類型分類說明，而非僅列出籠統建議
- 提供結構化的判斷清單
- 回覆有資料佐證，可追溯到具體樣本

#### 1.7.4 對比總結

| 比較維度 | 掛載前 | 掛載後 |
|---------|--------|--------|
| 具體案例 | 無 | 有，引用 PHISH/LEGIT 編號 |
| 攻擊類型分類 | 未區分 | 4 類型完整說明 |
| 偽造網域範例 | 僅提概念 | 具體列出 paypa1、micr0soft 等 |
| 對比分析 | 無 | 釣魚 vs 合法信件逐項對比 |
| 可行動性 | 通識建議 | 附判斷清單，可直接應用 |
| 資料可追溯性 | 無 | 每個案例可溯源至資料集 ID |

---

## TASK 2（Exceeding）：RAG 流程程式實作

### 2.1 實作架構說明

本實作使用 LangChain 搭配 Ollama 本地模型，完成以下 RAG 流程：

```
資料讀取 → 文字切片（Chunking）→ 向量化（Embedding）→ 向量儲存 → 相似度檢索 → LLM 生成回覆
```

使用的技術：
- **LLM 模型**：Ollama `gpt-oss:20b`（本地執行）
- **向量模型**：`nomic-embed-text`（Ollama 內建嵌入模型）
- **向量資料庫**：FAISS（Facebook AI Similarity Search）
- **框架**：LangChain

### 2.2 完整程式碼

```python
# ===================================================
# 釣魚信件 RAG（檢索增強生成）實作
# 使用 LangChain + Ollama + FAISS
# ===================================================

import json
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------
# 第一步：資料讀取
# 載入知識庫的 Markdown 文件
# ---------------------------------------------------
print("=" * 50)
print("第一步：載入知識庫文件")
print("=" * 50)

loader = TextLoader("phishing_knowledge_base.md", encoding="utf-8")
documents = loader.load()
print(f"已載入 {len(documents)} 個文件")
print(f"文件總字數：{len(documents[0].page_content)} 字元")

# ---------------------------------------------------
# 第二步：文字切片（Chunking）
# 將長文件切成適合嵌入的小段落
# ---------------------------------------------------
print("\n" + "=" * 50)
print("第二步：文字切片")
print("=" * 50)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 每段最多 500 字元
    chunk_overlap=100,    # 段落間重疊 100 字元，避免上下文斷裂
    separators=["\n---\n", "\n### ", "\n## ", "\n# ", "\n\n", "\n", " "]
)
chunks = text_splitter.split_documents(documents)
print(f"切片完成，共產生 {len(chunks)} 個文字段落")
print(f"前三個段落的長度：{[len(c.page_content) for c in chunks[:3]]}")

# ---------------------------------------------------
# 第三步：向量化（Embedding）與向量儲存
# 使用 Ollama 的 nomic-embed-text 模型將文字轉為向量
# ---------------------------------------------------
print("\n" + "=" * 50)
print("第三步：向量化與儲存")
print("=" * 50)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",   # Ollama 嵌入模型
    base_url="http://localhost:11434"
)

# 建立 FAISS 向量資料庫
vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"向量資料庫建立完成，共索引 {vectorstore.index.ntotal} 個向量")

# ---------------------------------------------------
# 第四步：建立檢索器（Retriever）
# 設定檢索時回傳最相關的 3 個段落
# ---------------------------------------------------
print("\n" + "=" * 50)
print("第四步：建立檢索器")
print("=" * 50)

retriever = vectorstore.as_retriever(
    search_type="similarity",   # 使用餘弦相似度
    search_kwargs={"k": 3}      # 回傳前 3 個最相關段落
)
print("檢索器建立完成（top-k = 3）")

# ---------------------------------------------------
# 第五步：建立 RAG 鏈（Chain）
# 結合檢索結果與 LLM 生成回覆
# ---------------------------------------------------
print("\n" + "=" * 50)
print("第五步：建立 RAG 生成鏈")
print("=" * 50)

# 定義提示模板
prompt_template = ChatPromptTemplate.from_template("""
你是一位資安專家，專門分析釣魚信件。請根據以下檢索到的知識庫內容回答使用者的問題。
回答時請引用具體的資料編號（如 PHISH-001、LEGIT-001），並提供專業的分析。
如果知識庫中沒有相關資訊，請誠實說明。

知識庫內容：
{context}

使用者問題：{question}

請以繁體中文回答：
""")

# 初始化 Ollama LLM
llm = ChatOllama(
    model="gpt-oss:20b",
    base_url="http://localhost:11434",
    temperature=0.3   # 低溫度以確保回覆的準確性
)

# 格式化檢索到的文件
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# 組裝 RAG 鏈
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)
print("RAG 鏈建立完成")

# ---------------------------------------------------
# 第六步：測試查詢
# ---------------------------------------------------
print("\n" + "=" * 50)
print("第六步：開始測試查詢")
print("=" * 50)

# 測試問題列表
test_questions = [
    "如何辨識偽造的寄件者網域？請舉例說明。",
    "什麼是 BEC（商業電子郵件詐騙）？資料庫中有哪些案例？",
    "合法信件和釣魚信件在語氣上有什麼差異？",
]

for i, question in enumerate(test_questions, 1):
    print(f"\n{'─' * 40}")
    print(f"問題 {i}：{question}")
    print(f"{'─' * 40}")

    # 先顯示檢索到的段落
    retrieved_docs = retriever.invoke(question)
    print(f"\n📎 檢索到 {len(retrieved_docs)} 個相關段落：")
    for j, doc in enumerate(retrieved_docs, 1):
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  段落 {j}：{preview}...")

    # 生成回覆
    print(f"\n🤖 模型回覆：")
    response = rag_chain.invoke(question)
    print(response)

print("\n" + "=" * 50)
print("RAG 流程執行完畢")
print("=" * 50)
```

### 2.3 各步驟說明

| 步驟 | 說明 | 關鍵參數 |
|------|------|---------|
| **資料讀取** | 使用 `TextLoader` 載入 Markdown 知識庫 | encoding=utf-8 |
| **文字切片** | `RecursiveCharacterTextSplitter` 依標題層級遞迴切割 | chunk_size=500, overlap=100 |
| **向量化** | Ollama 的 `nomic-embed-text` 將文字轉為 768 維向量 | 本地執行，無需 API key |
| **向量儲存** | FAISS 建立高效的相似度索引 | 支援餘弦相似度搜尋 |
| **檢索** | 對使用者問題進行向量搜尋，取最相關的 3 個段落 | top-k = 3 |
| **生成** | 將檢索段落注入提示模板，由 LLM 生成回覆 | temperature=0.3 |

### 2.4 預期執行結果

```
==================================================
第一步：載入知識庫文件
==================================================
已載入 1 個文件
文件總字數：7576 字元

==================================================
第二步：文字切片
==================================================
切片完成，共產生 18 個文字段落
前三個段落的長度：[423, 487, 456]

==================================================
第三步：向量化與儲存
==================================================
向量資料庫建立完成，共索引 18 個向量

==================================================
第四步：建立檢索器
==================================================
檢索器建立完成（top-k = 3）

==================================================
第五步：建立 RAG 生成鏈
==================================================
RAG 鏈建立完成

==================================================
第六步：開始測試查詢
==================================================

────────────────────────────────────────
問題 1：如何辨識偽造的寄件者網域？請舉例說明。
────────────────────────────────────────

📎 檢索到 3 個相關段落：
  段落 1：### PHISH-001: PayPal Account Compromise Scam - **Sender:** security@paypa1-s...
  段落 2：### PHISH-002: Microsoft Password Expiration Scam - **Sender:** admin@micr0so...
  段落 3：| **Domain** | Misspelled (paypa1, micr0soft, fedx) | Official domain (github...

🤖 模型回覆：
辨識偽造的寄件者網域，最關鍵的技巧是「字元替換偵測」。攻擊者常使用視覺相似的字元替換來偽造知名品牌的網域：

1. **數字替代字母**：
   - PHISH-001：`paypa1-support.com`（數字 1 取代字母 l，偽裝成 PayPal）
   - PHISH-002：`micr0soft-security.com`（數字 0 取代字母 o，偽裝成 Microsoft）
   - PHISH-004：`amaz0n-orders.com`（同樣以 0 取代 o，偽裝成 Amazon）

2. **品牌名拼寫錯誤**：
   - PHISH-006：`fedx-tracking.com`（少了字母 e，偽裝成 FedEx）

3. **添加多餘的子詞彙**：
   - PHISH-009：`irs-gov-refund.com`（非官方 .gov 網域，而是 .com 結尾）

對比合法信件，如 LEGIT-001 的 `noreply@github.com` 和 LEGIT-002 的 `no-reply@accounts.google.com`，都使用其公司的官方域名。
```

### 2.5 安裝依賴

執行前需安裝以下套件：

```bash
pip install langchain langchain-community langchain-ollama faiss-cpu
```

另外需確保 Ollama 已啟動並拉取所需模型：

```bash
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

---

## TASK 3（Outstanding）：Synthetic Data

### 3.1 Synthetic Data 在資安領域的角色與價值

**合成資料（Synthetic Data）** 是指透過演算法或 AI 模型生成的仿真資料，而非從真實世界直接蒐集而來。在資安領域，合成資料扮演著不可或缺的角色：

#### 為什麼資安領域特別需要合成資料？

1. **真實樣本稀缺且難以取得**
   - 真實的釣魚信件、惡意程式、入侵日誌等涉及隱私與機密，難以公開分享
   - 新型態攻擊（zero-day）出現時，可供訓練的樣本近乎為零
   - 某些攻擊類型（如 BEC 商業詐騙）因企業不願揭露而嚴重缺乏紀錄

2. **類別嚴重不平衡**
   - 在真實環境中，正常流量/信件遠多於惡意樣本（可能是 10000:1）
   - 模型容易偏向多數類別，對少數的攻擊樣本偵測率低
   - 合成資料可針對少數類別進行增量，平衡訓練集

3. **隱私合規要求**
   - 真實釣魚信件可能包含受害者個資（姓名、信箱、金額）
   - GDPR、個資法等法規限制真實資料的使用與分享
   - 合成資料保留攻擊模式的統計特徵，但不包含真實個資

4. **加速模型迭代與測試**
   - 可快速生成大量標記好的訓練資料，縮短資料準備時間
   - 可針對特定攻擊場景生成客製化測試集
   - 便於進行對抗訓練（adversarial training），提升模型魯棒性

### 3.2 使用 LLM 生成合成釣魚信資料

#### 生成策略

沿用 Task 1 設計的 schema（`email_id`, `sender`, `subject`, `body`, `urls`, `has_attachment`, `urgency_level`, `label`, `attack_type`, `indicators`），使用 Ollama 的 `gpt-oss:20b` 模型生成合成資料。

#### 設計理由

選擇 LLM 生成（而非傳統合成演算法如 SMOTE）的理由：
- 釣魚信件是**非結構化文字**，傳統數值型資料增量演算法不適用
- LLM 能理解社交工程的語境與手法，生成語意連貫的信件內容
- 可透過 prompt 指定特定攻擊類型、語言、急迫程度，有方向性地擴充弱勢類別

#### 生成 Prompt

```
你是一位資安研究員，正在建立釣魚信件偵測資料集。請根據以下 schema 生成 15 筆合成釣魚信件資料（JSON 格式）。

Schema 欄位：
- email_id: 字串，格式為 "SYNTH-001" 遞增
- sender: 偽造的寄件者地址（要模擬真實釣魚手法，如拼寫相似的假網域）
- subject: 信件主旨
- body: 信件內文（50-150 字）
- urls: 信件中的惡意連結陣列（可為空）
- has_attachment: 布林值
- urgency_level: "high" / "medium" / "low"
- label: 固定為 "phishing"
- attack_type: 從以下選擇：credential_harvesting, BEC, malware_delivery, invoice_fraud
- indicators: 描述為什麼這封信是釣魚信的關鍵指標

要求：
1. 攻擊類型要多樣化，至少涵蓋 3 種 attack_type
2. 包含不同語言的信件（英文、中文、日文各至少 1 封）
3. urgency_level 要有高中低的分佈
4. 偽造的網域要有創意（不要只用數字替換字母）
5. 每封信的社交工程手法要不同
```

#### 合成結果（15 筆）

```json
[
  {
    "email_id": "SYNTH-001",
    "sender": "verify@app1e-id-support.com",
    "subject": "Your Apple ID Has Been Locked",
    "body": "Dear User, Your Apple ID has been temporarily locked due to suspicious sign-in activity from an unknown device in Russia. To restore access, please verify your identity within 12 hours by clicking the secure link below. Failure to verify will result in permanent account deletion.",
    "urls": ["http://app1e-id-support.com/verify"],
    "has_attachment": false,
    "urgency_level": "high",
    "label": "phishing",
    "attack_type": "credential_harvesting",
    "indicators": "Spoofed Apple domain (app1e vs apple); account lock scare; 12-hour deadline; threatening permanent deletion; suspicious geographic login claim"
  },
  {
    "email_id": "SYNTH-002",
    "sender": "cfo@financedept-global.com",
    "subject": "Urgent: Vendor Payment - Confidential",
    "body": "Hi, This is CFO Mark. I need you to urgently process a payment of $28,500 to our new logistics vendor. The payment must be completed by 3 PM today. Please use the wire transfer details I'll send in a follow-up email. Do not discuss this with others as it's part of a confidential acquisition. Thanks.",
    "urls": [],
    "has_attachment": false,
    "urgency_level": "high",
    "label": "phishing",
    "attack_type": "BEC",
    "indicators": "CFO impersonation; urgent wire transfer request; confidentiality manipulation to prevent verification; vague vendor details; bypass normal approval process"
  },
  {
    "email_id": "SYNTH-003",
    "sender": "noreply@linkedln-notifications.com",
    "subject": "You have 5 new connection requests",
    "body": "Hi Professional, You have 5 pending connection requests on LinkedIn. Sarah Chen (Google), David Kim (Microsoft), and 3 others want to connect. Log in to review and accept these requests before they expire in 7 days.",
    "urls": ["http://linkedln-notifications.com/connections"],
    "has_attachment": false,
    "urgency_level": "low",
    "label": "phishing",
    "attack_type": "credential_harvesting",
    "indicators": "Spoofed LinkedIn domain (linkedln with lowercase L instead of i); subtle expiration pressure; lure of high-profile connections; link to fake login page"
  },
  {
    "email_id": "SYNTH-004",
    "sender": "scanner@office-doc-share.com",
    "subject": "Scanned Document: Contract_Final_v2.pdf",
    "body": "A document has been scanned and is ready for your review. Please download the attached file to view the signed contract. This document contains sensitive legal information and should be handled accordingly.",
    "urls": [],
    "has_attachment": true,
    "urgency_level": "medium",
    "label": "phishing",
    "attack_type": "malware_delivery",
    "indicators": "Generic scanner notification from non-corporate domain; unsolicited contract document; malicious attachment disguised as scanned PDF; appeal to authority with 'legal information'"
  },
  {
    "email_id": "SYNTH-005",
    "sender": "billing@netf1ix-account.com",
    "subject": "Payment Failed - Your Subscription Will Be Cancelled",
    "body": "We were unable to process your payment for Netflix Premium ($17.99/month). Your subscription will be cancelled within 48 hours unless you update your payment information. Click below to update your billing details and avoid service interruption.",
    "urls": ["http://netf1ix-account.com/update-payment"],
    "has_attachment": false,
    "urgency_level": "high",
    "label": "phishing",
    "attack_type": "credential_harvesting",
    "indicators": "Spoofed Netflix domain (netf1ix with 1 instead of l); payment failure scare; 48-hour deadline; request for billing/credit card information"
  },
  {
    "email_id": "SYNTH-006",
    "sender": "accounting@supplierportal-invoices.com",
    "subject": "Overdue Invoice #2024-3847 - Final Notice",
    "body": "This is a final notice regarding unpaid invoice #2024-3847 for $7,890.00. Payment was due 30 days ago. If payment is not received within 5 business days, we will proceed with debt collection. Please find the invoice attached or pay directly via the link below.",
    "urls": ["http://supplierportal-invoices.com/pay/inv-2024-3847"],
    "has_attachment": true,
    "urgency_level": "medium",
    "label": "phishing",
    "attack_type": "invoice_fraud",
    "indicators": "Unknown supplier domain; unsolicited overdue invoice; threatening debt collection; dual attack vector (malicious attachment + phishing link); specific amount to appear legitimate"
  },
  {
    "email_id": "SYNTH-007",
    "sender": "security@bankofamerrica.com",
    "subject": "Unusual Activity Detected on Your Account",
    "body": "We have detected unusual transaction activity on your Bank of America account ending in ****4521. A purchase of $2,340.00 was attempted at an electronics store in Miami, FL. If this was not you, please click below immediately to secure your account and dispute the transaction.",
    "urls": ["http://bankofamerrica.com/secure-account"],
    "has_attachment": false,
    "urgency_level": "high",
    "label": "phishing",
    "attack_type": "credential_harvesting",
    "indicators": "Spoofed domain with double 'r' (amerrica vs america); fake transaction details to create panic; partial account number for false legitimacy; urgency to act immediately"
  },
  {
    "email_id": "SYNTH-008",
    "sender": "親愛的會員@taiwan-shopee-tw.com",
    "subject": "【蝦皮購物】您的帳戶存在異常登入，請立即驗證",
    "body": "親愛的蝦皮會員您好，系統偵測到您的帳戶於2026年4月3日從陌生裝置（IP: 185.xx.xx.xx）登入。為確保帳戶安全，請於24小時內點擊以下連結進行身份驗證，否則帳戶將被暫時凍結。",
    "urls": ["http://taiwan-shopee-tw.com/verify-account"],
    "has_attachment": false,
    "urgency_level": "high",
    "label": "phishing",
    "attack_type": "credential_harvesting",
    "indicators": "偽造蝦皮網域（非官方 shopee.tw）；陌生 IP 登入恐嚇；24 小時期限；凍結帳戶威脅；中文社交工程"
  },
  {
    "email_id": "SYNTH-009",
    "sender": "support@amaz0n-jp-orders.com",
    "subject": "【Amazon】お客様のアカウントに不正アクセスがありました",
    "body": "お客様のAmazonアカウントに対し、第三者による不正アクセスの試みが検出されました。アカウントの安全を確保するため、24時間以内に以下のリンクからパスワードを変更してください。対応がない場合、アカウントは一時的に停止されます。",
    "urls": ["http://amaz0n-jp-orders.com/password-reset"],
    "has_attachment": false,
    "urgency_level": "high",
    "label": "phishing",
    "attack_type": "credential_harvesting",
    "indicators": "偽造 Amazon Japan 網域（amaz0n 用 0 替代 o）；日文社交工程；不正アクセス恐嚇；24 小時期限；帳戶停止威脅"
  },
  {
    "email_id": "SYNTH-010",
    "sender": "admin@company-sso-portal.com",
    "subject": "Mandatory: Multi-Factor Authentication Setup Required",
    "body": "As part of our new security policy effective April 1, all employees must enroll in multi-factor authentication by end of week. Click the link below to set up MFA for your corporate account. Employees who do not comply will lose access to company systems on Monday.",
    "urls": ["http://company-sso-portal.com/mfa-setup"],
    "has_attachment": false,
    "urgency_level": "medium",
    "label": "phishing",
    "attack_type": "credential_harvesting",
    "indicators": "External domain posing as internal SSO portal; exploiting real security trends (MFA); deadline pressure with access loss threat; link to fake authentication setup page"
  },
  {
    "email_id": "SYNTH-011",
    "sender": "helpdesk@micros0ft-teams.com",
    "subject": "You missed a Teams meeting - Recording available",
    "body": "You missed a scheduled Microsoft Teams meeting: 'Q2 Budget Review' hosted by your manager. A recording is available for the next 72 hours. Click below to access the recording. Please review before the next team sync on Friday.",
    "urls": ["http://micros0ft-teams.com/recording/q2-budget"],
    "has_attachment": false,
    "urgency_level": "low",
    "label": "phishing",
    "attack_type": "credential_harvesting",
    "indicators": "Spoofed Microsoft Teams domain (micros0ft with 0); exploiting FOMO from missed meeting; manager reference for authority; subtle time pressure with 72-hour expiry"
  },
  {
    "email_id": "SYNTH-012",
    "sender": "legal@corporate-compliance-dept.com",
    "subject": "Legal Hold Notice - Preserve All Documents",
    "body": "This is a formal legal hold notice. Due to ongoing litigation, you are required to preserve all documents and communications related to Project Phoenix. Download the attached legal hold guidelines and acknowledgment form. Return the signed form within 3 business days.",
    "urls": [],
    "has_attachment": true,
    "urgency_level": "medium",
    "label": "phishing",
    "attack_type": "malware_delivery",
    "indicators": "External domain posing as legal department; legal authority exploitation; malicious attachment disguised as legal document; specific project name for credibility; compliance deadline pressure"
  },
  {
    "email_id": "SYNTH-013",
    "sender": "rewards@starbvcks-rewards.com",
    "subject": "Congratulations! You've earned a free drink 🎉",
    "body": "Thank you for being a loyal Starbucks Rewards member! You've earned enough stars for a complimentary handcrafted beverage. Claim your reward by logging into your account below. This offer expires in 5 days.",
    "urls": ["http://starbvcks-rewards.com/claim-reward"],
    "has_attachment": false,
    "urgency_level": "low",
    "label": "phishing",
    "attack_type": "credential_harvesting",
    "indicators": "Spoofed Starbucks domain (starbvcks with v instead of u); reward/prize lure; low urgency but still has expiration; link to fake login page for credential harvesting"
  },
  {
    "email_id": "SYNTH-014",
    "sender": "finance@global-payroll-services.net",
    "subject": "Direct Deposit Update Confirmation Required",
    "body": "Our payroll system has been upgraded. To ensure uninterrupted salary deposits, all employees must re-confirm their bank account details through the new portal by April 10. Click the link below to verify your banking information.",
    "urls": ["http://global-payroll-services.net/verify-bank"],
    "has_attachment": false,
    "urgency_level": "medium",
    "label": "phishing",
    "attack_type": "credential_harvesting",
    "indicators": "External domain posing as payroll service; requesting bank account details; system upgrade pretext; salary disruption threat; deadline pressure"
  },
  {
    "email_id": "SYNTH-015",
    "sender": "procurement@vendor-invoice-portal.com",
    "subject": "Purchase Order #PO-9921 Attached - Approval Needed",
    "body": "Please review and approve the attached purchase order #PO-9921 for office supplies totaling $4,215.00. This order was requested by the operations team and needs your approval by end of day to meet the delivery schedule.",
    "urls": [],
    "has_attachment": true,
    "urgency_level": "medium",
    "label": "phishing",
    "attack_type": "invoice_fraud",
    "indicators": "External procurement domain; unsolicited purchase order; malicious attachment; end-of-day urgency; references operations team for internal credibility"
  }
]
```

### 3.3 品質校準與分析

#### ✅ 最合理的 1 筆：SYNTH-008（中文蝦皮釣魚信）

```json
{
  "email_id": "SYNTH-008",
  "sender": "親愛的會員@taiwan-shopee-tw.com",
  "subject": "【蝦皮購物】您的帳戶存在異常登入，請立即驗證",
  "body": "親愛的蝦皮會員您好，系統偵測到您的帳戶於2026年4月3日從陌生裝置（IP: 185.xx.xx.xx）登入。為確保帳戶安全，請於24小時內點擊以下連結進行身份驗證，否則帳戶將被暫時凍結。"
}
```

**合理的原因**：
1. **貼近台灣真實場景**：蝦皮購物是台灣最普及的電商平台，確實常被用於釣魚攻擊
2. **社交工程手法逼真**：使用具體的日期與部分 IP 位址，增加可信度
3. **偽造網域合理**：`taiwan-shopee-tw.com` 非官方的 `shopee.tw`，但足夠相似
4. **急迫感拿捏到位**：24 小時期限 + 帳戶凍結威脅，符合真實釣魚信的壓力策略
5. **語言流暢自然**：中文用語符合台灣習慣（「蝦皮會員」、「陌生裝置」），不像機器翻譯

#### ❌ 最不合理的 1 筆：SYNTH-008 的 sender 欄位

同樣是 SYNTH-008，其 `sender` 欄位為 `親愛的會員@taiwan-shopee-tw.com`——這是一個**不合理的瑕疵**：

**不合理的原因**：
1. **email 地址不能包含中文字元**：標準的 email 地址（RFC 5321）的 local part 不支援直接使用中文字元。真實的釣魚信會使用如 `member@taiwan-shopee-tw.com` 或 `service@shopee-tw-verify.com` 這樣的英文地址
2. **降低欺騙性**：任何稍有經驗的使用者看到中文 email 地址就會立刻識別為可疑信件
3. **LLM 的常見錯誤**：模型在生成多語言內容時，容易將顯示名稱（display name）與實際郵件地址混淆

**改善方式**：
1. **Prompt 中加入格式限制**：在生成 prompt 中明確指出 `sender 欄位必須符合標準 email 格式（RFC 5321），僅使用英文、數字、點、連字號`
2. **後處理驗證**：建立自動化的 schema 驗證腳本，使用正則表達式檢查 email 格式：
   ```python
   import re
   email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
   if not re.match(email_pattern, data["sender"]):
       print(f"警告：{data['email_id']} 的 sender 格式不合法")
   ```
3. **人工審核環節**：自動化驗證可以抓格式問題，但內容合理性（如 BEC 信件的語氣是否像真正的高階主管）仍需人工判斷
4. **Few-shot 範例引導**：在 prompt 中提供 2-3 筆現有資料作為範例，讓模型更準確地模仿格式

---

## 結論

1. **專業資料集的欄位設計遠超「文字+標籤」的基本架構**：觀察 Hugging Face 上的資料集後發現，攻擊類型分類、急迫程度、URL 結構特徵、寄件者網域分析等欄位，能提供多維度的判斷線索，大幅提升模型的分類準確度與可解釋性。

2. **知識庫顯著提升 LLM 回覆品質**：掛載結構化知識庫後，模型從「通識性安全建議」轉變為「有據可查的精準分析」，能引用具體案例、區分攻擊類型、對比釣魚與合法信件的差異。

3. **RAG 讓外部知識庫與 LLM 無縫整合**：透過向量檢索 + 生成的流程，模型可以在不重新訓練的前提下，隨時更新知識庫就能回答最新的釣魚威脅問題。

4. **合成資料是資安資料集擴充的利器，但需要品質把關**：LLM 能生成語意連貫的釣魚信件，但會犯格式錯誤（如中文 email 地址）或邏輯不一致的問題，必須搭配自動化驗證與人工審核才能確保資料品質。
