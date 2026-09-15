# Regulatory Posture — IDDSI-Open

**Single source of truth for all external claims wording.** Any change to the wording in this file requires review before it propagates to the dataset card, datasheet, Hugging Face Space, marketing page, KOL receipt, or tender document. See §6 (Change control).

This document distills `r5_reg.md` and `r5_prior.md` (with context from `r5_feas.md` and `r5_value.md`). It does not introduce new regulatory positions; where a posture is not yet determined it is marked **TBD**.

**Languages:** English (§1–§6) and 繁體中文 (§7–§12). The two sections are kept paragraph-aligned so edits stay in sync.

---

## English

### 1. Intended-use statement (frozen wording)

The following block is the **frozen honest-claims wording** for the IDDSI-Open research preview. It is quoted verbatim from `dataschema/docs/DATASET_CARD.md` and `dataschema/docs/DATASHEET.md` (W10 report, 2026-08-31) and matches the safer public-demo boundary mandated by `r5_reg.md` §"Safer public-demo boundary".

> **Tests performed by LinguaLeap staff, not clinicians.** This is a research preview, not a validated safety product.

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

Honest claims for v0 (release shape locked 2026-09-07; the honest release card `hf/model_README.md` is the authority). The old dataset-plus-model first is DEAD and must not be used:

> The ambition was "the first publicly downloadable IDDSI-specific smartphone-food dataset and fine-tuned model released under explicit reuse licences, with physically tested Chinese/Cantonese soft-meal labels"; v0 does NOT meet it - the harvested dataset is withheld on licence grounds (of 587 unique events exactly 2 would ship cleanly; 592 rows carry unverified open-images/openverse licences, 7 are CC BY-SA against the BY-NC-SA main release, 109 have unresolved redistribution status) and the model weights are withheld because the photo-classification head failed its pre-registered gates (wk1 weighted kappa >= 0.70 measured 0.42; wk2 macro-F1 >= 0.75 measured 0.358, with dangerous under-classification at 33% against a <= 5% gate; the wk1 kill rule fired and photo-classification is dead as a product capability).

The ONE claim that stands in v0:

> First open-source smartphone grader for the 10 ml/10-second IDDSI Flow Test (v0 bench: MAE 0.065 ml on 100 videos, L0-L4).

The flow-test grader ships in v0, so the flow-test claim stands in v0; it is **not** "first video grader" (`r5_prior.md`). v0 ships no downloadable classifier and no downloadable dataset - only code, eval results, and the flow-test grader.

### 2. What the demo DOES and DOES NOT do

**DOES**

**v0 scope:** no downloadable classifier and no downloadable dataset ship in v0 (weights withheld on safety gates, dataset withheld on licence grounds — see §1). The photo path below is code plus recorded eval results only, and photo-classification is dead as a product capability.

- Accept a 45° plate photo with a standard fork / 15-mm reference and return an IDDSI level **suggestion** with calibrated **abstention** ("unclear — perform physical test") when confidence is low or capture is incomplete (`r5_feas.md` §"Capture protocol", §"Accuracy").
- Grade a video of the official 10 ml / 10-second IDDSI Flow Test as a **research tool**: detect the calibrated 61.5-mm syringe, release at t=0, meniscus at t=10, map residue to `<1`, `1–4`, `4–8`, `≥8 mL`, and abstain near boundaries or with bubbles/lumps (`r5_feas.md` §"Flow grader").
- Teach the official IDDSI test procedure rather than prescribe modifications (`r5_reg.md` §"Safer public-demo boundary").
- Show uncertainty explicitly in the UX (`r5_reg.md`).

**DOES NOT**

- **Cannot** measure flow, hardness, adhesiveness, cohesiveness, granulation, particle size, or temperature from a still photo. Surface appearance does not reveal these mechanical properties (`r5_prior.md` verdict; `r5_feas.md` §"Why photos fail").
- Is **not** a diagnostic and does not assess swallowing, aspiration risk, or suitability for any person (`r5_reg.md` safer-boundary wording).
- Is **not** for clinical decisions and **not** for patient-facing use. No patient profile, no diagnosis, no prescribed level, no "safe/unsafe/pass" output (`r5_reg.md`).
- Does **not** replace clinician-prescribed texture or physical IDDSI testing (`r5_value.md` final paragraph — replaces the older "食品不能代替藥物" wording).
- Does **not** perform official IDDSI tests; it estimates visual similarity to descriptors only (frozen disclaimer).

### 3. Never-claim list (from `r5_prior.md`, verbatim)

> Never claim: first IDDSI AI; first image classifier; first dataset/benchmark; clinically validated; diagnostic; medical-grade; or able to determine swallowing safety from a photograph.

Additional prohibitions from `r5_reg.md` and `r5_value.md`:

- **Never** claim "first IDDSI texture AI" — ViscoCam (2021), OptiTexture, and EdUHK's 2026 image system predate it (`r5_reg.md` final paragraph).
- **Never** say IDDSI-certified or IDDSI-endorsed; follow IDDSI's product-labelling rules (`r5_prior.md`).
- **Never** say "validated by HKU Swallowing Research Laboratory" without written authorisation.
- **Never** output "safe/unsafe/pass" or estimate aspiration risk (`dataschema/docs/DATASET_CARD.md`).
- **Never** rely on a "not diagnosis" disclaimer to override intended medical purpose — HK MDD treats medical-purpose software as a potential medical device regardless of disclaimer (`r5_prior.md`; `r5_value.md`).

### 4. Jurisdiction posture summary

Source: `r5_reg.md` §"Regulatory verdict". Overall verdict (verbatim):

> **As currently proposed, Snap-to-IDDSI is probably SaMD.** Although it does not diagnose dysphagia, it interprets an individual meal and recommends modification to mitigate a swallowing disorder. "Guidance only," "not a diagnosis," and "食品不能代替藥物" do not neutralize that intended medical purpose.

#### 4.1 Hong Kong — MDACS

Software intended for diagnosis, treatment, alleviation or support of a physiological process falls within the medical-device definition; standalone medical-purpose software is SaMD under TR-007. MDACS listing remains generally voluntary, although increasingly required for government procurement. A patient-meal verdict plus "thicken/blend/remove" likely "informs/drives clinical management."

**Posture:** Probably SaMD under TR-007. MDACS listing voluntary but increasingly required for government procurement. **Action:** keep the public demo inside the safer boundary in §2; seek an MDACS AI-medical-device classification opinion before any patient pilot (`r5_value.md` final paragraph; `r5_feas.md` §"Evaluation/regulation"). References: GN-00 definition, TR-007:2026, MDACS status page (links in `r5_reg.md`).

#### 4.2 Mainland China — NMPA

"Independent software" requires a medical purpose; classification follows intended purpose and risk. NMPA-derived policy treats AI providing only measurements/reference information generally as **Class II**, but immature AI offering lesion judgments, medication guidance or treatment planning as **Class III**. Patient-specific IDDSI grading with corrective action is therefore **at least plausibly Class II and potentially Class III**.

**Posture:** At least plausibly Class II, potentially Class III. **Action:** obtain a formal classification determination before any mainland release. References: SAMR Classification Rules, official AI classification explanation (links in `r5_reg.md`).

#### 4.3 European Union — MDR

If intended for dysphagia management, it is MDSW. Rule 11 makes software informing diagnostic/therapeutic decisions **Class IIa minimum**, rising to IIb/III with potential serious harm. Choking/aspiration consequences make IIb arguable.

**Posture:** MDSW; Rule 11 Class IIa minimum, IIb arguable given choking/aspiration harm. Reference: MDCG 2019-11 rev.1 (link in `r5_reg.md`).

#### 4.4 United States — FDA

Dysphagia-specific claims are not disease-unrelated "general wellness." The non-device CDS exclusion primarily protects transparent recommendations to healthcare professionals, not caregiver/patient directives.

**Posture:** Not general wellness; non-device CDS exclusion unlikely to apply to caregiver/patient-facing directives. References: FDA general-wellness guidance, FDA CDS guidance (links in `r5_reg.md`).

#### 4.5 Unassessed jurisdictions

**TBD.** No other jurisdiction has been assessed in `r5_reg.md`. Do not extrapolate.

### 5. What a regulated version would require

Quoted from `r5_reg.md`:

> A regulated version needs fixed intended use, formal classification, ISO 13485/14971, IEC 62304/62366, locked/versioned model, analytical and clinical validation, subgroup/external-site evidence, cybersecurity/privacy, human-factors work, adverse-event/post-market processes, and jurisdictional submission.

The current research preview does **not** satisfy these requirements and must not be presented as if it does.

### 6. Change control

**This file is the single source of truth for external claims wording.**

- Any wording change to the intended-use statement, the mandatory disclaimer, the honest-claims block, the never-claim list, or a jurisdiction posture paragraph **must** be made here first, reviewed, and only then propagated to:
  - `dataschema/docs/DATASET_CARD.md` and `DATASHEET.md` (EN + 繁中)
  - the Hugging Face Space page
  - any affiliated marketing page
  - the KOL receipt
  - any tender or procurement document
- The frozen honest-claims block in §1 is quoted verbatim from the dataset card; edits to it are a **claims change** and require the same review.
- If a new jurisdiction is assessed, add a new §4.x paragraph here before any external wording references it.
- Unassessed or uncertain items stay marked **TBD** — never fill them in by inference.

---

## 繁體中文

### 7. 預期用途聲明（凍結措辭）

以下為 IDDSI-Open 研究預覽的**凍結誠實聲明措辭**，逐字引自 `dataschema/docs/DATASET_CARD.md` 與 `dataschema/docs/DATASHEET.md`（W10 報告，2026-08-31），並符合 `r5_reg.md`「更安全的公開演示界線」之要求。

> **測試由 LinguaLeap 員工執行，並非臨床醫護人員。** 本項目為研究預覽，並非經驗證的安全產品。

> 本演示僅供烹飪教育研究之用。其估算與 IDDSI 描述符的視覺相似度；並不執行 IDDSI 官方測試、不評估吞嚥能力、不判斷食物對任何人士是否適合，亦不決定食物是否安全食用。

v0 誠實聲明（發佈形態於 2026-09-07 鎖定；以誠實發佈卡 `hf/model_README.md` 為準）。舊的數據集加模型之「首個」聲明已失效，不得使用：

> 原定目標是「首個公開可下載、專為 IDDSI 而設的智能手機食物數據集及微調模型，以明確的重用許可發佈，並附有經物理測試的中式／粵式軟餐標籤」；v0 並未達到——收穫數據集因授權原因被扣留（587 個獨立事件中僅 2 個可乾淨發佈；592 行帶有未驗證的 open-images／openverse 授權，7 個為 CC BY-SA 與 BY-NC-SA 主授權衝突，109 個重新分發狀態未解決），模型權重因照片分類頭未通過預先註冊的門檻而被扣留（wk1 加權 κ ≥ 0.70，實測 0.42；wk2 宏平均 F1 ≥ 0.75，實測 0.358，危險低估率 33%，門檻為 ≤ 5%；wk1 終止規則已觸發，照片分類作為產品能力已終止）。

v0 中仍成立的聲明：

> 首個開源智能手機評級工具，用於 10 毫升／10 秒 IDDSI 流動測試（v0 基準：100 段影片，MAE 0.065 毫升，L0–L4）。

流動測試評級工具已隨 v0 發佈，因此流動測試聲明在 v0 中成立；**不得**稱為「首個影片評級工具」（`r5_prior.md`）。v0 不附帶可下載的分類器，亦不附帶可下載的數據集——僅有程式碼、評估結果與流動測試評級工具。

### 8. 本演示能做與不能做的事

**能做**

**v0 範圍：** v0 不附帶可下載的分類器，亦不附帶可下載的數據集（權重因安全門檻被扣留，數據集因授權原因被扣留——見第 7 節）。以下照片路徑僅為程式碼加已記錄的評估結果，照片分類作為產品能力已終止。

- 接受以標準叉子／15 毫米參照物拍攝的 45° 餐碟照片，並返回 IDDSI 等級**建議**；當信心不足或拍攝不完整時，會按校準**棄權／不作判斷**（「不清晰——請進行物理測試」）（`r5_feas.md`「拍攝規程」、「準確度」）。
- 作為**研究工具**，為官方 10 毫升／10 秒 IDDSI 流動測試的影片評級：識別 61.5 毫米校準注射器、t=0 釋放、t=10 彎月面，將殘留量映射至 `<1`、`1–4`、`4–8`、`≥8 mL`，並在臨界值附近或出現氣泡／結塊時棄權（`r5_feas.md`「流動評級」）。
- 教授官方 IDDSI 測試程序，而非處方修改方案（`r5_reg.md`「更安全的公開演示界線」）。
- 在用戶介面明確顯示不確定性（`r5_reg.md`）。

**不能做**

- **無法**從靜態照片測量流動性、硬度、黏附性、凝聚性、顆粒感、粒子大小或溫度。表面外觀不能揭示這些機械特性（`r5_prior.md` 裁決；`r5_feas.md`「照片為何失效」）。
- **並非**診斷工具，不評估吞嚥能力、誤嚥風險或對任何人士的適合性（`r5_reg.md` 更安全界線措辭）。
- **不適用於**臨床決策，**亦不適用於**患者直接使用。不設患者檔案、不作診斷、不處方等級、不輸出「安全／不安全／合格」（`r5_reg.md`）。
- **不能**取代臨床醫護人員處方的質地或 IDDSI 物理測試（`r5_value.md` 末段——取代舊有「食品不能代替藥物」措辭）。
- **並不**執行 IDDSI 官方測試；僅估算與描述符的視覺相似度（凍結免責聲明）。

### 9. 永不聲明清單（逐字引自 `r5_prior.md`）

> 永不聲明：首個 IDDSI 人工智能；首個影像分類器；首個數據集／基準；經臨床驗證；具診斷能力；醫療級別；或能從照片判斷吞嚥安全。

來自 `r5_reg.md` 與 `r5_value.md` 的額外禁止事項：

- **永不**聲稱「首個 IDDSI 質地人工智能」——ViscoCam（2021）、OptiTexture 及 EdUHK 2026 年影像系統均早於本項目（`r5_reg.md` 末段）。
- **永不**聲稱獲 IDDSI 認證或認可；須遵守 IDDSI 產品標籤規則（`r5_prior.md`）。
- **永不**在未獲書面授權下聲稱「經港大吞嚥研究實驗室驗證」。
- **永不**輸出「安全／不安全／合格」或估算誤嚥風險（`dataschema/docs/DATASET_CARD.md`）。
- **永不**依賴「非診斷」免責聲明來推翻預期醫療用途——香港醫療儀器科視具醫療用途的軟件為潛在醫療儀器，不論免責聲明如何（`r5_prior.md`；`r5_value.md`）。

### 10. 各司法管轄區監管態勢摘要

來源：`r5_reg.md`「監管裁決」。總體裁決（逐字）：

> **按目前構思，Snap-to-IDDSI 很可能屬於 SaMD（獨立醫療軟件）。** 雖然它並不診斷吞嚥困難，但它解讀個別膳食並建議修改，以緩解吞嚥障礙。「僅供參考」、「並非診斷」及「食品不能代替藥物」均不能中和該預期醫療用途。

#### 10.1 香港 — MDACS

凡擬用於診斷、治療、緩解或支持生理過程的軟件，均屬醫療儀器定義範圍；具醫療用途的獨立軟件按 TR-007 屬 SaMD。MDACS 表列目前大致屬自願性質，但政府採購日益要求表列。針對患者膳食的判決加上「增稠／攪拌／移除」建議，很可能「告知／驅動臨床管理」。

**態勢：** 按 TR-007 很可能屬 SaMD。MDACS 表列屬自願，但政府採購日益要求。**行動：** 公開演示須守在第 8 節的更安全界線內；在任何患者試點前，先徵詢 MDACS 人工智能醫療儀器分類意見（`r5_value.md` 末段；`r5_feas.md`「評估／監管」）。參考：GN-00 定義、TR-007:2026、MDACS 狀態頁（連結見 `r5_reg.md`）。

#### 10.2 中國內地 — NMPA

「獨立軟件」須具醫療用途；分類依預期用途及風險而定。NMPA 衍生的政策一般將僅提供測量／參考資訊的人工智能列為**第二類**，但提供病變判斷、用藥指導或治療計劃的不成熟人工智能則列為**第三類**。因此，針對患者個別情況的 IDDSI 評級加糾正措施，**至少很可能屬第二類，並可能屬第三類**。

**態勢：** 至少很可能屬第二類，可能屬第三類。**行動：** 在內地發佈前，先取得正式分類裁定。參考：市場監管總局分類規則、官方人工智能分類解釋（連結見 `r5_reg.md`）。

#### 10.3 歐盟 — MDR

如擬用於吞嚥困難管理，即屬 MDSW。規則 11 將為診斷／治療決策提供資訊的軟件列為**至少 IIa 類**，如可能造成嚴重傷害則升至 IIb／III 類。哽喉／誤嚥的後果使 IIb 類成為可爭辯的分類。

**態勢：** MDSW；規則 11 下至少 IIa 類，基於哽喉／誤嚥傷害可爭辯為 IIb 類。參考：MDCG 2019-11 rev.1（連結見 `r5_reg.md`）。

#### 10.4 美國 — FDA

針對吞嚥困難的聲明並非與疾病無關的「一般健康」。非器械 CDS 豁免主要保障向醫護專業人員作出的透明建議，而非向照顧者／患者作出的指示。

**態勢：** 不屬一般健康；面向照顧者／患者的指示不大可能適用非器械 CDS 豁免。參考：FDA 一般健康指引、FDA CDS 指引（連結見 `r5_reg.md`）。

#### 10.5 未評估的司法管轄區

**待定（TBD）。** `r5_reg.md` 並未評估其他司法管轄區。切勿推斷。

### 11. 受監管版本所需條件

逐字引自 `r5_reg.md`：

> 受監管版本需要固定的預期用途、正式分類、ISO 13485/14971、IEC 62304/62366、鎖定／版本化的模型、分析與臨床驗證、亞組／外部場地證據、網絡安全／私隱、人因工程工作、不良事件／上市後流程，以及各司法管轄區的申報。

目前的研究預覽**並未**符合上述要求，亦不得以此形象示人。

### 12. 變更控制

**本文件是所有對外聲明措辭的單一真實來源。**

- 凡對預期用途聲明、強制免責聲明、誠實聲明區塊、永不聲明清單或司法管轄區態勢段落的措辭作出任何修改，**必須**先在此文件進行、經審閱，然後方可傳播至：
  - `dataschema/docs/DATASET_CARD.md` 與 `DATASHEET.md`（英文 + 繁中）
  - Hugging Face Space 頁面
  - 任何關聯市場推廣頁面
  - KOL 收據
  - 任何招標或採購文件
- 第 7 節的凍結誠實聲明區塊逐字引自數據集卡；對其作出修改即屬**聲明變更**，須經同樣審閱。
- 如評估了新的司法管轄區，須先在此新增第 10.x 段，方可於任何對外措辭中引用。
- 未評估或不確定的項目須保持標示為**待定（TBD）**——切勿以推斷填補。
