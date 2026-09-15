# Data Governance — IDDSI-Open

**Single source of truth for data governance: consent, takedown, attribution, and licence compatibility.** This document pairs with `docs/REGULATORY_POSTURE.md` (single source of truth for external claims wording) and implements the release checks in `dataschema/docs/LICENSE_PLAN.md`. This is a governance policy, not legal advice.

**Languages:** English (§1–§5) and 繁體中文 (§6–§10). The two sections are kept paragraph-aligned so edits stay in sync.

**Status:** research preview. Frozen disclaimer (quoted from `docs/REGULATORY_POSTURE.md` §1):

> **Tests performed by LinguaLeap staff, not clinicians.** This is a research preview, not a validated safety product.

---

## English

### 1. Data lanes and consent policy

IDDSI-Open has four data lanes. Consent treatment differs per lane:

| Lane | Who/what is in the data | Consent position |
|---|---|---|
| **Synthetic** (`synth/`) | AI-generated food images; no human subjects, no real photographs | No humans involved → **no consent needed**. Every item must carry `source=synthetic` and must never be represented as physically tested (`dataschema/docs/LICENSE_PLAN.md` check 3). |
| **Real-weak (harvested)** (`harvest/` → `realweak_pilot` on Spark) | Existing publicly licensed media (Nutrition5k, Wikimedia Commons, Open Images) | Individual consent is **not applicable** — the material is already published under licences that permit redistribution. **The licence terms bind instead.** Harvesting is fail-closed per `_pilot_realweak*/…/source_licenses.json`: only assets with an explicit allowlisted redistribution licence (CC0 / CC BY / CC BY-SA, all versions) and complete attribution are downloaded. Sources with no redistribution grant (Food101) or upstream rights conflicts (foodsense-dataset / Yelp) are blocked. |
| **Real-crowd (contributed)** (`crowd/`) | Media contributed by external contributors | **Informed opt-in required.** Contributors must be told, before upload, what the data will be used for and under what licence it will be released, and must actively consent. Upload consent must expressly cover public, noncommercial, share-alike redistribution of the media (`dataschema/docs/LICENSE_PLAN.md` check 4). Consent records remain private. Contributors keep a **right to withdraw** — exercised through the takedown process (§2). |
| **Real-tested (in-house physical tests)** | LinguaLeap staff performing IDDSI physical tests on food preparations | **Staff consent required** before participation; recordings are for internal research use and the public research preview. Consistent with the frozen wording in `docs/REGULATORY_POSTURE.md` §1: tests are performed by LinguaLeap staff, **not clinicians**, and the output is a research preview, not a validated safety product. |

No lane may contain patient data, patient profiles, diagnoses, or prescribed levels. If any harvested image is found to contain identifiable persons without a licence basis for redistribution of likeness, it is treated as a takedown candidate (§2) and removed.

### 2. Takedown policy

How an item is removed on request:

1. **Request intake.** A requester identifies the item by event ID, media SHA-256, source landing URL, or image copy. Requests are accepted from the original creator/licensor, a depicted person, or their authorised agent.
2. **Verification.** Match the item against `attribution.jsonl` (event ID, `media_sha256`, landing URL) and/or consent records (crowd lane) / staff consent records (tested lane).
3. **Removal.** The item is removed from **release manifests and media**: it is dropped from `events.jsonl`, `attribution.jsonl`, and every manifest, and the media file is deleted from the release package. Release builds must consume manifests only, never a raw media directory (see the 32-file finding in §4.4).
4. **Takedown log.** Every removal is recorded in a takedown log (repo-private) with: date, item identifier + SHA-256, requester, reason, action taken, and the dataset version in which the removal takes effect.
5. **Versioning.** The dataset is versioned (v0, v1, v2 …). A takedown lands in the next version; the version notes list removals by SHA-256 without exposing requester identity.
6. **Honest limit.** **Already-downloaded copies cannot be recalled.** Once a version is public, anyone who downloaded it retains it. We state this plainly in the takedown response. What we can do: removed SHA-256s are published in a revocation list per version, so downstream users can programmatically detect and drop removed items.

### 3. Attribution obligations

Per-licence obligations for every third-party item:

| Licence | Obligation when redistributing |
|---|---|
| **CC BY 4.0 / CC BY 2.0** | Attribution required: creator, licence name + link, source link, and indication of changes made (our pipeline records transformations: EXIF orientation, RGB JPEG conversion, metadata stripping). No ShareAlike obligation. |
| **CC BY-SA 4.0 / CC BY-SA 2.5** | Attribution required (same elements as CC BY) **plus ShareAlike**: adaptations must be offered under the same licence. See §4 for the compatibility consequence. |
| **CC0 1.0** | No attribution obligation (attribution is nonetheless provided as good practice). |

**`attribution.jsonl` (in each harvested dataset) is the source of truth** for creator, licence name, licence URL, landing URL, media path, and media SHA-256. The release rule from `source_licenses.json` is **fail closed**: only assets with an explicit allowlisted redistribution licence and **complete attribution** may be released; releases **must refuse any row lacking attribution** (build-time check, not manual review).

Recommended dataset-level attribution (from `dataschema/docs/LICENSE_PLAN.md`): "IDDSI-Open dataset, LinguaLeap Tech Limited (HK), version [VERSION], [URL], licensed CC BY-NC-SA 4.0." IDDSI trademark/attribution notices are preserved; LinguaLeap's tests are not IDDSI certification or endorsement.

### 4. Licence-compatibility audit (KEY SECTION)

#### 4.1 Verified pilot facts

Ground truth obtained directly from the realweak pilot on Spark (`~/iddsi/data/realweak_pilot/attribution.jsonl`), 2026-09-02. Note: **the W8 report (2026-08-31 07:55) and the work-package brief both describe a stale 100-event snapshot; the pilot files were extended at 2026-08-31 08:28 (Open Images added). The numbers below are the current verified counts.**

Command run:

```
python3 -c "
import json,collections
rows=[json.loads(l) for l in open('data/realweak_pilot/attribution.jsonl')]
print('rows:',len(rows))
uniq={r['media_path']:r for r in rows}
print('unique media:',len(uniq))
print('licence of unique media:',dict(collections.Counter(r['license_name'] for r in uniq.values())))
print('provider of unique media:',dict(collections.Counter(r['provider'] for r in uniq.values())))
"
```

Output:

```
rows: 129
unique media: 117
licence of unique media: {'CC BY-SA 2.5': 1, 'CC BY 4.0': 97, 'CC BY-SA 4.0': 3, 'CC0': 1, 'CC BY 2.0': 15}
provider of unique media: {'wikimedia': 5, 'nutrition5k': 97, 'open-images': 15}
```

Per-row counts (129 rows; 12 media files are referenced by more than one event):

```
python3 -c "
import json,collections
lic=collections.Counter(); prov=collections.Counter()
for l in open('data/realweak_pilot/attribution.jsonl'):
    d=json.loads(l); lic[d['license_name']]+=1; prov[d['provider']]+=1
print('licences:',dict(lic)); print('providers:',dict(prov)); print('total:',sum(lic.values()))
"
```

Output:

```
licences: {'CC BY-SA 2.5': 1, 'CC BY 4.0': 109, 'CC BY-SA 4.0': 3, 'CC0': 1, 'CC BY 2.0': 15}
providers: {'nutrition5k': 109, 'open-images': 15, 'wikimedia': 5}
total: 129
```

Summary: **129 events / 117 unique media; licences CC BY 4.0 ×97, CC BY 2.0 ×15, CC BY-SA 4.0 ×3, CC BY-SA 2.5 ×1, CC0 ×1 (unique media); providers nutrition5k ×97, open-images ×15, wikimedia ×5.** All 4 ShareAlike items (3× CC BY-SA 4.0, 1× CC BY-SA 2.5) are Wikimedia Commons files.

#### 4.2 Compatibility ruling

**CC BY-SA material cannot be re-licensed under CC BY-NC-SA.** The ShareAlike clause requires adaptations to be offered under the same (or, for BY-SA 4.0 from earlier versions, a compatible) licence; NonCommercial is an additional restriction that ShareAlike forbids imposing on the adapted material. This applies equally to CC BY-SA 2.5 and CC BY-SA 4.0. Therefore the 4 ShareAlike pilot items **must be either**:

- **(a) excluded from the CC BY-NC-SA release**, or
- **(b) kept in a separate BY-SA-licensed split** with its own manifest and its own licence file, where each item remains under its exact incoming licence (CC BY-SA 4.0 / CC BY-SA 2.5).

For CC BY and CC0 items there is no ShareAlike bar to inclusion in a BY-NC-SA dataset, but the **per-item licence survives**: attribution obligations continue to bind every downstream user of those items, and the manifest must make clear that third-party items remain under their original licences. Never silently relicense earlier media (`dataschema/docs/LICENSE_PLAN.md` check 7).

#### 4.3 Proposed release licence structure

| Artifact | Licence | Notes |
|---|---|---|
| **Main dataset** (synthetic + project-original content + CC BY / CC0 third-party content) | **CC BY-NC-SA 4.0** | Per-item licences preserved and disclosed via the attribution manifest; third-party items remain under their incoming licence (BY/CC0) with full attribution; BY-NC-SA governs project-original content (labels, annotations, schema records, synthetic media) and the collection as released. |
| **Optional BY-SA split** | Each item under its **incoming ShareAlike licence** (CC BY-SA 4.0 / CC BY-SA 2.5), split-level CC BY-SA 4.0 | Only if option (b) of §4.2 is chosen; separate manifest, separate licence file, clearly marked. |
| **Code, schema, validators, tests, capture docs** | **Apache License 2.0** (intended, per `dataschema/docs/LICENSE_PLAN.md`) | No `LICENSE` file exists at the repo root as of 2026-09-02 — **TBD**: add it before any public code release. Apache-2.0 must not be presented as granting rights in the dataset, IDDSI trademarks, or third-party media. |
| **Model weights** | **Apache-2.0 where legally permitted — intended, not yet granted** | Verify base-model terms and training-data rights first (`dataschema/docs/LICENSE_PLAN.md` check 5). |

Because of the NonCommercial restriction, public materials must say "publicly downloadable under an explicit reuse licence" and name CC BY-NC-SA 4.0 — never "open source" in the code sense (`dataschema/docs/LICENSE_PLAN.md` final paragraph).

#### 4.4 Housekeeping finding from the audit

Verified 2026-09-02: the pilot media directory contains **149 jpg files, of which 32 are on disk but unreferenced by `attribution.jsonl`** (leftovers from rejected/duplicate harvest attempts), and 12 media files are referenced by multiple event rows. Command output:

```
attribution rows: 117 on disk: 149
referenced but missing: 0
on disk but unreferenced: 32
```

Consequence (enforced as policy): **release builds must be driven exclusively by the manifests** — unreferenced media must never be swept into a release — and duplicate media references must be counted per media item, not per row, when computing licence totals.

### 5. Change control

**Any change to licence structure or claims goes through this file first, then `docs/REGULATORY_POSTURE.md`, before propagating anywhere else.**

- Changes to the release licence structure (§4.3), the compatibility ruling (§4.2), consent positions (§1), or takedown procedure (§2) are made here, reviewed, and only then propagated to release manifests, dataset cards (`dataschema/docs/`), the Hugging Face Space, and any tender document.
- Wording covered by `docs/REGULATORY_POSTURE.md` (intended-use statement, disclaimers, never-claim list, jurisdiction postures) is changed **only** in that file; this document quotes it but does not modify it.
- Never silently relicense already-released media; every release records its version, manifest checksums, exclusions, and exact licence files (`dataschema/docs/LICENSE_PLAN.md` check 7).
- Unassessed or uncertain items stay marked **TBD** (e.g., the missing root `LICENSE` file, weights licensing) — never filled in by inference.

---

## 繁體中文

### 6. 資料通道與同意政策

IDDSI-Open 設有四條資料通道，同意處理方式各有不同：

| 通道 | 資料內容 | 同定立場 |
|---|---|---|
| **合成（synthetic）**（`synth/`） | 人工智能生成的食物圖像；無人類受試者、非真實照片 | 不涉及人類 → **無需同意**。每項資料必須標註 `source=synthetic`，且**絕不得**聲稱經物理測試（`dataschema/docs/LICENSE_PLAN.md` 第 3 項檢查）。 |
| **真實-弱標籤（real-weak，擷取）**（`harvest/` → Spark 上之 `realweak_pilot`） | 現有、公開發佈且具授權的媒體（Nutrition5k、Wikimedia Commons、Open Images） | 個人同意**不適用**——素材本身已按允許再分發的授權條款公開發佈。**取而代之的是授權條款具有約束力。** 擷取程序按 `_pilot_realweak*/…/source_licenses.json` 採**失效關閉（fail-closed）**原則：只下載具明確允許清單再分發授權（CC0／CC BY／CC BY-SA 各版本）且署名完整的資產。無再分發授權（Food101）或上游權利衝突（foodsense-dataset／Yelp）的來源一律封鎖。 |
| **真實-群眾（real-crowd，貢獻）**（`crowd/`） | 外部貢獻者提供的媒體 | **必須知情並主動選擇加入（opt-in）。** 貢獻者上載前必須獲告知資料的用途及發佈授權，並須主動同意。上載同意必須明確涵蓋媒體的公開、非商業、相同方式分享再分發（`dataschema/docs/LICENSE_PLAN.md` 第 4 項檢查）。同意記錄保持私密。貢獻者保留**撤回權**——經下架程序（第 7 節）行使。 |
| **真實-實測（real-tested，內部物理測試）** | LinguaLeap 員工對食物製作執行 IDDSI 物理測試 | 參與前必須取得**員工同意**；錄像僅供內部研究使用及公開研究預覽。與 `docs/REGULATORY_POSTURE.md` 第 1 節凍結措辭一致：測試由 LinguaLeap 員工執行，**並非臨床醫護人員**，產出為研究預覽，並非經驗證的安全產品。 |

任何通道均不得含患者資料、患者檔案、診斷或處方等級。若發現任何擷取圖像含可識別人物而授權基礎不足以再分發其肖像，即視為下架候選（第 7 節）並予移除。

### 7. 下架政策

接獲要求後移除項目的程序：

1. **接收要求。** 要求人以事件 ID、媒體 SHA-256、來源落地頁 URL 或圖像副本識別項目。接受來自原創者／授權人、被攝者或其授權代理人的要求。
2. **核實。** 將項目與 `attribution.jsonl`（事件 ID、`media_sha256`、落地 URL）及／或（群眾通道）同意記錄、（實測通道）員工同意記錄比對。
3. **移除。** 項目從**發佈清單及媒體**中移除：從 `events.jsonl`、`attribution.jsonl` 及所有清單中剔除，並從發佈套件中刪除媒體檔案。發佈構建必須只讀取清單，絕不得直接讀取原始媒體目錄（見第 9.4 節之 32 個檔案發現）。
4. **下架記錄。** 每次移除均記錄於（儲存庫內部私存的）下架記錄：日期、項目識別碼 + SHA-256、要求人、原因、所採行動、移除生效的數據集版本。
5. **版本控制。** 數據集按版本發佈（v0、v1、v2……）。下架於下一版本生效；版本說明按 SHA-256 列出移除項目，但不披露要求人身分。
6. **誠實聲明。** **已下載的副本無法收回。** 版本一經公開，已下載者即持有該副本。我們在下架回覆中如實說明此點。可做的是：已移除項目的 SHA-256 會按版本載入撤銷清單，讓下游用戶能以程序方式偵測並剔除被移除項目。

### 8. 署名義務

每項第三方資料按授權承擔的義務：

| 授權 | 再分發義務 |
|---|---|
| **CC BY 4.0 / CC BY 2.0** | 須署名：創作者、授權名稱 + 連結、來源連結，並標明所作修改（本管線會記錄轉換操作：EXIF 方向校正、轉換為 RGB JPEG、元數據清除）。無相同方式分享義務。 |
| **CC BY-SA 4.0 / CC BY-SA 2.5** | 須署名（同 CC BY 各項）**另加相同方式分享（ShareAlike）**：改編作品須以同一授權提供。相容性後果見第 9 節。 |
| **CC0 1.0** | 無署名義務（惟作為良好慣例仍提供署名）。 |

**`attribution.jsonl`（每個擷取數據集內）是署名的真實來源（source of truth）**，涵蓋創作者、授權名稱、授權 URL、落地 URL、媒體路徑及媒體 SHA-256。`source_licenses.json` 的發佈規則為**失效關閉**：只有具明確允許清單再分發授權且**署名完整**的資產方可發佈；發佈時**必須拒絕任何缺乏署名的資料列**（構建時自動檢查，而非人手覆核）。

數據集層面建議署名（引自 `dataschema/docs/LICENSE_PLAN.md`）：「IDDSI-Open dataset, LinguaLeap Tech Limited (HK), version [VERSION], [URL], licensed CC BY-NC-SA 4.0.」。IDDSI 商標／署名通知一律保留；LinguaLeap 的測試並非 IDDSI 認證或認可。

### 9. 授權相容性審計（關鍵章節）

#### 9.1 已核實的試點事實

以下真實數據於 2026-09-02 直接從 Spark 上的 realweak 試點（`~/iddsi/data/realweak_pilot/attribution.jsonl`）取得。注意：**W8 報告（2026-08-31 07:55）及工作包簡報所述均為過時的 100 項事件快照；試點檔案於 2026-08-31 08:28 擴充（加入 Open Images）。以下為現行核實數字。**

執行之指令：

```
python3 -c "
import json,collections
rows=[json.loads(l) for l in open('data/realweak_pilot/attribution.jsonl')]
print('rows:',len(rows))
uniq={r['media_path']:r for r in rows}
print('unique media:',len(uniq))
print('licence of unique media:',dict(collections.Counter(r['license_name'] for r in uniq.values())))
print('provider of unique media:',dict(collections.Counter(r['provider'] for r in uniq.values())))
"
```

輸出：

```
rows: 129
unique media: 117
licence of unique media: {'CC BY-SA 2.5': 1, 'CC BY 4.0': 97, 'CC BY-SA 4.0': 3, 'CC0': 1, 'CC BY 2.0': 15}
provider of unique media: {'wikimedia': 5, 'nutrition5k': 97, 'open-images': 15}
```

按資料列統計（129 列；12 個媒體檔案被多於一個事件引用）：

```
python3 -c "
import json,collections
lic=collections.Counter(); prov=collections.Counter()
for l in open('data/realweak_pilot/attribution.jsonl'):
    d=json.loads(l); lic[d['license_name']]+=1; prov[d['provider']]+=1
print('licences:',dict(lic)); print('providers:',dict(prov)); print('total:',sum(lic.values()))
"
```

輸出：

```
licences: {'CC BY-SA 2.5': 1, 'CC BY 4.0': 109, 'CC BY-SA 4.0': 3, 'CC0': 1, 'CC BY 2.0': 15}
providers: {'nutrition5k': 109, 'open-images': 15, 'wikimedia': 5}
total: 129
```

摘要：**129 項事件／117 個唯一媒體；授權為 CC BY 4.0 ×97、CC BY 2.0 ×15、CC BY-SA 4.0 ×3、CC BY-SA 2.5 ×1、CC0 ×1（唯一媒體計）；來源為 nutrition5k ×97、open-images ×15、wikimedia ×5。** 全部 4 項 ShareAlike 項目（3× CC BY-SA 4.0、1× CC BY-SA 2.5）均為 Wikimedia Commons 檔案。

#### 9.2 相容性裁定

**CC BY-SA 素材不得改以 CC BY-NC-SA 授權發佈。** ShareAlike 條款要求改編作品以同一（或相容）授權提供；NonCommercial 屬額外限制，為 ShareAlike 所禁止施加於改編素材。此裁定同樣適用於 CC BY-SA 2.5 及 CC BY-SA 4.0。因此 4 項 ShareAlike 試點項目**必須**：

- **(a) 從 CC BY-NC-SA 發佈中剔除**，或
- **(b) 保留於獨立的 BY-SA 授權分片中**，附自有清單及授權檔案，且每項維持其原有確切授權（CC BY-SA 4.0／CC BY-SA 2.5）。

CC BY 及 CC0 項目納入 BY-NC-SA 數據集並無 ShareAlike 障礙，但**項目層面授權持續有效**：署名義務約束該等項目的每一位下游用戶，清單必須表明第三方項目維持原有授權。絕不得靜默地為已發佈媒體更改授權（`dataschema/docs/LICENSE_PLAN.md` 第 7 項檢查）。

#### 9.3 建議發佈授權結構

| 標的 | 授權 | 備註 |
|---|---|---|
| **主數據集**（合成 + 項目原創內容 + CC BY／CC0 第三方內容） | **CC BY-NC-SA 4.0** | 以署名清單保留並披露項目層面授權；第三方項目維持原有授權（BY／CC0）並附完整署名；BY-NC-SA 管轄項目原創內容（標籤、註釋、結構記錄、合成媒體）及整體合集之發佈。 |
| **可選 BY-SA 分片** | 每項按其**原有 ShareAlike 授權**（CC BY-SA 4.0／CC BY-SA 2.5），分片層面 CC BY-SA 4.0 | 僅在選擇第 9.2 節選項 (b) 時採用；獨立清單、獨立授權檔案、清晰標示。 |
| **程式碼、結構、驗證器、測試、拍攝文件** | **Apache License 2.0**（按 `dataschema/docs/LICENSE_PLAN.md` 屬意向性） | 截至 2026-09-02 儲存庫根目錄**並無 `LICENSE` 檔案——待定（TBD）**：公開發佈程式碼前必須加入。Apache-2.0 不得被呈現為授予數據集、IDDSI 商標或第三方媒體之權利。 |
| **模型權重** | **Apache-2.0（於法律允許範圍內）——屬意向，尚未授予** | 須先核實基礎模型條款及訓練資料權利（`dataschema/docs/LICENSE_PLAN.md` 第 5 項檢查）。 |

因 NonCommercial 限制，對外資料必須稱「以明確的重用授權公開提供」並指明 CC BY-NC-SA 4.0——絕不得稱程式碼意義上的「開源」（`dataschema/docs/LICENSE_PLAN.md` 末段）。

#### 9.4 審計中的整理髮現

2026-09-02 核實：試點媒體目錄含 **149 個 jpg 檔案，其中 32 個在磁碟上但未被 `attribution.jsonl` 引用**（屬被拒／重複擷取嘗試的殘留檔案），另有 12 個媒體檔案被多個事件資料列引用。指令輸出：

```
attribution rows: 117 on disk: 149
referenced but missing: 0
on disk but unreferenced: 32
```

後果（訂為政策）：**發佈構建必須完全由清單驅動**——未被引用的媒體絕不得納入發佈——而計算授權總數時，重複引用的媒體必須按媒體項目而非資料列計算。

### 10. 變更控制

**任何授權結構或聲明的變更，必須先經本文件，再經 `docs/REGULATORY_POSTURE.md`，方可傳播至其他地方。**

- 對發佈授權結構（第 9.3 節）、相容性裁定（第 9.2 節）、同意立場（第 6 節）或下架程序（第 7 節）的修改，均須先在此進行、經審閱，然後方可傳播至發佈清單、數據集卡（`dataschema/docs/`）、Hugging Face Space 及任何招標文件。
- 屬 `docs/REGULATORY_POSTURE.md` 管轄的措辭（預期用途聲明、免責聲明、永不聲明清單、司法管轄區態勢）**只可在該文件修改**；本文件僅引用而不修改之。
- 絕不靜默地為已發佈媒體更改授權；每次發佈均記錄版本、清單校驗和、排除項目及確切授權檔案（`dataschema/docs/LICENSE_PLAN.md` 第 7 項檢查）。
- 未評估或不確定的項目一律保持標示**待定（TBD）**（例如缺失的根目錄 `LICENSE` 檔案、權重授權）——絕不以推斷填補。
