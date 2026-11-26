# Discord Botによる雑談対話システム

ルールベース応答・日本語BERTによる感情分析（非タスク指向対話）と、状態遷移を用いた外部API連携・情報検索（タスク指向対話）を1つのボット内で統合したDiscord Botです。

大学（東京工科大学 コンピュータサイエンス学部）のプロジェクト演習にて開発し、本システムの設計・実装および評価をまとめたレポートを含む一連の成果が**学部長賞**を受賞しました。

A hybrid Japanese Discord chatbot that combines non-task-oriented chit-chat (morphological analysis + Japanese BERT sentiment analysis) with state-managed task-oriented utilities (live postal code API lookup and rich F1 driver stat cards).

---

## 主な機能 / Features

### 1. 非タスク指向対話（雑談モード / `State.DEFAULT`）
複数の対話戦略から応答候補を生成し、スコアに基づいて最適な応答を選択します。
* **BERT感情分析（ネガポジ判定）:** Hugging Face `llm-book/bert-base-japanese-v3-wrime-sentiment` を用いてユーザー発話の感情を推定し、共感応答を生成
* **形態素解析による話題展開:** `fugashi`（UniDic）で入力文から名詞を抽出し、質問形式の応答候補を生成
* **キーワード照合ルール:** `kw_matching_rule.txt` に定義されたルールベース応答
* **無難な相槌:** 会話を継続するためのフォールバック応答

### 2. タスク指向対話（状態遷移モード）
`enum.Flag` による状態管理でモードを切り替え、`END` で通常対話へ復帰します。
* **郵便番号住所検索 (`State.ZIPCODE`):** 7桁の郵便番号を正規表現で抽出し、外部Web API（[ZipCloud API](https://zipcloud.ibsnet.co.jp/)）へリクエストしてJSONレスポンスから住所を返却
* **F1ドライバー情報検索 (`State.FORMULA1`):** ドライバー名入力に対し、戦績・チームカラー・顔写真を含む視認性の高い `discord.Embed` カードを生成
* **ミニゲーム・ユーティリティ:** じゃんけん (`State.JANKEN`)、だじゃれ (`State.DAJARE`)、現在時刻取得、おみくじ・リラックス提案のランダム応答

---

## 使用技術 / Tech Stack

* **Language:** Python 3.8+
* **Bot Framework:** `discord.py`
* **NLP / ML:** Hugging Face `transformers` (`llm-book/bert-base-japanese-v3-wrime-sentiment`), `torch`, `fugashi` (UniDic)
* **External API:** ZipCloud Postal Code Search API (`urllib.request`, `json`)

---

## ファイル構成 / Project Structure

```text
.
├── bot.py                  # メインアプリケーション（対話制御・Discordクライアント）
├── kw_matching_rule.txt    # キーワード照合ルール定義
├── settings.example.py     # トークン・チャンネル設定のテンプレート
├── requirements.txt        # 依存ライブラリ一覧
├── .gitignore              # 機密情報（settings.py等）の除外設定
└── README.md
```

---

## セットアップ / Setup
git clone [https://github.com/](https://github.com/)<your-username>/discord-dialogue-bot.git

cd discord-dialogue-bot

pip install -r requirements.txt

python -m unidic download

cp settings.example.py settings.py

python bot.py

---

## 評価と今後の課題 / Evaluation & Future Improvements
* 状態遷移によるモード切り替え・END による復帰・外部API連携および Embed 表示は高い精度と実用性を確認済み。
* 一方、雑談モードにおける名詞抽出応答では一般的すぎる名詞に対しても質問を返してしまうケースが確認されたため、今後は品詞フィルタリングの精緻化や、対話履歴を保持できるLLMの導入による文脈理解の強化が課題。
