# jaxa_-hackathon_demo2

JAXA Earth API（NDVI/LST/降水量の月間値）と座標データを使って、
宇宙風の画像と音楽（WAV）を生成する Streamlit アプリです。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 実行

```bash
streamlit run app.py
```

ブラウザで表示されたアプリで以下を入力します。
- 緯度 / 経度
- 作成する月

アプリは指定月の **1か月前** のデータを取得して、
- 画像表示（PNGダウンロード可）
- 音楽再生（WAVダウンロード可）
を行います。

## py_compile の次にやること

`python -m py_compile app.py` は「文法エラーがない」確認です。
次は以下を実施してください。

1. 依存ライブラリをインストールする
2. `streamlit run app.py` でアプリを起動する
3. 座標と月を入力して生成ボタンを押す
4. 画像表示・音再生・ダウンロードを確認する


## 404 エラーが出る場合

JAXA 側のAPIパスや公開形態が変わっていると、`404 Not Found` になることがあります。
このアプリでは **API設定（404が出る場合）** からエンドポイントURLを複数指定できます（1行1URL）。
取得失敗時はエラー詳細を表示し、最後に座標ベース代替データで生成を継続します。
