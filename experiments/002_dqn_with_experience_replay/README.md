# DQN（Experience Replay あり）実験

**Experience Replay** を使う DQN の実験です。遷移 (s, a, r, s', done) を Replay バッファに貯め、ランダムにサンプリングしてミニバッチで TD 学習します。終端状態（done）では次状態の Q を使わないようにしており、CartPole で学習の挙動を確認するためのコードです。

## フォルダ構成

```
002_dqn_with_experience_replay/
├── README.md          # このファイル
├── main.py             # 1 回分の学習実行（引数で出力 JSON パスを指定）
├── agent.py            # DQN エージェント（Replay あり・ε-greedy・ミニバッチ TD 学習）
├── replay_memory.py    # Experience Replay 用のバッファ（遷移の保存・サンプリング）
├── run_experiments.py  # 複数回実験をまとめて実行するランナー
└── results/            # 実験結果の JSON を保存するディレクトリ
    ├── run001.json
    ├── run002.json
    └── ...
```

## 各ファイルの役割

| ファイル | 役割 |
|----------|------|
| **agent.py** | Experience Replay を使う DQN エージェント。Q ネットワーク（3 層 MLP）、ε-greedy による行動選択、バッファからサンプルしたミニバッチでの TD 更新を実装。終端（done）のときは TD 目標を reward のみにする。 |
| **replay_memory.py** | 遷移をリングバッファで保持し、`sample(batch_size)` でランダムにバッチを返す。state / action / reward / next_state / done をまとめて返す。 |
| **main.py** | CartPole 環境で 1 本の学習を実行し、各遷移を `memorize` でバッファに追加しつつ `update` でミニバッチ学習。結果を指定パスの JSON に出力。10 エピソード連続で「成功」するか、最大エピソード数に達するまで学習。 |
| **run_experiments.py** | `main.py` を `results/run001.json` ～ `results/run010.json` の 10 通りで順に実行する実験ランナー。 |

## 実行方法

### 1 回だけ実験する

出力先の JSON パスを引数で指定して `main.py` を実行します。

```bash
cd experiments/002_dqn_with_experience_replay
python main.py results/run001.json
```

別名で保存する例:

```bash
python main.py results/my_run.json
```

### 複数回まとめて実験する（run001 ～ run010）

`run_experiments.py` を実行すると、`main.py` が 10 回連続で実行され、それぞれ `results/run001.json` ～ `results/run010.json` に保存されます。

```bash
cd experiments/002_dqn_with_experience_replay
python run_experiments.py
```

## 出力 JSON の形式

各実行で保存される JSON は次の形式です（001 と同じ）。

```json
{
  "episode_at_10_consecutive_success": 42,
  "episodes": [
    {
      "episode_index": 0,
      "steps": 10,
      "success": false
    },
    {
      "episode_index": 1,
      "steps": 12,
      "success": false
    }
  ]
}
```

| キー | 説明 |
|------|------|
| **episode_at_10_consecutive_success** | 10 エピソード連続で「成功」したときのエピソード番号（1 始まり）。未達の場合は `null`。 |
| **episodes** | 各エピソードの記録の配列。 |
| **episodes[].episode_index** | エピソード番号（0 始まり）。 |
| **episodes[].steps** | そのエピソードで進んだステップ数。 |
| **episodes[].success** | そのエピソードが成功（steps ≥ 196）かどうか。 |

## 成功条件・終了条件

- **1 エピソードの最大ステップ数**: 200（`main.py` 内の `MAX_STEPS`）。
- **成功**: 1 エピソードで 196 ステップ以上進めた場合（ほぼ最後まで立て続けた場合）。
- **学習の打ち切り**: 成功を 10 エピソード連続で達成したとき、または最大エピソード数（デフォルト 1000）に達したとき。

## 必要な環境

- Python 3
- [Gymnasium](https://gymnasium.farama.org/)（`gymnasium`）
- [PyTorch](https://pytorch.org/)（`torch`）

インストール例:

```bash
pip install gymnasium torch
```

## 描画について

`main.py` の `HUMAN_RENDER_MODE = True` にすると、学習中に CartPole のウィンドウが表示されます。実験を速く回したい場合は `False` のままにしてください。
