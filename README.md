# reinforcement-learning-notes

このリポジトリでは私が（深層）強化学習について勉強した内容を、なるべく分かりやすくまとめています。

勉強ノートは `notes` フォルダの中にマークダウン形式のファイルとして保存しています。

一部のノートについては、今後 YouTube チャンネル「[ゆうきラボ | Yuuki Lab](https://www.youtube.com/@YuukiLab)」で解説する予定です。

## 目次（ざっくり）

- `notes/` フォルダ: 強化学習そのものの理論・実装ノート
- `experiments/` フォルダ: PyTorch や実装の細かい挙動を確認するための実験ノート群

## experiments フォルダについて

`experiments` フォルダには、PyTorch の基本的な使い方や、強化学習の実装を読み解くときに
つまずきやすいポイントを小さなコード例で検証するためのノートブックを置いています。

PyTorch やテンソルまわりの挙動をまず手を動かしながら確認したい方は、
ここから読み始めると本編ノート（`notes/` 以下）も理解しやすくなると思います。

## experiments 内の主なノートブック

- [`experiments/pytorch_tensor_basics_1.ipynb`](experiments/pytorch_tensor_basics_1.ipynb)
  - PyTorch のテンソルを扱うときに **つまずきやすい基本** をまとめたノートです。
  - `torch.arange`、ストライド（`stride`）、メモリ上での「連続（contiguous）」、
    そして `view` / `reshape` の違い、`gather` の使い方などを、
    小さなテンソルを使って一つずつ確認できる構成になっています。
  - 強化学習のコードを読むときによく出てくるテンソル操作の「なぜ？」を解消することが目的です。

- [`experiments/pytorch_tensor_basics_2.ipynb`](experiments/pytorch_tensor_basics_2.ipynb)
  - PyTorch の `max` 関数／メソッドに焦点を当てたノートです。
  - 「テンソル全体の最大値」と「次元 `dim` に沿った最大値（`max(dim)`）の違い」や、
    `max(1)[0]` / `max(1)[1]` の意味を、Q 値テーブルを例に取りながら丁寧に説明しています。
  - 最後に、Q 学習（DQN）で現れる $\\max_a Q(s_{t+1}, a)$ の項と、
    PyTorch コード上の `max(1)[0]` がどのように対応しているかも、
    小さなテンソル例を通して確認できます。

- [`experiments/pytorch_tensor_basics_3.ipynb`](experiments/pytorch_tensor_basics_3.ipynb)
  - PyTorch のテンソル連結まわり（`torch.cat` / `torch.stack`）に焦点を当てたノートです。
  - 「既存の次元を伸ばす `cat`」と「新しい次元を増やして積み重ねる `stack`」の違いを、
    1 次元・2 次元テンソルの小さな例で順番に確認できます。
  - `cat` の名前の由来（concatenate）や、`dim` の解釈（行方向・列方向）もあわせて整理しているので、
    強化学習のネットワーク実装でテンソルを結合するときの混乱を減らすことを目的にしています。

- [`experiments/test001.ipynb`](experiments/test001.ipynb)
  - ちょっとした動作確認や試行錯誤に使っているサンドボックス的なノートです。
  - 内容は安定していない可能性があるため、基本的には `pytorch_tensor_basics_*.ipynb` から
    読み始めることをおすすめします。

