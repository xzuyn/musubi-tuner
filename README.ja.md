# Musubi Tuner

[English](./README.md) | [日本語](./README.ja.md)

## 目次

<details>
<summary>クリックすると展開します</summary>

- [はじめに](#はじめに)
    - [スポンサー](#スポンサー)
    - [スポンサー募集のお知らせ](#スポンサー募集のお知らせ)
    - [最近の更新](#最近の更新)
    - [リリースについて](#リリースについて)
    - [AIコーディングエージェントを使用する開発者の方へ](#AIコーディングエージェントを使用する開発者の方へ)
- [概要](#概要)
    - [ハードウェア要件](#ハードウェア要件)
    - [特徴](#特徴)
    - [ドキュメント](#ドキュメント)
- [インストール](#インストール)
    - [pipによるインストール](#pipによるインストール)
    - [uvによるインストール](#uvによるインストール)
    - [Linux/MacOS](#linuxmacos)
    - [Windows](#windows)
- [モデルのダウンロード](#モデルのダウンロード)
- [使い方](#使い方)
    - [データセット設定](#データセット設定)
    - [事前キャッシュと学習](#事前キャッシュと学習)
    - [Accelerateの設定](#Accelerateの設定)
    - [学習と推論](#学習と推論)
- [その他](#その他)
    - [SageAttentionのインストール方法](#SageAttentionのインストール方法)
    - [PyTorchのバージョンについて](#PyTorchのバージョンについて)
- [免責事項](#免責事項)
- [コントリビューションについて](#コントリビューションについて)
- [ライセンス](#ライセンス)
</details>

## はじめに

このリポジトリは、HunyuanVideo、Wan2.1/2.2、FramePack、FLUX.1 Kontext、Qwen-ImageのLoRA学習用のコマンドラインツールです。このリポジトリは非公式であり、公式のHunyuanVideo、Wan2.1/2.2、FramePack、FLUX.1 Kontext、Qwen-Imageのリポジトリとは関係ありません。

*リポジトリは開発中です。*

### スポンサー

このプロジェクトを支援してくださる企業・団体の皆様に深く感謝いたします。

<a href="https://aihub.co.jp/">
  <img src="./images/logo_aihub.png" alt="AiHUB株式会社" title="AiHUB株式会社" height="100px">
</a>

### スポンサー募集のお知らせ

このプロジェクトがお役に立ったなら、ご支援いただけると嬉しく思います。 [GitHub Sponsors](https://github.com/sponsors/kohya-ss/)で受け付けています。

### 最近の更新

GitHub Discussionsを有効にしました。コミュニティのQ&A、知識共有、技術情報の交換などにご利用ください。バグ報告や機能リクエストにはIssuesを、質問や経験の共有にはDiscussionsをご利用ください。[Discussionはこちら](https://github.com/kohya-ss/musubi-tuner/discussions)

- 2025/10/05
    - エポックの切替をcollate_fnからDataLoaderの取得ループ開始前に変更しました。[PR #601](https://github.com/kohya-ss/musubi-tuner/pull/601)
    - これまでの実装ではエポックの最初のデータ取得後に、ARBバケットがシャッフルされます。そのため、エポックの最初のデータは前エポックのARBソート順で取得されます。これによりエポック内でデータの重複や欠落が起きていました。
    - 各DataSetでは`__getitem__`で共有エポックの変化を検知した直後にARBバケットをシャッフルします。これにより先頭サンプルから新しい順序で取得され、重複・欠落は解消されます。
    - シャッフルタイミングが前倒しになったため、同一シードでも旧実装と同一のサンプル順序にはなりません
    - **学習全体への影響**
        - この修正はエポック境界における先頭サンプルの取り違いを解消するものです。複数エポックで見れば、各サンプルは最終的に欠落・重複なく使用されるため、学習全体に与える影響は軽微です。変更点は「エポック内の消費順序の整合性」を高めるものであり、長期的な学習挙動は同条件では実質的に変わりません（※極端に少ないエポック数や早期打ち切りの場合は順序による差異が観測される可能性があります）。

    - [高度な設定のドキュメント](./docs/advanced_config.md#using-configuration-files-to-specify-training-options--設定ファイルを使用した学習オプションの指定)に、学習時のオプションを設定ファイルで指定する方法を追加しました。[PR #630](https://github.com/kohya-ss/musubi-tuner/pull/630)
    - ドキュメント構成を整理しました。データセット設定に関するドキュメントを`docs/dataset_config.md`に移動しました。

- 2025/10/03
    - 各学習スクリプトで用いられているblock swap機構を改善し、Windows環境における共有GPUメモリの使用量を大きく削減しました。[PR #585](https://github.com/kohya-ss/musubi-tuner/pull/585)
        - block swapのoffload先を共有GPUメモリからCPUメモリに変更しました。これによりトータルでのメモリ使用量は変わりませんが、共有GPUメモリの使用量が大幅に削減されます。
        - たとえば32GBのメインメモリでは、offloadできるのは16GBまででしたが、今回の変更により「32GB-他の使用量」までoffloadできるようになります。
        - 学習速度はわずかに低下します。技術的詳細は[PR #585](https://github.com/kohya-ss/musubi-tuner/pull/585)を参照してください。

- 2025/09/30
    - Qwen-Image-Edit-2509のLoRA学習で複数枚の制御画像を正しく取り扱えない不具合を修正しました。[PR #612](https://github.com/kohya-ss/musubi-tuner/pull/612)

- 2025/09/28
    - [Qwen-Image-Edit-2509](https://github.com/QwenLM/Qwen-Image)の学習、推論に対応しました。[PR #590](https://github.com/kohya-ss/musubi-tuner/pull/590) 詳細は[Qwen-Imageのドキュメント](./docs/qwen_image.md)を参照してください。
        - 複数枚の制御画像を同時に使用できます。Qwen-Image-Edit-2509公式では3枚までですが、Musubi Tunerでは任意の枚数を指定できます（正しく動作するのは3枚までです）。
        - DiTモデルの重みが異なるほか、キャッシュ、学習、推論の各スクリプトに、`--edit_plus`オプションが追加されています。

- 2025/09/24
    - Wan2.2のLoRA学習および推論スクリプトに`--force_v2_1_time_embedding`オプションを追加しました。[PR #586](https://github.com/kohya-ss/musubi-tuner/pull/586) このオプションを指定することでVRAM使用量を削減できます。詳細は[Wanのドキュメント](./docs/wan.md#training--学習)を参照してください。
    
- 2025/09/23
    - `--fp8_scaled`オプションを指定した時の量子化方法を、per-tensorからblock-wise scalingに変更しました。[PR #575](https://github.com/kohya-ss/musubi-tuner/pull/575) [Discussion #564](https://github.com/kohya-ss/musubi-tuner/discussions/564)も参照してください。
        - これによりFP8量子化の精度が向上し、各モデル（HunyuanVideoを除く）学習の安定、推論精度の向上が期待できます。学習、推論速度はわずかに低下します。
        - Qwen-ImageのLoRA学習では、量子化対象モジュールの見直しにより、学習に必要なVRAMが5GB程度削減されます。
        - 詳細は[高度な設定のドキュメント](./docs/advanced_config.md#fp8-weight-optimization-for-models--モデルの重みのfp8への最適化)を参照してください。

### リリースについて

Musubi Tunerの解説記事執筆や、関連ツールの開発に取り組んでくださる方々に感謝いたします。このプロジェクトは開発中のため、互換性のない変更や機能追加が起きる可能性があります。想定外の互換性問題を避けるため、参照用として[リリース](https://github.com/kohya-ss/musubi-tuner/releases)をお使いください。

最新のリリースとバージョン履歴は[リリースページ](https://github.com/kohya-ss/musubi-tuner/releases)で確認できます。

### AIコーディングエージェントを使用する開発者の方へ

このリポジトリでは、ClaudeやGeminiのようなAIエージェントが、プロジェクトの概要や構造を理解しやすくするためのエージェント向け文書（プロンプト）を用意しています。

これらを使用するためには、プロジェクトのルートディレクトリに各エージェント向けの設定ファイルを作成し、明示的に読み込む必要があります。

**セットアップ手順:**

1.  プロジェクトのルートに `CLAUDE.md` や `GEMINI.md` ファイルを作成します。
2.  `CLAUDE.md` に以下の行を追加して、リポジトリが推奨するプロンプトをインポートします（現在、両者はほぼ同じ内容です）：

    ```markdown
    @./.ai/claude.prompt.md
    ```

    Geminiの場合はこちらです：

    ```markdown
    @./.ai/gemini.prompt.md
    ```

3.  インポートした行の後に、必要な指示を適宜追加してください（例：`Always respond in Japanese.`）。

このアプローチにより、共有されたプロジェクトのコンテキストを活用しつつ、エージェントに与える指示を各ユーザーが自由に制御できます。`CLAUDE.md` と `GEMINI.md` はすでに `.gitignore` に記載されているため、リポジトリにコミットされることはありません。

## 概要

### ハードウェア要件

- VRAM: 静止画での学習は12GB以上推奨、動画での学習は24GB以上推奨。
    - *アーキテクチャ、解像度等の学習設定により異なります。*12GBでは解像度 960x544 以下とし、`--blocks_to_swap`、`--fp8_llm`等の省メモリオプションを使用してください。
- メインメモリ: 64GB以上を推奨、32GB+スワップで動作するかもしれませんが、未検証です。

### 特徴

- 省メモリに特化
- Windows対応（Linuxでの動作報告もあります）
- マルチGPU学習（[Accelerate](https://huggingface.co/docs/accelerate/index)を使用）、ドキュメントは後日追加予定

### ドキュメント

各アーキテクチャの詳細、設定、高度な機能については、以下のドキュメントを参照してください。

**アーキテクチャ別:**
- [HunyuanVideo](./docs/hunyuan_video.md)
- [Wan2.1/2.2](./docs/wan.md)
- [Wan2.1/2.2 (1フレーム推論)](./docs/wan_1f.md)
- [FramePack](./docs/framepack.md)
- [FramePack (1フレーム推論)](./docs/framepack_1f.md)
- [FLUX.1 Kontext](./docs/flux_kontext.md)
- [Qwen-Image](./docs/qwen_image.md)

**共通設定・その他:**
- [データセット設定](./docs/dataset_config.md)
- [高度な設定](./docs/advanced_config.md)
- [学習中のサンプル生成](./docs/sampling_during_training.md)
- [ツールとユーティリティ](./docs/tools.md)

## インストール

### pipによるインストール

Python 3.10以上を使用してください（3.10で動作確認済み）。

適当な仮想環境を作成し、ご利用のCUDAバージョンに合わせたPyTorchとtorchvisionをインストールしてください。

PyTorchはバージョン2.5.1以上を使用してください（[補足](#PyTorchのバージョンについて)）。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

以下のコマンドを使用して、必要な依存関係をインストールします。

```bash
pip install -e .
```

オプションとして、FlashAttention、SageAttention（**推論にのみ使用できます**、インストール方法は[こちら](#SageAttentionのインストール方法)を参照）を使用できます。

また、`ascii-magic`（データセットの確認に使用）、`matplotlib`（timestepsの可視化に使用）、`tensorboard`（学習ログの記録に使用）、`prompt-toolkit`を必要に応じてインストールしてください。

`prompt-toolkit`をインストールするとWan2.1およびFramePackのinteractive modeでの編集に、自動的に使用されます。特にLinux環境でプロンプトの編集が容易になります。

```bash
pip install ascii-magic matplotlib tensorboard prompt-toolkit
```

### uvによるインストール

uvを使用してインストールすることもできますが、uvによるインストールは試験的なものです。フィードバックを歓迎します。

#### Linux/MacOS

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

表示される指示に従い、pathを設定してください。

#### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

表示される指示に従い、PATHを設定するか、この時点でシステムを再起動してください。

## モデルのダウンロード

モデルのダウンロード手順はアーキテクチャによって異なります。詳細は[ドキュメント](#ドキュメント)セクションにある、各アーキテクチャのドキュメントを参照してください。

## 使い方

### データセット設定

[こちら](./docs/dataset_config.md)を参照してください。

### 事前キャッシュ

事前キャッシュの手順の詳細は、[ドキュメント](#ドキュメント)セクションにある各アーキテクチャのドキュメントを参照してください。

### Accelerateの設定

`accelerate config`を実行して、Accelerateの設定を行います。それぞれの質問に、環境に応じた適切な値を選択してください（値を直接入力するか、矢印キーとエンターで選択、大文字がデフォルトなので、デフォルト値でよい場合は何も入力せずエンター）。GPU 1台での学習の場合、以下のように答えてください。

```txt
- In which compute environment are you running?: This machine
- Which type of machine are you using?: No distributed training
- Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)?[yes/NO]: NO
- Do you wish to optimize your script with torch dynamo?[yes/NO]: NO
- Do you want to use DeepSpeed? [yes/NO]: NO
- What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]: all
- Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: NO
- Do you wish to use mixed precision?: bf16
```

※場合によって ``ValueError: fp16 mixed precision requires a GPU`` というエラーが出ることがあるようです。この場合、6番目の質問（
``What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:``）に「0」と答えてください。（id `0`、つまり1台目のGPUが使われます。）

### 学習と推論

学習と推論の手順はアーキテクチャによって大きく異なります。詳細な手順については、[ドキュメント](#ドキュメント)セクションにある対応するアーキテクチャのドキュメント、および各種の設定のドキュメントを参照してください。

## その他

### SageAttentionのインストール方法

sdbds氏によるWindows対応のSageAttentionのwheelが https://github.com/sdbds/SageAttention-for-windows で公開されています。triton をインストールし、Python、PyTorch、CUDAのバージョンが一致する場合は、[Releases](https://github.com/sdbds/SageAttention-for-windows/releases)からビルド済みwheelをダウンロードしてインストールすることが可能です。sdbds氏に感謝します。

参考までに、以下は、SageAttentionをビルドしインストールするための簡単な手順です。Microsoft Visual C++ 再頒布可能パッケージを最新にする必要があるかもしれません。

1. Pythonのバージョンに応じたtriton 3.1.0のwhellを[こちら](https://github.com/woct0rdho/triton-windows/releases/tag/v3.1.0-windows.post5)からダウンロードしてインストールします。

2. Microsoft Visual Studio 2022かBuild Tools for Visual Studio 2022を、C++のビルドができるよう設定し、インストールします。（上のRedditの投稿を参照してください）。

3. 任意のフォルダにSageAttentionのリポジトリをクローンします。
    ```shell
    git clone https://github.com/thu-ml/SageAttention.git
    ```

4. スタートメニューから Visual Studio 2022 内の `x64 Native Tools Command Prompt for VS 2022` を選択してコマンドプロンプトを開きます。

5. venvを有効にし、SageAttentionのフォルダに移動して以下のコマンドを実行します。DISTUTILSが設定されていない、のようなエラーが出た場合は `set DISTUTILS_USE_SDK=1`としてから再度実行してください。
    ```shell
    python setup.py install
    ```

以上でSageAttentionのインストールが完了です。

### PyTorchのバージョンについて

`--attn_mode`に`torch`を指定する場合、2.5.1以降のPyTorchを使用してください（それより前のバージョンでは生成される動画が真っ黒になるようです）。

古いバージョンを使う場合、xformersやSageAttentionを使用してください。

## 免責事項

このリポジトリは非公式であり、サポートされているアーキテクチャの公式リポジトリとは関係ありません。また、このリポジトリは開発中で、実験的なものです。テストおよびフィードバックを歓迎しますが、以下の点にご注意ください：

- 実際の稼働環境での動作を意図したものではありません
- 機能やAPIは予告なく変更されることがあります
- いくつもの機能が未検証です
- 動画学習機能はまだ開発中です

問題やバグについては、以下の情報とともにIssueを作成してください：

- 問題の詳細な説明
- 再現手順
- 環境の詳細（OS、GPU、VRAM、Pythonバージョンなど）
- 関連するエラーメッセージやログ

## コントリビューションについて

コントリビューションを歓迎します。 [CONTRIBUTING.md](./CONTRIBUTING.md)および[CONTRIBUTING.ja.md](./CONTRIBUTING.ja.md)をご覧ください。

## ライセンス

`hunyuan_model`ディレクトリ以下のコードは、[HunyuanVideo](https://github.com/Tencent/HunyuanVideo)のコードを一部改変して使用しているため、そちらのライセンスに従います。

`wan`ディレクトリ以下のコードは、[Wan2.1](https://github.com/Wan-Video/Wan2.1)のコードを一部改変して使用しています。ライセンスはApache License 2.0です。

`frame_pack`ディレクトリ以下のコードは、[frame_pack](https://github.com/lllyasviel/FramePack)のコードを一部改変して使用しています。ライセンスはApache License 2.0です。

他のコードはApache License 2.0に従います。一部Diffusersのコードをコピー、改変して使用しています。
