# Tutorial de Diffusers

Série de tutoriais de Hugging Face Diffusers, com scripts na pasta `scripts` e instalação local usando `uv` e PyTorch com CUDA 13.2.

## Pré-requisitos

- Python instalado.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) instalado e disponível no terminal.
- GPU NVIDIA e driver compatível com a versão CUDA do PyTorch instalada.

## 1. Obter o repositório

```powershell
git clone https://github.com/DEVAIEXP/diffusers-tutorial.git
cd diffusers-tutorial
```

Se você já está na pasta do projeto, siga para a criação do ambiente virtual.

## 2. Criar e ativar o ambiente virtual

No Windows, usando PowerShell:

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
```

No Linux ou macOS, a ativação é feita com `source .venv/bin/activate`. A instalação CUDA abaixo é destinada a máquinas com GPU NVIDIA compatível.

## 3. Instalar as dependências

Instale Diffusers, Transformers e Torchvision na mesma linha, usando o índice do PyTorch para CUDA 13.2:

```powershell
uv pip install "diffusers[torch]" transformers torchvision --extra-index-url https://download.pytorch.org/whl/cu132
```

O extra `diffusers[torch]` inclui as dependências de integração com PyTorch. O `transformers` fornece os componentes de modelos de texto, e o `torchvision` oferece utilitários de visão computacional.

### Suporte opcional à quantização

Para acompanhar os tutoriais que utilizam quantização com `bitsandbytes`, instale também essa biblioteca com o ambiente virtual ativado:

```powershell
uv pip install bitsandbytes
```

### Suporte opcional à compilação com Triton

A etapa 4 (`scripts/quickstart/tutorial_step_4.py`) utiliza `compile_repeated_blocks(fullgraph=True)` para compilar os blocos repetidos do transformer com `torch.compile`. Para acompanhar essa etapa, instale o Triton com o ambiente virtual ativado, usando o comando correspondente ao seu sistema operacional.

**Windows (PowerShell):**

```powershell
uv pip install "triton-windows==3.7.1.post27"
```

**Linux:**

```bash
uv pip install "triton>=3.7,<3.8"
```

### Instalação opcional do SDNQ para a etapa 5

Para acompanhar a etapa 5 da série, instale também a biblioteca `sdnq` com o ambiente virtual ativado:

```powershell
uv pip install sdnq
```

## 4. Verificar a instalação

```powershell
python -c "import torch, diffusers, transformers, torchvision; print('PyTorch:', torch.__version__); print('CUDA do PyTorch:', torch.version.cuda); print('GPU disponível:', torch.cuda.is_available())"
```

Confira se `CUDA do PyTorch` corresponde a `13.2` e se `GPU disponível` retorna `True` antes de executar o exemplo com CUDA.

## 5. Acompanhar a série de tutoriais

Os scripts da série ficam na pasta `scripts`, organizados por tutorial. Execute os comandos a partir da raiz do repositório, com o ambiente virtual ativado.

### Quickstart — etapa 1

O script `scripts/quickstart/tutorial_step_1.py` carrega o modelo `Tongyi-MAI/Z-Image` com `DiffusionPipeline` e gera uma imagem a partir de um prompt de texto, usando CUDA e 25 passos de inferência.

```powershell
python scripts/quickstart/tutorial_step_1.py
```

Na primeira execução, os arquivos do modelo são baixados. A imagem gerada é salva em `outputs/z_image/result.png`. A pasta `outputs/` está no `.gitignore` para manter os resultados gerados fora do versionamento.

## Referências

- [Documentação do Diffusers](https://huggingface.co/docs/diffusers/installation)
- [Documentação do uv](https://docs.astral.sh/uv/)
- [Índice oficial do PyTorch para CUDA 13.2](https://download.pytorch.org/whl/cu132/)
