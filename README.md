# Tutorial de Diffusers

Instalação local do Hugging Face Diffusers usando `uv` e PyTorch com CUDA 13.2.

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

## 4. Verificar a instalação

```powershell
python -c "import torch, diffusers, transformers, torchvision; print('PyTorch:', torch.__version__); print('CUDA do PyTorch:', torch.version.cuda); print('GPU disponível:', torch.cuda.is_available())"
```

Confira se `CUDA do PyTorch` corresponde a `13.2` e se `GPU disponível` retorna `True` antes de executar o exemplo com CUDA.

## 5. Executar o exemplo

```powershell
python pass1.py
```

Na primeira execução, os arquivos do modelo são baixados. O exemplo salva a imagem gerada em `outputs/z_image/base_example_42.png`.

## Referências

- [Documentação do Diffusers](https://huggingface.co/docs/diffusers/installation)
- [Documentação do uv](https://docs.astral.sh/uv/)
- [Índice oficial do PyTorch para CUDA 13.2](https://download.pytorch.org/whl/cu132/)
