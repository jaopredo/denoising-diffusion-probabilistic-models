.PHONY: setup run

# Pacotes que o uv sync NÃO deve instalar nem mexer: quem cuida deles é o uv pip
TORCH_PKGS = --no-install-package torch \
             --no-install-package torchvision \
             --no-install-package triton

setup:
	uv sync --inexact $(TORCH_PKGS)
	uv pip install --reinstall torch torchvision --torch-backend=auto
	uv pip install "diffusers[torch]"