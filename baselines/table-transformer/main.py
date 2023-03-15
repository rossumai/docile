import table_transformer
from pytorch_lightning.cli import LightningCLI


def cli_main():
    cli = LightningCLI(table_transformer.TableDetr)  # noqa: F841


if __name__ == "__main__":
    cli_main()
