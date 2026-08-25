from pathlib import Path
import click


@click.group()
def cli():
    """Model training CLI."""
    pass


@cli.command()
@click.option(
    "--config",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Training configuration file.",
)
@click.option(
    "--resume",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Resume from checkpoint.",
)
def train(config: Path, resume: Path | None):
    """Train a model."""
    # TODO:
    # cfg = load_config(config)
    # trainer = Trainer(cfg)
    # trainer.train(resume=resume)
    raise NotImplementedError


@cli.command()
@click.option(
    "--checkpoint",
    type=click.Path(path_type=Path, exists=True),
    required=True,
)
@click.option(
    "--config",
    type=click.Path(path_type=Path, exists=True),
    required=True,
)
def evaluate(checkpoint: Path, config: Path):
    """Evaluate a trained model."""
    # TODO:
    # cfg = load_config(config)
    # evaluator = Evaluator(cfg)
    # evaluator.evaluate(checkpoint)
    raise NotImplementedError


@cli.command()
@click.option(
    "--checkpoint",
    type=click.Path(path_type=Path, exists=True),
    required=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
)
def export(checkpoint: Path, output: Path):
    """Export a model for inference."""
    # TODO:
    # export_model(checkpoint, output)
    raise NotImplementedError


@cli.command("list-checkpoints")
@click.option(
    "--directory",
    type=click.Path(path_type=Path, exists=True),
    default=Path("checkpoints"),
)
def list_checkpoints(directory: Path):
    """List available checkpoints."""
    # TODO:
    raise NotImplementedError


@cli.command()
@click.option(
    "--config",
    type=click.Path(path_type=Path, exists=True),
    required=True,
)
def validate(config: Path):
    """Validate a training configuration."""
    # TODO:
    raise NotImplementedError


if __name__ == "__main__":
    cli()
