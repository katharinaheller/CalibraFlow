from pathlib import Path
import logging

from core.loaders.csv_dataset_loader import CsvDatasetLoader
from core.loaders.loader_orchestrator import LoaderOrchestrator
from core.loaders.dataset_ids import DatasetId


def configure_logging() -> None:
    # configures root logger for the application
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def main() -> None:
    # sets up pipeline and logs basic dataset information
    configure_logging()
    logger = logging.getLogger(__name__)

    base_path = Path(__file__).resolve().parent  # project root if main.py is located there
    csv_loader = CsvDatasetLoader(base_path=base_path)
    orchestrator = LoaderOrchestrator(csv_loader)

    synthetic_data = orchestrator.load_all_synthetic()

    for ds_id, df in synthetic_data.items():
        logger.info("=== %s ===", ds_id.value)
        logger.info("Rows: %s, Columns: %s", df.height, df.width)
        logger.info("Schema: %s", df.schema)
        # logs first rows for transparency without dumping full dataset
        logger.info("Head preview:\n%s", df.head(3))


if __name__ == "__main__":
    main()
