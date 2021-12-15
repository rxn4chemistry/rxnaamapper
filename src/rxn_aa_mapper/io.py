"""I/O utilities."""
import random
from itertools import takewhile
from typing import Generator, List

from loguru import logger


def lines_generator(filepath: str, skip_first_n_lines: int = 0) -> Generator:
    """Generate continuosly lines from file.

    Args:
        filepath: path to the file.
        skip_first_n_lines: skip a fixed number of lines at the beginning of the file.
            Defaults to 0.

    Yields:
        a line from the file, None every time the iterator goes through all file lines.
    """
    while True:
        with open(filepath, "rt") as fp:
            for _ in range(skip_first_n_lines):
                _ = next(fp)
            for line in fp:
                yield line
            logger.info(f"looping through file {filepath} completed")
            yield None


def random_lines_from_filepath(
    filepath: str, number_of_lines: int, seed: int = 42, skip_first_n_lines: int = 0
) -> List[str]:
    """Sample a fixed number of lines from a file randomly.

    Args:
        filepath: path to the file.
        number_of_lines: number of lines to sample.
        seed: seed for random sampling. Defaults to 42.
        skip_first_n_lines: skip a fixed number of lines at the beginning of the file.
            Defaults to 0.

    Returns:
        random lines sampled from the file.
    """
    sampled_lines: List[str] = []
    random.seed(seed)
    file_lines = 0
    while len(sampled_lines) <= number_of_lines:
        lines = lines_generator(filepath, skip_first_n_lines=skip_first_n_lines)
        for line_index, line in enumerate(lines, 0):
            if line is None and file_lines < 1:
                if line_index < 1:
                    logger.warning(f"file: {filepath} is empty!")
                    break
                file_lines = line_index
                if file_lines < number_of_lines:
                    logger.warning(f"returning all lines from file {filepath}")
                    sampled_lines = [
                        line.strip()
                        for line in takewhile(lambda line: line is not None, lines)
                    ]
                    break
            elif line:
                if random.random() >= 0.5:
                    sampled_lines.append(line.strip())
                    if len(sampled_lines) >= number_of_lines:
                        break
        else:
            continue
        break
    return sampled_lines[:number_of_lines]
