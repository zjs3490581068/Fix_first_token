"""Two fixed tokens followed by at most 16382 generated tokens, one sample per problem."""
from .common import main

CUE = " To determine"
PREFIX = [2014, 8253]
CONDITIONS = [("Base_two_token_To_determine", CUE, PREFIX)]


if __name__ == "__main__":
    main("experiments.two_token", "two_token.json", CONDITIONS)
