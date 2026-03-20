import os
import sys
import argparse

def setup_paths():
    """
    Add project root (2 levels up) to PYTHONPATH
    """
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if ROOT not in sys.path:
        sys.path.append(ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Run VAE script")

    # List of strings argument
    parser.add_argument(
        "--regions",
        nargs="+",                # accepts 1 or more values
        type=str,
        required=True,
        help="List of region names"
    )

    parser.add_argument(
        "--path_models",
        type=str,
        required=False,
        help="List of subjects"
    )

    parser.add_argument(
        "--test_UKB",
        type=str,
        required=True,
        help="Path to test UKB dataset"
    )



    return parser.parse_args()


def main():
    args = parse_args()

    # Now you can safely import
    from beta_vae import VAE

    print("Running main...")
    print("Regions:", args.regions)

    # Example usage
    for region in args.regions:
        print(f"Processing {region}")



    print("Result:")


if __name__ == "__main__":
    setup_paths()
    main()