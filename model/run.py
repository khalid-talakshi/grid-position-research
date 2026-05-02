from model.basic import run_model as run_basic
from model.street import run_model as run_street
from model.era import run_model as run_era

if __name__ == "__main__":
    print("Running basic model...")
    run_basic()
    print("Basic model completed!\n")

    print("Running street model...")
    run_street()
    print("Street model completed!\n")

    print("Running era model...")
    run_era()
    print("Era model completed!")