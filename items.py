import random as rd


def get_test_data(N=200, seed=42):
    """
    Generate test data with a fixed seed for reproducibility
    N: number of items to generate (default is 4)
    seed: random seed (default is 42)
    """
    rd.seed(seed)  # Set fixed seed

    items = []
    for i in range(N):
        item = {
            "id": i + 1,
            "width": rd.randint(1, 8),
            "height": rd.randint(1, 8)
        }
        items.append(item)
    return items


# Example usage
if __name__ == "__main__":
    data = get_test_data()
    print(data)
