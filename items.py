import random as rd
import json  # 添加json导入


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


def save_to_json(data, filename):
    """保存数据到JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# Example usage
if __name__ == "__main__":
    data = get_test_data()
    save_to_json(data, 'generated_data.json')
    print(data)
