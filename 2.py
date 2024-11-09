import numpy as np

# Create the 30x30 matrix from your data
matrix = [
    [0.000, 0.000, 4.000, 0.000, 22.000, 2.000, 1.000, 4.000, 1.000, 55.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.000,
     0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 81.000, 1.000, 0.000, 0.000, 1.000, 22.000, 2.000, 1.000],
    [0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 1.000, 1.000, 2.000, 0.000, 9.000, 0.000, 0.000, 67.000, 2.000, 0.000,
     1.000, 0.000, 0.000, 0.000, 2.000, 0.000, 0.000, 0.000, 0.000, 20.000, 0.000, 0.000, 0.000, 0.000],
    [4.000, 0.000, 0.000, 68.000, 20.000, 12.000, 0.000, 0.000, 16.000, 2.000, 0.000, 0.000, 0.000, 2.000, 1.000, 2.000,
     1.000, 0.000, 0.000, 0.000, 23.000, 0.000, 0.000, 41.000, 1.000, 1.000, 0.000, 1.000, 3.000, 1.000],
    [0.000, 0.000, 68.000, 0.000, 0.000, 0.000, 22.000, 5.000, 93.000, 1.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000,
     0.000, 0.000, 1.000, 2.000, 1.000, 1.000, 0.000, 0.000, 0.000, 2.000, 0.000, 2.000, 0.000, 1.000],
    [22.000, 1.000, 20.000, 0.000, 0.000, 0.000, 0.000, 13.000, 0.000, 23.000, 0.000, 0.000, 2.000, 1.000, 0.000,
     17.000, 2.000, 1.000, 0.000, 1.000, 0.000, 1.000, 19.000, 0.000, 34.000, 10.000, 1.000, 1.000, 30.000, 1.000],
    [2.000, 0.000, 12.000, 0.000, 0.000, 0.000, 1.000, 12.000, 0.000, 0.000, 0.000, 25.000, 1.000, 0.000, 0.000, 0.000,
     86.000, 0.000, 18.000, 3.000, 0.000, 3.000, 0.000, 1.000, 0.000, 3.000, 20.000, 0.000, 1.000, 11.000],
    [1.000, 1.000, 0.000, 22.000, 0.000, 1.000, 0.000, 1.000, 1.000, 32.000, 0.000, 5.000, 0.000, 0.000, 19.000, 5.000,
     1.000, 5.000, 0.000, 20.000, 0.000, 21.000, 0.000, 42.000, 0.000, 20.000, 1.000, 1.000, 1.000, 0.000],
    [4.000, 1.000, 0.000, 5.000, 13.000, 12.000, 1.000, 0.000, 76.000, 0.000, 1.000, 1.000, 1.000, 0.000, 7.000, 0.000,
     1.000, 0.000, 0.000, 0.000, 1.000, 21.000, 1.000, 51.000, 1.000, 0.000, 0.000, 0.000, 0.000, 2.000],
    [1.000, 2.000, 16.000, 93.000, 0.000, 0.000, 1.000, 76.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
     2.000, 1.000, 0.000, 0.000, 1.000, 0.000, 0.000, 1.000, 0.000, 2.000, 1.000, 2.000, 1.000, 0.000],
    [55.000, 0.000, 2.000, 1.000, 23.000, 0.000, 32.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000, 30.000,
     19.000, 1.000, 0.000, 0.000, 0.000, 1.000, 1.000, 6.000, 0.000, 1.000, 0.000, 0.000, 22.000, 0.000, 4.000],
    [0.000, 9.000, 0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 53.000, 93.000, 1.000, 1.000,
     1.000, 0.000, 6.000, 0.000, 0.000, 1.000, 0.000, 1.000, 2.000, 30.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 25.000, 5.000, 1.000, 0.000, 0.000, 0.000, 0.000, 18.000, 22.000, 1.000, 2.000,
     1.000, 0.000, 49.000, 1.000, 0.000, 0.000, 0.000, 0.000, 3.000, 45.000, 26.000, 0.000, 0.000, 1.000],
    [0.000, 0.000, 0.000, 0.000, 2.000, 1.000, 0.000, 1.000, 0.000, 2.000, 53.000, 18.000, 0.000, 0.000, 0.000, 0.000,
     1.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 22.000, 0.000, 94.000, 3.000, 1.000, 0.000],
    [0.000, 67.000, 2.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 93.000, 22.000, 0.000, 0.000, 1.000, 0.000,
     1.000, 1.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000, 8.000, 0.000, 1.000, 0.000, 0.000],
    [1.000, 2.000, 1.000, 0.000, 0.000, 0.000, 19.000, 7.000, 0.000, 30.000, 1.000, 1.000, 0.000, 1.000, 0.000, 2.000,
     0.000, 0.000, 0.000, 1.000, 0.000, 23.000, 2.000, 5.000, 0.000, 0.000, 11.000, 0.000, 92.000, 1.000],
    [1.000, 0.000, 2.000, 0.000, 17.000, 0.000, 5.000, 0.000, 0.000, 19.000, 1.000, 2.000, 0.000, 0.000, 2.000, 0.000,
     0.000, 0.000, 1.000, 96.000, 2.000, 0.000, 0.000, 0.000, 22.000, 7.000, 0.000, 1.000, 0.000, 22.000],
    [0.000, 1.000, 1.000, 0.000, 2.000, 86.000, 1.000, 1.000, 2.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 6.000, 0.000, 2.000, 1.000, 3.000, 0.000, 2.000, 87.000],
    [0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 5.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000,
     0.000, 0.000, 0.000, 0.000, 91.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 1.000, 0.000, 18.000, 0.000, 0.000, 0.000, 0.000, 6.000, 49.000, 0.000, 1.000, 0.000, 1.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 92.000, 1.000, 5.000, 0.000, 0.000, 25.000],
    [1.000, 0.000, 0.000, 2.000, 1.000, 3.000, 20.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.000, 0.000, 1.000, 96.000,
     0.000, 0.000, 0.000, 0.000, 49.000, 24.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 2.000, 23.000, 1.000, 0.000, 0.000, 0.000, 1.000, 1.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.000,
     0.000, 91.000, 0.000, 49.000, 0.000, 0.000, 0.000, 29.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 1.000, 1.000, 3.000, 21.000, 21.000, 0.000, 1.000, 1.000, 0.000, 0.000, 0.000, 23.000, 0.000,
     0.000, 0.000, 1.000, 24.000, 0.000, 0.000, 5.000, 2.000, 0.000, 10.000, 0.000, 85.000, 1.000, 0.000],
    [81.000, 0.000, 0.000, 0.000, 19.000, 0.000, 0.000, 1.000, 0.000, 6.000, 0.000, 0.000, 0.000, 0.000, 2.000, 0.000,
     6.000, 0.000, 0.000, 0.000, 0.000, 5.000, 0.000, 0.000, 14.000, 0.000, 22.000, 7.000, 3.000, 34.000],
    [1.000, 0.000, 41.000, 0.000, 0.000, 1.000, 42.000, 51.000, 1.000, 0.000, 1.000, 0.000, 0.000, 0.000, 5.000, 0.000,
     0.000, 0.000, 0.000, 1.000, 29.000, 2.000, 0.000, 0.000, 1.000, 15.000, 7.000, 0.000, 0.000, 1.000],
    [0.000, 0.000, 1.000, 0.000, 34.000, 0.000, 0.000, 1.000, 0.000, 1.000, 2.000, 3.000, 22.000, 2.000, 0.000, 22.000,
     2.000, 1.000, 92.000, 0.000, 0.000, 0.000, 14.000, 1.000, 0.000, 1.000, 0.000, 1.000, 0.000, 0.000],
    [0.000, 20.000, 1.000, 2.000, 10.000, 3.000, 20.000, 0.000, 2.000, 0.000, 30.000, 45.000, 0.000, 8.000, 0.000,
     7.000, 1.000, 0.000, 1.000, 0.000, 0.000, 10.000, 0.000, 15.000, 1.000, 0.000, 3.000, 1.000, 14.000, 6.000],
    [1.000, 0.000, 0.000, 0.000, 1.000, 20.000, 1.000, 0.000, 1.000, 0.000, 0.000, 26.000, 94.000, 0.000, 11.000, 0.000,
     3.000, 0.000, 5.000, 0.000, 0.000, 0.000, 22.000, 7.000, 0.000, 3.000, 0.000, 3.000, 0.000, 1.000],
    [22.000, 0.000, 1.000, 2.000, 1.000, 0.000, 1.000, 0.000, 2.000, 22.000, 0.000, 0.000, 3.000, 1.000, 0.000, 1.000,
     0.000, 0.000, 0.000, 0.000, 0.000, 85.000, 7.000, 0.000, 1.000, 1.000, 3.000, 0.000, 47.000, 0.000],
    [2.000,0.000,3.000,0.000,30.000,1.000,1.000,0.000,1.000,0.000,0.000,0.000,1.000,0.000,92.000,0.000,2.000,0.000,0.000,0.000,0.000,1.000,3.000,0.000,0.000,14.000,0.000,47.000,0.000,1.000],
    [1.000,0.000,1.000,1.000,1.000,11.000,0.000,2.000,0.000,4.000,0.000,1.000,0.000,0.000,1.000,22.000,87.000,0.000,25.000,0.000,0.000,0.000,34.000,1.000,0.000,6.000,1.000,0.000,1.000,0.000]
]
def get_strong_pairs(matrix, threshold=50):
    """找出所有强连接对"""
    pairs = []
    n = len(matrix)
    used_nodes = set()

    # 按强度从高到低排序所有连接
    connections = []
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] >= threshold:
                connections.append((i, j, matrix[i][j]))

    # 按强度排序
    connections.sort(key=lambda x: x[2], reverse=True)

    # 收集强连接对，避免节点重复使用
    for i, j, strength in connections:
        if i not in used_nodes and j not in used_nodes:
            pairs.append((i, j, strength))
            used_nodes.add(i)
            used_nodes.add(j)

    return pairs, used_nodes


def complete_sequence_greedy(matrix, strong_pairs, used_nodes):
    """使用贪婪方法补全序列"""
    n = len(matrix)
    sequence = []

    # 先把强连接对排好
    for pair in strong_pairs:
        if not sequence:
            sequence.extend([pair[0], pair[1]])
        else:
            if pair[0] not in sequence and pair[1] not in sequence:
                # 找最佳插入位置
                best_pos = 0
                best_score = -1
                for i in range(len(sequence) + 1):
                    # 尝试在位置i插入
                    test_seq = sequence[:i] + [pair[0], pair[1]] + sequence[i:]
                    score = evaluate_sequence(test_seq, matrix)
                    if score > best_score:
                        best_score = score
                        best_pos = i
                sequence[best_pos:best_pos] = [pair[0], pair[1]]

    # 处理剩余节点
    remaining = set(range(n)) - set(sequence)
    while remaining:
        best_node = None
        best_pos = None
        best_score = float('-inf')

        for node in remaining:
            # 尝试插入每个位置
            for i in range(len(sequence) + 1):
                test_seq = sequence[:i] + [node] + sequence[i:]
                score = evaluate_sequence(test_seq, matrix)
                if score > best_score:
                    best_score = score
                    best_node = node
                    best_pos = i

        sequence.insert(best_pos, best_node)
        remaining.remove(best_node)

    return sequence


def evaluate_sequence(sequence, matrix):
    """评估序列质量"""
    score = 0
    for i in range(len(sequence) - 1):
        score += matrix[sequence[i]][sequence[i + 1]]
    return score


def format_sequence_info(sequence, matrix):
    """格式化序列信息"""
    result = "Sequence Analysis:\n"
    result += "-" * 50 + "\n"

    # 总体信息
    total_score = evaluate_sequence(sequence, matrix)
    result += f"Total Connection Strength: {total_score:.1f}\n"
    result += f"Sequence Length: {len(sequence)}\n\n"

    # 完整序列
    result += "Complete Sequence:\n"
    result += " -> ".join(f"{node:2d}" for node in sequence) + "\n\n"

    # 连接详情
    result += "Connections in Sequence:\n"
    for i in range(len(sequence) - 1):
        node1 = sequence[i]
        node2 = sequence[i + 1]
        strength = matrix[node1][node2]
        result += f"{node1:2d}-{node2:<2d}: {strength:5.1f}\n"

    return result


# 主处理函数
def process_matrix(matrix, threshold=50):
    # 1. 找出强连接对
    strong_pairs, used_nodes = get_strong_pairs(matrix, threshold)

    # 2. 使用贪婪方法补全序列
    sequence = complete_sequence_greedy(matrix, strong_pairs, used_nodes)

    # 3. 生成分析报告
    report = "Strong Pairs (>= 50):\n"
    report += "-" * 50 + "\n"
    for i, j, strength in sorted(strong_pairs, key=lambda x: x[2], reverse=True):
        report += f"{i:2d}-{j:<2d}: {strength:5.1f}\n"

    report += "\n" + format_sequence_info(sequence, matrix)

    return report


# 运行分析
result = process_matrix(matrix)
print(result)

# 保存到文件
with open('matrix_analysis.txt', 'w') as f:
    f.write(result)