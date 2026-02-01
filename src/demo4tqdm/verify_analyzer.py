from code_analyzer import CodeAnalyzer

def main():
    analyzer = CodeAnalyzer()
    target_file = "./test_sample.py"
    
    print(f"Analyzing {target_file}...")
    result = analyzer.analyze_file(target_file)
    
    class_count = len(result.classes)
    func_count = len(result.functions)
    method_count = sum(len(c.methods) for c in result.classes)
    
    print("-" * 30)
    print(f"类定义数：{class_count}")
    print(f"函数定义数：{func_count}")
    print(f"方法定义数：{method_count}")
    print("-" * 30)

    # expected quota
    expected_classes = 2
    expected_funcs = 3
    expected_methods = 3
    
    if class_count == expected_classes and func_count == expected_funcs and method_count == expected_methods:
        print("输出与预期结果一致。")
    else:
        print("输出与预期结果不一致。")

if __name__ == "__main__":
    main()
