# 完全复制自父目录
import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:  # 有则改用radon计算
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit

    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False


@dataclass
class FunctionInfo:
    name: str
    lineno: int  # line number
    end_lineno: int
    args_count: int
    decorators: List[str] = field(default_factory=list)
    docstring: str = None
    complexity: int = 0  # 环形复杂度


@dataclass
class ClassInfo:
    name: str
    lineno: int
    end_lineno: int
    methods: List[FunctionInfo] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)  # 基类
    decorators: List[str] = field(default_factory=list)
    docstring: str = None


@dataclass
class FileAnalysis:
    filepath: str
    lines_of_code: int  # 有效代码行数
    imports: List[str] = field(default_factory=list)
    from_imports: List[Dict] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    global_variables: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    maintainability_index: float = 0.0  # 维护性指标


class CodeAnalyzer:
    """
    Python源代码静态分析器，使用ast解析提取代码结构信息与复杂度指标。
    """

    def analyze_file(self, filepath: str) -> FileAnalysis:
        """
        分析单个Python文件，提取其结构信息与代码指标。

        Args:
            filepath (str): Python文件的路径

        Returns:
            FileAnalysis: 包含文件结构和指标的分析结果
        """
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        lines_of_code = len(
            [
                l
                for l in source.split("\n")
                if l.strip() and not l.strip().startswith("#")
            ]
        )
        tree = ast.parse(source)  # 构建ast语法树

        result = FileAnalysis(filepath=filepath, lines_of_code=lines_of_code)

        # 获取依赖关系
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    result.from_imports.append(
                        {"module": module, "name": alias.name, "alias": alias.asname}
                    )

        # 查找函数、类和全局变量
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result.functions.append(self._analyze_function(node))
            elif isinstance(node, ast.ClassDef):
                result.classes.append(self._analyze_class(node))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        result.global_variables.append(target.id)

        if RADON_AVAILABLE:
            cc_results = cc_visit(source)
            if cc_results:
                result.complexity_score = sum(b.complexity for b in cc_results) / len(
                    cc_results
                )
            result.maintainability_index = mi_visit(source, True)

        return result

    def _analyze_function(self, node) -> FunctionInfo:
        """分析函数节点，获取函数信息。"""
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                decorators.append(dec.func.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(dec.attr)
        docstring = ast.get_docstring(node)
        return FunctionInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            args_count=len(node.args.args),
            decorators=decorators,
            docstring=docstring,
            complexity=self._calculate_complexity(node),
        )

    def _analyze_class(self, node) -> ClassInfo:
        """分析类节点，获取类的信息及其方法。"""
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._analyze_function(item))

        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)

        decorators = [
            dec.id for dec in node.decorator_list if isinstance(dec, ast.Name)
        ]
        docstring = ast.get_docstring(node)

        return ClassInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            methods=methods,
            bases=bases,
            decorators=decorators,
            docstring=docstring[:100] + "..."
            if docstring and len(docstring) > 100
            else docstring,
        )

    def _calculate_complexity(self, node) -> int:
        """
        计算函数的环形复杂度。
        当一个函数的环形复杂度较大时，通常就意味着这个函数承担了太多的责任，需要把它拆分成多个小函数。
        """
        complexity = 1  # 基础复杂度为1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1  # 分支+1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1  # 异常+1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1  # 等效多个表达式
            elif isinstance(
                child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
            ):
                complexity += 1  # 各种推导式等+1
        return complexity

    def count_definitions(self, analysis: FileAnalysis) -> Dict[str, int]:
        """
        统计文件中各类代码定义的数量。即一个文件里有多少个类、函数、变量等等。
        """
        total_methods = sum(len(c.methods) for c in analysis.classes)
        return {
            "functions": len(analysis.functions),
            "classes": len(analysis.classes),
            "methods": total_methods,
            "imports": len(analysis.imports) + len(analysis.from_imports),
            "global_variables": len(analysis.global_variables),
            "lines_of_code": analysis.lines_of_code,
        }

    def analyze_directory(self, dir_path: str) -> List[FileAnalysis]:
        """
        递归扫描整个文件夹并跳过 __pycache__ 目录.
        """
        results = []
        for py_file in Path(dir_path).rglob("*.py"):
            if "__pycache__" not in str(py_file):
                results.append(self.analyze_file(str(py_file)))
        return results

    def get_complexity_report(self, analyses: List[FileAnalysis]) -> pd.DataFrame:
        """
        生成代码复杂度统计报告表。

        Args:
            analyses (List[FileAnalysis]): 文件分析结果列表

        Returns:
            pd.DataFrame: 各文件的复杂度统计表，包含代码行数、函数/类数量、平均/最大复杂度等
        """
        rows = []
        for analysis in analyses:
            all_functions = list(analysis.functions)
            for cls in analysis.classes:
                all_functions.extend(cls.methods)

            avg_complexity = (
                sum(f.complexity for f in all_functions) / len(all_functions)
                if all_functions
                else 0
            )
            max_complexity = max((f.complexity for f in all_functions), default=0)

            rows.append(
                {
                    "file": os.path.basename(analysis.filepath),
                    "lines_of_code": analysis.lines_of_code,
                    "functions": len(analysis.functions),
                    "classes": len(analysis.classes),
                    "methods": sum(len(c.methods) for c in analysis.classes),
                    "avg_complexity": round(avg_complexity, 2),
                    "max_complexity": max_complexity,
                    "maintainability_index": round(analysis.maintainability_index, 2),
                }
            )
        return pd.DataFrame(rows)

    def find_complex_functions(
        self, analyses: List[FileAnalysis], threshold: int = 10
    ) -> pd.DataFrame:
        """
        查找环形复杂度超过传入阈值的函数。

        Args:
            analyses (List[FileAnalysis]): 文件分析结果列表
            threshold (int): 环形复杂度阈值，默认为10

        Returns:
            pd.DataFrame: 高复杂度函数列表，按复杂度降序排列
        """
        complex_funcs = []
        for analysis in analyses:
            filename = os.path.basename(analysis.filepath)
            for func in analysis.functions:
                if func.complexity >= threshold:
                    complex_funcs.append(
                        {
                            "file": filename,
                            "function": func.name,
                            "line": func.lineno,
                            "complexity": func.complexity,
                            "type": "function",
                        }
                    )
            for cls in analysis.classes:
                for method in cls.methods:
                    if method.complexity >= threshold:
                        complex_funcs.append(
                            {
                                "file": filename,
                                "function": f"{cls.name}.{method.name}",
                                "line": method.lineno,
                                "complexity": method.complexity,
                                "type": "method",
                            }
                        )
        df = pd.DataFrame(complex_funcs)
        if not df.empty:
            df = df.sort_values("complexity", ascending=False).reset_index(drop=True)
        return df


def main():
    import sys

    if len(sys.argv) >= 2:
        target = sys.argv[1]
    else:  # 方便直接运行
        target = "../tqdm"
    analyzer = CodeAnalyzer()

    if os.path.isfile(target):
        result = analyzer.analyze_file(target)
        stats = analyzer.count_definitions(result)
        print("代码统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        results = analyzer.analyze_directory(target)
        print(f"共分析了 {len(results)} 个文件：")
        report = analyzer.get_complexity_report(results)
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
