# 主程序，整合Git分析、代码分析和可视化模块，并生成完整的项目分析报告。

import argparse
import os
from datetime import datetime

from code_analyzer import CodeAnalyzer
from git_analyzer import GitAnalyzer
from visualizer import Visualizer


def run_git_analysis(repo_path: str, output_dir: str, visualizer: Visualizer):
    """执行Git日志分析并生成相关可视化图表"""
    analyzer = GitAnalyzer(repo_path)

    commits = analyzer.get_all_commits()
    commits.to_csv(
        os.path.join(output_dir, "commits.csv"), index=False, encoding="utf-8-sig"
    )

    author_stats = analyzer.get_commit_stats_by_author(commits)
    author_stats.to_csv(
        os.path.join(output_dir, "author_stats.csv"), index=False, encoding="utf-8-sig"
    )

    yearly = analyzer.get_yearly_summary(commits)
    yearly.to_csv(
        os.path.join(output_dir, "yearly_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    churn = analyzer.calculate_code_churn(commits, window="M")
    churn.to_csv(
        os.path.join(output_dir, "code_churn.csv"), index=False, encoding="utf-8-sig"
    )

    weekday_stats, hour_stats = analyzer.get_commit_time_distribution(commits)

    # 生成可视化图表
    visualizer.plot_commit_timeline(commits, title="tqdm Commit Timeline")
    visualizer.plot_contributor_heatmap(commits, title="tqdm Contributor Activity")
    visualizer.plot_author_distribution(author_stats, title="tqdm Top Contributors")
    visualizer.plot_weekly_pattern(
        weekday_stats, hour_stats, title="tqdm Development Habits"
    )
    visualizer.plot_yearly_summary(yearly, title="tqdm Yearly Summary")
    visualizer.plot_code_churn(churn, title="tqdm Code Churn")

    # 公共汽车系数
    bus_trend = []
    for year in sorted(commits["date"].dt.year.unique()):
        year_commits = commits[commits["date"].dt.year == year]
        if len(year_commits) >= 10:
            year_bus = analyzer.calculate_bus_factor(year_commits)
            bus_trend.append(
                {"period": str(year), "bus_factor": year_bus["bus_factor"]}
            )
    if bus_trend:
        visualizer.plot_bus_factor_trend(bus_trend, title="tqdm Bus Factor Trend")

    return commits, author_stats, yearly


def run_code_analysis(repo_path: str, output_dir: str, visualizer: Visualizer):
    """执行静态代码分析并生成复杂度图表"""
    analyzer = CodeAnalyzer()
    tqdm_src = os.path.join(repo_path, "tqdm")

    analyses = analyzer.analyze_directory(tqdm_src)

    complexity_report = analyzer.get_complexity_report(analyses)
    complexity_report.to_csv(
        os.path.join(output_dir, "complexity_report.csv"),
        index=False,
        encoding="utf-8",
    )

    complex_funcs = analyzer.find_complex_functions(analyses, threshold=5)
    if not complex_funcs.empty:
        complex_funcs.to_csv(
            os.path.join(output_dir, "complex_functions.csv"),
            index=False,
            encoding="utf-8",
        )

    visualizer.plot_file_complexity_comparison(
        complexity_report, title="tqdm File Complexity"
    )
    return complexity_report


def generate_report(output_dir: str, commits, author_stats, yearly, complexity):
    """生成Markdown分析报告"""
    report_lines = [
        "# tqdm 开源项目分析报告",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 概览",
        "",
        f"- 首次提交: {commits['date'].min().strftime('%Y-%m-%d')}",
        f"- 最新提交: {commits['date'].max().strftime('%Y-%m-%d')}",
        f"- 总提交: {len(commits)}",
        f"- 贡献者: {len(author_stats)}",
        "",
        "## 2. Top 10 贡献者",
        "",
        "| 排名 | 贡献者 | 提交数 | 代码量 |",
        "| --- | --- | --- | --- |",
    ]
    for i, row in author_stats.head(10).iterrows():
        report_lines.append(
            f"| {i + 1} | {row['author']} | {row['commits']} | {row['total_lines']} |"
        )

    report_lines.extend(
        [
            "",
            "## 3. 年度发展",
            "",
            "| 年份 | 提交 | 贡献者 | 新增 | 删除 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in yearly.iterrows():
        report_lines.append(
            f"| {row['year']} | {row['commits']} | {row['unique_authors']} | {row['insertions']} | {row['deletions']} |"
        )

    if complexity is not None:
        report_lines.extend(
            [
                "",
                "## 4. 复杂度 Top 10",
                "",
                "| 文件 | 行数 | 平均 | 最高 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for _, row in complexity.nlargest(10, "lines_of_code").iterrows():
            report_lines.append(
                f"| {row['file']} | {row['lines_of_code']} | {row['avg_complexity']} | {row['max_complexity']} |"
            )

    with open(
        os.path.join(output_dir, "analysis_report.md"), "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(report_lines))


def main():
    parser = argparse.ArgumentParser(description="tqdm分析工具")
    parser.add_argument("--repo", "-r", default="../tqdm", help="仓库路径")
    parser.add_argument("--output", "-o", default="./output", help="输出目录")
    parser.add_argument("--skip-git", action="store_true", help="跳过Git分析")
    parser.add_argument("--skip-code", action="store_true", help="跳过代码分析")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    visualizer = Visualizer(output_dir=args.output)
    commits = author_stats = yearly = complexity = None

    if not args.skip_git:
        commits, author_stats, yearly = run_git_analysis(
            args.repo, args.output, visualizer
        )
    if not args.skip_code:
        complexity = run_code_analysis(args.repo, args.output, visualizer)
    if commits is not None:
        generate_report(args.output, commits, author_stats, yearly, complexity)

    print(f"\n分析已完成，结果保存在: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
